import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler
import configparser
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm  # 用于字体管理
import seaborn as sns
from Orion_model import make_model
from tqdm import tqdm
import torch.distributed as dist
import torch.multiprocessing as mp
import argparse
import logging
from datetime import datetime

# 设置日志（输出到终端和文件）
# 设置日志（输出到终端和文件）
def setup_logging(config_file, local_rank):
    config = configparser.ConfigParser()
    config.read(config_file)
    dataset_name = config['Data']['dataset_name']
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = '/root/python_on_hyy/Orion/results/测试日志'
    os.makedirs(log_dir, exist_ok=True)
    log_filename = os.path.join(log_dir, f'{dataset_name}_{current_time}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_filename)
        ]
    )
    logger = logging.getLogger(__name__)
    if local_rank == 0:
        logger.info(f"Logging to file: {log_filename}")
    return logger

# 在全局作用域定义 logger
logger = None  # 初始值为 None，稍后由 setup_logging 设置

# 清理 GPU 缓存
torch.cuda.empty_cache()

# 设置 matplotlib 的字体，优先使用 Arial，如果不可用则回退到系统字体
def setup_matplotlib_fonts():
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    if 'Arial' in available_fonts:
        plt.rcParams['font.family'] = 'Arial'
        logger.info("Using Arial font for matplotlib.")
    else:
        fallback_fonts = ['DejaVu Sans', 'Liberation Sans', 'sans-serif']
        for font in fallback_fonts:
            if font in available_fonts:
                plt.rcParams['font.family'] = font
                logger.info(f"Arial font not found. Using {font} instead.")
                break
        else:
            logger.warning("Arial and common fallback fonts not found. Using matplotlib default font.")
            plt.rcParams['font.family'] = 'sans-serif'

def setup_distributed(local_rank, world_size):
    try:
        dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=local_rank)
        torch.cuda.set_device(local_rank)
        logger.info(f"Process {local_rank}/{world_size}: Distributed environment initialized successfully.")
    except Exception as e:
        logger.error(f"Process {local_rank}: Failed to initialize distributed environment: {e}")
        raise

def cleanup_distributed():
    try:
        if dist.is_initialized():
            logger.info("Cleaning up distributed environment...")
            dist.barrier()
            dist.destroy_process_group()
            logger.info("Distributed environment cleaned up successfully.")
    except Exception as e:
        logger.error(f"Failed to clean up distributed environment: {e}")
        raise

def masked_mae_loss(y_pred, y_true, mask=None, use_loss_mask=False):
    if use_loss_mask and mask is not None:
        mask = mask.float()
        mask = mask / (mask.mean(dim=(0, 2), keepdim=True).clamp(min=1e-6))
    else:
        mask = (y_true != 0).float()
        mask = mask / (mask.mean(dim=(0, 2), keepdim=True).clamp(min=1e-6))

    loss = torch.abs(y_pred - y_true)
    loss = loss * mask
    loss[torch.isnan(loss)] = 0
    return loss

def evaluate(model, data_loader, DEVICE, mean_target, std_target, num_nodes, num_for_predict, phase="test", metric_method="mask", local_rank=0):
    model.eval()
    with torch.no_grad():
        loss_per_node = torch.zeros(num_nodes).to(DEVICE)
        mae_per_node = torch.zeros(num_nodes).to(DEVICE)
        rmse_per_node = torch.zeros(num_nodes).to(DEVICE)
        mape_per_node = torch.zeros(num_nodes).to(DEVICE)
        # 新增：按时间步存储 MAE、RMSE、MAPE
        mae_per_step = torch.zeros(num_for_predict).to(DEVICE)
        rmse_per_step = torch.zeros(num_for_predict).to(DEVICE)
        mape_per_step = torch.zeros(num_for_predict).to(DEVICE)
        total_samples = 0

        for batch_idx, batch in enumerate(tqdm(data_loader, desc=f"Evaluating ({phase})", disable=local_rank != 0)):
            x_h, x_d, x_w, target = batch[:4]
            batch_mask = batch[4] if len(batch) > 4 else None

            x_h, x_d, x_w, target = (x_h.to(DEVICE), x_d.to(DEVICE),
                                     x_w.to(DEVICE), target.to(DEVICE))
            if batch_mask is not None:
                batch_mask = batch_mask.to(DEVICE)

            y_pred = model(x_h, x_d, x_w)

            y_pred_raw = y_pred * std_target.to(DEVICE) + mean_target.to(DEVICE)
            y_true_raw = target * std_target.to(DEVICE) + mean_target.to(DEVICE)
            y_true_raw = torch.clamp(y_true_raw, min=0)
            y_pred_raw = torch.clamp(y_pred_raw, min=0)

            if batch_idx == 0 and local_rank == 0:
                logger.info(f"Eval batch {batch_idx}: y_pred min: {y_pred.min().item():.6f}, max: {y_pred.max().item():.6f}")
                logger.info(f"Eval batch {batch_idx}: target min: {target.min().item():.6f}, max: {target.max().item():.6f}")
                logger.info(f"Eval batch {batch_idx}: y_true_raw min: {y_true_raw.min().item():.6f}, max: {y_true_raw.max().item():.6f}")
                logger.info(f"Eval batch {batch_idx}: y_pred_raw min: {y_pred_raw.min().item():.6f}, max: {y_pred_raw.max().item():.6f}")

            if metric_method == "mask" and batch_mask is not None:
                mask = batch_mask.float()
                mask = mask / (mask.mean(dim=(0, 2), keepdim=True).clamp(min=1e-6))
            else:
                mask = (y_true_raw != 0).float()
                mask = mask / (mask.mean(dim=(0, 2), keepdim=True).clamp(min=1e-6))

            # 原有计算：总体 MAE、RMSE、MAPE（所有 12 步一起计算）
            batch_loss = torch.abs(y_pred_raw - y_true_raw) * mask
            batch_loss[torch.isnan(batch_loss)] = 0
            batch_loss = batch_loss.mean(dim=(0, 2))
            loss_per_node += batch_loss * y_pred.shape[0]

            batch_mae = (torch.abs(y_pred_raw - y_true_raw) * mask).sum(dim=(0, 2)) / mask.sum(dim=(0, 2)).clamp(min=1e-6)
            mae_per_node += batch_mae * y_pred.shape[0]

            batch_rmse = torch.sqrt(((y_pred_raw - y_true_raw) ** 2 * mask).sum(dim=(0, 2)) / mask.sum(dim=(0, 2)).clamp(min=1e-6))
            rmse_per_node += batch_rmse * y_pred.shape[0]

            mape_mask = (y_true_raw != 0).float()
            non_zero_counts = mape_mask.sum(dim=(0, 2))
            mape_value = torch.zeros(num_nodes).to(DEVICE)
            valid_nodes = non_zero_counts > 0
            if valid_nodes.any():
                mape_value[valid_nodes] = (
                    torch.abs((y_pred_raw - y_true_raw) / (y_true_raw + 1e-6)) * mape_mask
                ).sum(dim=(0, 2))[valid_nodes] / non_zero_counts[valid_nodes].clamp(min=1e-6)
            mape_per_node += mape_value * y_pred.shape[0]

            # 新增：按时间步计算 MAE、RMSE、MAPE
            for t in range(num_for_predict):
                # MAE
                batch_mae_t = (torch.abs(y_pred_raw[:, :, t] - y_true_raw[:, :, t]) * mask[:, :, t]).sum() / mask[:, :, t].sum().clamp(min=1e-6)
                mae_per_step[t] += batch_mae_t * y_pred.shape[0]

                # RMSE
                batch_rmse_t = torch.sqrt(((y_pred_raw[:, :, t] - y_true_raw[:, :, t]) ** 2 * mask[:, :, t]).sum() / mask[:, :, t].sum().clamp(min=1e-6))
                rmse_per_step[t] += batch_rmse_t * y_pred.shape[0]

                # MAPE
                mape_mask_t = (y_true_raw[:, :, t] != 0).float()
                non_zero_count_t = mape_mask_t.sum()
                mape_value_t = 0
                if non_zero_count_t > 0:
                    mape_value_t = (
                        torch.abs((y_pred_raw[:, :, t] - y_true_raw[:, :, t]) / (y_true_raw[:, :, t] + 1e-6)) * mape_mask_t
                    ).sum() / non_zero_count_t.clamp(min=1e-6)
                mape_per_step[t] += mape_value_t * y_pred.shape[0]

            if batch_idx == 0 and local_rank == 0:
                non_zero_ratios = non_zero_counts / (y_pred.shape[0] * y_pred.shape[2])
                logger.info(f"Non-zero counts per node: {non_zero_counts.cpu().numpy()}")
                logger.info(f"Non-zero ratios per node: {non_zero_ratios.cpu().numpy()}")

            total_samples += y_pred.shape[0]

        if dist.is_initialized():
            total_samples_tensor = torch.tensor(total_samples, device=DEVICE)
            dist.all_reduce(total_samples_tensor, op=dist.ReduceOp.SUM)
            total_samples = total_samples_tensor.item()

            dist.all_reduce(loss_per_node, op=dist.ReduceOp.SUM)
            dist.all_reduce(mae_per_node, op=dist.ReduceOp.SUM)
            dist.all_reduce(rmse_per_node, op=dist.ReduceOp.SUM)
            dist.all_reduce(mape_per_node, op=dist.ReduceOp.SUM)
            # 同步每一步的指标
            dist.all_reduce(mae_per_step, op=dist.ReduceOp.SUM)
            dist.all_reduce(rmse_per_step, op=dist.ReduceOp.SUM)
            dist.all_reduce(mape_per_step, op=dist.ReduceOp.SUM)

        loss_per_node = loss_per_node / total_samples
        mae_per_node = mae_per_node / total_samples
        rmse_per_node = rmse_per_node / total_samples
        mape_per_node = mape_per_node / total_samples * 100

        # 每一步的指标
        mae_per_step = mae_per_step / total_samples
        rmse_per_step = rmse_per_step / total_samples
        mape_per_step = mape_per_step / total_samples * 100

        loss = loss_per_node.mean().item()
        mae = mae_per_node.mean().item()
        rmse = rmse_per_node.mean().item()
        mape_values = mape_per_node.cpu().numpy()
        valid_mape = mape_values[~np.isnan(mape_values)]
        mape = valid_mape.mean() if len(valid_mape) > 0 else float('nan')

    return (loss, mae, rmse, mape,
            loss_per_node.cpu().numpy(), mae_per_node.cpu().numpy(),
            rmse_per_node.cpu().numpy(), mape_per_node.cpu().numpy(),
            mae_per_step.cpu().numpy(), rmse_per_step.cpu().numpy(), mape_per_step.cpu().numpy())

def load_data(config_file, use_distributed=False, local_rank=0, world_size=1):
    config = configparser.ConfigParser()
    config.read(config_file)

    data_config = config['Data']
    training_config = config['Training']

    graph_signal_matrix_filename = data_config['graph_signal_matrix_filename']
    num_of_vertices = int(data_config['num_of_vertices'])
    points_per_hour = int(data_config['points_per_hour'])
    num_for_predict = int(data_config['num_for_predict'])
    num_of_weeks = int(training_config['num_of_weeks'])
    num_of_days = int(training_config['num_of_days'])
    num_of_hours = int(training_config['num_of_hours'])
    num_workers = int(training_config.get('num_workers', 8))

    file = os.path.basename(graph_signal_matrix_filename).split('.')[0]
    dirpath = os.path.dirname(graph_signal_matrix_filename)
    filename = os.path.join(dirpath, f'{file}_r{num_of_hours}_d{num_of_days}_w{num_of_weeks}_Orion.npz')
    if local_rank == 0:
        logger.info(f"Loading data from: {filename}")

    data = np.load(filename)

    test_x_h = torch.FloatTensor(data['test_x_h'])
    test_x_d = torch.FloatTensor(data['test_x_d'])
    test_x_w = torch.FloatTensor(data['test_x_w'])
    test_target = torch.FloatTensor(data['test_target'])
    test_mask = torch.FloatTensor(data['test_mask'])

    adj_matrix = torch.FloatTensor(data['adj_matrix'])

    mean_flow = torch.FloatTensor(data['mean_flow'])
    std_flow = torch.FloatTensor(data['std_flow'])
    mean_occupancy = torch.FloatTensor(data['mean_occupancy'])
    std_occupancy = torch.FloatTensor(data['std_occupancy'])
    mean_speed = torch.FloatTensor(data['mean_speed'])
    std_speed = torch.FloatTensor(data['std_speed'])
    mean_target = torch.FloatTensor(data['mean_target'])
    std_target = torch.FloatTensor(data['std_target'])

    batch_size = int(training_config['batch_size'])

    test_dataset = TensorDataset(test_x_h, test_x_d, test_x_w, test_target, test_mask)

    if use_distributed:
        test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=local_rank, shuffle=False)
    else:
        test_sampler = None

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             sampler=test_sampler, num_workers=num_workers, pin_memory=True)

    stats = {
        'mean_flow': mean_flow,
        'std_flow': std_flow,
        'mean_occupancy': mean_occupancy,
        'std_occupancy': std_occupancy,
        'mean_speed': mean_speed,
        'std_speed': std_speed,
        'mean_target': mean_target,
        'std_target': std_target
    }

    return test_loader, num_of_vertices, num_for_predict, stats, adj_matrix

def test_and_visualize(local_rank, world_size, config_file, DEVICE):
    if world_size > 1:
        setup_distributed(local_rank, world_size)

    # 设置绘图样式
    sns.set_style("whitegrid")
    setup_matplotlib_fonts()  # 设置字体
    plt.rcParams['font.size'] = 14
    plt.rcParams['axes.labelsize'] = 16
    plt.rcParams['axes.titlesize'] = 18
    plt.rcParams['legend.fontsize'] = 14
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['axes.linewidth'] = 1.2
    plt.rcParams['grid.linestyle'] = '--'
    plt.rcParams['grid.alpha'] = 0.7

    config = configparser.ConfigParser()
    config.read(config_file)

    data_config = config['Data']
    training_config = config['Training']
    Orion_config = config['Orion']
    test_config = config['Test']

    visualize_nodes = training_config.getint('visualize_nodes', -1)
    use_distributed = training_config.getboolean('use_distributed', False)

    num_of_vertices = int(data_config['num_of_vertices'])
    num_for_predict = int(data_config['num_for_predict'])
    len_input = int(data_config['len_input'])
    in_channels = int(training_config['in_channels'])
    points_per_hour = int(data_config['points_per_hour'])
    num_of_hours = int(training_config['num_of_hours'])
    num_of_days = int(training_config['num_of_days'])
    num_of_weeks = int(training_config['num_of_weeks'])

    nb_block = int(training_config['nb_block'])
    num_heads_spatial = int(Orion_config['num_heads_spatial'])
    num_heads_temporal = int(Orion_config['num_heads_temporal'])
    num_heads_fusion = int(Orion_config['num_heads_fusion'])
    spatial_dropout = float(Orion_config['spatial_dropout'])
    temporal_dropout = float(Orion_config['temporal_dropout'])
    fusion_dropout = float(Orion_config['fusion_dropout'])
    ff_dropout = float(Orion_config['ff_dropout'])
    ff_hidden_dim = int(Orion_config['ff_hidden_dim'])

    results_path = test_config['results_path']
    use_best_model = test_config.getboolean('use_best_model', True)
    node_id = int(test_config['node_id'])
    save_attention_weights = Orion_config.getboolean('save_attention_weights')
    attention_visualization_path = Orion_config['attention_visualization_path']
    use_metric_mask = training_config.getboolean('use_metric_mask', False)
    metric_method = "mask" if use_metric_mask else "unmask"

    test_loader, num_of_vertices, num_for_predict, stats, adj_matrix = load_data(
        config_file, use_distributed=use_distributed, local_rank=local_rank, world_size=world_size
    )

    if local_rank == 0:
        logger.info(f"mean_flow: {stats['mean_flow'].item()}")
        logger.info(f"std_flow: {stats['std_flow'].item()}")
        logger.info(f"mean_occupancy: {stats['mean_occupancy'].item()}")
        logger.info(f"std_occupancy: {stats['std_occupancy'].item()}")
        logger.info(f"mean_speed: {stats['mean_speed'].item()}")
        logger.info(f"std_speed: {stats['std_speed'].item()}")
        logger.info(f"mean_target shape: {stats['mean_target'].shape}")
        logger.info(f"std_target shape: {stats['std_target'].shape}")

    model = make_model(DEVICE, in_channels=in_channels, out_channels=64, num_nodes=num_of_vertices,
                       num_timesteps_h=len_input, num_timesteps_d=len_input, num_timesteps_w=len_input,
                       num_for_predict=num_for_predict, nb_block=nb_block,
                       num_heads_spatial=num_heads_spatial, num_heads_temporal=num_heads_temporal,
                       num_heads_fusion=num_heads_fusion, spatial_dropout=spatial_dropout,
                       temporal_dropout=temporal_dropout, fusion_dropout=fusion_dropout,
                       ff_dropout=ff_dropout, ff_hidden_dim=ff_hidden_dim, adj_matrix=adj_matrix,
                       use_distributed=use_distributed, local_rank=local_rank)

    if use_best_model:
        model_path = os.path.join(results_path, 'best_model.pth')
        if local_rank == 0:
            logger.info(f"Loading best model from: {model_path}")
        state_dict = torch.load(model_path, map_location=DEVICE, weights_only=True)
    else:
        checkpoint_path = os.path.join(results_path, 'checkpoint.pth')
        if local_rank == 0:
            logger.info(f"Loading last checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        state_dict = checkpoint['model_state_dict']

    if use_distributed:
        model.module.load_state_dict(state_dict)
    else:
        model.load_state_dict(state_dict)

    model.eval()

    (test_loss, test_mae, test_rmse, test_mape,
     test_loss_per_node, test_mae_per_node,
     test_rmse_per_node, test_mape_per_node,
     mae_per_step, rmse_per_step, mape_per_step) = evaluate(
        model, test_loader, DEVICE, stats['mean_target'], stats['std_target'], num_of_vertices, num_for_predict,
        phase="test", metric_method=metric_method, local_rank=local_rank
    )

    if local_rank == 0:
        # 原有输出：总体 12 步的 MAE、RMSE、MAPE
        logger.info(f"Test Loss: {test_loss:.4f}, MAE: {test_mae:.4f}, RMSE: {test_rmse:.4f}, MAPE: {test_mape:.2f}%")

        # 新增输出：每一步的 MAE、RMSE、MAPE
        logger.info("\n=== Per-Step Evaluation Metrics ===")
        for t in range(num_for_predict):
            logger.info(f"Evaluate best model on test data for horizon {t+1}, "
                        f"Test MAE: {mae_per_step[t]:.4f}, "
                        f"Test MAPE: {mape_per_step[t]:.2f}%, "
                        f"Test RMSE: {rmse_per_step[t]:.4f}")

        # 新增输出：12 步的平均 MAE、RMSE、MAPE
        avg_mae = np.mean(mae_per_step)
        avg_rmse = np.mean(rmse_per_step)
        avg_mape = np.mean(mape_per_step)
        logger.info(f"On average over {num_for_predict} horizons, "
                    f"Test MAE: {avg_mae:.4f}, "
                    f"Test MAPE: {avg_mape:.2f}%, "
                    f"Test RMSE: {avg_rmse:.4f}")

        # 原有输出：每个节点的指标
        logger.info("\nTest Metrics per Node:")
        if node_id == -1:
            for node in range(num_of_vertices):
                logger.info(f"Node {node}: Loss: {test_loss_per_node[node]:.4f}, MAE: {test_mae_per_node[node]:.4f}, "
                            f"RMSE: {test_rmse_per_node[node]:.4f}, MAPE: {test_mape_per_node[node]:.2f}%")
        elif 0 <= node_id < num_of_vertices:
            logger.info(f"Node {node_id}: Loss: {test_loss_per_node[node_id]:.4f}, MAE: {test_mae_per_node[node_id]:.4f}, "
                        f"RMSE: {test_rmse_per_node[node_id]:.4f}, MAPE: {test_mape_per_node[node_id]:.2f}%")
        else:
            logger.warning(f"Invalid node_id: {node_id}. Must be between 0 and {num_of_vertices-1} or -1 for all nodes.")

    all_preds = []
    all_labels = []
    all_inputs_h = []
    all_inputs_d = []
    all_inputs_w = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            x_h, x_d, x_w, target, batch_mask = batch
            x_h, x_d, x_w, target = (x_h.to(DEVICE), x_d.to(DEVICE),
                                     x_w.to(DEVICE), target.to(DEVICE))
            batch_mask = batch_mask.to(DEVICE)

            y_pred = model(x_h, x_d, x_w)

            y_pred_raw = y_pred * stats['std_target'].to(DEVICE) + stats['mean_target'].to(DEVICE)
            y_true_raw = target * stats['std_target'].to(DEVICE) + stats['mean_target'].to(DEVICE)
            y_pred_raw = torch.clamp(y_pred_raw, min=0)
            y_true_raw = torch.clamp(y_true_raw, min=0)

            # 修复语法错误：正确解包 stats 字典
            mean_flow = stats['mean_flow']
            std_flow = stats['std_flow']
            mean_occupancy = stats['mean_occupancy']
            std_occupancy = stats['std_occupancy']
            mean_speed = stats['mean_speed']
            std_speed = stats['std_speed']

            x_h_raw = torch.zeros_like(x_h)
            x_h_raw[:, :, 0, :] = x_h[:, :, 0, :] * std_flow.to(DEVICE) + mean_flow.to(DEVICE)
            if in_channels == 3:
                x_h_raw[:, :, 1, :] = x_h[:, :, 1, :] * std_occupancy.to(DEVICE) + mean_occupancy.to(DEVICE)
                x_h_raw[:, :, 2, :] = x_h[:, :, 2, :] * std_speed.to(DEVICE) + mean_speed.to(DEVICE)

            x_d_raw = torch.zeros_like(x_d)
            x_d_raw[:, :, 0, :] = x_d[:, :, 0, :] * std_flow.to(DEVICE) + mean_flow.to(DEVICE)
            if in_channels == 3:
                x_d_raw[:, :, 1, :] = x_d[:, :, 1, :] * std_occupancy.to(DEVICE) + mean_occupancy.to(DEVICE)
                x_d_raw[:, :, 2, :] = x_d[:, :, 2, :] * std_speed.to(DEVICE) + mean_speed.to(DEVICE)

            x_w_raw = torch.zeros_like(x_w)
            x_w_raw[:, :, 0, :] = x_w[:, :, 0, :] * std_flow.to(DEVICE) + mean_flow.to(DEVICE)
            if in_channels == 3:
                x_w_raw[:, :, 1, :] = x_w[:, :, 1, :] * std_occupancy.to(DEVICE) + mean_occupancy.to(DEVICE)
                x_w_raw[:, :, 2, :] = x_w[:, :, 2, :] * std_speed.to(DEVICE) + mean_speed.to(DEVICE)

            if batch_idx == 0:
                all_preds.append(y_pred_raw[0].cpu().numpy())
                all_labels.append(y_true_raw[0].cpu().numpy())
                all_inputs_h.append(x_h_raw[0].cpu().numpy())
                all_inputs_d.append(x_d_raw[0].cpu().numpy())
                all_inputs_w.append(x_w_raw[0].cpu().numpy())

            if local_rank == 0:
                logger.info("\n=== Verifying Cycle Data Extraction ===")
                logger.info(f"points_per_hour: {points_per_hour}, num_of_hours: {num_of_hours}, num_of_days: {num_of_days}, num_of_weeks: {num_of_weeks}")
                logger.info(f"len_input: {len_input}")

                timesteps_per_hour = points_per_hour
                timesteps_per_day = timesteps_per_hour * 24
                timesteps_per_week = timesteps_per_day * 7

                logger.info(f"\nNode {node_id} Flow Values (last timestep of each cycle):")
                logger.info(f"x_h (hourly, last timestep, should be current time): {x_h_raw[0, node_id, 0, -1]:.4f}")
                logger.info(f"x_d (daily, last timestep, should be {num_of_days} day(s) ago): {x_d_raw[0, node_id, 0, -1]:.4f}")
                logger.info(f"x_w (weekly, last timestep, should be {num_of_weeks} week(s) ago): {x_w_raw[0, node_id, 0, -1]:.4f}")

                logger.info(f"\nx_h flow (hourly cycle, last {len_input} timesteps):")
                logger.info(x_h_raw[0, node_id, 0, :].cpu().numpy())
                logger.info(f"x_d flow (daily cycle, {num_of_days} day(s) ago):")
                logger.info(x_d_raw[0, node_id, 0, :].cpu().numpy())
                logger.info(f"x_w flow (weekly cycle, {num_of_weeks} week(s) ago):")
                logger.info(x_w_raw[0, node_id, 0, :].cpu().numpy())

                if in_channels == 3:
                    logger.info(f"\nBatch {batch_idx}, occupy (raw, x_h): {x_h_raw[0, node_id, 1, :].cpu().numpy()}")
                    logger.info(f"Batch {batch_idx}, speed (raw, x_h): {x_h_raw[0, node_id, 2, :].cpu().numpy()}")

            if save_attention_weights:
                # 修改注意力可视化的保存路径
                dataset_name = data_config['dataset_name']
                current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
                test_attention_path = f"/root/python_on_hyy/Orion/results/visualization/test_heat_map/{dataset_name}_{current_time}"
                os.makedirs(test_attention_path, exist_ok=True)
                try:
                    logger.info(f"use_distributed: {use_distributed}")
                    logger.info(f"torch.distributed.is_initialized(): {torch.distributed.is_initialized()}")
                    logger.info(f"Model type: {type(model)}")
                    if use_distributed:
                        logger.info(f"Model.module type: {type(model.module)}")
                        has_interpret = hasattr(model.module, 'interpret')
                        logger.info(f"Model.module has interpret method: {has_interpret}")
                        if not has_interpret:
                            logger.warning("Available methods in model.module: " + str([attr for attr in dir(model.module) if not attr.startswith('_')]))
                        else:
                            model.module.interpret(
                                x_h[:1], x_d[:1], x_w[:1],
                                test_attention_path,
                                epoch=0,
                                config=config,
                                visualize_nodes=visualize_nodes,
                                local_rank=local_rank
                            )
                    else:
                        has_interpret = hasattr(model, 'interpret')
                        logger.info(f"Model has interpret method: {has_interpret}")
                        if not has_interpret:
                            logger.warning("Available methods in model: " + str([attr for attr in dir(model) if not attr.startswith('_')]))
                        else:
                            model.interpret(
                                x_h[:1], x_d[:1], x_w[:1],
                                test_attention_path,
                                epoch=0,
                                config=config,
                                visualize_nodes=visualize_nodes,
                                local_rank=local_rank
                            )
                    logger.info(f"Attention weights visualization saved to {test_attention_path}")
                except AttributeError as e:
                    logger.error(f"Error calling interpret during test: {e}")
                    logger.error("Ensure that the Orion class has an 'interpret' method defined.")
                    raise
                except Exception as e:
                    logger.error(f"Unexpected error during test interpret: {e}")
                    raise

            break

    if local_rank == 0:
        pred_sample = all_preds[0]
        label_sample = all_labels[0]
        input_h_sample = all_inputs_h[0]
        input_d_sample = all_inputs_d[0]
        input_w_sample = all_inputs_w[0]

        logger.info(f"\nNode {node_id} x_h last timestep (flow): {input_h_sample[node_id, 0, -1]}")
        logger.info(f"Node {node_id} Ground Truth first timestep: {label_sample[node_id, 0]}")

        logger.info("12-Step Predictions for Node 0:")
        logger.info(pred_sample[0])
        logger.info("12-Step Labels for Node 0:")
        logger.info(label_sample[0])

        logger.info("12-Step Predictions for Node 5:")
        logger.info(pred_sample[5])
        logger.info("12-Step Labels for Node 5:")
        logger.info(label_sample[5])

        # 修改折线图可视化的保存路径
        dataset_name = data_config['dataset_name']
        current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = f"/root/python_on_hyy/Orion/results/visualization/test_line_chart/{dataset_name}_{current_time}"
        os.makedirs(output_dir, exist_ok=True)

        colors = {
            'flow': '#1f77b4',
            'predicted': '#ff4d4f',
        }

        nodes_to_visualize = range(num_of_vertices) if node_id == -1 else [node_id]

        for node in nodes_to_visualize:
            if not (0 <= node < num_of_vertices):
                logger.warning(f"Skipping invalid node_id: {node}. Must be between 0 and {num_of_vertices-1}.")
                continue

            fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

            time_steps_input = np.array(range(-len_input, 0))
            time_steps_pred = np.array(range(0, num_for_predict))

            flow_combined = np.concatenate([input_h_sample[node, 0, :], label_sample[node]])
            time_combined = np.concatenate([time_steps_input, time_steps_pred])

            ax.plot(time_combined, flow_combined,
                    color=colors['flow'], linestyle='-', linewidth=2.5, label='Ground Truth Flow', marker='o', markersize=5)
            ax.plot(time_steps_pred, pred_sample[node],
                    color=colors['predicted'], linestyle='--', linewidth=2.5, label='Predicted Flow', marker='^', markersize=5)

            ax.set_xlabel('Time Step', fontweight='bold')
            ax.set_ylabel('Flow', fontweight='bold')
            ax.set_title(f'Flow Prediction for Node {node}', pad=15, fontweight='bold')

            flow_min = min(flow_combined.min(), pred_sample[node].min())
            flow_max = max(flow_combined.max(), pred_sample[node].max())
            flow_margin = (flow_max - flow_min) * 0.2 if flow_max > flow_min else 0.1
            ax.set_ylim(flow_min - flow_margin, flow_max + flow_margin)
            ax.set_yticks(np.linspace(flow_min - flow_margin, flow_max + flow_margin, 6))

            ax.grid(True, linestyle='--', alpha=0.7)

            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=True, edgecolor='black')

            plt.tight_layout()

            output_path = os.path.join(output_dir, f'node_{node}_flow_prediction.png')
            plt.savefig(output_path, bbox_inches='tight')
            plt.close()
            logger.info(f"Saved Flow visualization for Node {node} at: {output_path}")

    if world_size > 1:
        cleanup_distributed()

def main():
    parser = argparse.ArgumentParser(description="Test script for Orion model")
    parser.add_argument("--config", default='configurations/Orion_config.conf', type=str,
                        help="Path to the configuration file (default: configurations/Orion_config.conf)")
    parser.add_argument("--local-rank", type=int, default=int(os.environ.get('LOCAL_RANK', 0)),
                        help="Local rank for distributed training (default: 0 or from env LOCAL_RANK)")
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(args.config)

    training_config = config['Training']
    use_distributed = training_config.getboolean('use_distributed', False)

    local_rank = int(os.environ.get('LOCAL_RANK', args.local_rank))
    world_size = int(os.environ.get('WORLD_SIZE', 1))

    # 设置 logger
    global logger
    logger = setup_logging(args.config, local_rank)

    if local_rank >= torch.cuda.device_count():
        logger.error(f"local_rank {local_rank} is greater than available GPUs {torch.cuda.device_count()}")
        raise ValueError(f"local_rank {local_rank} is greater than available GPUs {torch.cuda.device_count()}")
    DEVICE = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if local_rank == 0:
        logger.info(f"Using device: {DEVICE}, World size: {world_size}")

    if world_size > 1 and use_distributed:
        if 'MASTER_ADDR' not in os.environ:
            os.environ['MASTER_ADDR'] = 'localhost'
        if 'MASTER_PORT' not in os.environ:
            os.environ['MASTER_PORT'] = '12355'
        test_and_visualize(local_rank, world_size, args.config, DEVICE)
    else:
        test_and_visualize(local_rank, world_size, args.config, DEVICE)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Main process failed with exception: {e}")
        raise
    finally:
        if 'WORLD_SIZE' in os.environ and int(os.environ['WORLD_SIZE']) > 1:
            cleanup_distributed()
        torch.cuda.empty_cache()
        logger.info("Program exited, GPU memory cleared.")