import os
import signal
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler
import argparse
import configparser
import time
from Orion_model import make_model
import matplotlib.pyplot as plt
from torch import amp
from tqdm import tqdm
import csv
import torch.distributed as dist
import torch.multiprocessing as mp
import logging
from datetime import datetime

# 设置日志（输出到终端和文件）
# 设置日志（输出到终端和文件）
def setup_logging(config_file, local_rank):
    config = configparser.ConfigParser()
    config.read(config_file)
    dataset_name = config['Data']['dataset_name']
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = '/root/python_on_hyy/Orion/results/训练日志'
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
torch.serialization.add_safe_globals({'torch': torch})

def setup_distributed(local_rank, world_size):
    """
    初始化分布式训练环境。
    
    Args:
        local_rank (int): 当前进程的本地 rank。
        world_size (int): 总进程数（GPU 数量）。
    """
    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=local_rank)
    torch.cuda.set_device(local_rank)
    logger.info(f"Process {local_rank}/{world_size}: Distributed environment initialized successfully.")

def cleanup_distributed():
    """
    清理分布式训练环境。
    """
    try:
        if dist.is_initialized():
            logger.info("Cleaning up distributed environment...")
            # 移除 dist.barrier()，直接销毁进程组
            dist.destroy_process_group()
            logger.info("Distributed environment cleaned up successfully.")
    except Exception as e:
        logger.error(f"Failed to clean up distributed environment: {e}")
        # 不抛出异常，确保进程可以继续退出

def shutdown_dataloader(dataloader):
    """
    关闭 DataLoader 的工作进程。
    
    Args:
        dataloader (DataLoader): 需要关闭的 DataLoader 对象。
    """
    if dataloader is not None:
        if hasattr(dataloader, '_iterator') and dataloader._iterator is not None:
            dataloader._iterator._shutdown_workers()
        if hasattr(dataloader, 'sampler') and isinstance(dataloader.sampler, DistributedSampler):
            dataloader.sampler.set_epoch(0)  # 重置 sampler 状态

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

def evaluate(model, data_loader, DEVICE, mean_target, std_target, num_nodes, num_for_predict, phase="val", metric_method="mask", local_rank=0):
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
            x_h, x_d, x_w, target, batch_mask = batch
            x_h, x_d, x_w, target = x_h.to(DEVICE), x_d.to(DEVICE), x_w.to(DEVICE), target.to(DEVICE)
            batch_mask = batch_mask.to(DEVICE)
            
            y_pred = model(x_h, x_d, x_w)
            
            y_pred_raw = y_pred * std_target.to(DEVICE) + mean_target.to(DEVICE)
            y_true_raw = target * std_target.to(DEVICE) + mean_target.to(DEVICE)
            y_true_raw = torch.clamp(y_true_raw, min=0)
            y_pred_raw = torch.clamp(y_pred_raw, min=0)  # 优化：移除 max 限制
            
            if batch_idx == 0 and local_rank == 0:
                logger.info(f"Eval batch {batch_idx}: y_pred min: {y_pred.min().item():.6f}, max: {y_pred.max().item():.6f}")
                logger.info(f"Eval batch {batch_idx}: target min: {target.min().item():.6f}, max: {target.max().item():.6f}")
                logger.info(f"Eval batch {batch_idx}: y_true_raw min: {y_true_raw.min().item():.6f}, max: {y_true_raw.max().item():.6f}")
                logger.info(f"Eval batch {batch_idx}: y_pred_raw min: {y_pred_raw.min().item():.6f}, max: {y_pred_raw.max().item():.6f}")
            
            if metric_method == "mask":
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
        
        # 分布式训练中，同步所有进程的指标
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
    shuffle_data = data_config.getboolean('shuffle_data', True)  # 新增：是否打乱数据
    
    num_of_weeks = int(training_config['num_of_weeks'])
    num_of_days = int(training_config['num_of_days'])
    num_of_hours = int(training_config['num_of_hours'])
    num_workers = int(training_config.get('num_workers', 8))  # 从配置文件读取，默认为 8
    
    file66 = os.path.basename(graph_signal_matrix_filename).split('.')[0]
    dirpath = os.path.dirname(graph_signal_matrix_filename)
    filename = os.path.join(dirpath, f'{file66}_r{num_of_hours}_d{num_of_days}_w{num_of_weeks}_Orion.npz')
    if local_rank == 0:
        logger.info(f"Loading data from: {filename}")
    
    data = np.load(filename)
    
    train_x_h = torch.FloatTensor(data['train_x_h'])
    train_x_d = torch.FloatTensor(data['train_x_d'])
    train_x_w = torch.FloatTensor(data['train_x_w'])
    train_target = torch.FloatTensor(data['train_target'])
    train_mask = torch.FloatTensor(data['train_mask'])
    
    val_x_h = torch.FloatTensor(data['val_x_h'])
    val_x_d = torch.FloatTensor(data['val_x_d'])
    val_x_w = torch.FloatTensor(data['val_x_w'])
    val_target = torch.FloatTensor(data['val_target'])
    val_mask = torch.FloatTensor(data['val_mask'])
    
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
    
    test_sample_sizes = [t.shape[0] for t in [test_x_h, test_x_d, test_x_w, test_target, test_mask]]
    min_test_samples = min(test_sample_sizes)
    if not all(size == test_sample_sizes[0] for size in test_sample_sizes):
        if local_rank == 0:
            logger.info(f"Test tensor sizes mismatch: {test_sample_sizes}. Adjusting to minimum size: {min_test_samples}")
        test_x_h = test_x_h[:min_test_samples]
        test_x_d = test_x_d[:min_test_samples]
        test_x_w = test_x_w[:min_test_samples]
        test_target = test_target[:min_test_samples]
        test_mask = test_mask[:min_test_samples]
    
    train_sample_sizes = [t.shape[0] for t in [train_x_h, train_x_d, train_x_w, train_target, train_mask]]
    min_train_samples = min(train_sample_sizes)
    if not all(size == train_sample_sizes[0] for size in train_sample_sizes):
        if local_rank == 0:
            logger.info(f"Train tensor sizes mismatch: {train_sample_sizes}. Adjusting to minimum size: {min_train_samples}")
        train_x_h = train_x_h[:min_train_samples]
        train_x_d = train_x_d[:min_train_samples]
        train_x_w = train_x_w[:min_train_samples]
        train_target = train_target[:min_train_samples]
        train_mask = train_mask[:min_train_samples]
    
    val_sample_sizes = [t.shape[0] for t in [val_x_h, val_x_d, val_x_w, val_target, val_mask]]
    min_val_samples = min(val_sample_sizes)
    if not all(size == val_sample_sizes[0] for size in val_sample_sizes):
        if local_rank == 0:
            logger.info(f"Val tensor sizes mismatch: {val_sample_sizes}. Adjusting to minimum size: {min_val_samples}")
        val_x_h = val_x_h[:min_val_samples]
        val_x_d = val_x_d[:min_val_samples]
        val_x_w = val_x_w[:min_val_samples]
        val_target = val_target[:min_val_samples]
        val_mask = val_mask[:min_val_samples]
    
    train_dataset = TensorDataset(train_x_h, train_x_d, train_x_w, train_target, train_mask)
    val_dataset = TensorDataset(val_x_h, val_x_d, val_x_w, val_target, val_mask)
    test_dataset = TensorDataset(test_x_h, test_x_d, test_x_w, test_target, test_mask)
    
    if use_distributed:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=local_rank, shuffle=shuffle_data)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=local_rank, shuffle=False)
        test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=local_rank, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None
        test_sampler = None
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=(train_sampler is None and shuffle_data),
                              sampler=train_sampler, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            sampler=val_sampler, num_workers=num_workers, pin_memory=True)
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
    
    return (train_loader, val_loader, test_loader, num_of_vertices, 
            num_for_predict, points_per_hour, num_of_hours, num_of_days, num_of_weeks,
            stats, adj_matrix)

def train(local_rank, world_size, config_file, DEVICE):
    if world_size > 1:
        setup_distributed(local_rank, world_size)
    
    # 设置信号处理
    def signal_handler(sig, frame, train_loader, val_loader, test_loader):
        logger.info(f"Process {local_rank}: Received signal {sig}, cleaning up...")
        # 关闭 DataLoader 的工作进程
        shutdown_dataloader(train_loader)
        shutdown_dataloader(val_loader)
        shutdown_dataloader(test_loader)
        if world_size > 1:
            cleanup_distributed()
        torch.cuda.empty_cache()
        logger.info(f"Process {local_rank}: GPU memory cleared.")
        sys.exit(0)

    config = configparser.ConfigParser()
    config.read(config_file)
    
    data_config = config['Data']
    training_config = config['Training']
    Orion_config = config['Orion']
    test_config = config['Test']
    
    load_checkpoint = training_config.getboolean('load_checkpoint')
    seed = int(training_config['seed'])
    eval_node_id = int(training_config['eval_node_id'])
    use_loss_mask = training_config.getboolean('use_loss_mask', False)
    use_metric_mask = training_config.getboolean('use_metric_mask', False)
    visualize_nodes = training_config.getint('visualize_nodes', -1)
    metric_method = "mask" if use_metric_mask else "unmask"
    use_distributed = training_config.getboolean('use_distributed', False)
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    num_of_vertices = int(data_config['num_of_vertices'])
    num_for_predict = int(data_config['num_for_predict'])
    len_input = int(data_config['len_input'])
    
    batch_size = int(training_config['batch_size'])
    learning_rate = float(training_config['learning_rate'])
    epochs = int(training_config['epochs'])
    start_epoch = int(training_config['start_epoch'])
    loss_function = training_config['loss_function']
    results_path = test_config['results_path']
    patience = int(training_config.get('patience', 10))
    
    in_channels = int(training_config['in_channels'])
    nb_block = int(training_config['nb_block'])
    num_heads_spatial = int(Orion_config['num_heads_spatial'])
    num_heads_temporal = int(Orion_config['num_heads_temporal'])
    num_heads_fusion = int(Orion_config['num_heads_fusion'])
    spatial_dropout = float(Orion_config['spatial_dropout'])
    temporal_dropout = float(Orion_config['temporal_dropout'])
    fusion_dropout = float(Orion_config['fusion_dropout'])
    ff_dropout = float(Orion_config['ff_dropout'])
    ff_hidden_dim = int(Orion_config['ff_hidden_dim'])
    # save_attention_weights = Orion_config.getboolean('save_attention_weights')  # 注释掉
    # attention_visualization_path = Orion_config['attention_visualization_path']  # 注释掉
    
    (train_loader, val_loader, test_loader, num_of_vertices, num_for_predict, _, _, _, _, 
     stats, adj_matrix) = load_data(config_file, use_distributed=use_distributed, local_rank=local_rank, world_size=world_size)
    
    if local_rank == 0:
        logger.info(f"Number of batches in train_loader: {len(train_loader)}")
    
    # 确保 adj_matrix 在正确的设备上
    adj_matrix = adj_matrix.to(DEVICE)
    
    # 创建模型（DDP 包装在 make_model 中完成）
    model = make_model(DEVICE, in_channels=in_channels, out_channels=64, num_nodes=num_of_vertices,
                       num_timesteps_h=len_input, num_timesteps_d=len_input, num_timesteps_w=len_input,
                       num_for_predict=num_for_predict, nb_block=nb_block,
                       num_heads_spatial=num_heads_spatial, num_heads_temporal=num_heads_temporal,
                       num_heads_fusion=num_heads_fusion, spatial_dropout=spatial_dropout,
                       temporal_dropout=temporal_dropout, fusion_dropout=fusion_dropout,
                       ff_dropout=ff_dropout, ff_hidden_dim=ff_hidden_dim, adj_matrix=adj_matrix,
                       use_distributed=use_distributed, local_rank=local_rank)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = masked_mae_loss if loss_function == 'mae' else nn.MSELoss()
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6
    )
    
    scaler = amp.GradScaler('cuda')
    
    checkpoint_path = os.path.join(results_path, 'checkpoint.pth')
    if not load_checkpoint and os.path.exists(checkpoint_path) and local_rank == 0:
        os.remove(checkpoint_path)
        logger.info(f"Deleted existing checkpoint: {checkpoint_path}")
    
    train_losses = []
    val_losses = []
    
    log_file_path = os.path.join(results_path, 'training_log.csv')
    if local_rank == 0 and not os.path.exists(log_file_path):
        with open(log_file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Epoch', 'Train Loss', 'Val Loss', 'Val MAE', 'Val RMSE', 'Val MAPE',
                             'Test Loss', 'Test MAE', 'Test RMSE', 'Test MAPE'])
    
    if load_checkpoint and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        if use_distributed:
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']
        prev_train_loss = checkpoint.get('prev_train_loss', None)
        prev_val_mae = checkpoint.get('prev_val_mae', None)
        prev_val_rmse = checkpoint.get('prev_val_rmse', None)
        prev_val_mape = checkpoint.get('prev_val_mape', None)
        patience_counter = checkpoint['patience_counter']
        train_losses = checkpoint.get('train_losses', [])
        val_losses = checkpoint.get('val_losses', [])
        if local_rank == 0:
            logger.info(f"Resuming from epoch {start_epoch}, best_val_loss: {best_val_loss:.4f}")
    else:
        best_val_loss = float('inf')
        prev_train_loss = None
        prev_val_mae = None
        prev_val_rmse = None
        prev_val_mape = None
        patience_counter = 0
    
    # 注册信号处理
    signal.signal(signal.SIGINT, lambda sig, frame: signal_handler(sig, frame, train_loader, val_loader, test_loader))
    signal.signal(signal.SIGTERM, lambda sig, frame: signal_handler(sig, frame, train_loader, val_loader, test_loader))
    
    try:
        for epoch in range(start_epoch, epochs):
            if use_distributed:
                train_loader.sampler.set_epoch(epoch)
            
            model.train()
            epoch_loss_per_node = torch.zeros(num_of_vertices).to(DEVICE)
            total_train_samples = 0
            start_time = time.time()
            
            with tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch", disable=local_rank != 0) as t:
                for batch_idx, batch in enumerate(t):
                    batch_start = time.time()
                    x_h, x_d, x_w, target, batch_mask = batch
                    x_h, x_d, x_w, target = x_h.to(DEVICE), x_d.to(DEVICE), x_w.to(DEVICE), target.to(DEVICE)
                    batch_mask = batch_mask.to(DEVICE)
                    
                    optimizer.zero_grad()
                    with amp.autocast('cuda'):
                        y_pred = model(x_h, x_d, x_w)
                        y_pred_raw = y_pred * stats['std_target'].to(DEVICE) + stats['mean_target'].to(DEVICE)
                        y_true_raw = target * stats['std_target'].to(DEVICE) + stats['mean_target'].to(DEVICE)
                        y_true_raw = torch.clamp(y_true_raw, min=0)
                        y_pred_raw = torch.clamp(y_pred_raw, min=0)
                        
                        # 直接使用 criterion 计算损失，不加权
                        loss = criterion(y_pred_raw, y_true_raw, mask=batch_mask, use_loss_mask=use_loss_mask)
                        loss = loss.mean()  # 确保 loss 是一个标量
                        
                        # 计算每个节点的未加权损失（与 Val Loss 一致）
                        batch_loss = torch.abs(y_pred_raw - y_true_raw)
                        if use_loss_mask:
                            batch_mask = batch_mask / (batch_mask.mean(dim=(0, 2), keepdim=True).clamp(min=1e-6))
                        else:
                            batch_mask = (y_true_raw != 0).float()
                            batch_mask = batch_mask / (batch_mask.mean(dim=(0, 2), keepdim=True).clamp(min=1e-6))
                        batch_loss = batch_loss * batch_mask
                        batch_loss[torch.isnan(batch_loss)] = 0
                        batch_loss = batch_loss.mean(dim=(0, 2))
                        epoch_loss_per_node += batch_loss * y_pred.shape[0]
                    
                    if batch_idx == 0 and epoch == start_epoch and local_rank == 0:
                        logger.info(f"Batch {batch_idx}, Memory allocated: {torch.cuda.memory_allocated(DEVICE)/1024**3:.2f} GiB")
                        logger.info(f"Batch {batch_idx}, x_h shape: {x_h.shape}")
                        logger.info(f"Batch {batch_idx}, x_d shape: {x_d.shape}")
                        logger.info(f"Batch {batch_idx}, x_w shape: {x_w.shape}")
                        logger.info(f"Batch {batch_idx}, target shape: {target.shape}")
                        logger.info(f"Batch {batch_idx}, batch_mask shape: {batch_mask.shape}")
                        logger.info(f"Batch {batch_idx}, y_pred shape: {y_pred.shape}")
                        logger.info(f"Batch {batch_idx}, y_true_raw min: {y_true_raw.min().item():.6f}, max: {y_true_raw.max().item():.6f}")
                        logger.info(f"Batch {batch_idx}, y_pred_raw min: {y_pred_raw.min().item():.6f}, max: {y_pred_raw.max().item():.6f}")
                    
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    
                    total_train_samples += y_pred.shape[0]
                    if local_rank == 0:
                        t.set_postfix(loss=loss.item())
            
            if dist.is_initialized():
                total_samples_tensor = torch.tensor(total_train_samples, device=DEVICE)
                dist.all_reduce(total_samples_tensor, op=dist.ReduceOp.SUM)
                total_train_samples = total_samples_tensor.item()
                dist.all_reduce(epoch_loss_per_node, op=dist.ReduceOp.SUM)
            
            epoch_loss_per_node = epoch_loss_per_node / total_train_samples
            avg_train_loss = epoch_loss_per_node.mean().item()
            
            (val_loss, val_mae, val_rmse, val_mape,
             val_loss_per_node, val_mae_per_node,
             val_rmse_per_node, val_mape_per_node,
             _, _, _) = evaluate(
                model, val_loader, DEVICE, stats['mean_target'], stats['std_target'], num_of_vertices, num_for_predict,
                phase="val", metric_method=metric_method, local_rank=local_rank
            )
            
            if local_rank == 0:
                train_losses.append(avg_train_loss)
                val_losses.append(val_loss)
                
                # 保留 loss 可视化
                plt.figure(figsize=(10, 5))
                plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss', marker='o')
                plt.plot(range(1, len(val_losses) + 1), val_losses, label='Val Loss', marker='o')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.title('Train and Val Loss over Epochs')
                plt.legend()
                plt.grid(True)
                loss_plot_path = os.path.join(results_path, 'loss_plot.png')
                plt.savefig(loss_plot_path)
                plt.close()
                logger.info(f"Loss plot saved at: {loss_plot_path}")
                
                train_loss_str = f"Train Loss: {avg_train_loss:.4f}"
                if prev_train_loss is not None:
                    train_loss_str += f" (prev: {prev_train_loss:.4f}, diff: {avg_train_loss - prev_train_loss:.4f})"
                val_loss_str = f"Val Loss: {val_loss:.4f}"
                val_mae_str = f"Val MAE: {val_mae:.4f}"
                if prev_val_mae is not None:
                    val_mae_str += f" (prev: {prev_val_mae:.4f}, diff: {val_mae - prev_val_mae:.4f})"
                val_rmse_str = f"Val RMSE: {val_rmse:.4f}"
                if prev_val_rmse is not None:
                    val_rmse_str += f" (prev: {prev_val_rmse:.4f}, diff: {val_rmse - prev_val_rmse:.4f})"
                val_mape_str = f"Val MAPE: {val_mape:.2f}%"
                if prev_val_mape is not None:
                    val_mape_str += f" (prev: {prev_val_mape:.2f}%, diff: {val_mape - prev_val_mape:.2f})"
                
                logger.info(f"Epoch {epoch+1}/{epochs}, {train_loss_str}, {val_loss_str}, {val_mae_str}, {val_rmse_str}, {val_mape_str}, "
                            f"Time: {time.time() - start_time:.2f} seconds, "
                            f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
                
                logger.info("\nValidation Metrics per Node:")
                if eval_node_id == -1:
                    for node in range(num_of_vertices):
                        logger.info(f"Node {node}: Loss: {val_loss_per_node[node]:.4f}, MAE: {val_mae_per_node[node]:.4f}, "
                                    f"RMSE: {val_rmse_per_node[node]:.4f}, MAPE: {val_mape_per_node[node]:.2f}%")
                elif 0 <= eval_node_id < num_of_vertices:
                    logger.info(f"Node {eval_node_id}: Loss: {val_loss_per_node[eval_node_id]:.4f}, MAE: {val_mae_per_node[eval_node_id]:.4f}, "
                                f"RMSE: {val_rmse_per_node[eval_node_id]:.4f}, MAPE: {val_mape_per_node[eval_node_id]:.2f}%")
                else:
                    logger.warning(f"Invalid eval_node_id: {eval_node_id}. Must be between 0 and {num_of_vertices-1} or -1 for all nodes.")
                
                # 保留 CSV 记录
                with open(log_file_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([epoch + 1, avg_train_loss, val_loss, val_mae, val_rmse, val_mape,
                                     '-', '-', '-', '-'])
                
                prev_train_loss = avg_train_loss
                prev_val_mae = val_mae
                prev_val_rmse = val_rmse
                prev_val_mape = val_mape
            
            scheduler.step(val_mae)
            
            # 保存最佳模型（移除注意力权重可视化）
            if val_mae < best_val_loss:
                best_val_loss = val_mae
                if local_rank == 0:  # 仅在 rank 0 上保存模型
                    state_dict = model.module.state_dict() if use_distributed else model.state_dict()
                    torch.save(state_dict, os.path.join(results_path, 'best_model.pth'))
                    logger.info(f"New best model saved with Val MAE: {best_val_loss:.4f}")
                    
                    # 注释掉注意力权重可视化
                    # if save_attention_weights:
                    #     os.makedirs(attention_visualization_path, exist_ok=True)
                    #     try:
                    #         logger.info(f"use_distributed: {use_distributed}")
                    #         logger.info(f"torch.distributed.is_initialized(): {torch.distributed.is_initialized()}")
                    #         logger.info(f"Model type: {type(model)}")
                    #         if use_distributed:
                    #             logger.info(f"Model.module type: {type(model.module)}")
                    #             has_interpret = hasattr(model.module, 'interpret')
                    #             logger.info(f"Model.module has interpret method: {has_interpret}")
                    #             if not has_interpret:
                    #                 logger.warning("Available methods in model.module: " + str([attr for attr in dir(model.module) if not attr.startswith('_')]))
                    #             else:
                    #                 model.module.interpret(
                    #                     x_h[:1], x_d[:1], x_w[:1],
                    #                     attention_visualization_path,
                    #                     epoch + 1,
                    #                     config,
                    #                     visualize_nodes=visualize_nodes,
                    #                     local_rank=local_rank
                    #                 )
                    #         else:
                    #             has_interpret = hasattr(model, 'interpret')
                    #             logger.info(f"Model has interpret method: {has_interpret}")
                    #             if not has_interpret:
                    #                 logger.warning("Available methods in model: " + str([attr for attr in dir(model) if not attr.startswith('_')]))
                    #             else:
                    #                 model.interpret(
                    #                     x_h[:1], x_d[:1], x_w[:1],
                    #                     attention_visualization_path,
                    #                     epoch + 1,
                    #                     config,
                    #                     visualize_nodes=visualize_nodes,
                    #                     local_rank=local_rank
                    #                 )
                    #         logger.info(f"Attention weights visualization saved to {attention_visualization_path}")
                    #     except AttributeError as e:
                    #         logger.error(f"Error calling interpret: {e}")
                    #         logger.error("Ensure that the Orion class has an 'interpret' method defined.")
                    #         raise
                    #     except Exception as e:
                    #         logger.error(f"Unexpected error during interpret: {e}")
                    #         raise
                patience_counter = 0
            else:
                if local_rank == 0:
                    patience_counter += 1
                    logger.info(f"Patience counter: {patience_counter}/{patience}")
                    if patience_counter >= patience:
                        logger.info("Early stopping triggered!")
                        break
            
            if local_rank == 0:
                state_dict = model.module.state_dict() if use_distributed else model.state_dict()
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': state_dict,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'best_val_loss': best_val_loss,
                    'prev_train_loss': prev_train_loss,
                    'prev_val_mae': prev_val_mae,
                    'prev_val_rmse': prev_val_rmse,
                    'prev_val_mape': prev_val_mape,
                    'patience_counter': patience_counter,
                    'train_losses': train_losses,
                    'val_losses': val_losses,
                    'stats': stats
                }, checkpoint_path)
    
    except Exception as e:
        logger.error(f"Training failed with exception: {e}")
        raise
    finally:
        # 确保在退出前关闭 DataLoader
        shutdown_dataloader(train_loader)
        shutdown_dataloader(val_loader)
        shutdown_dataloader(test_loader)
    
    best_model_path = os.path.join(results_path, 'best_model.pth')
    if local_rank == 0 and os.path.exists(best_model_path):
        state_dict = torch.load(best_model_path, map_location=DEVICE, weights_only=True)
        if use_distributed:
            model.module.load_state_dict(state_dict)
        else:
            model.load_state_dict(state_dict)
    
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
        
        # 注释掉测试阶段的注意力权重可视化
        # if save_attention_weights:
        #     test_attention_path = os.path.join(attention_visualization_path, 'test')
        #     os.makedirs(test_attention_path, exist_ok=True)
        #     try:
        #         logger.info(f"use_distributed: {use_distributed}")
        #         logger.info(f"torch.distributed.is_initialized(): {torch.distributed.is_initialized()}")
        #         logger.info(f"Model type: {type(model)}")
        #         if use_distributed:
        #             logger.info(f"Model.module type: {type(model.module)}")
        #             has_interpret = hasattr(model.module, 'interpret')
        #             logger.info(f"Model.module has interpret method: {has_interpret}")
        #             if not has_interpret:
        #                 logger.warning("Available methods in model.module: " + str([attr for attr in dir(model.module) if not attr.startswith('_')]))
        #             else:
        #                 model.module.interpret(
        #                     x_h[:1], x_d[:1], x_w[:1],
        #                     test_attention_path,
        #                     epoch=0,
        #                     config=config,
        #                     visualize_nodes=visualize_nodes,
        #                     local_rank=local_rank
        #                 )
        #         else:
        #             has_interpret = hasattr(model, 'interpret')
        #             logger.info(f"Model has interpret method: {has_interpret}")
        #             if not has_interpret:
        #                 logger.warning("Available methods in model: " + str([attr for attr in dir(model) if not attr.startswith('_')]))
        #             else:
        #                 model.interpret(
        #                     x_h[:1], x_d[:1], x_w[:1],
        #                     test_attention_path,
        #                     epoch=0,
        #                     config=config,
        #                     visualize_nodes=visualize_nodes,
        #                     local_rank=local_rank
        #                 )
        #         logger.info(f"Test attention weights visualization saved to {test_attention_path}")
        #     except AttributeError as e:
        #         logger.error(f"Error calling interpret during test: {e}")
        #         logger.error("Ensure that the Orion class has an 'interpret' method defined.")
        #         raise
        #     except Exception as e:
        #         logger.error(f"Unexpected error during test interpret: {e}")
        #         raise
        
        # 保留 CSV 记录
        with open(log_file_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Final', avg_train_loss, val_loss, val_mae, val_rmse, val_mape,
                             test_loss, test_mae, test_rmse, test_mape])
        
        logger.info("\nTest Metrics per Node:")
        if eval_node_id == -1:
            for node in range(num_of_vertices):
                logger.info(f"Node {node}: Loss: {test_loss_per_node[node]:.4f}, MAE: {test_mae_per_node[node]:.4f}, "
                            f"RMSE: {test_rmse_per_node[node]:.4f}, MAPE: {test_mape_per_node[node]:.2f}%")
        elif 0 <= eval_node_id < num_of_vertices:
            logger.info(f"Node {eval_node_id}: Loss: {test_loss_per_node[eval_node_id]:.4f}, MAE: {test_mae_per_node[eval_node_id]:.4f}, "
                        f"RMSE: {test_rmse_per_node[eval_node_id]:.4f}, MAPE: {test_mape_per_node[eval_node_id]:.2f}%")
        else:
            logger.warning(f"Invalid eval_node_id: {eval_node_id}. Must be between 0 and {num_of_vertices-1} or -1 for all nodes.")
    
    if world_size > 1:
        cleanup_distributed()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default='configurations/Orion_config.conf', type=str,
                        help="configuration file path")
    parser.add_argument("--local-rank", "--local_rank", type=int, default=int(os.environ.get('LOCAL_RANK', 0)),
                        help="Local rank for distributed training")
    args = parser.parse_args()
    
    config = configparser.ConfigParser()
    config.read(args.config)
    
    # 从环境变量中获取 local_rank 和 world_size
    local_rank = int(os.environ.get('LOCAL_RANK', args.local_rank))
    world_size = int(os.environ.get('WORLD_SIZE', 1))  # torchrun 会设置 WORLD_SIZE
    
    # 设置 logger
    global logger
    logger = setup_logging(args.config, local_rank)

    # 调试：打印配置文件路径和内容
    if local_rank == 0:
        logger.info(f"Config file path: {args.config}")
        logger.info(f"Sections in config: {config.sections()}")
    
    training_config = config['Training']
    use_distributed = training_config.getboolean('use_distributed', False)
    
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
        train(local_rank, world_size, args.config, DEVICE)
    else:
        train(local_rank, world_size, args.config, DEVICE)

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