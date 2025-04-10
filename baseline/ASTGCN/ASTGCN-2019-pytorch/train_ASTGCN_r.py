#!/usr/bin/env python
# coding: utf-8
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from time import time
import shutil
import argparse
import configparser
from model.ASTGCN_r import make_model
from lib.utils import load_graphdata_channel1, get_adjacency_matrix, compute_val_loss_mstgcn, predict_and_save_results_mstgcn
from tensorboardX import SummaryWriter
from lib.metrics import masked_mape_np, masked_mae, masked_mse, masked_rmse, masked_mae_test, masked_rmse_test
import logging
from datetime import datetime

# 设置日志
def setup_logging(params_path):
    log_dir = os.path.join(params_path, 'logs')
    try:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
            print(f"Created log directory: {log_dir}")
        else:
            print(f"Log directory already exists: {log_dir}")
        
        # 检查目录权限
        if not os.access(log_dir, os.W_OK):
            raise PermissionError(f"No write permission for directory: {log_dir}")
        
        log_file = os.path.join(log_dir, f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        print(f"Log file path: {log_file}")
        
        # 检查文件写入权限
        try:
            with open(log_file, 'a') as f:
                f.write("Test write permission\n")
            print(f"Successfully tested write permission for log file: {log_file}")
        except Exception as e:
            raise Exception(f"Failed to write to log file: {e}")
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    except Exception as e:
        print(f"Failed to set up logging: {e}")
        raise
    logger = logging.getLogger('ASTGCN')
    return logger, log_file

parser = argparse.ArgumentParser()
parser.add_argument("--config", default='configurations/PEMS08_astgcn.conf', type=str,
                    help="configuration file path")
args = parser.parse_args()
config = configparser.ConfigParser()
print('Read configuration file: %s' % (args.config))
config.read(args.config)
data_config = config['Data']
training_config = config['Training']

adj_filename = data_config['adj_filename']
graph_signal_matrix_filename = data_config['graph_signal_matrix_filename']
if config.has_option('Data', 'id_filename'):
    id_filename = data_config['id_filename']
else:
    id_filename = None

num_of_vertices = int(data_config['num_of_vertices'])
points_per_hour = int(data_config['points_per_hour'])
num_for_predict = int(data_config['num_for_predict'])
len_input = int(data_config['len_input'])
dataset_name = data_config['dataset_name']
in_channels = int(data_config['in_channels'])

model_name = training_config['model_name']

ctx = training_config['ctx']
os.environ["CUDA_VISIBLE_DEVICES"] = ctx
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda:0')
print("CUDA:", USE_CUDA, DEVICE)

learning_rate = float(training_config['learning_rate'])
epochs = int(training_config['epochs'])
start_epoch = int(training_config['start_epoch'])
batch_size = int(training_config['batch_size'])
num_of_weeks = int(training_config['num_of_weeks'])
num_of_days = int(training_config['num_of_days'])
num_of_hours = int(training_config['num_of_hours'])
time_strides = num_of_hours
nb_chev_filter = int(training_config['nb_chev_filter'])
nb_time_filter = int(training_config['nb_time_filter'])
nb_block = int(training_config['nb_block'])
K = int(training_config['K'])
loss_function = training_config['loss_function']
metric_method = training_config['metric_method']
missing_value = float(training_config['missing_value'])

# 计算总的输入时间步数
total_periods = num_of_weeks + num_of_days + num_of_hours
total_timesteps = len_input * total_periods

folder_dir = '%s_h%dd%dw%d_channel%d_%e' % (model_name, num_of_hours, num_of_days, num_of_weeks, in_channels, learning_rate)
print('folder_dir:', folder_dir)
params_path = os.path.join('experiments', dataset_name, folder_dir)
print('params_path:', params_path)

# 设置日志
logger, log_file_path = setup_logging(params_path)

# 加载数据
train_loader, train_target_tensor, val_loader, val_target_tensor, test_loader, test_target_tensor, _mean, _std = load_graphdata_channel1(
    graph_signal_matrix_filename, num_of_hours, num_of_days, num_of_weeks, DEVICE, batch_size, in_channels, num_workers=8)

adj_mx, distance_mx = get_adjacency_matrix(adj_filename, num_of_vertices, id_filename)

# 构建模型
net = make_model(DEVICE, nb_block, in_channels, K, nb_chev_filter, nb_time_filter, time_strides, adj_mx,
                 num_for_predict, total_timesteps, num_of_vertices)

def evaluate_on_test(net, test_loader, test_target_tensor, metric_method, _mean, _std):
    '''
    Compute MAE, MAPE, RMSE for each horizon (1 to num_for_predict).
    '''
    net.eval()
    with torch.no_grad():
        test_loader_length = len(test_loader)
        test_target_tensor = test_target_tensor.cpu().numpy()
        prediction = []
        for batch_index, batch_data in enumerate(test_loader):
            encoder_inputs, labels = batch_data
            encoder_inputs = encoder_inputs.to(DEVICE)  # 将输入移动到 GPU
            outputs = net(encoder_inputs)
            prediction.append(outputs.detach().cpu().numpy())
            if batch_index % 100 == 0:
                logger.info('predicting testing set batch %s / %s' % (batch_index + 1, test_loader_length))
        prediction = np.concatenate(prediction, 0)  # (batch, N, T')
        prediction_length = prediction.shape[2]
        mae_list, mape_list, rmse_list = [], [], []
        for i in range(prediction_length):
            print(f"Evaluating horizon {i+1}/{prediction_length}")
            if metric_method == 'mask':
                mae = masked_mae_test(test_target_tensor[:, :, i], prediction[:, :, i], 0.0)
                rmse = masked_rmse_test(test_target_tensor[:, :, i], prediction[:, :, i], 0.0)
                mape = masked_mape_np(test_target_tensor[:, :, i], prediction[:, :, i], 0)
            else:
                mae = np.mean(np.abs(test_target_tensor[:, :, i] - prediction[:, :, i]))
                rmse = np.sqrt(np.mean((test_target_tensor[:, :, i] - prediction[:, :, i]) ** 2))
                mape = masked_mape_np(test_target_tensor[:, :, i], prediction[:, :, i], 0)
            logger.info('Horizon %02d, MAE: %.2f, MAPE: %.4f, RMSE: %.2f' % (i + 1, mae, mape, rmse))
            mae_list.append(mae)
            mape_list.append(mape)
            rmse_list.append(rmse)
        # 平均值
        avg_mae = np.mean(mae_list)
        avg_mape = np.mean(mape_list)
        avg_rmse = np.mean(rmse_list)
        logger.info('Average over %d horizons, MAE: %.2f, MAPE: %.4f, RMSE: %.2f' % (prediction_length, avg_mae, avg_mape, avg_rmse))

def train_main():
    # 保存 logs 目录中的现有日志文件
    log_dir = os.path.join(params_path, 'logs')
    existing_log_files = []
    if os.path.exists(log_dir):
        existing_log_files = [os.path.join(log_dir, f) for f in os.listdir(log_dir) if f.startswith('training_') and f.endswith('.log')]
    
    if (start_epoch == 0) and (not os.path.exists(params_path)):
        os.makedirs(params_path)
        logger.info('create params directory %s' % (params_path))
    elif (start_epoch == 0) and (os.path.exists(params_path)):
        # 临时移动日志文件到项目根目录下的临时目录（避免被 params_path 删除）
        temp_log_dir = os.path.join(os.getcwd(), 'temp_logs')
        if existing_log_files:
            os.makedirs(temp_log_dir, exist_ok=True)
            for log_file in existing_log_files:
                shutil.move(log_file, temp_log_dir)
                logger.info('Moved log file to temp directory: %s' % log_file)
        # 删除 params_path
        shutil.rmtree(params_path)
        os.makedirs(params_path)
        logger.info('delete the old one and create params directory %s' % (params_path))
        # 恢复日志文件
        if existing_log_files:
            os.makedirs(log_dir, exist_ok=True)
            for log_file in os.listdir(temp_log_dir):
                shutil.move(os.path.join(temp_log_dir, log_file), log_dir)
                logger.info('Restored log file: %s' % os.path.join(log_dir, log_file))
            shutil.rmtree(temp_log_dir)
        else:
            # 如果没有日志文件，确保 logs 目录存在
            os.makedirs(log_dir, exist_ok=True)
    elif (start_epoch > 0) and (os.path.exists(params_path)):
        logger.info('train from params directory %s' % (params_path))
    else:
        raise SystemExit('Wrong type of model!')

    logger.info('param list:')
    logger.info('CUDA\t%s' % DEVICE)
    logger.info('in_channels\t%s' % in_channels)
    logger.info('nb_block\t%s' % nb_block)
    logger.info('nb_chev_filter\t%s' % nb_chev_filter)
    logger.info('nb_time_filter\t%s' % nb_time_filter)
    logger.info('time_strides\t%s' % time_strides)
    logger.info('batch_size\t%s' % batch_size)
    logger.info('graph_signal_matrix_filename\t%s' % graph_signal_matrix_filename)
    logger.info('start_epoch\t%s' % start_epoch)
    logger.info('epochs\t%s' % epochs)
    masked_flag = 0
    criterion = nn.L1Loss().to(DEVICE)
    criterion_masked = masked_mae
    if loss_function == 'masked_mse':
        criterion_masked = masked_mse
        masked_flag = 1
    elif loss_function == 'masked_mae':
        criterion_masked = masked_mae
        masked_flag = 1
    elif loss_function == 'mae':
        criterion = nn.L1Loss().to(DEVICE)
        masked_flag = 0
    elif loss_function == 'rmse':
        criterion = nn.MSELoss().to(DEVICE)
        masked_flag = 0
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    sw = SummaryWriter(logdir=params_path, flush_secs=5)
    logger.info(str(net))

    logger.info('Net\'s state_dict:')
    total_param = 0
    for param_tensor in net.state_dict():
        logger.info('%s\t%s' % (param_tensor, net.state_dict()[param_tensor].size()))
        total_param += np.prod(net.state_dict()[param_tensor].size())
    logger.info('Net\'s total params: %d' % total_param)

    logger.info('Optimizer\'s state_dict:')
    for var_name in optimizer.state_dict():
        logger.info('%s\t%s' % (var_name, optimizer.state_dict()[var_name]))

    global_step = 0
    best_epoch = 0
    best_val_loss = np.inf

    start_time = time()

    # 添加 GPU 利用率监控
    try:
        import pynvml
        pynvml.nvmlInit()
        device = pynvml.nvmlDeviceGetHandleByIndex(0)  # 假设使用 GPU 0
    except Exception as e:
        logger.info(f"Failed to initialize pynvml: {e}")
        device = None

    if start_epoch > 0:
        params_filename = os.path.join(params_path, 'epoch_%s.params' % start_epoch)
        net.load_state_dict(torch.load(params_filename))
        logger.info('start epoch: %d' % start_epoch)
        logger.info('load weight from: %s' % params_filename)

    # 训练模型
    logger.info('===== Training Start =====')
    for epoch in range(start_epoch, epochs):
        params_filename = os.path.join(params_path, 'epoch_%s.params' % epoch)
        if masked_flag:
            val_loss = compute_val_loss_mstgcn(net, val_loader, criterion_masked, masked_flag, missing_value, sw, epoch, device=DEVICE)
        else:
            val_loss = compute_val_loss_mstgcn(net, val_loader, criterion, masked_flag, missing_value, sw, epoch, device=DEVICE)

        # 记录每个 epoch 的验证损失
        logger.info('Epoch %d, Validation Loss: %.2f' % (epoch + 1, val_loss))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(net.state_dict(), params_filename)
            logger.info('save parameters to file: %s' % params_filename)

        net.train()
        epoch_loss = 0
        num_batches = 0
        for batch_index, batch_data in enumerate(train_loader):
            batch_start_time = time()
            encoder_inputs, labels = batch_data
            encoder_inputs = encoder_inputs.to(DEVICE)  # 将输入移动到 GPU
            labels = labels.to(DEVICE)  # 将标签移动到 GPU
            optimizer.zero_grad()
            outputs = net(encoder_inputs)
            if masked_flag:
                loss = criterion_masked(outputs, labels, missing_value)
            else:
                loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            training_loss = loss.item()
            epoch_loss += training_loss
            num_batches += 1
            global_step += 1
            sw.add_scalar('training_loss', training_loss, global_step)
            batch_time = time() - batch_start_time
            if global_step % 1000 == 0:
                logger.info('global step: %d, training loss: %.2f, time: %.2fs' % (global_step, training_loss, time() - start_time))
        # 记录每个 epoch 的平均训练损失
        avg_epoch_loss = epoch_loss / num_batches
        logger.info('Epoch %d, Average Training Loss: %.2f' % (epoch + 1, avg_epoch_loss))

    logger.info('===== Training End =====')
    logger.info('best epoch: %d, best val loss: %.2f' % (best_epoch, best_val_loss))

    # 在测试集上应用最优模型
    logger.info('===== Testing Start =====')
    predict_main(best_epoch, test_loader, test_target_tensor, metric_method, _mean, _std, 'test')
    logger.info('===== Testing End =====')

def predict_main(global_step, data_loader, data_target_tensor, metric_method, _mean, _std, type):
    params_filename = os.path.join(params_path, 'epoch_%s.params' % global_step)
    logger.info('load weight from: %s' % params_filename)
    net.load_state_dict(torch.load(params_filename))
    predict_and_save_results_mstgcn(net, data_loader, data_target_tensor, global_step, metric_method, _mean, _std, params_path, type, device=DEVICE)
    # 添加测试评估
    evaluate_on_test(net, data_loader, data_target_tensor, metric_method, _mean, _std)

if __name__ == "__main__":
    train_main()