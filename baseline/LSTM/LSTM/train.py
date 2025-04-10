import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import configparser
import argparse
import logging
from datetime import datetime
import gc

from model import LSTM

def setup_logging(log_path):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger('LSTM_Trainer')

def compute_metrics(pred, target):
    """Compute MAE, RMSE, MAPE for each horizon (denormalized)."""
    # pred: (B, 12, 1), target: (B, 12, 1)
    mae = torch.mean(torch.abs(pred - target), dim=0).squeeze().cpu().numpy()  # (12,)
    rmse = torch.sqrt(torch.mean((pred - target) ** 2, dim=0)).squeeze().cpu().numpy()  # (12,)
    mape = torch.mean(torch.abs((pred - target) / (target + 1e-6)), dim=0).squeeze().cpu().numpy() * 100  # (12,)
    return mae, rmse, mape

def compute_avg_metrics(pred, target):
    """Compute average MAE, RMSE, MAPE over all horizons (denormalized)."""
    mae = torch.mean(torch.abs(pred - target)).item()
    rmse = torch.sqrt(torch.mean((pred - target) ** 2)).item()
    mape = torch.mean(torch.abs((pred - target) / (target + 1e-6))).item() * 100
    return mae, rmse, mape

def train_model(config_file):
    config = configparser.ConfigParser()
    print(f"Attempting to read config file: {config_file}")
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file {config_file} does not exist!")
    
    config.read(config_file)
    if not config.sections():
        raise ValueError(f"Config file {config_file} is empty or not properly formatted!")
    print(f"Config sections loaded: {config.sections()}")
    
    data_config = config['Data']
    train_config = config['Training']
    lstm_config = config['LSTM']
    test_config = config['Test']
    
    dataset_name = data_config['dataset_name']
    data_path = os.path.join("/root/python_on_hyy/data_for_benchmark", f"{dataset_name}_LSTM.npz")
    log_path = test_config['log_path']
    results_path = test_config['results_path']
    os.makedirs(results_path, exist_ok=True)
    
    logger = setup_logging(log_path)
    logger.info(f"Starting training for {dataset_name}")
    
    use_distributed = train_config.getboolean('use_distributed') and torch.cuda.device_count() > 1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if use_distributed:
        dist.init_process_group(backend='nccl')
        local_rank = dist.get_rank()
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
    else:
        local_rank = 0
    
    logger.info(f"Loading data from {data_path}")
    data = np.load(data_path)
    train_x, train_y = data['train_x'], data['train_y']
    val_x, val_y = data['val_x'], data['val_y']
    test_x, test_y = data['test_x'], data['test_y']
    stats = {'mean': data['mean'], 'std': data['std'], 'mean_y': data['mean_y'], 'std_y': data['std_y']}
    
    logger.info(f"Original train_x shape: {train_x.shape}, train_y shape: {train_y.shape}")
    logger.info(f"Original test_x shape: {test_x.shape}, test_y shape: {test_y.shape}")
    if len(train_y.shape) == 2:
        train_y = train_y[..., None]
    if len(val_y.shape) == 2:
        val_y = val_y[..., None]
    if len(test_y.shape) == 2:
        test_y = test_y[..., None]
    logger.info(f"Fixed train_x shape: {train_x.shape}, train_y shape: {train_y.shape}")
    logger.info(f"Fixed test_x shape: {test_x.shape}, test_y shape: {test_y.shape}")
    logger.info(f"mean_y shape: {stats['mean_y'].shape}, std_y shape: {stats['std_y'].shape}")
    
    train_dataset = TensorDataset(torch.FloatTensor(train_x), torch.FloatTensor(train_y))
    val_dataset = TensorDataset(torch.FloatTensor(val_x), torch.FloatTensor(val_y))
    test_dataset = TensorDataset(torch.FloatTensor(test_x), torch.FloatTensor(test_y))
    
    batch_size = int(train_config['batch_size'])
    num_workers = min(int(train_config['num_workers']), 4)
    logger.info(f"Using batch_size: {batch_size}, num_workers: {num_workers}")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
    
    model = LSTM(
        in_channels=int(data_config['in_channels']),
        hidden_size=int(lstm_config['hidden_size']),
        num_layers=int(lstm_config['num_layers']),
        dropout=float(lstm_config['dropout'])
    ).to(device)
    if use_distributed:
        model = DDP(model, device_ids=[local_rank])
    
    optimizer = optim.Adam(model.parameters(), lr=float(train_config['learning_rate']))
    criterion = nn.L1Loss()
    
    mean_y = torch.FloatTensor(stats['mean_y']).to(device).view(1, 12, 1)  # (1, 12, 1)
    std_y = torch.FloatTensor(stats['std_y']).to(device).view(1, 12, 1)    # (1, 12, 1)
    
    epochs = int(train_config['epochs'])
    patience = int(train_config['patience'])
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        total_denorm_loss = 0
        train_preds, train_targets = [], []
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            if i == 0 and local_rank == 0:
                logger.info(f"Batch 0, pred shape: {pred.shape}, target shape: {y.shape}")
            
            pred_denorm = pred * std_y + mean_y
            y_denorm = y * std_y + mean_y
            
            denorm_loss = criterion(pred_denorm, y_denorm)
            denorm_loss.backward()
            optimizer.step()
            total_denorm_loss += denorm_loss.item() * x.size(0)
            
            train_preds.append(pred_denorm.cpu())
            train_targets.append(y_denorm.cpu())
            
            del x, y, pred, pred_denorm, y_denorm, denorm_loss
            torch.cuda.empty_cache()
            if i % 100 == 0 and local_rank == 0:
                logger.info(f"Epoch {epoch+1}/{epochs}, Batch {i}/{len(train_loader)}, MAE: {total_denorm_loss / ((i + 1) * batch_size):.2f}")
        
        total_denorm_loss /= len(train_loader.dataset)
        train_preds = torch.cat(train_preds, dim=0)
        train_targets = torch.cat(train_targets, dim=0)
        train_mae, train_rmse, train_mape = compute_avg_metrics(train_preds, train_targets)
        
        if local_rank == 0:
            logger.info(f"Epoch {epoch+1}/{epochs}, Train MAE: {train_mae:.2f}, RMSE: {train_rmse:.2f}, MAPE: {train_mape:.2f}%")
        
        model.eval()
        val_denorm_loss = 0
        val_preds, val_targets = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                pred_denorm = pred * std_y + mean_y
                y_denorm = y * std_y + mean_y
                val_denorm_loss += criterion(pred_denorm, y_denorm).item() * x.size(0)
                val_preds.append(pred_denorm.cpu())
                val_targets.append(y_denorm.cpu())
                del x, y, pred, pred_denorm, y_denorm
                torch.cuda.empty_cache()
        
        val_denorm_loss /= len(val_loader.dataset)
        val_preds = torch.cat(val_preds, dim=0)
        val_targets = torch.cat(val_targets, dim=0)
        val_mae, val_rmse, val_mape = compute_avg_metrics(val_preds, val_targets)
        
        if local_rank == 0:
            logger.info(f"Epoch {epoch+1}/{epochs}, Val MAE: {val_mae:.2f}, RMSE: {val_rmse:.2f}, MAPE: {val_mape:.2f}%")
        
        if val_denorm_loss < best_val_loss:
            best_val_loss = val_denorm_loss
            patience_counter = 0
            if local_rank == 0:
                torch.save(model.state_dict(), os.path.join(results_path, 'best_model.pt'))
        else:
            patience_counter += 1
            if patience_counter >= patience and local_rank == 0:
                logger.info("Early stopping triggered")
                break
    
    if local_rank == 0:
        model.load_state_dict(torch.load(os.path.join(results_path, 'best_model.pt'), weights_only=True))
        model.eval()
        test_preds, test_targets = [], []
        with torch.no_grad():
            for i, (x, y) in enumerate(test_loader):
                x, y = x.to(device), y.to(device)
                pred = model(x)
                if i == 0:
                    logger.info(f"Test Batch 0, pred shape: {pred.shape}, target shape: {y.shape}")
                pred_denorm = pred * std_y + mean_y
                y_denorm = y * std_y + mean_y
                test_preds.append(pred_denorm.cpu())
                test_targets.append(y_denorm.cpu())
                del x, y, pred, pred_denorm, y_denorm
                torch.cuda.empty_cache()
        
        test_preds = torch.cat(test_preds, dim=0)
        test_targets = torch.cat(test_targets, dim=0)
        
        logger.info(f"test_preds shape: {test_preds.shape}, test_targets shape: {test_targets.shape}")
        
        mae, rmse, mape = compute_metrics(test_preds, test_targets)
        
        logger.info(f"MAE shape: {mae.shape}, RMSE shape: {rmse.shape}, MAPE shape: {mape.shape}")
        
        avg_mae = np.mean(mae)
        avg_rmse = np.mean(rmse)
        avg_mape = np.mean(mape)
        
        logger.info("Test Results:")
        for h in range(len(mae)):
            logger.info(f"Horizon {h+1:02d}, MAE: {mae[h]:.2f}, MAPE: {mape[h]:.2f}%, RMSE: {rmse[h]:.2f}")
        logger.info(f"Average over {len(mae)} horizons, MAE: {avg_mae:.2f}, MAPE: {avg_mape:.2f}%, RMSE: {avg_rmse:.2f}")
    
    if use_distributed:
        dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    args = parser.parse_args()
    train_model(args.config)