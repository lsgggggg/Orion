import os
import numpy as np
import argparse
import configparser
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm  # 添加进度条以监控进度

def search_data(sequence_length, num_of_depend, label_start_idx, num_for_predict, len_input, units, points_per_hour):
    """
    Search for valid indices for a given period (week, day, hour), ensuring the sampled data aligns with the target time slot.

    Args:
        sequence_length (int): Total length of the data sequence.
        num_of_depend (int): Number of dependencies (e.g., number of weeks, days, or hours).
        label_start_idx (int): Starting index of the target (e.g., start of the prediction window).
        num_for_predict (int): Number of time steps to predict (e.g., 12 for 1 hour).
        len_input (int): Length of the input sequence (e.g., 12 for 1 hour of historical data).
        units (int): Number of hours in the period (e.g., 7*24 for a week, 24 for a day, 1 for an hour).
        points_per_hour (int): Number of time steps per hour (e.g., 12).

    Returns:
        List of tuples: Each tuple contains (start_idx, end_idx) for a dependency.
    """
    if points_per_hour < 0:
        raise ValueError("points_per_hour should be greater than 0!")
    
    if label_start_idx + num_for_predict > sequence_length:
        return None
    
    points_per_period = units * points_per_hour

    x_idx = []
    if units == 1:  # Special case for hour periodicity
        for i in range(1, num_of_depend + 1):
            end_idx = label_start_idx - (i - 1) * points_per_period
            start_idx = end_idx - len_input  # 使用 len_input 而不是 num_for_predict
            if start_idx >= 0 and end_idx <= sequence_length:
                x_idx.append((start_idx, end_idx))
            else:
                return None
    else:
        # For week and day periodicity, align with the same time slot
        for i in range(1, num_of_depend + 1):
            period_end_idx = label_start_idx - i * points_per_period
            start_idx = period_end_idx - len_input  # 使用 len_input 而不是 num_for_predict
            end_idx = period_end_idx
            if start_idx >= 0 and end_idx <= sequence_length:
                x_idx.append((start_idx, end_idx))
            else:
                return None
    
    if len(x_idx) != num_of_depend:
        return None
    
    return x_idx[::-1]

def get_sample_indices(data_sequence, num_of_weeks, num_of_days, num_of_hours, 
                       label_start_idx, num_for_predict, len_input, points_per_hour=12, device='cpu'):
    """
    Generate samples for recent, daily, and weekly periods, ensuring alignment with the target's time slot.

    Args:
        data_sequence (torch.Tensor): Shape (T, N, F), the input data sequence on the specified device.
        num_of_weeks (int): Number of weeks to sample.
        num_of_days (int): Number of days to sample.
        num_of_hours (int): Number of hours to sample.
        label_start_idx (int): Starting index of the target.
        num_for_predict (int): Number of time steps to predict.
        len_input (int): Length of the input sequence.
        points_per_hour (int): Number of time steps per hour.
        device (str): Device to perform tensor operations ('cpu' or 'cuda').

    Returns:
        Tuple: (week_sample, day_sample, hour_sample, target)
    """
    week_sample, day_sample, hour_sample = None, None, None
    
    if label_start_idx + num_for_predict > data_sequence.shape[0]:
        return None, None, None, None
    
    # Weekly-periodic segment
    if num_of_weeks > 0:
        week_indices = search_data(data_sequence.shape[0], num_of_weeks, 
                                   label_start_idx, num_for_predict, len_input, 7 * 24, points_per_hour)
        if not week_indices:
            return None, None, None, None
        week_sample = torch.cat([data_sequence[i:j] for i, j in week_indices], dim=0).to(device)
    
    # Daily-periodic segment
    if num_of_days > 0:
        day_indices = search_data(data_sequence.shape[0], num_of_days, 
                                  label_start_idx, num_for_predict, len_input, 24, points_per_hour)
        if not day_indices:
            return None, None, None, None
        day_sample = torch.cat([data_sequence[i:j] for i, j in day_indices], dim=0).to(device)
    
    # Recent segment
    if num_of_hours > 0:
        hour_indices = search_data(data_sequence.shape[0], num_of_hours, 
                                   label_start_idx, num_for_predict, len_input, 1, points_per_hour)
        if not hour_indices:
            return None, None, None, None
        hour_sample = torch.cat([data_sequence[i:j] for i, j in hour_indices], dim=0).to(device)
        # Ensure hour_sample's last time step aligns with label_start_idx
        if hour_sample.shape[0] > 0:
            expected_end_idx = label_start_idx
            actual_end_idx = hour_indices[-1][1]
            if actual_end_idx != expected_end_idx:
                print(f"Warning: hour_sample end index {actual_end_idx} does not match label_start_idx {label_start_idx}")
                # Adjust hour_sample to align with label_start_idx
                if actual_end_idx < label_start_idx:
                    padding_length = label_start_idx - actual_end_idx
                    if label_start_idx < data_sequence.shape[0]:
                        padding_data = data_sequence[actual_end_idx:label_start_idx]
                        hour_sample = torch.cat([hour_sample, padding_data], dim=0).to(device)
                    else:
                        return None, None, None, None
                elif actual_end_idx > label_start_idx:
                    excess_length = actual_end_idx - label_start_idx
                    hour_sample = hour_sample[:-excess_length]
    
    # Target
    target = data_sequence[label_start_idx:label_start_idx + num_for_predict].to(device)
    
    return week_sample, day_sample, hour_sample, target

def linear_interpolation_gpu(data, device='cuda'):
    """
    Apply linear interpolation to fill NaN and zero values in the data along the time axis (axis=0) on GPU.
    
    Args:
        data (ndarray): Shape (T, N, F), the input data sequence.
        device (str): Device to perform tensor operations ('cpu' or 'cuda').
    
    Returns:
        ndarray: Data with NaN and zero values filled using linear interpolation.
    """
    # Convert data to torch tensor and move to GPU
    data_tensor = torch.from_numpy(data).float().to(device)  # (T, N, F)
    T, N, F = data_tensor.shape
    
    # Create a mask for invalid values (NaN or zero)
    invalid_mask = (data_tensor == 0) | torch.isnan(data_tensor)  # True where invalid
    
    # Replace invalid values with NaN for interpolation
    data_tensor[invalid_mask] = float('nan')
    
    # Reshape to (N*F, T) for parallel processing
    data_reshaped = data_tensor.permute(1, 2, 0).reshape(N * F, T)  # (N*F, T)
    invalid_mask_reshaped = invalid_mask.permute(1, 2, 0).reshape(N * F, T)  # (N*F, T)
    
    # Find indices of valid (non-NaN) values
    valid_indices = torch.where(~invalid_mask_reshaped)  # Tuple of (row_idx, col_idx)
    row_idx, col_idx = valid_indices
    
    # If no valid values in a row, fill with zeros
    if row_idx.numel() == 0:
        data_reshaped.fill_(0)
        data_tensor = data_reshaped.reshape(N, F, T).permute(2, 0, 1)
        return data_tensor.cpu().numpy()
    
    # For each row, perform linear interpolation
    for i in tqdm(range(N * F), desc="Linear interpolation on GPU"):
        # Get valid indices for this row
        valid_cols = col_idx[row_idx == i]
        if valid_cols.numel() == 0:
            # No valid values in this row, fill with 0
            data_reshaped[i] = 0
            continue
        
        # Get invalid indices for this row
        invalid_cols = torch.where(invalid_mask_reshaped[i])[0]
        if invalid_cols.numel() == 0:
            continue  # No interpolation needed
        
        # Get valid values and their indices
        valid_vals = data_reshaped[i, valid_cols]
        
        # For each invalid position, find the nearest valid points
        for invalid_col in invalid_cols:
            # Find the nearest previous and next valid indices
            prev_idx = valid_cols[valid_cols < invalid_col]
            next_idx = valid_cols[valid_cols > invalid_col]
            
            if prev_idx.numel() == 0 and next_idx.numel() == 0:
                # No valid points, fill with 0
                data_reshaped[i, invalid_col] = 0
            elif prev_idx.numel() == 0:
                # Only next valid point exists, fill with that value
                data_reshaped[i, invalid_col] = data_reshaped[i, next_idx[0]]
            elif next_idx.numel() == 0:
                # Only previous valid point exists, fill with that value
                data_reshaped[i, invalid_col] = data_reshaped[i, prev_idx[-1]]
            else:
                # Linear interpolation between prev_idx[-1] and next_idx[0]
                t0, t1 = prev_idx[-1], next_idx[0]
                v0, v1 = data_reshaped[i, t0], data_reshaped[i, t1]
                t = (invalid_col - t0).float() / (t1 - t0).float()
                data_reshaped[i, invalid_col] = v0 + (v1 - v0) * t
    
    # Reshape back to (T, N, F)
    data_tensor = data_reshaped.reshape(N, F, T).permute(2, 0, 1)
    
    # Convert back to numpy
    data_filled = data_tensor.cpu().numpy()
    return data_filled

def read_and_generate_dataset(graph_signal_matrix_filename, adj_filename, num_of_weeks, num_of_days, 
                              num_of_hours, num_for_predict, points_per_hour=12, len_input=12, 
                              in_channels=3, shuffle_data=True, save=False, device='cuda'):
    """
    Read and preprocess the dataset, generating samples for training, validation, and testing.

    Args:
        graph_signal_matrix_filename (str): Path to the .npz file containing the graph signal matrix.
        adj_filename (str): Path to the adjacency matrix file (.csv).
        num_of_weeks (int): Number of weeks for weekly periodicity.
        num_of_days (int): Number of days for daily periodicity.
        num_of_hours (int): Number of hours for hourly periodicity.
        num_for_predict (int): Number of time steps to predict.
        points_per_hour (int): Number of time steps per hour.
        len_input (int): Length of the input sequence.
        in_channels (int): Number of input features (1 or 3).
        shuffle_data (bool): Whether to shuffle the data before splitting.
        save (bool): Whether to save the processed data.
        device (str): Device to perform tensor operations ('cpu' or 'cuda').

    Returns:
        dict: Processed data dictionary containing train, val, test sets, stats, and adjacency matrix.
    """
    # Load traffic data from .npz
    data_seq = np.load(graph_signal_matrix_filename)['data']  # Shape: (T, N, F)
    
    # Create a mask for original zero or NaN values
    print("Creating mask for zero or NaN values...")
    mask = np.ones_like(data_seq, dtype=np.float32)  # 1 for valid, 0 for invalid
    mask[np.isnan(data_seq)] = 0  # Mark NaN as invalid
    mask[data_seq == 0] = 0      # Mark zeros as invalid
    
    # Preprocess data: Apply linear interpolation to fill NaN and zero values
    print("Before preprocessing (linear interpolation):")
    print(f"data_seq min: {np.nanmin(data_seq, axis=(0, 1))}, max: {np.nanmax(data_seq, axis=(0, 1))}")
    print(f"Number of NaN values: {np.isnan(data_seq).sum()}")
    print(f"Number of zero values: {(data_seq == 0).sum()}")
    
    data_seq = linear_interpolation_gpu(data_seq, device=device)
    
    # Ensure non-negative after interpolation
    data_seq = np.maximum(data_seq, 0)
    
    print("After preprocessing (linear interpolation):")
    print(f"data_seq min: {data_seq.min(axis=(0, 1))}, max: {data_seq.max(axis=(0, 1))}")
    print(f"Number of NaN values: {np.isnan(data_seq).sum()}")
    print(f"Number of zero values: {(data_seq == 0).sum()}")
    
    # Load and convert adjacency matrix from edge list to adjacency matrix
    edges = pd.read_csv(adj_filename, header=0).values
    num_nodes = int(config['Data']['num_of_vertices'])
    adj_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    
    # 从配置文件中读取 KNN 参数
    knn_k = int(config['Data'].get('knn_k', 3))  # 默认 K=3
    print(f"Using KNN K: {knn_k}")
    
    # 从配置文件中读取 cost_threshold（可选）
    cost_threshold = float(config['Data'].get('cost_threshold', float('inf')))  # 默认无阈值限制
    print(f"Using cost_threshold: {cost_threshold}")
    
    # 提取所有 cost 值以进行标准化
    costs = np.array([float(row[2]) for row in edges])
    if len(costs) == 0:
        print("Warning: No edges found in the adjacency file.")
    else:
        # Min-Max 标准化 cost
        cost_min = costs.min()
        cost_max = costs.max()
        if cost_max == cost_min:
            print("Warning: All cost values are the same, setting normalized costs to 0.")
            normalized_costs = np.zeros_like(costs)
        else:
            normalized_costs = (costs - cost_min) / (cost_max - cost_min)
        
        print(f"Before normalization - cost min: {cost_min}, max: {cost_max}")
        print(f"After normalization - normalized cost min: {normalized_costs.min()}, max: {normalized_costs.max()}")
        
        # 构建邻接列表：对于每个节点，记录其所有邻居及对应的 cost
        adj_list = [[] for _ in range(num_nodes)]  # 每个节点存储 (to_node, cost, normalized_cost)
        for idx, row in enumerate(edges):
            from_node, to_node, cost = row
            from_node = int(from_node)
            to_node = int(to_node)
            cost = float(cost)
            normalized_cost = normalized_costs[idx]
            
            if 0 <= from_node < num_nodes and 0 <= to_node < num_nodes:
                if cost <= cost_threshold:  # 应用 cost_threshold 过滤（如果有）
                    adj_list[from_node].append((to_node, cost, normalized_cost))
                    adj_list[to_node].append((from_node, cost, normalized_cost))  # 无向图
            else:
                print(f"Warning: Invalid node indices {from_node}, {to_node} - skipped")
        
        # 统计应用阈值后保留的边数量
        total_edges = sum(len(neighbors) for neighbors in adj_list) // 2  # 无向图，边被双向记录
        print(f"Total edges after applying cost_threshold: {total_edges}")
        
        # KNN：对每个节点的邻居按 cost 排序，保留 K 个最近的邻居
        retained_edges = 0
        for from_node in range(num_nodes):
            if not adj_list[from_node]:
                continue
            # 按 cost 升序排序（cost 越小，距离越近）
            neighbors = sorted(adj_list[from_node], key=lambda x: x[1])
            # 保留 K 个最近的邻居
            k_neighbors = neighbors[:min(knn_k, len(neighbors))]
            for to_node, cost, normalized_cost in k_neighbors:
                # 计算权重：距离近（cost 小）权重大，使用 1 / (cost + epsilon)
                epsilon = 1e-6  # 防止除以 0
                weight = 1.0 / (cost + epsilon)
                adj_matrix[from_node, to_node] = weight
                retained_edges += 1
        
        # 因为是无向图，边被双向记录，实际边数需要除以 2
        retained_edges = retained_edges // 2
        print(f"Retained edges after applying KNN (K={knn_k}): {retained_edges}")
        print(f"Adjacency matrix shape: {adj_matrix.shape}")
        print(f"Adjacency matrix min: {adj_matrix.min()}, max: {adj_matrix.max()}")
    
    # Convert data_seq to torch tensor and move to device
    data_seq_tensor = torch.from_numpy(data_seq).float().to(device)
    mask_tensor = torch.from_numpy(mask).float().to(device)  # 将掩码也转换为张量
    
    all_samples = []
    all_masks = []  # 用于存储每个样本的掩码
    for idx in tqdm(range(data_seq.shape[0]), desc="Generating samples"):
        sample = get_sample_indices(data_seq_tensor, num_of_weeks, num_of_days, num_of_hours, 
                                    idx, num_for_predict, len_input, points_per_hour, device=device)
        if all(s is None for s in sample[:-1]):
            continue
        
        week_sample, day_sample, hour_sample, target = sample
        
        # 获取对应的掩码
        target_mask = mask_tensor[idx:idx + num_for_predict, :, 0]  # 只取 flow 通道
        
        # Adjust input length to len_input
        if week_sample is not None and week_sample.shape[0] > len_input:
            week_sample = week_sample[-len_input:]
        if day_sample is not None and day_sample.shape[0] > len_input:
            day_sample = day_sample[-len_input:]
        if hour_sample is not None and hour_sample.shape[0] > len_input:
            hour_sample = hour_sample[-len_input:]
        
        sample_list = []
        if num_of_weeks > 0 and week_sample is not None:
            week_sample = week_sample.unsqueeze(0).permute(0, 2, 3, 1)  # (1, N, F, T_w)
            sample_list.append(week_sample)
        if num_of_days > 0 and day_sample is not None:
            day_sample = day_sample.unsqueeze(0).permute(0, 2, 3, 1)  # (1, N, F, T_d)
            sample_list.append(day_sample)
        if num_of_hours > 0 and hour_sample is not None:
            hour_sample = hour_sample.unsqueeze(0).permute(0, 2, 3, 1)  # (1, N, F, T_h)
            sample_list.append(hour_sample)
        
        target = target.unsqueeze(0).permute(0, 2, 3, 1)[:, :, 0, :]  # (1, N, T_pred)
        sample_list.append(target)
        all_masks.append(target_mask.unsqueeze(0))  # (1, T_pred, N)
        
        # Move tensors to CPU and convert to numpy for storage
        sample_list = [s.cpu().numpy() if s is not None else None for s in sample_list]
        all_samples.append(sample_list)
    
    # Shuffle data if specified
    if shuffle_data:
        indices = np.arange(len(all_samples))
        np.random.seed(int(config['Training']['seed']))  # 确保可重复性
        np.random.shuffle(indices)
        all_samples = [all_samples[i] for i in indices]
        all_masks = [all_masks[i] for i in indices]
    
    # Split dataset: 60% train, 20% val, 20% test
    split_line1 = int(len(all_samples) * 0.6)
    split_line2 = int(len(all_samples) * 0.8)
    
    training_set = [np.concatenate(i, axis=0) for i in zip(*all_samples[:split_line1])]
    validation_set = [np.concatenate(i, axis=0) for i in zip(*all_samples[split_line1:split_line2])]
    testing_set = [np.concatenate(i, axis=0) for i in zip(*all_samples[split_line2:])]
    
    # 处理掩码：将 CUDA 张量移到 CPU 并转换为 NumPy 数组
    all_masks_cpu = [m.cpu().numpy() for m in all_masks]  # 先将每个张量移到 CPU 并转为 NumPy
    train_mask = np.concatenate(all_masks_cpu[:split_line1], axis=0).transpose(0, 2, 1)  # (B_train, N, T_pred)
    val_mask = np.concatenate(all_masks_cpu[split_line1:split_line2], axis=0).transpose(0, 2, 1)  # (B_val, N, T_pred)
    test_mask = np.concatenate(all_masks_cpu[split_line2:], axis=0).transpose(0, 2, 1)  # (B_test, N, T_pred)
    
    # Assign data
    train_x_w, train_x_d, train_x_h, train_target = training_set
    val_x_w, val_x_d, val_x_h, val_target = validation_set
    test_x_w, test_x_d, test_x_h, test_target = testing_set
    
    # Print raw data statistics for debugging
    print("\nRaw data statistics:")
    print(f"train_target min: {train_target.min()}, max: {train_target.max()}")
    print(f"val_target min: {val_target.min()}, max: {val_target.max()}")
    print(f"test_target min: {test_target.min()}, max: {test_target.max()}")
    print(f"train_x_h (flow) min: {train_x_h[:, :, 0, :].min()}, max: {train_x_h[:, :, 0, :].max()}")
    if in_channels > 1:  # 只有当特征数大于1时才打印占用率和速度
        print(f"train_x_h (occupancy) min: {train_x_h[:, :, 1, :].min()}, max: {train_x_h[:, :, 1, :].max()}")
        print(f"train_x_h (speed) min: {train_x_h[:, :, 2, :].min()}, max: {train_x_h[:, :, 2, :].max()}")
    
    # Normalization (including target)
    stats, train_x_w_norm, train_x_d_norm, train_x_h_norm, train_target_norm = normalization(
        train_x_w, train_x_d, train_x_h, train_target, in_channels=in_channels, device=device)
    _, val_x_w_norm, val_x_d_norm, val_x_h_norm, val_target_norm = normalization(
        val_x_w, val_x_d, val_x_h, val_target, stats=stats, in_channels=in_channels, device=device)
    _, test_x_w_norm, test_x_d_norm, test_x_h_norm, test_target_norm = normalization(
        test_x_w, test_x_d, test_x_h, test_target, stats=stats, in_channels=in_channels, device=device)
    
    all_data = {
        'train': {
            'x_w': train_x_w_norm,
            'x_d': train_x_d_norm,
            'x_h': train_x_h_norm,
            'target': train_target_norm,
            'mask': train_mask  # Add mask for training set
        },
        'val': {
            'x_w': val_x_w_norm,
            'x_d': val_x_d_norm,
            'x_h': val_x_h_norm,
            'target': val_target_norm,
            'mask': val_mask  # Add mask for validation set
        },
        'test': {
            'x_w': test_x_w_norm,
            'x_d': test_x_d_norm,
            'x_h': test_x_h_norm,
            'target': test_target_norm,
            'mask': test_mask  # Add mask for test set
        },
        'stats': {
            'mean_flow': stats['mean_flow'],
            'std_flow': stats['std_flow'],  # Flow 统计量
            'mean_occupancy': stats['mean_occupancy'],
            'std_occupancy': stats['std_occupancy'],  # Occupancy 统计量
            'mean_speed': stats['mean_speed'],
            'std_speed': stats['std_speed'],  # Speed 统计量
            'mean_target': stats['mean_target'],
            'std_target': stats['std_target']  # Target 统计量
        },
        'adj_matrix': adj_matrix  # Add adjacency matrix to all_data
    }
    
    # Print shapes for debugging
    for split in ['train', 'val', 'test']:
        print(f"{split} x_w: {all_data[split]['x_w'].shape}")
        print(f"{split} x_d: {all_data[split]['x_d'].shape}")
        print(f"{split} x_h: {all_data[split]['x_h'].shape}")
        print(f"{split} target: {all_data[split]['target'].shape}")
        print(f"{split} mask: {all_data[split]['mask'].shape}")
    print(f"adj_matrix: {all_data['adj_matrix'].shape}")
    
    # Save processed data with explicit key names
    if save:
        file = os.path.basename(graph_signal_matrix_filename).split('.')[0]
        dirpath = os.path.dirname(graph_signal_matrix_filename)
        filename = os.path.join(dirpath, f'{file}_r{num_of_hours}_d{num_of_days}_w{num_of_weeks}_Orion.npz')
        print('Save file:', filename)
        # Use np.savez instead of np.savez_compressed to reduce CPU overhead
        np.savez(filename,
                 train_x_w=all_data['train']['x_w'],
                 train_x_d=all_data['train']['x_d'],
                 train_x_h=all_data['train']['x_h'],
                 train_target=all_data['train']['target'],
                 train_mask=all_data['train']['mask'],  # Save mask
                 val_x_w=all_data['val']['x_w'],
                 val_x_d=all_data['val']['x_d'],
                 val_x_h=all_data['val']['x_h'],
                 val_target=all_data['val']['target'],
                 val_mask=all_data['val']['mask'],  # Save mask
                 test_x_w=all_data['test']['x_w'],
                 test_x_d=all_data['test']['x_d'],
                 test_x_h=all_data['test']['x_h'],
                 test_target=all_data['test']['target'],
                 test_mask=all_data['test']['mask'],  # Save mask
                 adj_matrix=all_data['adj_matrix'],  # Save adjacency matrix
                 mean_flow=all_data['stats']['mean_flow'],
                 std_flow=all_data['stats']['std_flow'],
                 mean_occupancy=all_data['stats']['mean_occupancy'],
                 std_occupancy=all_data['stats']['std_occupancy'],
                 mean_speed=all_data['stats']['mean_speed'],
                 std_speed=all_data['stats']['std_speed'],
                 mean_target=all_data['stats']['mean_target'],
                 std_target=all_data['stats']['std_target'])
    
    return all_data

def normalization(x_w, x_d, x_h, target=None, stats=None, in_channels=3, device='cuda'):
    """
    Normalize the data using global statistics on the specified device.

    Args:
        x_w (ndarray): Weekly input data.
        x_d (ndarray): Daily input data.
        x_h (ndarray): Hourly input data.
        target (ndarray, optional): Target data.
        stats (dict, optional): Precomputed statistics.
        in_channels (int): Number of input features (1 or 3).
        device (str): Device to perform tensor operations ('cpu' or 'cuda').

    Returns:
        tuple: (stats, x_w_norm, x_d_norm, x_h_norm, target_norm)
    """
    # Convert inputs to torch tensors and move to device
    x_w_tensor = torch.from_numpy(x_w).float().to(device) if x_w is not None else None
    x_d_tensor = torch.from_numpy(x_d).float().to(device) if x_d is not None else None
    x_h_tensor = torch.from_numpy(x_h).float().to(device) if x_h is not None else None
    target_tensor = torch.from_numpy(target).float().to(device) if target is not None else None
    
    if stats is None:
        # Compute global statistics for each feature
        if target_tensor is not None:
            mean_target = target_tensor.mean().item()  # Convert to scalar
            std_target = target_tensor.std().item()
            std_target = max(std_target, 1.0)  # 避免除以 0
        else:
            mean_target, std_target = None, None
        
        mean_flow = mean_target
        std_flow = std_target
        
        # Compute mean and std for occupancy (index 1) if exists
        mean_occupancy = x_h_tensor[:, :, 1, :].mean().item() if in_channels > 1 else 0.0
        std_occupancy = (x_h_tensor[:, :, 1, :].std() + 1e-6).item() if in_channels > 1 else 1.0
        
        # Compute mean and std for speed (index 2) if exists
        mean_speed = x_h_tensor[:, :, 2, :].mean().item() if in_channels > 1 else 0.0
        std_speed = (x_h_tensor[:, :, 2, :].std() + 1e-6).item() if in_channels > 1 else 1.0
        
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
    
    def normalize(x, is_target=False):
        if x is None:
            return None
        if is_target:
            mean_target_tensor = torch.tensor(stats['mean_target'], device=device)
            std_target_tensor = torch.tensor(stats['std_target'], device=device)
            x_normalized = (x - mean_target_tensor) / std_target_tensor
        else:
            x_normalized = x.clone()
            mean_flow_tensor = torch.tensor(stats['mean_flow'], device=device)
            std_flow_tensor = torch.tensor(stats['std_flow'], device=device)
            
            x_normalized[:, :, 0, :] = (x[:, :, 0, :] - mean_flow_tensor) / std_flow_tensor
            if in_channels > 1:  # 只有当特征数大于1时才对占用率和速度归一化
                mean_occupancy_tensor = torch.tensor(stats['mean_occupancy'], device=device)
                std_occupancy_tensor = torch.tensor(stats['std_occupancy'], device=device)
                mean_speed_tensor = torch.tensor(stats['mean_speed'], device=device)
                std_speed_tensor = torch.tensor(stats['std_speed'], device=device)
                
                x_normalized[:, :, 1, :] = (x[:, :, 1, :] - mean_occupancy_tensor) / std_occupancy_tensor
                x_normalized[:, :, 2, :] = (x[:, :, 2, :] - mean_speed_tensor) / std_speed_tensor
        return x_normalized.cpu().numpy()
    
    x_w_norm = normalize(x_w_tensor) if x_w_tensor is not None else None
    x_d_norm = normalize(x_d_tensor) if x_d_tensor is not None else None
    x_h_norm = normalize(x_h_tensor) if x_h_tensor is not None else None
    target_norm = normalize(target_tensor, is_target=True) if target_tensor is not None else None

    # 在 normalization 函数后打印归一化后的统计信息
    print("\nNormalized data statistics:")
    print(f"target_norm min: {target_norm.min():.6f}, max: {target_norm.max():.6f}")
    print(f"x_h_norm (flow) min: {x_h_norm[:, :, 0, :].min():.6f}, max: {x_h_norm[:, :, 0, :].max():.6f}")
    if in_channels > 1:  # 只有当特征数大于1时才打印占用率和速度
        print(f"x_h_norm (occupancy) min: {x_h_norm[:, :, 1, :].min():.6f}, max: {x_h_norm[:, :, 1, :].max():.6f}")
        print(f"x_h_norm (speed) min: {x_h_norm[:, :, 2, :].min():.6f}, max: {x_h_norm[:, :, 2, :].max():.6f}")
    
    return stats, x_w_norm, x_d_norm, x_h_norm, target_norm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default='configurations/Orion_config.conf', type=str,
                        help="configuration file path")
    args = parser.parse_args()
    config = configparser.ConfigParser()
    print('Read configuration file: %s' % args.config)
    config.read(args.config)
    
    data_config = config['Data']
    training_config = config['Training']
    
    adj_filename = data_config['adj_filename']
    graph_signal_matrix_filename = data_config['graph_signal_matrix_filename']
    num_of_vertices = int(data_config['num_of_vertices'])
    points_per_hour = int(data_config['points_per_hour'])
    num_for_predict = int(data_config['num_for_predict'])
    len_input = int(data_config['len_input'])
    shuffle_data = data_config.getboolean('shuffle_data')  # 读取是否打乱数据
    
    num_of_weeks = int(training_config['num_of_weeks'])
    num_of_days = int(training_config['num_of_days'])
    num_of_hours = int(training_config['num_of_hours'])
    in_channels = int(training_config['in_channels'])  # 读取输入特征数量
    
    # Determine device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    all_data = read_and_generate_dataset(graph_signal_matrix_filename, adj_filename, num_of_weeks, num_of_days,
                                         num_of_hours, num_for_predict, points_per_hour, len_input,
                                         in_channels=in_channels, shuffle_data=shuffle_data, save=True, device=device)