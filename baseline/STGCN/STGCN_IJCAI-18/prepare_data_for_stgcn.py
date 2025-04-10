import os
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

# Create output directory if it doesn't exist
output_dir = "/root/python_on_hyy/data_for_benchmark"
os.makedirs(output_dir, exist_ok=True)

# Define dataset configurations
DATASETS = {
    "PEMS03": {"nodes": 358, "time_interval": 5, "features": ["flow"], "feature_dim": 1},
    "PEMS04": {"nodes": 307, "time_interval": 5, "features": ["flow", "speed", "occ"], "feature_dim": 3},
    "PEMS08": {"nodes": 170, "time_interval": 5, "features": ["flow", "speed", "occ"], "feature_dim": 3},
    "NYCBike2_part1": {"nodes": 200, "time_interval": 30, "features": ["flow", "in_flow", "out_flow", "weather"], "feature_dim": 4},
    "NYCTaxi_part1": {"nodes": 200, "time_interval": 30, "features": ["flow", "temperature", "time_index"], "feature_dim": 3},
    "Taxi_BJ_hist12_pred12_group1": {"nodes": 128, "time_interval": 30, "features": ["flow", "temperature", "wind_speed", "weather_index"], "feature_dim": 4}
}

# Paths to raw data
RAW_DATA_PATH = "/root/python_on_hyy/data"

def load_npz_data(dataset_name):
    """Load .npz data file."""
    npz_path = os.path.join(RAW_DATA_PATH, f"{dataset_name}.npz")
    data = np.load(npz_path)
    return data['data']  # Shape: (T, N, F)

def load_csv_adj(dataset_name):
    """Load adjacency matrix from .csv file."""
    csv_path = os.path.join(RAW_DATA_PATH, f"{dataset_name}.csv")
    # Convert edge list to adjacency matrix
    adj_df = pd.read_csv(csv_path)
    n_nodes = DATASETS[dataset_name]["nodes"]
    adj_matrix = np.zeros((n_nodes, n_nodes))
    for _, row in adj_df.iterrows():
        from_node, to_node, cost = int(row['from']), int(row['to']), row['cost']
        adj_matrix[from_node, to_node] = cost
        adj_matrix[to_node, from_node] = cost  # Assuming undirected graph
    return adj_matrix  # Shape: (N, N)

def linear_interpolation(data):
    """Fill missing values using linear interpolation."""
    T, N, F = data.shape
    for n in range(N):
        for f in range(F):
            series = data[:, n, f]
            mask = np.isnan(series)
            if np.any(mask):
                x = np.arange(T)
                x_valid = x[~mask]
                y_valid = series[~mask]
                if len(x_valid) > 1:  # Need at least 2 points for interpolation
                    f_interp = interp1d(x_valid, y_valid, kind='linear', fill_value="extrapolate")
                    series[mask] = f_interp(x[mask])
                else:
                    series[mask] = 0  # If not enough points, fill with 0
            data[:, n, f] = series
    return data

def z_score(data):
    """Apply Z-Score normalization."""
    # Compute mean and std across time, samples, and nodes (axis 0, 1, 2)
    mean = np.mean(data, axis=(0, 1, 2), keepdims=True)
    std = np.std(data, axis=(0, 1, 2), keepdims=True)
    std[std == 0] = 1  # Avoid division by zero
    return (data - mean) / std, mean, std

def prepare_data_for_stgcn(dataset_name, n_his=12, n_pred=12):
    """Prepare data for STGCN."""
    config = DATASETS[dataset_name]
    n_route = config["nodes"]
    time_interval = config["time_interval"]
    C_0 = config["feature_dim"]  # Number of features

    # Load data
    data = load_npz_data(dataset_name)  # Shape: (T, N, F)
    adj_matrix = load_csv_adj(dataset_name)  # Shape: (N, N)

    # Fill missing values
    data = linear_interpolation(data)

    # Calculate day_slot based on time interval
    day_slot = int(24 * 60 / time_interval)  # e.g., 288 for 5 min, 48 for 30 min

    # Split data into train/val/test (60%-20%-20%)
    total_days = len(data) // day_slot
    n_train = int(total_days * 0.6)
    n_val = int(total_days * 0.2)
    n_test = total_days - n_train - n_val

    # Generate sequences
    def seq_gen(len_seq, data_seq, offset):
        n_frame = n_his + n_pred
        n_slot = day_slot - n_frame + 1
        tmp_seq = np.zeros((len_seq * n_slot, n_frame, n_route, C_0))
        for i in range(len_seq):
            for j in range(n_slot):
                sta = (i + offset) * day_slot + j
                end = sta + n_frame
                # Boundary check to avoid index out of bounds
                if end > len(data_seq):
                    end = len(data_seq)
                if sta >= end:  # Skip if start index exceeds end index
                    continue
                tmp_seq[i * n_slot + j, :end-sta, :, :] = data_seq[sta:end, :, :]
        return tmp_seq

    seq_train = seq_gen(n_train, data, 0)
    seq_val = seq_gen(n_val, data, n_train)
    seq_test = seq_gen(n_test, data, n_train + n_val)

    # Z-Score normalization (only on training data)
    x_train, mean, std = z_score(seq_train)
    x_val = (seq_val - mean) / std
    x_test = (seq_test - mean) / std

    # Save processed data
    np.savez(os.path.join(output_dir, f"{dataset_name}_data.npz"),
             train=x_train, val=x_val, test=x_test, mean=mean, std=std)
    np.save(os.path.join(output_dir, f"{dataset_name}_adj.npy"), adj_matrix)

    return day_slot, C_0

# Process all datasets
for dataset_name in DATASETS.keys():
    day_slot, C_0 = prepare_data_for_stgcn(dataset_name)
    print(f"Processed {dataset_name} with day_slot={day_slot}, features={C_0}")