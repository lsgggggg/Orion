
import os
import numpy as np

# Define dataset configurations (same as in prepare_data_for_stgcn.py)
DATASETS = {
    "PEMS03": {"nodes": 358, "time_interval": 5, "features": ["flow"], "feature_dim": 1},
    "PEMS04": {"nodes": 307, "time_interval": 5, "features": ["flow", "speed", "occ"], "feature_dim": 3},
    "PEMS08": {"nodes": 170, "time_interval": 5, "features": ["flow", "speed", "occ"], "feature_dim": 3},
    "NYCBike2_part1": {"nodes": 200, "time_interval": 30, "features": ["flow", "in_flow", "out_flow", "weather"], "feature_dim": 4},
    "NYCTaxi_part1": {"nodes": 200, "time_interval": 30, "features": ["flow", "temperature", "time_index"], "feature_dim": 3},
    "Taxi_BJ_hist12_pred12_group1": {"nodes": 128, "time_interval": 30, "features": ["flow", "temperature", "wind_speed", "weather_index"], "feature_dim": 4}
}

# Data directory
DATA_DIR = "/root/python_on_hyy/data_for_benchmark"

# STGCN parameters (same as used in training)
N_HIS = 12
N_PRED = 12
N_FRAME = N_HIS + N_PRED  # Total time steps in a sequence

def check_data_file(dataset_name):
    """Check the format and shape of prepared data for a single dataset."""
    config = DATASETS[dataset_name]
    n_route = config["nodes"]
    time_interval = config["time_interval"]
    C_0 = config["feature_dim"]
    day_slot = int(24 * 60 / time_interval)  # e.g., 288 for 5 min, 48 for 30 min

    print(f"\nChecking dataset: {dataset_name}")
    print(f"Expected n_route: {n_route}, C_0: {C_0}, day_slot: {day_slot}")

    # Check .npz file
    npz_path = os.path.join(DATA_DIR, f"{dataset_name}_data.npz")
    if not os.path.exists(npz_path):
        print(f"ERROR: {npz_path} does not exist!")
        return False

    data = np.load(npz_path)
    required_keys = ['train', 'val', 'test', 'mean', 'std']
    for key in required_keys:
        if key not in data:
            print(f"ERROR: {key} not found in {npz_path}!")
            return False

    # Check shapes of train, val, test
    for split in ['train', 'val', 'test']:
        array = data[split]
        expected_shape = (array.shape[0], N_FRAME, n_route, C_0)
        if array.shape[1:] != (N_FRAME, n_route, C_0):
            print(f"ERROR: {split} shape {array.shape} does not match expected {expected_shape}!")
            return False
        # Check for NaN or Inf values
        if np.any(np.isnan(array)) or np.any(np.isinf(array)):
            print(f"ERROR: {split} contains NaN or Inf values!")
            return False
        print(f"{split} shape: {array.shape} - OK")

    # Check mean and std shapes
    for stat in ['mean', 'std']:
        array = data[stat]
        expected_shape = (1, 1, 1, C_0)
        if array.shape != expected_shape:
            print(f"ERROR: {stat} shape {array.shape} does not match expected {expected_shape}!")
            return False
        print(f"{stat} shape: {array.shape} - OK")

    # Check adjacency matrix
    adj_path = os.path.join(DATA_DIR, f"{dataset_name}_adj.npy")
    if not os.path.exists(adj_path):
        print(f"ERROR: {adj_path} does not exist!")
        return False

    adj_matrix = np.load(adj_path)
    expected_shape = (n_route, n_route)
    if adj_matrix.shape != expected_shape:
        print(f"ERROR: Adjacency matrix shape {adj_matrix.shape} does not match expected {expected_shape}!")
        return False
    # Check for NaN or Inf values in adjacency matrix
    if np.any(np.isnan(adj_matrix)) or np.any(np.isinf(adj_matrix)):
        print(f"ERROR: Adjacency matrix contains NaN or Inf values!")
        return False
    print(f"Adjacency matrix shape: {adj_matrix.shape} - OK")

    print(f"Dataset {dataset_name} passed all checks!")
    return True

def main():
    """Check all datasets."""
    all_passed = True
    for dataset_name in DATASETS.keys():
        if not check_data_file(dataset_name):
            all_passed = False

    if all_passed:
        print("\nAll datasets passed the checks! Ready for STGCN training.")
    else:
        print("\nSome datasets failed the checks. Please fix the issues before training.")

if __name__ == "__main__":
    main()