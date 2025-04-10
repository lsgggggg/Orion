import os
import numpy as np
from tqdm import tqdm

def linear_interpolation(data):
    """CPU-based linear interpolation for NaN and zero values."""
    data = data.copy()
    T, N, F = data.shape
    for n in range(N):
        for f in range(F):
            series = data[:, n, f]
            mask = (series == 0) | np.isnan(series)
            if np.all(mask):
                series[mask] = 0
            else:
                valid_idx = np.where(~mask)[0]
                if len(valid_idx) > 0:
                    interp = np.interp(np.arange(T), valid_idx, series[valid_idx])
                    series[mask] = interp[mask]
            data[:, n, f] = series
    return data

def generate_samples(data, len_input=12, num_for_predict=12):
    """Generate samples: (B, N, F, T) -> (B, 1, F, 12) for each node, y as (B, 12)."""
    T, N, F = data.shape
    samples = []
    targets = []
    for t in range(T - len_input - num_for_predict + 1):
        for n in range(N):
            x = data[t:t + len_input, n, :].reshape(1, 1, F, len_input)  # (1, 1, F, 12)
            y = data[t + len_input:t + len_input + num_for_predict, n, 0].reshape(1, num_for_predict)  # (1, 12)
            samples.append(x)
            targets.append(y)
    samples = np.concatenate(samples, axis=0)  # (B, 1, F, 12)
    targets = np.concatenate(targets, axis=0)  # (B, 12)
    print(f"Generated samples: {len(samples)}, targets: {len(targets)}")
    print(f"Samples shape: {samples.shape}, Targets shape: {targets.shape}")
    return samples, targets

def normalize(data, stats=None):
    """Normalize data using mean and std."""
    if stats is None:
        mean = data.mean(axis=0, keepdims=True)  # (1, 12) for y
        std = data.std(axis=0, keepdims=True) + 1e-6  # (1, 12) for y
        stats = {'mean': mean.squeeze(0), 'std': std.squeeze(0)}  # (12,) for saving
    data_norm = (data - stats['mean']) / stats['std']
    return stats, data_norm

def process_dataset(filename, dataset_name, in_channels, save_dir):
    """Process a single dataset."""
    data = np.load(filename)['data']  # (T, N, F)
    print(f"Processing {dataset_name}: Original shape {data.shape}")

    # Fill NaN and zeros
    data = linear_interpolation(data)
    data = np.maximum(data, 0)

    # Generate samples
    x, y = generate_samples(data)
    print(f"Generated samples: x {x.shape}, y {y.shape}")

    # Split: 60% train, 20% val, 20% test
    num_samples = x.shape[0]
    split1 = int(num_samples * 0.6)
    split2 = int(num_samples * 0.8)
    indices = np.arange(num_samples)
    np.random.seed(42)
    np.random.shuffle(indices)

    train_x, train_y = x[indices[:split1]], y[indices[:split1]]
    val_x, val_y = x[indices[split1:split2]], y[indices[split1:split2]]
    test_x, test_y = x[indices[split2:]], y[indices[split2:]]

    # Normalize
    stats, train_x = normalize(train_x)
    _, val_x = normalize(val_x, stats)
    _, test_x = normalize(test_x, stats)
    stats_y, train_y = normalize(train_y)  # y单独归一化
    _, val_y = normalize(val_y, stats_y)
    _, test_y = normalize(test_y, stats_y)

    # Save
    save_path = os.path.join(save_dir, f"{dataset_name}_LSTM.npz")
    np.savez(save_path,
             train_x=train_x, train_y=train_y,
             val_x=val_x, val_y=val_y,
             test_x=test_x, test_y=test_y,
             mean=stats['mean'], std=stats['std'],
             mean_y=stats_y['mean'], std_y=stats_y['std'])
    print(f"Saved to {save_path}")
    print(f"mean_y shape: {stats_y['mean'].shape}, std_y shape: {stats_y['std'].shape}")

if __name__ == "__main__":
    datasets = [
        ("PEMS03", "/root/python_on_hyy/data/PEMS03.npz", 1),
        ("PEMS04", "/root/python_on_hyy/data/PEMS04.npz", 3),
        ("PEMS08", "/root/python_on_hyy/data/PEMS08.npz", 3),
        ("NYCBike2_part1", "/root/python_on_hyy/data/NYCBike2_part1.npz", 4),
        ("NYCTaxi_part1", "/root/python_on_hyy/data/NYCTaxi_part1.npz", 3),
        ("Taxi_BJ_hist12_pred12_group1", "/root/python_on_hyy/data/Taxi_BJ_hist12_pred12_group1.npz", 4),
    ]

    save_dir = "/root/python_on_hyy/data_for_benchmark"
    os.makedirs(save_dir, exist_ok=True)

    for dataset_name, filename, in_channels in datasets:
        process_dataset(filename, dataset_name, in_channels, save_dir)