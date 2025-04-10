import os
import pandas as pd
import numpy as np
import argparse
from datetime import datetime, timedelta

# 数据集信息
DATASETS = {
    "PEMS03": {
        "npz_file": "/root/python_on_hyy/data/PEMS03.npz",
        "csv_file": "/root/python_on_hyy/data/PEMS03.csv",
        "shape": (26208, 358, 1),
        "interval_minutes": 5,
        "features": ["flow"]
    },
    "PEMS04": {
        "npz_file": "/root/python_on_hyy/data/PEMS04.npz",
        "csv_file": "/root/python_on_hyy/data/PEMS04.csv",
        "shape": (16992, 307, 3),
        "interval_minutes": 5,
        "features": ["flow", "speed", "occupancy"]
    },
    "PEMS08": {
        "npz_file": "/root/python_on_hyy/data/PEMS08.npz",
        "csv_file": "/root/python_on_hyy/data/PEMS08.csv",
        "shape": (17856, 170, 3),
        "interval_minutes": 5,
        "features": ["flow", "speed", "occupancy"]
    },
    "NYCBike2_part1": {
        "npz_file": "/root/python_on_hyy/data/NYCBike2_part1.npz",
        "csv_file": "/root/python_on_hyy/data/NYCBike2_part1.csv",
        "shape": (11952, 200, 4),
        "interval_minutes": 30,
        "features": ["flow", "feature1", "feature2", "feature3"]
    },
    "NYCTaxi_part1": {
        "npz_file": "/root/python_on_hyy/data/NYCTaxi_part1.npz",
        "csv_file": "/root/python_on_hyy/data/NYCTaxi_part1.csv",
        "shape": (11952, 200, 3),
        "interval_minutes": 30,
        "features": ["flow", "feature1", "feature2"]
    },
    "Taxi_BJ_hist12_pred12_group1": {
        "npz_file": "/root/python_on_hyy/data/Taxi_BJ_hist12_pred12_group1.npz",
        "csv_file": "/root/python_on_hyy/data/Taxi_BJ_hist12_pred12_group1.csv",
        "shape": (14163, 128, 4),
        "interval_minutes": 30,
        "features": ["flow", "temperature", "wind_speed", "weather_index"]
    }
}

OUTPUT_DIR = "/root/python_on_hyy/data_for_benchmark"

def ensure_directory(directory):
    """确保目录存在，如果不存在则创建"""
    if not os.path.exists(directory):
        os.makedirs(directory)

def convert_npz_to_hdf5(dataset_name, dataset_info):
    """将.npz文件转换为HDF5格式"""
    # 读取.npz文件
    data_dict = np.load(dataset_info["npz_file"])
    # 假设数据存储在键 'data' 中（根据实际情况调整）
    data = data_dict['data']  # 如果键名不是 'data'，请根据 print(data_dict.files) 的输出调整
    num_timesteps, num_nodes, num_features = dataset_info["shape"]
    
    # 验证数据形状
    if data.shape != dataset_info["shape"]:
        raise ValueError(f"Data shape {data.shape} does not match expected shape {dataset_info['shape']} for {dataset_name}")
    
    # 生成时间戳
    start_time = datetime(2023, 1, 1, 0, 0)  # 假设起始时间为2023-01-01 00:00
    interval = timedelta(minutes=dataset_info["interval_minutes"])
    timestamps = [start_time + i * interval for i in range(num_timesteps)]
    
    # 创建DataFrame，包含所有特征
    # 将数据展平为 (num_timesteps, num_nodes * num_features)
    data_flat = data.reshape(num_timesteps, num_nodes * num_features)
    df = pd.DataFrame(
        data_flat,
        index=timestamps,
        columns=[f"sensor_{i}_feat_{j}" for i in range(num_nodes) for j in range(num_features)]
    )
    
    # 保存为HDF5文件
    output_hdf5 = os.path.join(OUTPUT_DIR, dataset_name, f"{dataset_name}.h5")
    ensure_directory(os.path.dirname(output_hdf5))
    df.to_hdf(output_hdf5, key='df', mode='w')
    print(f"Saved {dataset_name} to {output_hdf5}")

def generate_sensor_ids(dataset_name, num_nodes):
    """生成graph_sensor_ids.txt"""
    sensor_ids = [f"sensor_{i}" for i in range(num_nodes)]
    output_file = os.path.join(OUTPUT_DIR, dataset_name, "graph_sensor_ids.txt")
    with open(output_file, 'w') as f:
        f.write(','.join(sensor_ids))
    print(f"Saved sensor IDs for {dataset_name} to {output_file}")

def copy_distances_csv(dataset_name, csv_file):
    """复制.csv文件到目标目录"""
    output_csv = os.path.join(OUTPUT_DIR, dataset_name, f"distances_{dataset_name}.csv")
    os.system(f"cp {csv_file} {output_csv}")
    print(f"Copied distances CSV for {dataset_name} to {output_csv}")

def generate_graph_seq2seq_io_data(
        df, x_offsets, y_offsets, add_time_in_day=False, add_day_in_week=False, scaler=None
):
    """从DataFrame生成序列到序列的输入输出数据（与generate_training_data.py相同）"""
    num_samples, num_columns = df.shape
    # 计算节点数量和特征数量
    num_features = len(DATASETS[df.name]["features"])  # 从 DATASETS 中获取特征数量
    num_nodes = num_columns // num_features
    
    # 重塑数据为 (num_samples, num_nodes, num_features)
    data = df.values.reshape(num_samples, num_nodes, num_features)
    
    # 输入 x 包含所有特征
    x_data = data
    
    # 输出 y 仅包含第一个特征（流量）
    y_data = data[:, :, 0:1]  # 只取第 0 个特征（流量）
    
    x, y = [], []
    min_t = abs(min(x_offsets))
    max_t = abs(num_samples - abs(max(y_offsets)))
    for t in range(min_t, max_t):
        x_t = x_data[t + x_offsets, ...]  # 包含所有特征
        y_t = y_data[t + y_offsets, ...]  # 仅包含流量
        x.append(x_t)
        y.append(y_t)
    x = np.stack(x, axis=0)  # 形状为 (num_samples, seq_len, num_nodes, num_features)
    y = np.stack(y, axis=0)  # 形状为 (num_samples, horizon, num_nodes, 1)
    return x, y

def generate_train_val_test(dataset_name, hdf5_file):
    """为每个数据集生成train/val/test的npz文件"""
    df = pd.read_hdf(hdf5_file)
    df.name = dataset_name  # 将 dataset_name 附加到 df，以便在 generate_graph_seq2seq_io_data 中使用
    x_offsets = np.sort(np.arange(-11, 1, 1))  # 历史12个时间步
    y_offsets = np.sort(np.arange(1, 13, 1))  # 预测未来12个时间步
    x, y = generate_graph_seq2seq_io_data(
        df,
        x_offsets=x_offsets,
        y_offsets=y_offsets,
        add_time_in_day=False,  # 保持 False
        add_day_in_week=False,  # 保持 False
    )

    print(f"{dataset_name} - x shape: {x.shape}, y shape: {y.shape}")
    num_samples = x.shape[0]
    num_test = round(num_samples * 0.2)
    num_train = round(num_samples * 0.6)  # 60%训练
    num_val = num_samples - num_test - num_train  # 20%验证

    # 划分数据集
    x_train, y_train = x[:num_train], y[:num_train]
    x_val, y_val = x[num_train:num_train + num_val], y[num_train:num_train + num_val]
    x_test, y_test = x[-num_test:], y[-num_test:]

    # 保存到npz文件
    dataset_output_dir = os.path.join(OUTPUT_DIR, dataset_name)
    for cat in ["train", "val", "test"]:
        _x, _y = locals()["x_" + cat], locals()["y_" + cat]
        print(f"{dataset_name} {cat} - x: {_x.shape}, y: {_y.shape}")
        np.savez_compressed(
            os.path.join(dataset_output_dir, f"{cat}.npz"),
            x=_x,
            y=_y,
            x_offsets=x_offsets.reshape(list(x_offsets.shape) + [1]),
            y_offsets=y_offsets.reshape(list(y_offsets.shape) + [1]),
        )

def main():
    # 创建输出目录
    ensure_directory(OUTPUT_DIR)

    # 处理每个数据集
    for dataset_name, dataset_info in DATASETS.items():
        print(f"\nProcessing {dataset_name}...")
        
        # 转换为HDF5格式
        convert_npz_to_hdf5(dataset_name, dataset_info)
        
        # 生成sensor_ids
        num_nodes = dataset_info["shape"][1]
        generate_sensor_ids(dataset_name, num_nodes)
        
        # 复制distances CSV
        copy_distances_csv(dataset_name, dataset_info["csv_file"])
        
        # 生成train/val/test数据
        hdf5_file = os.path.join(OUTPUT_DIR, dataset_name, f"{dataset_name}.h5")
        generate_train_val_test(dataset_name, hdf5_file)

if __name__ == "__main__":
    main()