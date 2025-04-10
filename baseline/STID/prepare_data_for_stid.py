import os
import json
import shutil
import numpy as np
import pandas as pd
import pickle

# 确保输出目录存在
output_base_dir = "/root/python_on_hyy/data_for_benchmark"
if not os.path.exists(output_base_dir):
    os.makedirs(output_base_dir)

# 数据集配置
datasets = [
    {
        "name": "PEMS03",
        "npz_path": "/root/python_on_hyy/data/PEMS03.npz",
        "csv_path": "/root/python_on_hyy/data/PEMS03.csv",
        "shape": (26208, 358, 1),
        "features": ["traffic_flow"],
        "frequency": 5,  # 5分钟
        "steps_per_day": 288  # 1440分钟 / 5分钟 = 288
    },
    {
        "name": "PEMS04",
        "npz_path": "/root/python_on_hyy/data/PEMS04.npz",
        "csv_path": "/root/python_on_hyy/data/PEMS04.csv",
        "shape": (16992, 307, 3),
        "features": ["traffic_flow", "occupancy", "speed"],
        "frequency": 5,
        "steps_per_day": 288
    },
    {
        "name": "PEMS08",
        "npz_path": "/root/python_on_hyy/data/PEMS08.npz",
        "csv_path": "/root/python_on_hyy/data/PEMS08.csv",
        "shape": (17856, 170, 3),
        "features": ["traffic_flow", "occupancy", "speed"],
        "frequency": 5,
        "steps_per_day": 288
    },
    {
        "name": "NYCBike2_part1",
        "npz_path": "/root/python_on_hyy/data/NYCBike2_part1.npz",
        "csv_path": "/root/python_on_hyy/data/NYCBike2_part1.csv",
        "shape": (11952, 200, 4),
        "features": ["traffic_flow", "unknown1", "unknown2", "unknown3"],
        "frequency": 30,
        "steps_per_day": 48  # 1440分钟 / 30分钟 = 48
    },
    {
        "name": "NYCTaxi_part1",
        "npz_path": "/root/python_on_hyy/data/NYCTaxi_part1.npz",
        "csv_path": "/root/python_on_hyy/data/NYCTaxi_part1.csv",
        "shape": (11952, 200, 3),
        "features": ["traffic_flow", "unknown1", "unknown2"],
        "frequency": 30,
        "steps_per_day": 48
    },
    {
        "name": "Taxi_BJ_hist12_pred12_group1",
        "npz_path": "/root/python_on_hyy/data/Taxi_BJ_hist12_pred12_group1.npz",
        "csv_path": "/root/python_on_hyy/data/Taxi_BJ_hist12_pred12_group1.csv",
        "shape": (14163, 128, 4),
        "features": ["traffic_flow", "temperature", "wind_speed", "weather_index"],
        "frequency": 30,
        "steps_per_day": 48
    }
]

# 通用设置
regular_settings = {
    "INPUT_LEN": 12,
    "OUTPUT_LEN": 12,
    "TRAIN_VAL_TEST_RATIO": [0.6, 0.2, 0.2],
    "NORM_EACH_CHANNEL": False,
    "RESCALE": True,
    "METRICS": ["MAE", "RMSE", "MAPE"],
    "NULL_VAL": 0.0
}

def generate_adj_mx(csv_path, num_nodes, output_path):
    """从 CSV 文件生成邻接矩阵并保存为 pickle 文件"""
    df = pd.read_csv(csv_path)
    adj_mx = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for _, row in df.iterrows():
        from_node, to_node, cost = int(row['from']), int(row['to']), float(row['cost'])
        if from_node < num_nodes and to_node < num_nodes:
            adj_mx[from_node, to_node] = cost
            adj_mx[to_node, from_node] = cost  # 无向图
    with open(output_path, 'wb') as f:
        pickle.dump([list(range(num_nodes)), list(range(num_nodes)), adj_mx], f)
    print(f"Adjacency matrix saved to {output_path}")

def load_and_preprocess_data(dataset):
    """加载和预处理数据"""
    data = np.load(dataset["npz_path"])["data"]
    assert data.shape == dataset["shape"], f"Data shape mismatch for {dataset['name']}"
    return data

def add_temporal_features(data, steps_per_day):
    """添加时间特征：time_of_day 和 day_of_week"""
    l, n, _ = data.shape
    feature_list = [data]

    # 添加 time_of_day
    time_of_day = np.array([i % steps_per_day / steps_per_day for i in range(l)])
    time_of_day_tiled = np.tile(time_of_day, [1, n, 1]).transpose((2, 1, 0))
    feature_list.append(time_of_day_tiled)

    # 添加 day_of_week
    day_of_week = np.array([(i // steps_per_day) % 7 / 7 for i in range(l)])
    day_of_week_tiled = np.tile(day_of_week, [1, n, 1]).transpose((2, 1, 0))
    feature_list.append(day_of_week_tiled)

    data_with_features = np.concatenate(feature_list, axis=-1)  # L x N x (C+2)
    return data_with_features

def save_data(data, output_dir):
    """保存预处理后的数据"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    file_path = os.path.join(output_dir, "data.dat")
    fp = np.memmap(file_path, dtype="float32", mode="w+", shape=data.shape)
    fp[:] = data[:]
    fp.flush()
    del fp
    print(f"Data saved to {file_path}")

def save_graph(dataset, output_dir):
    """保存邻接矩阵"""
    output_graph_path = os.path.join(output_dir, "adj_mx.pkl")
    generate_adj_mx(dataset["csv_path"], dataset["shape"][1], output_graph_path)

def save_description(dataset, data, output_dir):
    """保存数据集描述文件"""
    description = {
        "name": dataset["name"],
        "domain": "traffic flow",
        "shape": data.shape,
        "num_time_steps": data.shape[0],
        "num_nodes": data.shape[1],
        "num_features": data.shape[2],
        "feature_description": dataset["features"] + ["time_of_day", "day_of_week"],
        "has_graph": True,
        "frequency (minutes)": dataset["frequency"],
        "regular_settings": regular_settings
    }
    description_path = os.path.join(output_dir, "desc.json")
    with open(description_path, "w") as f:
        json.dump(description, f, indent=4)
    print(f"Description saved to {description_path}")

def main():
    for dataset in datasets:
        print(f"Processing dataset: {dataset['name']}")
        output_dir = os.path.join(output_base_dir, dataset["name"])

        # 加载和预处理数据
        data = load_and_preprocess_data(dataset)

        # 添加时间特征
        data_with_features = add_temporal_features(data, dataset["steps_per_day"])

        # 保存数据
        save_data(data_with_features, output_dir)

        # 保存邻接矩阵
        save_graph(dataset, output_dir)

        # 保存描述文件
        save_description(dataset, data_with_features, output_dir)

if __name__ == "__main__":
    main()