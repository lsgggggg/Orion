import os
import numpy as np
import pandas as pd
import argparse
import pickle
from tqdm import tqdm

def generate_graph_seq2seq_data(data, num_nodes, flow_dims=[0], seq_length_x=12, seq_length_y=12):
    T, actual_nodes, features = data.shape
    if actual_nodes != num_nodes:
        raise ValueError(f"Data has {actual_nodes} nodes, but expected {num_nodes} nodes based on data shape.")
    
    min_t = seq_length_x - 1
    max_t = T - seq_length_y
    x, y = [], []
    
    for t in tqdm(range(min_t, max_t), desc="Generating samples"):
        # 输入 x 包含所有特征
        x_t = data[t - seq_length_x + 1 : t + 1, :, :]  # [seq_length_x, num_nodes, features]
        # 输出 y 只包含流量（第0维）
        y_t = data[t + 1 : t + 1 + seq_length_y, :, flow_dims]  # [seq_length_y, num_nodes, len(flow_dims)]
        x.append(x_t.transpose(1, 0, 2))  # [num_nodes, seq_length_x, features]
        y.append(y_t.transpose(1, 0, 2))  # [num_nodes, seq_length_y, 1]
    
    x = np.stack(x, axis=0)  # [num_samples, num_nodes, seq_length_x, features]
    y = np.stack(y, axis=0)  # [num_samples, num_nodes, seq_length_y, 1]
    return x, y

def process_all_datasets(data_configs, output_dir="data/processed"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for config in data_configs:
        print(f"Processing {config['name']}...")
        data = np.load(config['npz_file'])['data']
        edges = pd.read_csv(config['csv_file']).values
        num_nodes = data.shape[1]  # 使用数据本身的节点数
        flow_dims = [0]  # 只预测第0维（流量）
        
        x, y = generate_graph_seq2seq_data(data, num_nodes, flow_dims=flow_dims)
        print(f"x shape: {x.shape}, y shape: {y.shape}")
        
        num_samples = x.shape[0]
        num_train = int(num_samples * 0.6)
        num_val = int(num_samples * 0.2)
        num_test = num_samples - num_train - num_val
        
        x_train, y_train = x[:num_train], y[:num_train]
        x_val, y_val = x[num_train:num_train + num_val], y[num_train:num_train + num_val]
        x_test, y_test = x[num_train + num_val:], y[num_train + num_val:]
        
        output_file = os.path.join(output_dir, f"{config['name']}_graphwavenet.npz")
        np.savez_compressed(
            output_file,
            x_train=x_train, y_train=y_train,
            x_val=x_val, y_val=y_val,
            x_test=x_test, y_test=y_test
        )
        print(f"Saved to {output_file}")
        
        adj_mx = np.zeros((num_nodes, num_nodes), dtype=np.float32)
        for from_node, to_node, cost in edges:
            from_node, to_node = int(from_node), int(to_node)
            if 0 <= from_node < num_nodes and 0 <= to_node < num_nodes:
                adj_mx[from_node, to_node] = 1.0 / (float(cost) + 1e-6)
        adj_mx = adj_mx + adj_mx.T
        rowsum = np.sum(adj_mx, axis=1)
        rowsum[rowsum == 0] = 1.0
        adj_mx = adj_mx / rowsum[:, None]
        sensor_ids = [str(i) for i in range(num_nodes)]
        sensor_id_to_ind = {str(i): i for i in range(num_nodes)}
        adj_file = os.path.join(output_dir, f"{config['name']}_adj_mx.pkl")
        with open(adj_file, 'wb') as f:
            pickle.dump([sensor_ids, sensor_id_to_ind, adj_mx], f)
        print(f"Adjacency matrix saved to {adj_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/root/python_on_hyy/data", help="Root directory of data")
    parser.add_argument("--output_dir", type=str, default="data/processed", help="Output directory")
    args = parser.parse_args()

    data_configs = [
        {"name": "PEMS03", "npz_file": os.path.join(args.data_dir, "PEMS03.npz"), "csv_file": os.path.join(args.data_dir, "PEMS03.csv")},
        {"name": "PEMS04", "npz_file": os.path.join(args.data_dir, "PEMS04.npz"), "csv_file": os.path.join(args.data_dir, "PEMS04.csv")},
        {"name": "PEMS08", "npz_file": os.path.join(args.data_dir, "PEMS08.npz"), "csv_file": os.path.join(args.data_dir, "PEMS08.csv")},
        {"name": "NYCBike2_part1", "npz_file": os.path.join(args.data_dir, "NYCBike2_part1.npz"), "csv_file": os.path.join(args.data_dir, "NYCBike2_part1.csv")},
        {"name": "NYCTaxi_part1", "npz_file": os.path.join(args.data_dir, "NYCTaxi_part1.npz"), "csv_file": os.path.join(args.data_dir, "NYCTaxi_part1.csv")},
        {"name": "Taxi_BJ_hist12_pred12_group1", "npz_file": os.path.join(args.data_dir, "Taxi_BJ_hist12_pred12_group1.npz"), "csv_file": os.path.join(args.data_dir, "Taxi_BJ_hist12_pred12_group1.csv")},
    ]
    
    process_all_datasets(data_configs, args.output_dir)