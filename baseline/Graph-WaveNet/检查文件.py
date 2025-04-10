import numpy as np
data_configs = [
    {"name": "PEMS03", "npz_file": "/root/python_on_hyy/Graph-WaveNet/data/PEMS03.npz", "num_nodes": 358},
    {"name": "PEMS04", "npz_file": "/root/python_on_hyy/Graph-WaveNet/data/PEMS04.npz", "num_nodes": 307},
    {"name": "PEMS08", "npz_file": "/root/python_on_hyy/Graph-WaveNet/data/PEMS08.npz", "num_nodes": 170},
    {"name": "NYCBike2_part1", "npz_file": "/root/python_on_hyy/Graph-WaveNet/data/NYCBike2_part1.npz", "num_nodes": 250},
    {"name": "NYCTaxi_part1", "npz_file": "/root/python_on_hyy/Graph-WaveNet/data/NYCTaxi_part1.npz", "num_nodes": 266},
    {"name": "Taxi_BJ_hist12_pred12_group1", "npz_file": "/root/python_on_hyy/Graph-WaveNet/data/Taxi_BJ_hist12_pred12_group1.npz", "num_nodes": 256},
]

for config in data_configs:
    data = np.load(config['npz_file'])['data']
    print(f"{config['name']}: shape = {data.shape}, sample values = {data[0, :5, :]}")