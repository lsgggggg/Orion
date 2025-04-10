from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import pandas as pd
import pickle


def get_adjacency_matrix(distance_df, sensor_ids, normalized_k=0.1, dataset_name="PEMS08"):
    """
    :param distance_df: data frame with three columns: [from, to, distance].
    :param sensor_ids: list of sensor ids.
    :param normalized_k: entries that become lower than normalized_k after normalization are set to zero for sparsity.
    :param dataset_name: name of the dataset (e.g., "PEMS08"), used to determine ID prefix.
    :return:
    """
    num_sensors = len(sensor_ids)
    dist_mx = np.zeros((num_sensors, num_sensors), dtype=np.float32)
    dist_mx[:] = np.inf
    # Builds sensor id to index map.
    sensor_id_to_ind = {}
    
    # 从sensor_ids中提取数字部分用于匹配
    numeric_id_to_sensor_id = {}
    for i, sensor_id in enumerate(sensor_ids):
        sensor_id_to_ind[sensor_id] = i
        # 假设格式为 "sensor_X" 提取X部分
        if sensor_id.startswith("sensor_"):
            numeric_id = sensor_id.split("_")[1]
            numeric_id_to_sensor_id[numeric_id] = sensor_id

    # 调试：打印 sensor_id_to_ind
    print("Sensor IDs from graph_sensor_ids.txt:", list(sensor_id_to_ind.keys())[:10], "...")
    print("Extracted numeric IDs:", list(numeric_id_to_sensor_id.keys())[:10], "...")

    # 调试：打印 distance_df 中的传感器 ID
    from_ids = set(distance_df['from'].astype(str))
    to_ids = set(distance_df['to'].astype(str))
    all_ids_in_distances = from_ids.union(to_ids)
    print("Sensor IDs in distances.csv:", list(all_ids_in_distances)[:10], "...")

    # Fills cells in the matrix with distances.
    for row in distance_df.values:
        from_id, to_id, distance = row[0], row[1], row[2]
        # 确保 ID 是字符串类型
        from_id, to_id = str(from_id), str(to_id)
        
        # 尝试通过数字ID匹配到完整的sensor_id
        from_sensor_id = numeric_id_to_sensor_id.get(from_id)
        to_sensor_id = numeric_id_to_sensor_id.get(to_id)
        
        if from_sensor_id is None or to_sensor_id is None:
            continue
            
        if from_sensor_id not in sensor_id_to_ind or to_sensor_id not in sensor_id_to_ind:
            continue
            
        dist_mx[sensor_id_to_ind[from_sensor_id], sensor_id_to_ind[to_sensor_id]] = distance

    # 调试：检查未匹配的传感器 ID
    matched_ids = set(numeric_id_to_sensor_id.keys())
    unmatched_ids = all_ids_in_distances - matched_ids
    print("Unmatched sensor IDs (in distances.csv but not matched to sensor_ids):", list(unmatched_ids)[:20], "...")
    print("Number of matched IDs:", len(matched_ids & all_ids_in_distances))

    # Calculates the standard deviation as theta.
    distances = dist_mx[~np.isinf(dist_mx)].flatten()
    print("Number of valid distances:", len(distances))

    if len(distances) == 0:
        print("Warning: No valid distances found. Using an identity matrix as the adjacency matrix.")
        adj_mx = np.eye(num_sensors, dtype=np.float32)  # 单位矩阵
        std = 1.0  # 设置一个默认值，避免后续计算问题
    else:
        print("Distances mean:", np.mean(distances))
        print("Distances min:", np.min(distances))
        print("Distances max:", np.max(distances))
        std = distances.std()
        print("Standard deviation of distances:", std)
        if std == 0:
            print("Warning: Standard deviation is 0. Using default std=1e-6.")
            std = 1e-6

        # 高斯核计算邻接矩阵
        adj_mx = np.exp(-np.square(dist_mx / std))

    # 确保对角线为 0
    np.fill_diagonal(adj_mx, 0)

    # 检查邻接矩阵是否存在 NaN 或 Inf
    if np.any(np.isnan(adj_mx)):
        print("Warning: Adjacency matrix contains NaN values. Replacing with 0.")
        adj_mx = np.nan_to_num(adj_mx, nan=0.0)
    if np.any(np.isinf(adj_mx)):
        print("Warning: Adjacency matrix contains Inf values. Replacing with 0.")
        adj_mx = np.nan_to_num(adj_mx, posinf=0.0, neginf=0.0)

    # Make the adjacent matrix symmetric by taking the max.
    adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])

    # Sets entries that lower than a threshold, i.e., k, to zero for sparsity.
    adj_mx[adj_mx < normalized_k] = 0

    # 再次检查邻接矩阵
    print("Adjacency matrix shape:", adj_mx.shape)
    print("Adjacency matrix mean:", np.mean(adj_mx))
    print("Adjacency matrix min:", np.min(adj_mx))
    print("Adjacency matrix max:", np.max(adj_mx))
    print("Contains NaN:", np.any(np.isnan(adj_mx)))
    print("Contains Inf:", np.any(np.isinf(adj_mx)))

    return sensor_ids, sensor_id_to_ind, adj_mx


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sensor_ids_filename', type=str, default='data/sensor_graph/graph_sensor_ids.txt',
                        help='File containing sensor ids separated by comma.')
    parser.add_argument('--distances_filename', type=str, default='data/sensor_graph/distances_la_2012.csv',
                        help='CSV file containing sensor distances with three columns: [from, to, distance].')
    parser.add_argument('--normalized_k', type=float, default=0.1,
                        help='Entries that become lower than normalized_k after normalization are set to zero for sparsity.')
    parser.add_argument('--output_pkl_filename', type=str, default='data/sensor_graph/adj_mat.pkl',
                        help='Path of the output file.')
    args = parser.parse_args()

    with open(args.sensor_ids_filename) as f:
        sensor_ids = f.read().strip().split(',')
    print(f"Loaded {len(sensor_ids)} sensor IDs from {args.sensor_ids_filename}")

    distance_df = pd.read_csv(args.distances_filename, dtype={'from': 'str', 'to': 'str'})
    print(f"Loaded {len(distance_df)} distance entries from {args.distances_filename}")

    # 从 distances_filename 中提取数据集名称
    dataset_name = args.distances_filename.split('/')[-1].split('_')[1].split('.')[0]  # 例如 "distances_PEMS08.csv" -> "PEMS08"
    print(f"Dataset name: {dataset_name}")

    normalized_k = args.normalized_k
    sensor_ids, sensor_id_to_ind, adj_mx = get_adjacency_matrix(distance_df, sensor_ids, normalized_k, dataset_name)

    # Save to pickle file.
    with open(args.output_pkl_filename, 'wb') as f:
        pickle.dump([sensor_ids, sensor_id_to_ind, adj_mx], f, protocol=2)
    print(f"Saved adjacency matrix to {args.output_pkl_filename}")