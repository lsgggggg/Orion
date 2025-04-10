import os
import json
import pickle
import numpy as np

# 数据集配置（与 prepare_data_for_stid.py 一致）
datasets = [
    {
        "name": "PEMS03",
        "original_shape": (26208, 358, 1),
        "expected_features": 3,  # 1个原始特征 + 2个时间特征
        "steps_per_day": 288,
        "traffic_max": 10000,  # 假设流量最大值（根据 PEMS 数据集常识）
        "feature_ranges": {
            "traffic_flow": (0, 10000),  # 流量范围
        }
    },
    {
        "name": "PEMS04",
        "original_shape": (16992, 307, 3),
        "expected_features": 5,  # 3个原始特征 + 2个时间特征
        "steps_per_day": 288,
        "traffic_max": 10000,
        "feature_ranges": {
            "traffic_flow": (0, 10000),
            "occupancy": (0, 100),  # 占用率通常为百分比
            "speed": (0, 120),  # 速度（km/h），假设最大为 120
        }
    },
    {
        "name": "PEMS08",
        "original_shape": (17856, 170, 3),
        "expected_features": 5,  # 3个原始特征 + 2个时间特征
        "steps_per_day": 288,
        "traffic_max": 10000,
        "feature_ranges": {
            "traffic_flow": (0, 10000),
            "occupancy": (0, 100),
            "speed": (0, 120),
        }
    },
    {
        "name": "NYCBike2_part1",
        "original_shape": (11952, 200, 4),
        "expected_features": 6,  # 4个原始特征 + 2个时间特征
        "steps_per_day": 48,
        "traffic_max": 500,  # 共享单车流量较小，假设最大为 500
        "feature_ranges": {
            "traffic_flow": (0, 500),
            "unknown1": (0, float('inf')),  # 未知特征，暂不限制
            "unknown2": (0, float('inf')),
            "unknown3": (0, float('inf')),
        }
    },
    {
        "name": "NYCTaxi_part1",
        "original_shape": (11952, 200, 3),
        "expected_features": 5,  # 3个原始特征 + 2个时间特征
        "steps_per_day": 48,
        "traffic_max": 1000,  # 出租车流量，假设最大为 1000
        "feature_ranges": {
            "traffic_flow": (0, 1000),
            "unknown1": (0, float('inf')),
            "unknown2": (0, float('inf')),
        }
    },
    {
        "name": "Taxi_BJ_hist12_pred12_group1",
        "original_shape": (14163, 128, 4),
        "expected_features": 6,  # 4个原始特征 + 2个时间特征
        "steps_per_day": 48,
        "traffic_max": 1000,
        "feature_ranges": {
            "traffic_flow": (0, 1000),
            "temperature": (-30, 50),  # 北京温度范围（摄氏度）
            "wind_speed": (0, 30),  # 风速（m/s），假设最大为 30
            "weather_index": (0, 10),  # 天气类型索引，假设为 0-10
        }
    }
]

# 数据目录
data_base_dir = "/root/python_on_hyy/data_for_benchmark"

def check_files_exist(dataset):
    """检查必要文件是否存在"""
    dataset_dir = os.path.join(data_base_dir, dataset["name"])
    required_files = ["data.dat", "adj_mx.pkl", "desc.json"]
    
    print(f"\nChecking files for {dataset['name']}...")
    for file_name in required_files:
        file_path = os.path.join(dataset_dir, file_name)
        if not os.path.exists(file_path):
            print(f"❌ Error: {file_name} is missing in {dataset_dir}")
            return False
        else:
            print(f"✔ {file_name} exists")
    return True

def check_data_shape(dataset):
    """检查 data.dat 的形状"""
    dataset_dir = os.path.join(data_base_dir, dataset["name"])
    desc_path = os.path.join(dataset_dir, "desc.json")
    
    # 读取描述文件
    with open(desc_path, "r") as f:
        desc = json.load(f)
    
    data_shape = tuple(desc["shape"])
    expected_shape = (dataset["original_shape"][0], dataset["original_shape"][1], dataset["expected_features"])
    
    print(f"\nChecking data shape for {dataset['name']}...")
    print(f"Expected shape: {expected_shape}")
    print(f"Actual shape: {data_shape}")
    
    if data_shape != expected_shape:
        print(f"❌ Error: Data shape mismatch for {dataset['name']}")
        return False
    
    # 加载数据并进一步检查
    data_path = os.path.join(dataset_dir, "data.dat")
    data = np.memmap(data_path, dtype="float32", mode="r", shape=data_shape)
    print(f"✔ Data shape matches expected shape")
    return data

def check_temporal_features(dataset, data):
    """检查时间特征 time_of_day 和 day_of_week"""
    T, N, C = data.shape
    steps_per_day = dataset["steps_per_day"]
    
    print(f"\nChecking temporal features for {dataset['name']}...")
    
    # 检查 time_of_day（倒数第二个特征）
    time_of_day = data[:, 0, -2]  # 取第一个节点的时间特征
    expected_time_of_day = np.array([i % steps_per_day / steps_per_day for i in range(T)])
    
    if not np.allclose(time_of_day, expected_time_of_day, atol=1e-5):
        print(f"❌ Error: time_of_day feature is incorrect for {dataset['name']}")
        return False
    print(f"✔ time_of_day feature is correct")
    
    # 检查 day_of_week（最后一个特征）
    day_of_week = data[:, 0, -1]  # 取第一个节点的时间特征
    expected_day_of_week = np.array([(i // steps_per_day) % 7 / 7 for i in range(T)])
    
    if not np.allclose(day_of_week, expected_day_of_week, atol=1e-5):
        print(f"❌ Error: day_of_week feature is incorrect for {dataset['name']}")
        return False
    print(f"✔ day_of_week feature is correct")
    
    # 检查时间特征分布（确保均匀性）
    time_of_day_bins = np.histogram(time_of_day, bins=10, range=(0, 1))[0]
    expected_bin_count = T / 10  # 假设均匀分布
    if np.any(np.abs(time_of_day_bins - expected_bin_count) > 0.1 * expected_bin_count):
        print(f"❌ Warning: time_of_day distribution is not uniform for {dataset['name']}")
        print(f"Bin counts: {time_of_day_bins}")
    else:
        print(f"✔ time_of_day distribution is uniform")
    
    day_of_week_bins = np.histogram(day_of_week, bins=7, range=(0, 1))[0]
    expected_bin_count = T / 7
    if np.any(np.abs(day_of_week_bins - expected_bin_count) > 0.1 * expected_bin_count):
        print(f"❌ Warning: day_of_week distribution is not uniform for {dataset['name']}")
        print(f"Bin counts: {day_of_week_bins}")
    else:
        print(f"✔ day_of_week distribution is uniform")
    
    return True

def check_adj_matrix(dataset):
    """检查邻接矩阵 adj_mx.pkl"""
    dataset_dir = os.path.join(data_base_dir, dataset["name"])
    adj_path = os.path.join(dataset_dir, "adj_mx.pkl")
    
    with open(adj_path, "rb") as f:
        adj_data = pickle.load(f)
    
    # adj_mx.pkl 包含 [node_ids, node_ids, adj_matrix]
    adj_matrix = adj_data[2]
    expected_shape = (dataset["original_shape"][1], dataset["original_shape"][1])
    
    print(f"\nChecking adjacency matrix for {dataset['name']}...")
    print(f"Expected shape: {expected_shape}")
    print(f"Actual shape: {adj_matrix.shape}")
    
    if adj_matrix.shape != expected_shape:
        print(f"❌ Error: Adjacency matrix shape mismatch for {dataset['name']}")
        return False
    print(f"✔ Adjacency matrix shape is correct")
    
    # 检查对称性（无向图）
    if not np.allclose(adj_matrix, adj_matrix.T, atol=1e-5):
        print(f"❌ Error: Adjacency matrix is not symmetric for {dataset['name']}")
        return False
    print(f"✔ Adjacency matrix is symmetric")
    
    # 检查对角线元素（应为 0）
    if not np.allclose(np.diag(adj_matrix), 0, atol=1e-5):
        print(f"❌ Error: Diagonal elements of adjacency matrix are not zero for {dataset['name']}")
        return False
    print(f"✔ Diagonal elements are zero")
    
    # 检查边的权重范围（假设 cost 为距离或权重，应为非负且合理）
    non_zero_weights = adj_matrix[adj_matrix > 0]
    if len(non_zero_weights) == 0:
        print(f"❌ Error: Adjacency matrix has no edges for {dataset['name']}")
        return False
    
    if np.any(non_zero_weights < 0):
        print(f"❌ Error: Adjacency matrix contains negative weights for {dataset['name']}")
        return False
    print(f"✔ No negative weights in adjacency matrix")
    
    # 假设最大权重为 100（根据 PEMS 数据集的 cost 通常为距离）
    max_weight = 100
    if np.any(non_zero_weights > max_weight):
        print(f"❌ Error: Adjacency matrix contains weights larger than {max_weight} for {dataset['name']}")
        return False
    print(f"✔ Weights are within reasonable range [0, {max_weight}]")
    
    # 检查稀疏性（交通网络通常稀疏，边的数量应远小于 N^2）
    N = adj_matrix.shape[0]
    num_edges = np.sum(adj_matrix > 0)
    sparsity = num_edges / (N * N)
    if sparsity > 0.1:  # 假设稀疏性阈值为 10%
        print(f"❌ Warning: Adjacency matrix is too dense for {dataset['name']}, sparsity = {sparsity:.4f}")
    else:
        print(f"✔ Adjacency matrix is sparse, sparsity = {sparsity:.4f}")
    
    return True

def check_description(dataset):
    """检查 desc.json 的元信息"""
    dataset_dir = os.path.join(data_base_dir, dataset["name"])
    desc_path = os.path.join(dataset_dir, "desc.json")
    
    with open(desc_path, "r") as f:
        desc = json.load(f)
    
    print(f"\nChecking description file for {dataset['name']}...")
    
    # 检查 shape
    expected_shape = (dataset["original_shape"][0], dataset["original_shape"][1], dataset["expected_features"])
    if tuple(desc["shape"]) != expected_shape:
        print(f"❌ Error: Shape in desc.json does not match expected shape for {dataset['name']}")
        return False
    print(f"✔ Shape in desc.json is correct")
    
    # 检查 num_nodes
    if desc["num_nodes"] != dataset["original_shape"][1]:
        print(f"❌ Error: num_nodes in desc.json does not match expected value for {dataset['name']}")
        return False
    print(f"✔ num_nodes in desc.json is correct")
    
    # 检查 num_features
    if desc["num_features"] != dataset["expected_features"]:
        print(f"❌ Error: num_features in desc.json does not match expected value for {dataset['name']}")
        return False
    print(f"✔ num_features in desc.json is correct")
    
    # 检查 feature_description
    expected_features = ["traffic_flow"] + ["unknown" + str(i) for i in range(1, dataset["original_shape"][2])] + ["time_of_day", "day_of_week"]
    if dataset["name"] == "PEMS04" or dataset["name"] == "PEMS08":
        expected_features = ["traffic_flow", "occupancy", "speed", "time_of_day", "day_of_week"]
    elif dataset["name"] == "Taxi_BJ_hist12_pred12_group1":
        expected_features = ["traffic_flow", "temperature", "wind_speed", "weather_index", "time_of_day", "day_of_week"]
    
    if desc["feature_description"] != expected_features:
        print(f"❌ Error: feature_description in desc.json does not match expected value for {dataset['name']}")
        print(f"Expected: {expected_features}")
        print(f"Actual: {desc['feature_description']}")
        return False
    print(f"✔ feature_description in desc.json is correct")
    return True

def check_data_values(dataset, data):
    """检查数据值范围"""
    T, N, C = data.shape
    print(f"\nChecking data values for {dataset['name']}...")
    
    # 检查每个特征的范围
    for feature_idx, feature_name in enumerate(dataset["feature_ranges"].keys()):
        feature_data = data[:, :, feature_idx]
        min_val, max_val = dataset["feature_ranges"][feature_name]
        
        if np.any(feature_data < min_val) or np.any(feature_data > max_val):
            print(f"❌ Error: {feature_name} values out of range [{min_val}, {max_val}] in {dataset['name']}")
            print(f"Min value: {np.min(feature_data)}, Max value: {np.max(feature_data)}")
            return False
        print(f"✔ {feature_name} values are within range [{min_val}, {max_val}]")
    
    # 检查时间特征范围
    time_of_day = data[:, :, -2]
    day_of_week = data[:, :, -1]
    
    if np.any(time_of_day < 0) or np.any(time_of_day > 1):
        print(f"❌ Error: time_of_day values out of range [0, 1] in {dataset['name']}")
        return False
    print(f"✔ time_of_day values are within range [0, 1]")
    
    if np.any(day_of_week < 0) or np.any(day_of_week > 1):
        print(f"❌ Error: day_of_week values out of range [0, 1] in {dataset['name']}")
        return False
    print(f"✔ day_of_week values are within range [0, 1]")
    
    # 检查流量值的分布（避免异常集中）
    traffic_flow = data[:, :, 0]
    traffic_bins = np.histogram(traffic_flow, bins=10, range=(0, dataset["traffic_max"]))[0]
    if np.any(traffic_bins == 0):
        print(f"❌ Warning: Traffic flow distribution has empty bins for {dataset['name']}")
        print(f"Bin counts: {traffic_bins}")
    else:
        print(f"✔ Traffic flow distribution is reasonable")
    
    return True

def main():
    all_passed = True
    for dataset in datasets:
        print(f"\n=== Checking dataset: {dataset['name']} ===")
        
        # 检查文件是否存在
        if not check_files_exist(dataset):
            all_passed = False
            continue
        
        # 检查数据形状
        data = check_data_shape(dataset)
        if data is False:
            all_passed = False
            continue
        
        # 检查时间特征
        if not check_temporal_features(dataset, data):
            all_passed = False
            continue
        
        # 检查邻接矩阵
        if not check_adj_matrix(dataset):
            all_passed = False
            continue
        
        # 检查描述文件
        if not check_description(dataset):
            all_passed = False
            continue
        
        # 检查数据值范围
        if not check_data_values(dataset, data):
            all_passed = False
            continue
        
        print(f"\n✅ All checks passed for {dataset['name']}")
    
    if all_passed:
        print("\n🎉 All datasets passed the checks!")
    else:
        print("\n❌ Some datasets failed the checks. Please review the errors above.")

if __name__ == "__main__":
    main()