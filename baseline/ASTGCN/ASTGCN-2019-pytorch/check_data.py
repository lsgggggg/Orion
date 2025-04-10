import os
import numpy as np
import configparser

# 数据集配置
DATASETS = {
    'PEMS03': {
        'npz_path': '/root/python_on_hyy/data_for_benchmark/PEMS03_r1_d1_w1_astcgn.npz',
        'points_per_hour': 12,  # 5分钟/次
        'num_of_vertices': 358,
        'features': 1,
        'len_input': 12,
        'num_for_predict': 12,
        'num_of_weeks': 1,
        'num_of_days': 1,
        'num_of_hours': 1,
    },
    'PEMS04': {
        'npz_path': '/root/python_on_hyy/data_for_benchmark/PEMS04_r1_d1_w1_astcgn.npz',
        'points_per_hour': 12,
        'num_of_vertices': 307,
        'features': 3,
        'len_input': 12,
        'num_for_predict': 12,
        'num_of_weeks': 1,
        'num_of_days': 1,
        'num_of_hours': 1,
    },
    'PEMS08': {
        'npz_path': '/root/python_on_hyy/data_for_benchmark/PEMS08_r1_d1_w1_astcgn.npz',
        'points_per_hour': 12,
        'num_of_vertices': 170,
        'features': 3,
        'len_input': 12,
        'num_for_predict': 12,
        'num_of_weeks': 1,
        'num_of_days': 1,
        'num_of_hours': 1,
    },
    'NYCBike2_part1': {
        'npz_path': '/root/python_on_hyy/data_for_benchmark/NYCBike2_part1_r1_d1_w1_astcgn.npz',
        'points_per_hour': 2,  # 30分钟/次
        'num_of_vertices': 200,
        'features': 4,
        'len_input': 12,
        'num_for_predict': 12,
        'num_of_weeks': 1,
        'num_of_days': 1,
        'num_of_hours': 1,
    },
    'NYCTaxi_part1': {
        'npz_path': '/root/python_on_hyy/data_for_benchmark/NYCTaxi_part1_r1_d1_w1_astcgn.npz',
        'points_per_hour': 2,
        'num_of_vertices': 200,
        'features': 3,
        'len_input': 12,
        'num_for_predict': 12,
        'num_of_weeks': 1,
        'num_of_days': 1,
        'num_of_hours': 1,
    },
    'Taxi_BJ_hist12_pred12_group1': {
        'npz_path': '/root/python_on_hyy/data_for_benchmark/Taxi_BJ_hist12_pred12_group1_r1_d1_w1_astcgn.npz',
        'points_per_hour': 2,
        'num_of_vertices': 128,
        'features': 4,
        'len_input': 12,
        'num_for_predict': 12,
        'num_of_weeks': 1,
        'num_of_days': 1,
        'num_of_hours': 1,
    },
}

def check_file_existence(dataset_name, config):
    npz_path = config['npz_path']
    if not os.path.exists(npz_path):
        print(f"[ERROR] {dataset_name}: .npz file does not exist at {npz_path}")
        return False
    print(f"[PASS] {dataset_name}: .npz file exists at {npz_path}")
    return True

def check_data_shapes(dataset_name, config, data):
    # 提取配置
    N = config['num_of_vertices']
    F = config['features']
    len_input = config['len_input']
    num_for_predict = config['num_for_predict']
    num_of_weeks = config['num_of_weeks']
    num_of_days = config['num_of_days']
    num_of_hours = config['num_of_hours']

    # 提取数据
    train_x = data['train_x']  # (B, N, F, T')
    val_x = data['val_x']
    test_x = data['test_x']
    train_target = data['train_target']  # (B, N, T)
    val_target = data['val_target']
    test_target = data['test_target']
    mean = data['mean']  # (1, 1, F, 1)
    std = data['std']  # (1, 1, F, 1)

    # 检查形状
    total_periods = num_of_weeks + num_of_days + num_of_hours
    expected_T = len_input * total_periods  # T' = len_input * (num_of_weeks + num_of_days + num_of_hours)
    checks = [
        (train_x.shape[1] == N, f"train_x N ({train_x.shape[1]}) != expected N ({N})"),
        (train_x.shape[2] == F, f"train_x F ({train_x.shape[2]}) != expected F ({F})"),
        (train_x.shape[3] == expected_T, f"train_x T' ({train_x.shape[3]}) != expected T' ({expected_T})"),
        (val_x.shape[1] == N, f"val_x N ({val_x.shape[1]}) != expected N ({N})"),
        (val_x.shape[2] == F, f"val_x F ({val_x.shape[2]}) != expected F ({F})"),
        (val_x.shape[3] == expected_T, f"val_x T' ({val_x.shape[3]}) != expected T' ({expected_T})"),
        (test_x.shape[1] == N, f"test_x N ({test_x.shape[1]}) != expected N ({N})"),
        (test_x.shape[2] == F, f"test_x F ({test_x.shape[2]}) != expected F ({F})"),
        (test_x.shape[3] == expected_T, f"test_x T' ({test_x.shape[3]}) != expected T' ({expected_T})"),
        (train_target.shape[1] == N, f"train_target N ({train_target.shape[1]}) != expected N ({N})"),
        (train_target.shape[2] == num_for_predict, f"train_target T ({train_target.shape[2]}) != num_for_predict ({num_for_predict})"),
        (val_target.shape[1] == N, f"val_target N ({val_target.shape[1]}) != expected N ({N})"),
        (val_target.shape[2] == num_for_predict, f"val_target T ({val_target.shape[2]}) != num_for_predict ({num_for_predict})"),
        (test_target.shape[1] == N, f"test_target N ({test_target.shape[1]}) != expected N ({N})"),
        (test_target.shape[2] == num_for_predict, f"test_target T ({test_target.shape[2]}) != num_for_predict ({num_for_predict})"),
        (mean.shape == (1, 1, F, 1), f"mean shape {mean.shape} != expected (1, 1, {F}, 1)"),
        (std.shape == (1, 1, F, 1), f"std shape {std.shape} != expected (1, 1, {F}, 1)"),
    ]

    all_passed = True
    for condition, error_msg in checks:
        if not condition:
            print(f"[ERROR] {dataset_name}: {error_msg}")
            all_passed = False
        else:
            print(f"[PASS] {dataset_name}: {error_msg.replace('!=', '==')}")
    return all_passed

def check_normalization(dataset_name, data):
    train_x = data['train_x']
    val_x = data['val_x']
    test_x = data['test_x']
    train_target = data['train_target']
    val_target = data['val_target']
    test_target = data['test_target']

    # 检查 train_x 是否归一化（均值约0，方差约1）
    train_x_mean = np.mean(train_x)
    train_x_std = np.std(train_x)
    val_x_mean = np.mean(val_x)
    val_x_std = np.std(val_x)
    test_x_mean = np.mean(test_x)
    test_x_std = np.std(test_x)

    # 打印 val_x 和 test_x 的均值和方差（仅供参考）
    print(f"[INFO] {dataset_name}: val_x mean: {val_x_mean}, val_x std: {val_x_std}")
    print(f"[INFO] {dataset_name}: test_x mean: {test_x_mean}, test_x std: {test_x_std}")

    # 检查 train_x, val_x, test_x 的值范围（放宽到 [-25, 25]）
    train_x_min = np.min(train_x)
    train_x_max = np.max(train_x)
    val_x_min = np.min(val_x)
    val_x_max = np.max(val_x)
    test_x_min = np.min(test_x)
    test_x_max = np.max(test_x)

    # target 不应归一化（流量值通常为正，且范围较大）
    train_target_min = np.min(train_target)
    train_target_max = np.max(train_target)

    checks = [
        (abs(train_x_mean) < 1e-5, f"train_x mean ({train_x_mean}) is not close to 0"),
        (abs(train_x_std - 1) < 1e-5, f"train_x std ({train_x_std}) is not close to 1"),
        (train_x_min > -25 and train_x_max < 25, f"train_x value range ({train_x_min}, {train_x_max}) is abnormal"),
        (val_x_min > -25 and val_x_max < 25, f"val_x value range ({val_x_min}, {val_x_max}) is abnormal"),
        (test_x_min > -25 and test_x_max < 25, f"test_x value range ({test_x_min}, {test_x_max}) is abnormal"),
        (train_target_min >= 0, f"train_target contains negative values (min: {train_target_min})"),
        (train_target_max > 1, f"train_target max ({train_target_max}) is too small, expected to be larger (not normalized)"),
    ]

    all_passed = True
    for condition, error_msg in checks:
        if not condition:
            print(f"[ERROR] {dataset_name}: {error_msg}")
            all_passed = False
        else:
            print(f"[PASS] {dataset_name}: {error_msg.replace('is not', 'is')}")
    return all_passed

def check_dataset(dataset_name, config):
    print(f"\n=== Checking dataset: {dataset_name} ===")
    
    # 检查文件存在性
    if not check_file_existence(dataset_name, config):
        return False

    # 加载数据
    data = np.load(config['npz_path'])
    
    # 检查数据形状
    if not check_data_shapes(dataset_name, config, data):
        return False
    
    # 检查归一化
    if not check_normalization(dataset_name, data):
        return False

    print(f"[SUCCESS] {dataset_name}: All checks passed!")
    return True

if __name__ == "__main__":
    all_passed = True
    for dataset_name, config in DATASETS.items():
        if not check_dataset(dataset_name, config):
            all_passed = False
    
    if all_passed:
        print("\n=== Final Result: All datasets passed the checks! ===")
    else:
        print("\n=== Final Result: Some datasets failed the checks! Please review the errors above. ===")