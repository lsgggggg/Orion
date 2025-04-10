import os
import numpy as np
import argparse
import configparser
import pandas as pd

# 数据集配置
DATASETS = {
    'PEMS03': {
        'npz_path': '/root/python_on_hyy/data/PEMS03.npz',
        'csv_path': '/root/python_on_hyy/data/PEMS03.csv',
        'points_per_hour': 12,  # 5分钟/次
        'num_of_vertices': 358,
        'features': 1,
    },
    'PEMS04': {
        'npz_path': '/root/python_on_hyy/data/PEMS04.npz',
        'csv_path': '/root/python_on_hyy/data/PEMS04.csv',
        'points_per_hour': 12,
        'num_of_vertices': 307,
        'features': 3,
    },
    'PEMS08': {
        'npz_path': '/root/python_on_hyy/data/PEMS08.npz',
        'csv_path': '/root/python_on_hyy/data/PEMS08.csv',
        'points_per_hour': 12,
        'num_of_vertices': 170,
        'features': 3,
    },
    'NYCBike2_part1': {
        'npz_path': '/root/python_on_hyy/data/NYCBike2_part1.npz',
        'csv_path': '/root/python_on_hyy/data/NYCBike2_part1.csv',
        'points_per_hour': 2,  # 30分钟/次
        'num_of_vertices': 200,
        'features': 4,
    },
    'NYCTaxi_part1': {
        'npz_path': '/root/python_on_hyy/data/NYCTaxi_part1.npz',
        'csv_path': '/root/python_on_hyy/data/NYCTaxi_part1.csv',
        'points_per_hour': 2,
        'num_of_vertices': 200,
        'features': 3,
    },
    'Taxi_BJ_hist12_pred12_group1': {
        'npz_path': '/root/python_on_hyy/data/Taxi_BJ_hist12_pred12_group1.npz',
        'csv_path': '/root/python_on_hyy/data/Taxi_BJ_hist12_pred12_group1.csv',
        'points_per_hour': 2,
        'num_of_vertices': 128,
        'features': 4,
    },
}

def search_data(sequence_length, num_of_depend, label_start_idx,
                num_for_predict, units, points_per_hour):
    if points_per_hour < 0:
        raise ValueError("points_per_hour should be greater than 0!")
    if label_start_idx + num_for_predict > sequence_length:
        return None
    x_idx = []
    for i in range(1, num_of_depend + 1):
        start_idx = label_start_idx - points_per_hour * units * i
        end_idx = start_idx + num_for_predict
        if start_idx >= 0:
            x_idx.append((start_idx, end_idx))
        else:
            return None
    if len(x_idx) != num_of_depend:
        return None
    return x_idx[::-1]

def get_sample_indices(data_sequence, num_of_weeks, num_of_days, num_of_hours,
                       label_start_idx, num_for_predict, points_per_hour=12):
    week_sample, day_sample, hour_sample = None, None, None
    if label_start_idx + num_for_predict > data_sequence.shape[0]:
        return week_sample, day_sample, hour_sample, None
    if num_of_weeks > 0:
        week_indices = search_data(data_sequence.shape[0], num_of_weeks,
                                   label_start_idx, num_for_predict,
                                   7 * 24, points_per_hour)
        if not week_indices:
            return None, None, None, None
        week_sample = np.concatenate([data_sequence[i: j]
                                      for i, j in week_indices], axis=0)
    if num_of_days > 0:
        day_indices = search_data(data_sequence.shape[0], num_of_days,
                                  label_start_idx, num_for_predict,
                                  24, points_per_hour)
        if not day_indices:
            return None, None, None, None
        day_sample = np.concatenate([data_sequence[i: j]
                                     for i, j in day_indices], axis=0)
    if num_of_hours > 0:
        hour_indices = search_data(data_sequence.shape[0], num_of_hours,
                                   label_start_idx, num_for_predict,
                                   1, points_per_hour)
        if not hour_indices:
            return None, None, None, None
        hour_sample = np.concatenate([data_sequence[i: j]
                                      for i, j in hour_indices], axis=0)
    target = data_sequence[label_start_idx: label_start_idx + num_for_predict]
    return week_sample, day_sample, hour_sample, target

def read_and_generate_dataset(graph_signal_matrix_filename,
                             num_of_weeks, num_of_days,
                             num_of_hours, num_for_predict,
                             points_per_hour=12, save=False, dataset_name=None):
    data_seq = np.load(graph_signal_matrix_filename)['data']  # (T, N, F)
    all_samples = []
    for idx in range(data_seq.shape[0]):
        sample = get_sample_indices(data_seq, num_of_weeks, num_of_days,
                                    num_of_hours, idx, num_for_predict,
                                    points_per_hour)
        if ((sample[0] is None) and (sample[1] is None) and (sample[2] is None)):
            continue
        week_sample, day_sample, hour_sample, target = sample
        sample = []
        if num_of_weeks > 0:
            week_sample = np.expand_dims(week_sample, axis=0).transpose((0, 2, 3, 1))
            sample.append(week_sample)
        if num_of_days > 0:
            day_sample = np.expand_dims(day_sample, axis=0).transpose((0, 2, 3, 1))
            sample.append(day_sample)
        if num_of_hours > 0:
            hour_sample = np.expand_dims(hour_sample, axis=0).transpose((0, 2, 3, 1))
            sample.append(hour_sample)
        target = np.expand_dims(target, axis=0).transpose((0, 2, 3, 1))[:, :, 0, :]  # 只取流量
        sample.append(target)
        time_sample = np.expand_dims(np.array([idx]), axis=0)
        sample.append(time_sample)
        all_samples.append(sample)
    split_line1 = int(len(all_samples) * 0.6)
    split_line2 = int(len(all_samples) * 0.8)
    training_set = [np.concatenate(i, axis=0) for i in zip(*all_samples[:split_line1])]
    validation_set = [np.concatenate(i, axis=0) for i in zip(*all_samples[split_line1:split_line2])]
    testing_set = [np.concatenate(i, axis=0) for i in zip(*all_samples[split_line2:])]
    train_x = np.concatenate(training_set[:-2], axis=-1)
    val_x = np.concatenate(validation_set[:-2], axis=-1)
    test_x = np.concatenate(testing_set[:-2], axis=-1)
    train_target = training_set[-2]
    val_target = validation_set[-2]
    test_target = testing_set[-2]
    train_timestamp = training_set[-1]
    val_timestamp = validation_set[-1]
    test_timestamp = testing_set[-1]
    (stats, train_x_norm, val_x_norm, test_x_norm) = normalization(train_x, val_x, test_x)
    all_data = {
        'train': {
            'x': train_x_norm,
            'target': train_target,
            'timestamp': train_timestamp,
        },
        'val': {
            'x': val_x_norm,
            'target': val_target,
            'timestamp': val_timestamp,
        },
        'test': {
            'x': test_x_norm,
            'target': test_target,
            'timestamp': test_timestamp,
        },
        'stats': {
            '_mean': stats['_mean'],
            '_std': stats['_std'],
        }
    }
    print('train x:', all_data['train']['x'].shape)
    print('train target:', all_data['train']['target'].shape)
    print('train timestamp:', all_data['train']['timestamp'].shape)
    print()
    print('val x:', all_data['val']['x'].shape)
    print('val target:', all_data['val']['target'].shape)
    print('val timestamp:', all_data['val']['timestamp'].shape)
    print()
    print('test x:', all_data['test']['x'].shape)
    print('test target:', all_data['test']['target'].shape)
    print('test timestamp:', all_data['test']['timestamp'].shape)
    print()
    print('train data _mean :', stats['_mean'].shape, stats['_mean'])
    print('train data _std :', stats['_std'].shape, stats['_std'])
    if save:
        file = os.path.basename(graph_signal_matrix_filename).split('.')[0]
        dirpath = '/root/python_on_hyy/data_for_benchmark'
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
        filename = os.path.join(dirpath, f"{file}_r{num_of_hours}_d{num_of_days}_w{num_of_weeks}_astcgn")
        print('save file:', filename)
        np.savez_compressed(filename,
                            train_x=all_data['train']['x'], train_target=all_data['train']['target'],
                            train_timestamp=all_data['train']['timestamp'],
                            val_x=all_data['val']['x'], val_target=all_data['val']['target'],
                            val_timestamp=all_data['val']['timestamp'],
                            test_x=all_data['test']['x'], test_target=all_data['test']['target'],
                            test_timestamp=all_data['test']['timestamp'],
                            mean=all_data['stats']['_mean'], std=all_data['stats']['_std'])
    return all_data

def normalization(train, val, test):
    assert train.shape[1:] == val.shape[1:] and val.shape[1:] == test.shape[1:]
    mean = train.mean(axis=(0,1,3), keepdims=True)
    std = train.std(axis=(0,1,3), keepdims=True)
    print('mean.shape:', mean.shape)
    print('std.shape:', std.shape)
    def normalize(x):
        return (x - mean) / std
    train_norm = normalize(train)
    val_norm = normalize(val)
    test_norm = normalize(test)
    return {'_mean': mean, '_std': std}, train_norm, val_norm, test_norm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_of_weeks", type=int, default=1)
    parser.add_argument("--num_of_days", type=int, default=1)
    parser.add_argument("--num_of_hours", type=int, default=1)
    parser.add_argument("--num_for_predict", type=int, default=12)
    args = parser.parse_args()
    for dataset_name, config in DATASETS.items():
        print(f'Processing dataset: {dataset_name}')
        all_data = read_and_generate_dataset(
            graph_signal_matrix_filename=config['npz_path'],
            num_of_weeks=args.num_of_weeks,
            num_of_days=args.num_of_days,
            num_of_hours=args.num_of_hours,
            num_for_predict=args.num_for_predict,
            points_per_hour=config['points_per_hour'],
            save=True,
            dataset_name=dataset_name
        )