# @Time     : Jan. 10, 2019 15:26
# @Author   : Veritas YIN
# @FileName : data_utils.py
# @Version  : 1.0
# @IDE      : PyCharm
# @Github   : https://github.com/VeritasYin/Project_Orion
# @Modified : Mar. 31, 2025 by Grok 3 (xAI)

from utils.math_utils import z_score, z_inverse
import numpy as np
import pandas as pd

class Dataset(object):
    def __init__(self, data, stats):
        self.__data = data
        self.mean = stats['mean']
        self.std = stats['std']

    def get_data(self, type):
        return self.__data[type]

    def get_stats(self):
        return {'mean': self.mean, 'std': self.std}

    def get_len(self, type):
        return len(self.__data[type])

    def z_inverse(self, type):
        return self.__data[type] * self.std + self.mean

def seq_gen(len_seq, data_seq, offset, n_frame, n_route, day_slot, C_0=1):
    n_slot = day_slot - n_frame + 1
    tmp_seq = np.zeros((len_seq * n_slot, n_frame, n_route, C_0))
    for i in range(len_seq):
        for j in range(n_slot):
            sta = (i + offset) * day_slot + j
            end = sta + n_frame
            tmp_seq[i * n_slot + j, :, :, :] = np.reshape(data_seq[sta:end, :], [n_frame, n_route, C_0])
    return tmp_seq

def data_gen(dataset_name, n_route, n_frame=21, day_slot=288, C_0=1):
    data_path = f"/root/python_on_hyy/data_for_benchmark/{dataset_name}_data.npz"
    data = np.load(data_path)
    
    x_train = data['train']  # Shape: (num_samples, n_frame, n_route, C_0)
    x_val = data['val']
    x_test = data['test']
    x_stats = {'mean': data['mean'], 'std': data['std']}

    # Ensure the channel dimension matches C_0
    assert x_train.shape[-1] == C_0, f"Expected channel dimension {C_0}, but got {x_train.shape[-1]}"

    x_data = {'train': x_train, 'val': x_val, 'test': x_test}
    dataset = Dataset(x_data, x_stats)
    return dataset

def gen_batch(inputs, batch_size, dynamic_batch=False, shuffle=False):
    len_inputs = len(inputs)
    if shuffle:
        idx = np.arange(len_inputs)
        np.random.shuffle(idx)

    for start_idx in range(0, len_inputs, batch_size):
        end_idx = start_idx + batch_size
        if end_idx > len_inputs:
            if dynamic_batch:
                end_idx = len_inputs
            else:
                break
        if shuffle:
            slide = idx[start_idx:end_idx]
        else:
            slide = slice(start_idx, end_idx)
        yield inputs[slide]