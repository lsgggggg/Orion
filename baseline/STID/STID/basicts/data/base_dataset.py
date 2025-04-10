import json
from typing import List

import numpy as np

class BaseDataset:
    def __init__(self, dataset_name: str, train_val_test_ratio: List[float], mode: str, input_len: int, output_len: int, overlap: bool = True) -> None:
        self.dataset_name = dataset_name
        self.train_val_test_ratio = train_val_test_ratio
        self.mode = mode
        self.input_len = input_len
        self.output_len = output_len
        self.overlap = overlap

class TimeSeriesForecastingDataset(BaseDataset):
    def __init__(self, dataset_name: str, train_val_test_ratio: List[float], mode: str, input_len: int, output_len: int, overlap: bool = True) -> None:
        assert mode in ['train', 'valid', 'test'], f"Invalid mode: {mode}. Must be one of ['train', 'valid', 'test']."
        super().__init__(dataset_name, train_val_test_ratio, mode, input_len, output_len, overlap)

        # 修改数据路径
        self.data_file_path = f'/root/python_on_hyy/STID/datasets/{dataset_name}/data.dat'
        self.description_file_path = f'/root/python_on_hyy/STID/datasets/{dataset_name}/desc.json'
        self.description = self._load_description()
        self.steps_per_day = self.description.get("steps_per_day", 288)  # 动态读取
        self.data = self._load_data()
        # 调试信息：打印数据形状
        print(f"Dataset {dataset_name} ({mode} mode) loaded with shape: {self.data.shape}")

    def _load_description(self) -> dict:
        try:
            with open(self.description_file_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError as e:
            raise FileNotFoundError(f'Description file not found: {self.description_file_path}') from e
        except json.JSONDecodeError as e:
            raise ValueError(f'Error decoding JSON file: {self.description_file_path}') from e

    def _load_data(self) -> np.ndarray:
        try:
            data = np.memmap(self.data_file_path, dtype='float32', mode='r', shape=tuple(self.description['shape']))
        except (FileNotFoundError, ValueError) as e:
            raise ValueError(f'Error loading data file: {self.data_file_path}') from e

        total_len = len(data)
        train_len = int(total_len * self.train_val_test_ratio[0])
        valid_len = int(total_len * self.train_val_test_ratio[1])

        if self.mode == 'train':
            offset = self.output_len if self.overlap else 0
            return data[:train_len + offset].copy()
        elif self.mode == 'valid':
            offset_left = self.input_len - 1 if self.overlap else 0
            offset_right = self.output_len if self.overlap else 0
            return data[train_len - offset_left : train_len + valid_len + offset_right].copy()
        else:  # self.mode == 'test'
            offset = self.input_len - 1 if self.overlap else 0
            return data[train_len + valid_len - offset:].copy()

    def __getitem__(self, index: int) -> dict:
        history_data = self.data[index:index + self.input_len]
        future_data = self.data[index + self.input_len:index + self.input_len + self.output_len]
        return {'inputs': history_data, 'target': future_data}

    def __len__(self) -> int:
        return len(self.data) - self.input_len - self.output_len + 1