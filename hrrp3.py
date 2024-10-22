import os, torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from scipy.io import loadmat
import os.path
import pickle
from pathlib import Path
from typing import Any, Callable, Optional, Tuple, Union

import numpy as np
from PIL import Image

class gaf12_Dataset(Dataset):
    def __init__(
            self,
            root: Union[str, Path],
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
    ) -> None:
        self.root = Path(root)
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        # Define label mapping
        self.label_map = {'EA-18G': 0, 'EP-3E': 1, 'F15': 2, 'F16': 3, 'F22': 4,
                          'F2': 5, 'F35': 6, 'F18': 7, 'IDF': 8, '捕食者': 9,
                          '全球鹰': 10, '幻影2000': 11}

        # Define the base folder for training and test data
        base_folder = "train" if self.train else "test"
        data_folder = self.root / base_folder

        self.data = []
        self.labels = []

        # Load data from .jpg files
        for file_path in data_folder.glob('*.jpg'):
            label_str = self.extract_label(file_path.name)
            label_idx = self.label_map[label_str]  # Convert label to index
            image = Image.open(file_path).convert('L')  # Convert to grayscale
            image = image.resize((84, 84), Image.Resampling.LANCZOS)
            self.labels.append(label_idx)
            self.data.append(np.array(image) / 255.0)

        # Convert lists to NumPy arrays if necessary
        self.data = np.array(self.data)
        self.labels = np.array(self.labels)

    def extract_label(self, file_name: str) -> str:
        # Extract label from file name (e.g., 'F15_001.jpg' -> 'F15')
        return file_name.split('_')[0]

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        image, label = self.data[index], self.labels[index]

        # 将 NumPy 数组转换为 PyTorch 张量
        image = torch.from_numpy(image).float()  # 确保是浮点张量

        # 如果定义了 transform 并且它是一个可调用的函数，则应用它
        if self.transform is not None and callable(self.transform):
            image = self.transform(image)

        # 如果定义了 target_transform 并且它是一个可调用的函数，则应用它
        if self.target_transform is not None and callable(self.target_transform):
            label = self.target_transform(label)

        return image, label

    def __len__(self) -> int:
        return len(self.data)

    def extra_repr(self) -> str:
        split = "Train" if self.train is True else "Test"
        return f"Split: {split}"

class hrrp3_Dataset(Dataset):
    def __init__(
            self,
            root: Union[str, Path],
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            maxpool_size: int = 84  # 定义最大池化的目标大小
    ) -> None:
        self.root = Path(root)
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.maxpool_size = maxpool_size  # 存储最大池化的目标大小

        # 定义标签映射
        self.label_map = {
            'EA-18G': 0, 'EP-3E': 1, 'F15': 2, 'F16': 3, 'F22': 4,
            'F2': 5, 'F35': 6, 'F18': 7, 'IDF': 8, '捕食者': 9,
            '全球鹰': 10, '幻影2000': 11
        }

        # 定义训练和测试数据的基础文件夹
        base_folder = "train" if self.train else "test"
        data_folder = self.root / base_folder

        self.data = []
        self.labels = []

        # 加载.mat文件中的数据
        for file_path in data_folder.glob('*.mat'):
            label_str = self.extract_label(file_path.name)
            label_idx = self.label_map[label_str]  # 将标签转换为索引
            hrrp_sequence = self.load_mat(file_path)
            # 应用最大池化
            hrrp_sequence = self.maxpool(hrrp_sequence)
            self.data.append(hrrp_sequence)
            self.labels.append(label_idx)

        # 将列表转换为NumPy数组（如果需要）
        self.data = np.array(self.data)
        self.labels = np.array(self.labels)

    def extract_label(self, file_name: str) -> str:
        # 从文件名中提取标签（例如，'F15_001.mat' -> 'F15'）
        return file_name.split('_')[0]

    def load_mat(self, file_path: Path) -> np.ndarray:
        # 加载.mat文件并返回HRRP序列
        mat = loadmat(file_path)
        return abs(mat['CoHH'].flatten())

    def maxpool(self, hrrp_sequence: np.ndarray) -> np.ndarray:
        # 应用最大池化操作
        # 假设hrrp_sequence是一维数组，需要压缩到self.maxpool_size长度
        if len(hrrp_sequence) > self.maxpool_size:
            # 计算步长，确保步长为整数
            stride = (len(hrrp_sequence) - self.maxpool_size + 1) // self.maxpool_size + 1
            # 应用最大池化
            pooled_sequence = hrrp_sequence[:self.maxpool_size]  # 确保不会超出数组边界
            for i in range(1, stride):
                pooled_sequence = np.maximum(pooled_sequence, hrrp_sequence[
                                                              i * self.maxpool_size:i * self.maxpool_size + self.maxpool_size])
        else:
            pooled_sequence = hrrp_sequence
        return pooled_sequence

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        hrrp_sequence, label = self.data[index], self.labels[index]

        # 将NumPy数组转换为PyTorch张量
        hrrp_sequence = torch.from_numpy(hrrp_sequence).float()  # 确保是浮点张量

        # 如果定义了transform并且它是一个可调用的函数，则应用它
        if self.transform is not None and callable(self.transform):
            hrrp_sequence = self.transform(hrrp_sequence)

        # 如果定义了target_transform并且它是一个可调用的函数，则应用它
        if self.target_transform is not None and callable(self.target_transform):
            label = self.target_transform(label)

        return hrrp_sequence, label

    def __len__(self) -> int:
        return len(self.data)

    def extra_repr(self) -> str:
        split = "Train" if self.train is True else "Test"
        return f"Split: {split}"

# class hrrp3_Dataset(Dataset):
#     def __init__(
#             self,
#             root: Union[str, Path],
#             train: bool = True,
#             transform: Optional[Callable] = None,
#             target_transform: Optional[Callable] = None,
#     ) -> None:
#         self.root = Path(root)
#         self.train = train
#         self.transform = transform
#         self.target_transform = target_transform
#
#         # Define label mapping
#         self.label_map = {'EA-18G': 0, 'EP-3E': 1, 'F15': 2, 'F16': 3, 'F22': 4,
#                           'F2': 5, 'F35': 6, 'F18': 7, 'IDF': 8, '捕食者': 9,
#                           '全球鹰': 10, '幻影2000': 11}
#
#         # Define the base folder for training and test data
#         base_folder = "train" if self.train else "test"
#         data_folder = self.root / base_folder
#
#         self.data = []
#         self.labels = []
#
#         # Load data from .mat files
#         for file_path in data_folder.glob('*.mat'):
#             label_str = self.extract_label(file_path.name)
#             label_idx = self.label_map[label_str]  # Convert label to index
#             hrrp_sequence = self.load_mat(file_path)
#             self.data.append(hrrp_sequence)
#             self.labels.append(label_idx)
#
#         # Convert lists to NumPy arrays if necessary
#         self.data = np.array(self.data)
#         self.labels = np.array(self.labels)
#
#     def extract_label(self, file_name: str) -> str:
#         # Extract label from file name (e.g., 'F15_001.mat' -> 'F15')
#         return file_name.split('_')[0]
#
#     def load_mat(self, file_path: Path) -> np.ndarray:
#         # Load a .mat file and return the HRRP sequence
#         mat = loadmat(file_path)
#         return abs(mat['CoHH'].flatten())
#
#     def __getitem__(self, index: int) -> Tuple[Any, Any]:
#         hrrp_sequence, label = self.data[index], self.labels[index]
#
#         # 将 NumPy 数组转换为 PyTorch 张量
#         hrrp_sequence = torch.from_numpy(hrrp_sequence).float()  # 确保是浮点张量
#
#         # 如果定义了 transform 并且它是一个可调用的函数，则应用它
#         if self.transform is not None and callable(self.transform):
#             hrrp_sequence = self.transform(hrrp_sequence)
#
#         # 如果定义了 target_transform 并且它是一个可调用的函数，则应用它
#         if self.target_transform is not None and callable(self.target_transform):
#             label = self.target_transform(label)
#
#         return hrrp_sequence, label
#
#     def __len__(self) -> int:
#         return len(self.data)
#
#     def extra_repr(self) -> str:
#         split = "Train" if self.train is True else "Test"
#         return f"Split: {split}"

# Example usage:
if __name__ == '__main__':
    data_dir = 'data/hrrp3'  # Replace with the path to your HRRP3 data
    transform = None  # Define your transform if necessary
    dataset = hrrp3_Dataset(root=data_dir, train=True, transform=transform)
    print(dataset)
    # Now you can use dataset with a DataLoader or similar