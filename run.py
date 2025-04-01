from torch.utils.data import Dataset
import torch
import numpy as np


class EEGEyeNetDataset(Dataset):
    def __init__(self, data_file, transpose=True, x_range=(0, 800), y_range=(0, 600)):
        self.data_file = data_file
        self.x_range = x_range
        self.y_range = y_range
        print('loading data...')
        with np.load(self.data_file) as f:
            self.trainX = f['EEG']
            self.trainY = f['labels']

        # Lọc outlier
        x_coords = self.trainY[:, 1]
        y_coords = self.trainY[:, 2]
        mask = (x_coords >= x_range[0]) & (x_coords <= x_range[1]) & (y_coords >= y_range[0]) & (y_coords <= y_range[1])
        self.trainX = self.trainX[mask]
        self.trainY = self.trainY[mask]
        print(f"Filtered dataset: {len(self.trainX)} samples remaining")

        # Chuẩn hóa EEG
        self.trainX = (self.trainX - self.trainX.mean()) / self.trainX.std()

        # Chuẩn hóa tọa độ x, y về [0, 1]
        self.trainY[:, 1] = (self.trainY[:, 1] - x_range[0]) / (x_range[1] - x_range[0])
        self.trainY[:, 2] = (self.trainY[:, 2] - y_range[0]) / (y_range[1] - y_range[0])

        # Pad channels để chia hết cho patch size (8)
        current_channels = self.trainX.shape[2]
        target_channels = ((current_channels // 8 + 1) * 8)
        pad_amount = target_channels - current_channels
        self.trainX = np.pad(self.trainX, ((0, 0), (0, 0), (0, pad_amount)), mode='constant')
        print(f"Padded channels from {current_channels} to {target_channels}")
        print(self.trainY)
        if transpose:
            self.trainX = np.transpose(self.trainX, (0, 2, 1))[:, np.newaxis, :, :]

    def __getitem__(self, index):
        X = torch.from_numpy(self.trainX[index]).float()
        y = torch.from_numpy(self.trainY[index, 1:3]).float()
        return (X, y, index)

    def __len__(self):
        return len(self.trainX)