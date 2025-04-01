from torch.utils.data import Dataset
import torch
import numpy as np

class EEGEyeNetDataset(Dataset):
    def __init__(self, data_file, transpose=True):
        self.data_file = data_file
        print('loading data...')
        with np.load(self.data_file) as f:  # Load the data array
            self.trainX = f['EEG']
            self.trainY = f['labels']
        # Pad channels để chia hết cho patch size (8)
        current_channels = self.trainX.shape[2]
        target_channels = ((current_channels // 8 + 1) * 8)  # Làm tròn lên số chia hết cho 8
        pad_amount = target_channels - current_channels
        self.trainX = np.pad(self.trainX, ((0, 0), (0, 0), (0, pad_amount)), mode='constant')
        print(f"Padded channels from {current_channels} to {target_channels}")
        print(self.trainY)
        if transpose:
            self.trainX = np.transpose(self.trainX, (0, 2, 1))[:, np.newaxis, :, :]

    def __getitem__(self, index):
        # Đọc một mẫu dữ liệu
        X = torch.from_numpy(self.trainX[index]).float()
        y = torch.from_numpy(self.trainY[index, 1:3]).float()
        return (X, y, index)

    def __len__(self):
        # Số lượng mẫu trong dataset
        return len(self.trainX)