import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torch_xla  # Thư viện cho TPU
import torch_xla.core.xla_model as xm  # XLA device
import torch_xla.distributed.parallel_loader as pl  # DataLoader cho TPU
from torch.cuda.amp import autocast, GradScaler  # Mixed Precision (vẫn dùng được với TPU)
from tqdm import tqdm
import numpy as np
from models.EEGViT_pretrained import EEGViT_pretrained
from dataset.EEGEyeNet import EEGEyeNetDataset
from helper_functions import split

# Khởi tạo model và dataset
model = EEGViT_pretrained()
EEGEyeNet = EEGEyeNetDataset('./dataset/Position_task_with_dots_synchronised_min.npz')
batch_size = 64
n_epoch = 15
learning_rate = 1e-4

# Device TPU
device = xm.xla_device()
criterion = nn.MSELoss().to(device)

# Optimizer và Scheduler
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.1)

def train(model, optimizer, scheduler=None):
    '''
    Huấn luyện mô hình trên TPU với Mixed Precision
    '''
    scaler = GradScaler()  # Dùng cho Mixed Precision

    # Chia dữ liệu
    train_indices, val_indices, test_indices = split(EEGEyeNet.trainY[:, 0], 0.7, 0.15, 0.15)
    print('create dataloader...')

    train = Subset(EEGEyeNet, indices=train_indices)
    val = Subset(EEGEyeNet, indices=val_indices)
    test = Subset(EEGEyeNet, indices=test_indices)

    # DataLoader cho TPU
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val, batch_size=batch_size)
    test_loader = DataLoader(test, batch_size=batch_size)

    # Wrap DataLoader với ParallelLoader cho TPU
    train_loader = pl.ParallelLoader(train_loader, [device]).per_device_loader(device)
    val_loader = pl.ParallelLoader(val_loader, [device]).per_device_loader(device)
    test_loader = pl.ParallelLoader(test_loader, [device]).per_device_loader(device)

    # Chuyển model và criterion sang TPU
    model = model.to(device)
    criterion = criterion.to(device)

    train_losses, val_losses, test_losses = [], [], []
    print('training...')

    for epoch in range(n_epoch):
        model.train()
        epoch_train_loss = 0.0

        for i, (inputs, targets, index) in tqdm(enumerate(train_loader)):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            with autocast():  # Mixed Precision
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), targets.squeeze())

            scaler.scale(loss).backward()  # Scale loss cho MPT
            scaler.step(optimizer)         # Dùng optimizer.step() thay vì xm.optimizer_step()
            scaler.update()
            epoch_train_loss += loss.item()

            if i % 100 == 0:
                print(f"Epoch {epoch}, Batch {i}, Loss: {loss.item()}")

        # Đồng bộ hóa TPU sau mỗi epoch
        xm.master_print(f"Epoch {epoch} completed")
        epoch_train_loss /= len(train_loader)
        train_losses.append(epoch_train_loss)

        # Validation
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for inputs, targets, index in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                with autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs.squeeze(), targets.squeeze())
                val_loss += loss.item()

            val_loss /= len(val_loader)
            val_losses.append(val_loss)
            xm.master_print(f"Epoch {epoch}, Val Loss: {val_loss}")

        # Test
        with torch.no_grad():
            test_loss = 0.0
            for inputs, targets, index in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                with autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs.squeeze(), targets.squeeze())
                test_loss += loss.item()

            test_loss /= len(test_loader)
            test_losses.append(test_loss)
            xm.master_print(f"Epoch {epoch}, Test Loss: {test_loss}")

        if scheduler is not None:
            scheduler.step()

    # Lưu mô hình trên TPU
    xm.save(model.state_dict(), 'model_tpu.pth')

if __name__ == "__main__":
    train(model, optimizer, scheduler)