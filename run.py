from models.EEGViT_pretrained import EEGViT_pretrained
from dataset.EEGEyeNet import EEGEyeNetDataset
from helper_functions import split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.amp import autocast, GradScaler  # Cập nhật import
from tqdm import tqdm
import numpy as np

# Khởi tạo mô hình và dữ liệu
model = EEGViT_pretrained()
EEGEyeNet = EEGEyeNetDataset('./dataset/Position_task_with_dots_synchronised_min.npz')
batch_size = 128
n_epoch = 15
learning_rate = 1e-4

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=3, verbose=True)


def train(model, optimizer, criterion, scheduler=None):
    torch.cuda.empty_cache()
    scaler = GradScaler('cuda')  # Sửa deprecated warning

    # Chia dữ liệu
    train_indices, val_indices, test_indices = split(EEGEyeNet.trainY[:, 0], 0.7, 0.15, 0.15)
    train = Subset(EEGEyeNet, indices=train_indices)
    val = Subset(EEGEyeNet, indices=val_indices)
    test = Subset(EEGEyeNet, indices=test_indices)

    # DataLoader tối ưu cho P100
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val, batch_size=batch_size, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test, batch_size=batch_size, num_workers=4, pin_memory=True)

    # Thiết lập thiết bị
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = criterion.to(device)

    train_losses, val_losses, test_losses = [], [], []
    best_val_loss = float('inf')
    patience, patience_counter = 5, 0

    print(f"Using device: {device}")

    for epoch in range(n_epoch):
        model.train()
        epoch_train_loss = 0.0
        for i, (inputs, targets, index) in tqdm(enumerate(train_loader)):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            with autocast('cuda'):  # Mixed Precision
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), targets.squeeze())
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_train_loss += loss.item()
            if i % 100 == 0:
                print(f"Epoch {epoch}, Batch {i}, Loss: {loss.item()}")

        epoch_train_loss /= len(train_loader)
        train_losses.append(epoch_train_loss)

        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for inputs, targets, index in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                with autocast('cuda'):
                    outputs = model(inputs)
                    loss = criterion(outputs.squeeze(), targets.squeeze())
                val_loss += loss.item()
            val_loss /= len(val_loader)
            val_losses.append(val_loss)
            print(f"Epoch {epoch}, Val Loss: {val_loss}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), '/kaggle/working/best_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping triggered")
                    break

            test_loss = 0.0
            for inputs, targets, index in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                with autocast('cuda'):
                    outputs = model(inputs)
                    loss = criterion(outputs.squeeze(), targets.squeeze())
                test_loss += loss.item()
            test_loss /= len(test_loader)
            test_losses.append(test_loss)
            print(f"Epoch {epoch}, Test Loss: {test_loss}")

        if scheduler is not None:
            scheduler.step(val_loss)

    torch.save(model.state_dict(), '/kaggle/working/final_model.pth')
    print("Training completed. Best model saved as 'best_model.pth', final model as 'final_model.pth'")


if __name__ == "__main__":
    train(model, optimizer, criterion, scheduler)