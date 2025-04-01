from models.EEGViT_pretrained import EEGViT_pretrained
from models.EEGViT import EEGViT_raw
from models.ViTBase import ViTBase
from models.ViTBase_pretrained import ViTBase_pretrained
from helper_functions import split
from dataset.EEGEyeNet import EEGEyeNetDataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import numpy as np

# Khởi tạo mô hình và dataset
model = EEGViT_pretrained()
EEGEyeNet = EEGEyeNetDataset('./dataset/Position_task_with_dots_synchronised_min.npz', x_range=(0, 800), y_range=(0, 600))
batch_size = 256
n_epoch = 30
learning_rate = 5e-4

# Loss, optimizer và scheduler
criterion = nn.HuberLoss()  # Dùng HuberLoss
mse_criterion = nn.MSELoss()  # Để tính loss theo đơn vị gốc
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

# Warmup + CosineAnnealingLR
class WarmupCosineAnnealingLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_epochs, total_epochs, eta_min=0, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            return [base_lr * (self.last_epoch + 1) / self.warmup_epochs for base_lr in self.base_lrs]
        else:
            return [self.eta_min + (base_lr - self.eta_min) *
                    (1 + np.cos(np.pi * (self.last_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs))) / 2
                    for base_lr in self.base_lrs]

scheduler = WarmupCosineAnnealingLR(optimizer, warmup_epochs=3, total_epochs=n_epoch, eta_min=1e-6)

def train(model, optimizer, criterion, scheduler=None):
    torch.cuda.empty_cache()
    scaler = GradScaler('cuda')

    train_indices, val_indices, test_indices = split(EEGEyeNet.trainY[:, 0], 0.7, 0.15, 0.15)
    print('create dataloader...')

    train = Subset(EEGEyeNet, indices=train_indices)
    val = Subset(EEGEyeNet, indices=val_indices)
    test = Subset(EEGEyeNet, indices=test_indices)

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val, batch_size=batch_size, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test, batch_size=batch_size, num_workers=4, pin_memory=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    model = model.to(device)
    criterion = criterion.to(device)
    mse_criterion = mse_criterion.to(device)

    print(f"Number of GPUs available: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    print(f"Using device: {device}")

    train_losses = []
    val_losses = []
    test_losses = []
    best_val_loss = float('inf')
    patience, patience_counter = 5, 0

    print('training...')
    for epoch in range(n_epoch):
        model.train()
        epoch_train_loss = 0.0
        epoch_train_loss_original = 0.0

        for i, (inputs, targets, index) in tqdm(enumerate(train_loader)):
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            with autocast('cuda'):
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), targets.squeeze())  # Loss trên tọa độ chuẩn hóa

                # Tính loss theo đơn vị gốc để hiển thị
                outputs_original = outputs.clone()
                targets_original = targets.clone()
                # Chuyển về đơn vị gốc
                outputs_original[:, 0] = outputs_original[:, 0] * (EEGEyeNet.x_range[1] - EEGEyeNet.x_range[0]) + EEGEyeNet.x_range[0]
                outputs_original[:, 1] = outputs_original[:, 1] * (EEGEyeNet.y_range[1] - EEGEyeNet.y_range[0]) + EEGEyeNet.y_range[0]
                targets_original[:, 0] = targets_original[:, 0] * (EEGEyeNet.x_range[1] - EEGEyeNet.x_range[0]) + EEGEyeNet.x_range[0]
                targets_original[:, 1] = targets_original[:, 1] * (EEGEyeNet.y_range[1] - EEGEyeNet.y_range[0]) + EEGEyeNet.y_range[0]
                loss_original = mse_criterion(outputs_original.squeeze(), targets_original.squeeze())

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            epoch_train_loss += loss.item()
            epoch_train_loss_original += loss_original.item()

            if i % 10 == 0:
                print(f"Epoch {epoch}, Batch {i}, Loss (original unit): {loss_original.item()}")

        epoch_train_loss /= len(train_loader)
        epoch_train_loss_original /= len(train_loader)
        train_losses.append(epoch_train_loss)

        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            val_loss_original = 0.0
            for inputs, targets, index in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)

                with autocast('cuda'):
                    outputs = model(inputs)
                    loss = criterion(outputs.squeeze(), targets.squeeze())
                    outputs_original = outputs.clone()
                    targets_original = targets.clone()
                    outputs_original[:, 0] = outputs_original[:, 0] * (EEGEyeNet.x_range[1] - EEGEyeNet.x_range[0]) + EEGEyeNet.x_range[0]
                    outputs_original[:, 1] = outputs_original[:, 1] * (EEGEyeNet.y_range[1] - EEGEyeNet.y_range[0]) + EEGEyeNet.y_range[0]
                    targets_original[:, 0] = targets_original[:, 0] * (EEGEyeNet.x_range[1] - EEGEyeNet.x_range[0]) + EEGEyeNet.x_range[0]
                    targets_original[:, 1] = targets_original[:, 1] * (EEGEyeNet.y_range[1] - EEGEyeNet.y_range[0]) + EEGEyeNet.y_range[0]
                    loss_original = mse_criterion(outputs_original.squeeze(), targets_original.squeeze())
                val_loss += loss.item()
                val_loss_original += loss_original.item()

            val_loss /= len(val_loader)
            val_loss_original /= len(val_loader)
            val_losses.append(val_loss)
            print(f"Epoch {epoch}, Val Loss (original unit): {val_loss_original}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.module.state_dict(), '/kaggle/working/best_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping triggered")
                    break

        with torch.no_grad():
            test_loss = 0.0
            test_loss_original = 0.0
            for inputs, targets, index in test_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)

                with autocast('cuda'):
                    outputs = model(inputs)
                    loss = criterion(outputs.squeeze(), targets.squeeze())
                    outputs_original = outputs.clone()
                    targets_original = targets.clone()
                    outputs_original[:, 0] = outputs_original[:, 0] * (EEGEyeNet.x_range[1] - EEGEyeNet.x_range[0]) + EEGEyeNet.x_range[0]
                    outputs_original[:, 1] = outputs_original[:, 1] * (EEGEyeNet.y_range[1] - EEGEyeNet.y_range[0]) + EEGEyeNet.y_range[0]
                    targets_original[:, 0] = targets_original[:, 0] * (EEGEyeNet.x_range[1] - EEGEyeNet.x_range[0]) + EEGEyeNet.x_range[0]
                    targets_original[:, 1] = targets_original[:, 1] * (EEGEyeNet.y_range[1] - EEGEyeNet.y_range[0]) + EEGEyeNet.y_range[0]
                    loss_original = mse_criterion(outputs_original.squeeze(), targets_original.squeeze())
                test_loss += loss.item()
                test_loss_original += loss_original.item()

            test_loss /= len(test_loader)
            test_loss_original /= len(test_loader)
            test_losses.append(test_loss)
            print(f"Epoch {epoch}, Test Loss (original unit): {test_loss_original}")

        if scheduler is not None:
            scheduler.step()

    torch.save(model.module.state_dict(), '/kaggle/working/final_model.pth')
    print("Training completed. Best model saved as 'best_model.pth', final model as 'final_model.pth'")

if __name__ == "__main__":
    train(model, optimizer, criterion, scheduler)