import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from nafnet.NAFNet_arch import NAFNet
import torchvision.transforms.functional as TF
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import csv

# Enable performance tuning
torch.backends.cudnn.benchmark = True
torch.cuda.empty_cache()

def sanitize_image(image):
    image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
    image = (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-6)
    return image

class SARDataset(Dataset):
    def __init__(self, lr_dir, hr_dir, patch_size=128, overlap=32):
        self.lr_files = sorted([os.path.join(lr_dir, f) for f in os.listdir(lr_dir) if f.endswith(".npy")])
        self.hr_files = sorted([os.path.join(hr_dir, f) for f in os.listdir(hr_dir) if f.endswith(".npy")])
        self.patch_size = patch_size
        self.overlap = overlap

    def extract_patches(self, image):
        h, w = image.shape[-2:]
        stride = self.patch_size - self.overlap
        patches = []
        indices = []
        
        for i in range(0, h, stride):
            for j in range(0, w, stride):
                patch = image[:, i:i + self.patch_size, j:j + self.patch_size]

                pad_h = max(0, self.patch_size - patch.shape[-2])
                pad_w = max(0, self.patch_size - patch.shape[-1])

                if pad_h > 0 or pad_w > 0:
                    patch = F.pad(patch, (0, pad_w, 0, pad_h), mode='constant', value=0)

                patches.append(patch)
                indices.append((i, j))
        
        return patches, indices

    def __len__(self):
        return len(self.lr_files)

    def __getitem__(self, idx):
        lr_image = np.load(self.lr_files[idx]).real
        hr_image = np.load(self.hr_files[idx]).real

        lr_image = sanitize_image(lr_image)
        hr_image = sanitize_image(hr_image)

        lr_image = torch.tensor(lr_image).unsqueeze(0).float()
        hr_image = torch.tensor(hr_image).unsqueeze(0).float()

        lr_image = TF.resize(lr_image, hr_image.shape[-2:], interpolation=TF.InterpolationMode.BICUBIC)

        lr_patches, _ = self.extract_patches(lr_image)
        hr_patches, _ = self.extract_patches(hr_image)

        return lr_patches, hr_patches

def compute_metrics(sr, hr):
    sr = np.clip(sr, 0, 1)
    hr = np.clip(hr, 0, 1)
    return {
        "PSNR": psnr(hr, sr, data_range=1),
        "SSIM": ssim(hr, sr, data_range=1)
    }

def train_model():
    lr_dir = "./data/Train/LR"
    hr_dir = "./data/Train/HR"
    dataset = SARDataset(lr_dir, hr_dir, patch_size=512, overlap=128)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    nafnet_sr = NAFNet(img_channel=1, width=96, middle_blk_num=12, enc_blk_nums=[2, 4, 6, 8], dec_blk_nums=[2, 4, 6, 8])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nafnet_sr.to(device)

    criterion = nn.L1Loss()
    optimizer = optim.Adam(nafnet_sr.parameters(), lr=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    scaler = torch.amp.GradScaler()

    best_loss = float("inf")
    patience = 7
    no_improvement = 0
    loss_history = []

    num_epochs = 50
    for epoch in range(num_epochs):
        nafnet_sr.train()
        epoch_loss = 0.0

        for lr_patches, hr_patches in dataloader:
            for lr_patch, hr_patch in zip(lr_patches[0], hr_patches[0]):
                lr_patch, hr_patch = lr_patch.to(device), hr_patch.to(device)

                optimizer.zero_grad()
                with torch.amp.autocast(device_type="cuda"):
                    sr_patch = nafnet_sr(lr_patch.unsqueeze(0))
                    loss = criterion(sr_patch, hr_patch.unsqueeze(0))

                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(nafnet_sr.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

                epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / len(dataloader)
        loss_history.append(avg_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {avg_loss:.6f}", flush=True)

        if avg_loss < best_loss:
            best_loss = avg_loss
            no_improvement = 0
            torch.save(nafnet_sr.state_dict(), "./model/nafnet_sr_best.pth")
            print("Best model saved!", flush=True)
        else:
            no_improvement += 1
            if no_improvement >= patience:
                print("Early stopping triggered.", flush=True)
                break

    torch.save(nafnet_sr.state_dict(), "./model/nafnet_sr_final.pth")
    print("Final model saved.", flush=True)

    # Save loss curve
    plt.figure()
    plt.plot(loss_history, marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Epochs")
    plt.grid(True)
    plt.savefig("loss_curve.png")

    # Save raw values
    with open("loss_values.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "Loss"])
        for i, l in enumerate(loss_history):
            writer.writerow([i+1, l])

if __name__ == '__main__':
    train_model()
