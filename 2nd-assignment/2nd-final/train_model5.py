import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import numpy as np
import cv2
from tqdm import tqdm
import random
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

# Dataset
class NoisyImageDataset(Dataset):
    def __init__(self, image_dir, patch_size=128, noise_level=25):
        self.image_dir = image_dir
        self.patch_size = patch_size
        self.noise_level = noise_level
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

        # Augmentation
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(90),
        ])

    def add_mixed_noise(self, image):
        # Gaussian + Poisson noise
        gaussian_noise = torch.randn_like(image) * (self.noise_level / 255.0)
        poisson_noise = torch.poisson(image * 255) / 255.0 - image
        noisy_image = image + gaussian_noise + poisson_noise
        return torch.clamp(noisy_image, 0, 1)

    def __len__(self):
        return len(self.image_files) * 100  # 100 patches per image

    def __getitem__(self, idx):
        img_idx = idx // 100
        image_path = os.path.join(self.image_dir, self.image_files[img_idx])
        image = Image.open(image_path).convert('RGB')

        # Random crop
        i, j = torch.randint(0, max(1, image.size[1] - self.patch_size), (1,)), \
               torch.randint(0, max(1, image.size[0] - self.patch_size), (1,))
        
        image = transforms.functional.crop(image, i.item(), j.item(), self.patch_size, self.patch_size)

        # Convert to tensor and apply augmentation
        clean_tensor = self.transform(image)

        # Add mixed noise
        noisy_tensor = self.add_mixed_noise(clean_tensor)

        # Pass the noise level (as a tensor)
        noise_level_tensor = torch.tensor(self.noise_level / 255.0).float()

        return noisy_tensor, clean_tensor, noise_level_tensor


# Improved Denoising CNN inspired by FFDNet
class FFDNetDenoiser(nn.Module):
    def __init__(self, num_channels=3):
        super(FFDNetDenoiser, self).__init__()

        # Encoder
        self.conv1 = nn.Conv2d(num_channels + 1, 64, kernel_size=3, padding=1)  # Include noise level
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, num_channels, kernel_size=3, padding=1)

    def forward(self, x, noise_level):
        # Concatenate noise level to the image
        noise_level_tensor = noise_level.view(-1, 1, 1, 1).expand(-1, 1, x.size(2), x.size(3))  # Reshape for broadcasting
        x = torch.cat((x, noise_level_tensor), dim=1)  # Concatenate along channel dimension

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        out = torch.sigmoid(self.conv4(x))  # Output clamped to [0, 1]
        return out


# Training
def train_model(train_dir, output_dir='improved_model_checkpoints5', num_epochs=100, batch_size=16, learning_rate=0.0001):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Dataset and DataLoader
    dataset = NoisyImageDataset(train_dir, patch_size=128)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # Model, loss functions, optimizer, scheduler
    model = FFDNetDenoiser().to(device)
    mse_loss = nn.MSELoss().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

    # Training loop
    best_loss = float('inf')
    checkpoint_path = os.path.join(output_dir, 'best_denoising_model.pth')

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch + 1}/{num_epochs}')

        for noisy_imgs, clean_imgs, noise_levels in progress_bar:
            noisy_imgs, clean_imgs, noise_levels = noisy_imgs.to(device), clean_imgs.to(device), noise_levels.to(device)

            # Forward pass
            outputs = model(noisy_imgs, noise_levels)
            loss = mse_loss(outputs, clean_imgs)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})

        avg_loss = epoch_loss / len(dataloader)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Average Loss: {avg_loss:.6f}')

        # Update best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), checkpoint_path)
            print(f'Saved best model checkpoint to {checkpoint_path}')

        scheduler.step()  # Cosine annealing learning rate adjustment


if __name__ == "__main__":
    # Set the directory containing training images
    train_dir = "training_images"

    # Train the model
    train_model(train_dir, num_epochs=100)
