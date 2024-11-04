import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.cuda.amp as amp  # Import for mixed precision
import time

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

        return noisy_tensor, clean_tensor


# Improved Denoising CNN inspired by FFDNet
class FFDNetDenoiser(nn.Module):
    def __init__(self, num_channels=3):
        super(FFDNetDenoiser, self).__init__()

        # Encoder
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, num_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        out = torch.sigmoid(self.conv4(x))  # Output clamped to [0, 1]
        return out


# Training
def train_model(train_dir, output_dir='improved_model_checkpoints5', num_epochs=100, batch_size=32, learning_rate=0.0001):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Dataset and DataLoader with pin_memory
    dataset = NoisyImageDataset(train_dir, patch_size=128)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)  # Pin memory enabled

    # Model, loss functions, optimizer, scheduler
    model = FFDNetDenoiser().to(device)
    mse_loss = nn.MSELoss().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

    # Mixed precision scaler
    scaler = amp.GradScaler()  # Create a gradient scaler for mixed precision training

    # Training loop
    best_loss = float('inf')
    checkpoint_path = os.path.join(output_dir, 'best_denoising_model.pth')

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch + 1}/{num_epochs}')

        for noisy_imgs, clean_imgs in progress_bar:
            noisy_imgs, clean_imgs = noisy_imgs.to(device), clean_imgs.to(device)

            # Forward pass with mixed precision
            with amp.autocast():  # Automatic mixed precision context
                outputs = model(noisy_imgs)
                loss = mse_loss(outputs, clean_imgs)

            # Backward pass
            optimizer.zero_grad()
            scaler.scale(loss).backward()  # Scale the loss before backward pass
            scaler.step(optimizer)  # Update the weights
            scaler.update()  # Update the scale for next iteration

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

        # Optional: Profile the training loop
        if epoch == 0:  # Profile only for the first epoch for demonstration
            with torch.profiler.profile(profile_memory=True) as prof:
                for _ in range(10):  # Profile for a few batches
                    noisy_imgs, clean_imgs = next(iter(dataloader))
                    noisy_imgs, clean_imgs = noisy_imgs.to(device), clean_imgs.to(device)
                    with amp.autocast():
                        outputs = model(noisy_imgs)
            print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

if __name__ == "__main__":
    # Set the directory containing training images
    train_dir = "training_images"

    # Train the model
    train_model(train_dir, num_epochs=100)
