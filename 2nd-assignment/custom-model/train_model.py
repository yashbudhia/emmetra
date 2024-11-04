import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
from tqdm import tqdm

class NoisyImageDataset(Dataset):
    def __init__(self, image_dir, patch_size=64, noise_level=25):
        self.image_dir = image_dir
        self.patch_size = patch_size
        self.noise_level = noise_level
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    
    def add_noise(self, image):
        # Add Gaussian noise
        noise = torch.randn_like(image) * (self.noise_level / 255.0)
        noisy_image = image + noise
        return torch.clamp(noisy_image, 0, 1)
    
    def __len__(self):
        return len(self.image_files) * 100  # 100 patches per image
    
    def __getitem__(self, idx):
        img_idx = idx // 100
        image_path = os.path.join(self.image_dir, self.image_files[img_idx])
        
        # Load and convert to RGB
        image = Image.open(image_path).convert('RGB')
        
        # Random crop
        i, j = torch.randint(0, max(1, image.size[1] - self.patch_size), (1,)), \
               torch.randint(0, max(1, image.size[0] - self.patch_size), (1,))
        
        image = transforms.functional.crop(image, i.item(), j.item(), self.patch_size, self.patch_size)
        
        # Convert to tensor
        clean_tensor = self.transform(image)
        
        # Add noise
        noisy_tensor = self.add_noise(clean_tensor)
        
        return noisy_tensor, clean_tensor

class DenoisingCNN(nn.Module):
    def __init__(self, num_channels=3):
        super(DenoisingCNN, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(num_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        
        self.middle = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.middle(x)
        x = self.decoder(x)
        return x

def train_model(train_dir, output_dir='model_checkpoints', num_epochs=50, batch_size=32, learning_rate=0.001):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dataset and dataloader
    dataset = NoisyImageDataset(train_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    # Initialize model, loss function, and optimizer
    model = DenoisingCNN().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Load existing best model checkpoint if it exists
    best_loss = float('inf')
    checkpoint_path = os.path.join(output_dir, 'best_denoising_model.pth')
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        best_loss = checkpoint['loss']
        print(f"Resuming training from checkpoint with best loss: {best_loss:.6f}")
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for noisy_imgs, clean_imgs in progress_bar:
            noisy_imgs = noisy_imgs.to(device)
            clean_imgs = clean_imgs.to(device)
            
            # Forward pass
            outputs = model(noisy_imgs)
            loss = criterion(outputs, clean_imgs)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
        
        avg_loss = epoch_loss / len(dataloader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.6f}')
        
        # Save checkpoint if it's the best model so far
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, checkpoint_path)
            print(f'Saved best model checkpoint to {checkpoint_path}')
        
        # Save regular checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            epoch_checkpoint_path = os.path.join(output_dir, f'denoising_model_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, epoch_checkpoint_path)


def load_model(checkpoint_path, device='cpu'):
    # Initialize model
    model = DenoisingCNN().to(device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)  # strict=False to ignore mismatched keys
    
    # Optionally, load optimizer state if needed
    optimizer = optim.Adam(model.parameters())
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"Model loaded from {checkpoint_path} with strict=False")
    return model, optimizer

if __name__ == "__main__":
    # Set your training images directory
    train_dir = "training_images"  # Directory containing your training images
    
    # Train the model
    train_model(train_dir, num_epochs=50)
    
    # Load the best model checkpoint for testing/inference
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    best_model, optimizer = load_model('model_checkpoints/best_denoising_model.pth', device)
