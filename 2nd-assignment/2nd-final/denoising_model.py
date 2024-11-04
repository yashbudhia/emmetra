import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

class DenoisingCNN(nn.Module):
    def __init__(self, kernel_size=3):
        super(DenoisingCNN, self).__init__()
        # Encoder
        self.enc1 = nn.Conv2d(3, 64, kernel_size, padding=kernel_size//2)
        self.enc2 = nn.Conv2d(64, 128, kernel_size, padding=kernel_size//2)
        self.enc3 = nn.Conv2d(128, 256, kernel_size, padding=kernel_size//2)
        
        # Decoder
        self.dec3 = nn.Conv2d(256, 128, kernel_size, padding=kernel_size//2)
        self.dec2 = nn.Conv2d(128, 64, kernel_size, padding=kernel_size//2)
        self.dec1 = nn.Conv2d(64, 3, kernel_size, padding=kernel_size//2)
        
        self.activation = nn.ReLU()
        self.batch_norm64 = nn.BatchNorm2d(64)
        self.batch_norm128 = nn.BatchNorm2d(128)
        self.batch_norm256 = nn.BatchNorm2d(256)
        
        # Dropout for regularization
        self.dropout = nn.Dropout2d(0.2)

    def forward(self, x):
        # Encoder
        x1 = self.dropout(self.batch_norm64(self.activation(self.enc1(x))))
        x2 = self.dropout(self.batch_norm128(self.activation(self.enc2(x1))))
        x3 = self.dropout(self.batch_norm256(self.activation(self.enc3(x2))))
        
        # Decoder
        d3 = self.dropout(self.batch_norm128(self.activation(self.dec3(x3))))
        d2 = self.dropout(self.batch_norm64(self.activation(self.dec2(d3))))
        d1 = self.dec1(d2)
        
        return torch.sigmoid(d1)

class NoisyImageDataset(Dataset):
    def __init__(self, clean_images, noise_level=0.1):
        self.clean_images = clean_images
        self.noise_level = noise_level

    def __len__(self):
        return len(self.clean_images)

    def __getitem__(self, idx):
        clean_image = self.clean_images[idx]
        # Add Poisson noise
        noisy_image = np.random.poisson(clean_image / 255.0 * 255) / 255.0
        noisy_image = np.clip(noisy_image, 0, 1)
        
        # Convert to torch tensors
        clean_tensor = torch.from_numpy(clean_image).float().permute(2, 0, 1)
        noisy_tensor = torch.from_numpy(noisy_image).float().permute(2, 0, 1)
        
        return noisy_tensor, clean_tensor

def train_denoising_model(clean_images, epochs=100, batch_size=32, learning_rate=0.001):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create dataset and dataloader
    dataset = NoisyImageDataset(clean_images)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model
    model = DenoisingCNN().to(device)
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    
    # Training loop
    best_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, (noisy_imgs, clean_imgs) in enumerate(dataloader):
            noisy_imgs = noisy_imgs.to(device)
            clean_imgs = clean_imgs.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(noisy_imgs)
            loss = criterion(outputs, clean_imgs)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f'Epoch: {epoch+1}/{epochs}, Batch: {batch_idx}/{len(dataloader)}, '
                      f'Loss: {loss.item():.6f}')
        
        # Average loss for the epoch
        avg_loss = total_loss / len(dataloader)
        print(f'Epoch: {epoch+1}/{epochs}, Average Loss: {avg_loss:.6f}')
        
        # Learning rate scheduling
        scheduler.step(avg_loss)
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, 'best_denoising_model.pth')
    
    return model

def run_denoising_cnn(noisy_image, model_path='best_denoising_model.pth', device='cuda'):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = DenoisingCNN().to(device)
    
    # Load trained model
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Prepare input
    noisy_torch = torch.from_numpy(noisy_image).float().permute(2, 0, 1).unsqueeze(0).to(device)
    
    # Inference
    model.eval()
    with torch.no_grad():
        denoised_torch = model(noisy_torch)
    
    # Convert back to numpy
    denoised_np = denoised_torch.cpu().squeeze().permute(1, 2, 0).numpy()
    return np.clip(denoised_np, 0, 1)
