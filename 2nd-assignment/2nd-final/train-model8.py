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
import torchvision.models as models

class NoisyImageDataset(Dataset):
    def __init__(self, image_dir, patch_size=128, noise_level=25):
        self.image_dir = image_dir
        self.patch_size = patch_size
        self.noise_level = noise_level
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(90),
            transforms.ColorJitter(brightness=0.1, contrast=0.1)
        ])

    def add_mixed_noise(self, image):
        # Advanced noise model combining multiple noise types
        gaussian_noise = torch.randn_like(image) * (self.noise_level / 255.0)
        poisson_noise = torch.poisson(image * 255) / 255.0 - image
        speckle_noise = torch.randn_like(image) * image * (self.noise_level / 255.0)
        
        # Random noise mixture
        mix_weights = torch.rand(3)
        mix_weights = mix_weights / mix_weights.sum()
        
        noisy_image = (image + 
                      mix_weights[0] * gaussian_noise + 
                      mix_weights[1] * poisson_noise + 
                      mix_weights[2] * speckle_noise)
        return torch.clamp(noisy_image, 0, 1)

    def __getitem__(self, idx):
        img_idx = idx % len(self.image_files)
        image_path = os.path.join(self.image_dir, self.image_files[img_idx])
        image = Image.open(image_path).convert('RGB')
        
        # Random crop
        i, j = torch.randint(0, max(1, image.size[1] - self.patch_size), (1,)), \
               torch.randint(0, max(1, image.size[0] - self.patch_size), (1,))
        image = transforms.functional.crop(image, i.item(), j.item(), self.patch_size, self.patch_size)
        
        clean_tensor = self.transform(image)
        noisy_tensor = self.add_mixed_noise(clean_tensor)
        
        return noisy_tensor, clean_tensor

    def __len__(self):
        return len(self.image_files) * 100

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out

class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super(AttentionBlock, self).__init__()
        self.conv_query = nn.Conv2d(channels, channels//8, kernel_size=1)
        self.conv_key = nn.Conv2d(channels, channels//8, kernel_size=1)
        self.conv_value = nn.Conv2d(channels, channels, kernel_size=1)
        
    def forward(self, x):
        batch_size, channels, height, width = x.size()
        
        query = self.conv_query(x).view(batch_size, -1, height * width)
        key = self.conv_key(x).view(batch_size, -1, height * width)
        value = self.conv_value(x).view(batch_size, -1, height * width)
        
        attention = torch.bmm(query.permute(0, 2, 1), key)
        attention = F.softmax(attention, dim=-1)
        
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channels, height, width)
        
        return out + x

class ImprovedFFDNet(nn.Module):
    def __init__(self, num_channels=3):
        super(ImprovedFFDNet, self).__init__()
        
        # Encoder
        self.enc_conv1 = nn.Conv2d(num_channels, 64, kernel_size=3, padding=1)
        self.enc_conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2)
        self.enc_conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2)
        
        # Residual blocks with attention
        self.res_blocks = nn.ModuleList([
            ResidualBlock(256) for _ in range(6)
        ])
        self.attention = AttentionBlock(256)
        
        # Decoder
        self.dec_conv1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.dec_conv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.dec_conv3 = nn.Conv2d(64, num_channels, kernel_size=3, padding=1)
        
        # Feature extraction for perceptual loss
        vgg = models.vgg16(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(vgg.features)[:16]).eval()
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def forward(self, x):
        # Encoder
        enc1 = F.relu(self.enc_conv1(x))
        enc2 = F.relu(self.enc_conv2(enc1))
        enc3 = F.relu(self.enc_conv3(enc2))
        
        # Residual blocks with attention
        out = enc3
        for res_block in self.res_blocks:
            out = res_block(out)
        out = self.attention(out)
        
        # Decoder with skip connections
        dec1 = F.relu(self.dec_conv1(out))
        dec2 = F.relu(self.dec_conv2(dec1))
        dec3 = torch.sigmoid(self.dec_conv3(dec2))
        
        return dec3
    
    def get_features(self, x):
        return self.feature_extractor(x)

class CombinedLoss(nn.Module):
    def __init__(self, lambda_perceptual=0.1):
        super(CombinedLoss, self).__init__()
        self.lambda_perceptual = lambda_perceptual
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()

    def forward(self, pred, target, pred_features, target_features):
        # L1 loss
        l1 = self.l1_loss(pred, target)
        
        # MSE loss
        mse = self.mse_loss(pred, target)
        
        # Perceptual loss
        perceptual = self.mse_loss(pred_features, target_features)
        
        # Combined loss
        total_loss = l1 + mse + self.lambda_perceptual * perceptual
        
        return total_loss

def train_model(train_dir, output_dir='improved_model_checkpoints6', num_epochs=100, batch_size=16, learning_rate=0.0001):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Dataset and DataLoader
    dataset = NoisyImageDataset(train_dir, patch_size=128)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8)

    # Model and loss functions
    model = ImprovedFFDNet().to(device)
    criterion = CombinedLoss().to(device)
    
    # Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=learning_rate/100)

    best_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch + 1}/{num_epochs}')

        for noisy_imgs, clean_imgs in progress_bar:
            noisy_imgs, clean_imgs = noisy_imgs.to(device), clean_imgs.to(device)

            # Forward pass
            denoised_imgs = model(noisy_imgs)
            
            # Calculate features for perceptual loss
            with torch.no_grad():
                clean_features = model.get_features(clean_imgs)
            denoised_features = model.get_features(denoised_imgs)
            
            # Calculate loss
            loss = criterion(denoised_imgs, clean_imgs, denoised_features, clean_features)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()

            epoch_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})

        avg_loss = epoch_loss / len(dataloader)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Average Loss: {avg_loss:.6f}')

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, os.path.join(output_dir, 'best_denoising_model.pth'))
            print(f'Saved best model checkpoint with loss: {best_loss:.6f}')

        scheduler.step()

def denoise_image(model_path, input_image_path, output_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = ImprovedFFDNet().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load and preprocess image
    image = Image.open(input_image_path).convert('RGB')
    transform = transforms.ToTensor()
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Denoise image
    with torch.no_grad():
        denoised_tensor = model(image_tensor)
    
    # Save result
    denoised_image = transforms.ToPILImage()(denoised_tensor.squeeze().cpu())
    denoised_image.save(output_path)

if __name__ == "__main__":
    train_dir = "training_images"
    train_model(train_dir, num_epochs=100)