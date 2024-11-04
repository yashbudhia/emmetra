import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class ImprovedDenoiser(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        
        # Encoder
        self.conv1 = DoubleConv(in_channels, 64)
        self.conv2 = DoubleConv(64, 128)
        self.conv3 = DoubleConv(128, 256)
        
        # Decoder with skip connections
        self.upconv2 = DoubleConv(256 + 128, 128)
        self.upconv1 = DoubleConv(128 + 64, 64)
        
        # Final output
        self.final_conv = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Attention gates
        self.attention1 = AttentionGate(128, 256, 128)
        self.attention2 = AttentionGate(64, 128, 64)

    def forward(self, x):
        # Encoder
        conv1 = self.conv1(x)
        pool1 = F.max_pool2d(conv1, 2)
        
        conv2 = self.conv2(pool1)
        pool2 = F.max_pool2d(conv2, 2)
        
        conv3 = self.conv3(pool2)
        
        # Decoder with attention and skip connections
        up2 = F.interpolate(conv3, size=conv2.shape[2:], mode='bilinear', align_corners=True)
        attn2 = self.attention1(conv2, up2)
        up2 = torch.cat([up2, attn2], dim=1)
        up2 = self.upconv2(up2)
        
        up1 = F.interpolate(up2, size=conv1.shape[2:], mode='bilinear', align_corners=True)
        attn1 = self.attention2(conv1, up1)
        up1 = torch.cat([up1, attn1], dim=1)
        up1 = self.upconv1(up1)
        
        return self.final_conv(up1)

class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.84):
        super().__init__()
        self.alpha = alpha
        self.l1_loss = nn.L1Loss()
        self.ssim = SSIM()
        
    def forward(self, pred, target):
        l1_loss = self.l1_loss(pred, target)
        ssim_loss = 1 - self.ssim(pred, target)
        return self.alpha * l1_loss + (1 - self.alpha) * ssim_loss

class SSIM(nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super().__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = self.create_window(window_size)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()
        window = self.window.to(img1.device)
        
        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = self.create_window(self.window_size).to(img1.device).type_as(img1)
            self.window = window
            self.channel = channel

        return self._ssim(img1, img2, window, self.window_size, channel, self.size_average)

    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
        return gauss/gauss.sum()

    def create_window(self, window_size):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(1, 1, window_size, window_size)
        return window

    def _ssim(self, img1, img2, window, window_size, channel, size_average=True):
        mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2

        C1 = 0.01**2
        C2 = 0.03**2

        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)
        
# Dataset class with patch extraction and data augmentation
class DenoisingDataset(Dataset):
    def __init__(self, image_pairs, transform=None, patch_size=256):
        self.image_pairs = image_pairs
        self.transform = transform
        self.patch_size = patch_size

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        noisy_img, clean_img = self.image_pairs[idx]

        # Extract random patch of size 256x256
        _, h, w = noisy_img.size()
        top = torch.randint(0, h - self.patch_size + 1, (1,)).item()
        left = torch.randint(0, w - self.patch_size + 1, (1,)).item()

        noisy_patch = noisy_img[:, top:top + self.patch_size, left:left + self.patch_size]
        clean_patch = clean_img[:, top:top + self.patch_size, left:left + self.patch_size]

        if self.transform:
            noisy_patch = self.transform(noisy_patch)
            clean_patch = self.transform(clean_patch)

        return noisy_patch, clean_patch

# Initialize model, optimizer, and loss function
def initialize_model(in_channels=3, learning_rate=1e-4, weight_decay=1e-4, alpha=0.84, device='cuda'):
    model = ImprovedDenoiser(in_channels).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = CombinedLoss(alpha=alpha).to(device)
    return model, optimizer, criterion

# Training loop with validation
def train_model(train_data, val_data, batch_size=16, num_epochs=10, learning_rate=1e-4, alpha=0.84, device='cuda'):
    # Data augmentations with color jittering
    transform = transforms.Compose([
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.ToTensor()
    ])
    
    train_loader = DataLoader(DenoisingDataset(train_data, transform=transform, patch_size=256), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(DenoisingDataset(val_data, transform=transforms.ToTensor(), patch_size=256), batch_size=batch_size, shuffle=False)

    # Initialize model, optimizer, and loss function
    model, optimizer, criterion = initialize_model(learning_rate=learning_rate, weight_decay=1e-4, alpha=alpha, device=device)

    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        for noisy_imgs, clean_imgs in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{num_epochs}"):
            noisy_imgs, clean_imgs = noisy_imgs.to(device), clean_imgs.to(device)

            optimizer.zero_grad()
            outputs = model(noisy_imgs)
            loss = criterion(outputs, clean_imgs)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Training Loss: {avg_train_loss:.4f}")

        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for noisy_imgs, clean_imgs in tqdm(val_loader, desc="Validating"):
                noisy_imgs, clean_imgs = noisy_imgs.to(device), clean_imgs.to(device)
                outputs = model(noisy_imgs)
                loss = criterion(outputs, clean_imgs)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Validation Loss: {avg_val_loss:.4f}")

        # Save the model if validation loss improves
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_denoiser_model.pth')
            print(f"Model saved with validation loss: {best_val_loss:.4f}")

    print("Training complete.")
    return model

# Usage example:
# Assuming `train_data` and `val_data` are lists of (noisy_image, clean_image) pairs
device = 'cuda' if torch.cuda.is_available() else 'cpu'
trained_model = train_model(train_data, val_data, batch_size=16, num_epochs=10, learning_rate=1e-4, alpha=0.84, device=device)