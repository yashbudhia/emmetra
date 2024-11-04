import torch
import torchvision.transforms as transforms
from PIL import Image
import argparse
from pathlib import Path
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import os
import math

# Default paths
SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODEL_PATH = SCRIPT_DIR / "improved_model_checkpoints6/best_denoising_model.pth"
DEFAULT_INPUT_DIR = SCRIPT_DIR / "training_images"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "denoised_images"

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

def process_in_patches(model, image_tensor, patch_size=512, overlap=32):
    """Process image in patches to reduce memory usage."""
    device = image_tensor.device
    b, c, h, w = image_tensor.shape
    
    # Initialize output tensor
    output = torch.zeros_like(image_tensor)
    
    # Calculate effective patch size considering overlap
    effective_patch = patch_size - 2 * overlap
    
    # Calculate number of patches in each dimension
    num_patches_h = math.ceil((h - 2 * overlap) / effective_patch)
    num_patches_w = math.ceil((w - 2 * overlap) / effective_patch)
    
    # Process each patch
    for i in range(num_patches_h):
        for j in range(num_patches_w):
            # Calculate patch boundaries
            start_h = i * effective_patch
            start_w = j * effective_patch
            end_h = min(start_h + patch_size, h)
            end_w = min(start_w + patch_size, w)
            
            # Adjust start positions to account for image boundary
            start_h = max(0, end_h - patch_size)
            start_w = max(0, end_w - patch_size)
            
            # Extract patch
            patch = image_tensor[:, :, start_h:end_h, start_w:end_w]
            
            # Process patch
            with torch.cuda.amp.autocast(enabled=True):  # Enable mixed precision
                processed_patch = model(patch)
            
            # Calculate valid region (excluding overlap)
            valid_start_h = overlap if i > 0 else 0
            valid_start_w = overlap if j > 0 else 0
            valid_end_h = patch_size - overlap if i < num_patches_h - 1 else patch_size
            valid_end_w = patch_size - overlap if j < num_patches_w - 1 else patch_size
            
            # Calculate corresponding positions in output tensor
            out_start_h = start_h + valid_start_h
            out_start_w = start_w + valid_start_w
            out_end_h = start_h + valid_end_h
            out_end_w = start_w + valid_end_w
            
            # Copy valid region to output tensor
            output[:, :, out_start_h:out_end_h, out_start_w:out_end_w] = \
                processed_patch[:, :, valid_start_h:valid_end_h, valid_start_w:valid_end_w]
            
            # Clear cache after each patch
            torch.cuda.empty_cache()
    
    return output

def denoise_image(model_path, input_path, output_path, patch_size=512):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Clear CUDA cache before starting
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Load model
    model = ImprovedFFDNet().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load and preprocess image
    image = Image.open(input_path).convert('RGB')
    transform = transforms.ToTensor()
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Denoise image
    print(f"Denoising image: {input_path}")
    with torch.no_grad():
        if device == 'cuda':
            # Process large images in patches
            denoised_tensor = process_in_patches(model, image_tensor, patch_size=patch_size)
        else:
            # Process whole image at once if on CPU
            denoised_tensor = model(image_tensor)
    
    # Save result
    denoised_image = transforms.ToPILImage()(denoised_tensor.squeeze().cpu())
    denoised_image.save(output_path)
    print(f"Denoised image saved to: {output_path}")
    
    # Final cache clear
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def denoise_directory(model_path, input_dir, output_dir):
    """Denoise all images in the input directory"""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all image files
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    image_files = [f for f in input_dir.iterdir() if f.suffix.lower() in image_extensions]
    
    if not image_files:
        print(f"No images found in {input_dir}")
        return
    
    for image_path in image_files:
        output_path = output_dir / f"denoised_{image_path.name}"
        try:
            denoise_image(model_path, str(image_path), str(output_path))
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            continue

def main():
    parser = argparse.ArgumentParser(description='Denoise images using the trained model')
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL_PATH,
                      help='Path to the model checkpoint (.pth file)')
    parser.add_argument('--input', type=str, default=DEFAULT_INPUT_DIR,
                      help='Path to input image or directory')
    parser.add_argument('--output', type=str, default=DEFAULT_OUTPUT_DIR,
                      help='Path to output image or directory')
    parser.add_argument('--patch-size', type=int, default=512,
                      help='Size of patches for processing large images')
    
    args = parser.parse_args()
    
    # Set memory-efficient settings
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
    
    # Create directories if they don't exist
    DEFAULT_INPUT_DIR.mkdir(parents=True, exist_ok=True)
    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    if input_path.is_dir():
        print(f"Processing all images in directory: {input_path}")
        denoise_directory(args.model, input_path, output_path)
    else:
        if output_path.is_dir():
            output_path = output_path / f"denoised_{input_path.name}"
        denoise_image(args.model, str(input_path), str(output_path), args.patch_size)

if __name__ == "__main__":
    main()