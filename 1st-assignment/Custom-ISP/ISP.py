import numpy as np
from scipy.ndimage import convolve
import numpy.typing as npt
from imageio import imwrite

def load_bayer_image(path: str, width: int = 1920, height: int = 1280) -> npt.NDArray[np.uint16]:
    """Load a raw Bayer image with improved error handling."""
    try:
        with open(path, 'rb') as f:
            raw_data = np.fromfile(f, dtype=np.uint16)
        expected_size = width * height
        if len(raw_data) != expected_size:
            raise ValueError(f"Image size mismatch. Expected {expected_size} pixels, got {len(raw_data)}")
        return raw_data.reshape((height, width))
    except Exception as e:
        print(f"Error loading image: {e}")
        raise

def demosaic_advanced(bayer_image: npt.NDArray[np.uint16]) -> npt.NDArray[np.uint16]:
    """Perform advanced demosaicing with gradient-based and bilinear interpolation."""
    height, width = bayer_image.shape
    rgb_image = np.zeros((height, width, 3), dtype=np.uint16)
    
    # Masks for the Bayer pattern
    green1_mask = np.zeros_like(bayer_image, dtype=bool)
    green2_mask = np.zeros_like(bayer_image, dtype=bool)
    red_mask = np.zeros_like(bayer_image, dtype=bool)
    blue_mask = np.zeros_like(bayer_image, dtype=bool)
    
    green1_mask[0::2, 0::2] = True   
    green2_mask[1::2, 1::2] = True   
    red_mask[0::2, 1::2] = True      
    blue_mask[1::2, 0::2] = True     

    green_channel = np.zeros_like(bayer_image, dtype=np.uint16)
    red_channel = np.zeros_like(bayer_image, dtype=np.uint16)
    blue_channel = np.zeros_like(bayer_image, dtype=np.uint16)
    
    green_channel[green1_mask | green2_mask] = bayer_image[green1_mask | green2_mask]
    red_channel[red_mask] = bayer_image[red_mask]
    blue_channel[blue_mask] = bayer_image[blue_mask]

    # Bilinear kernel for interpolation
    bilinear_kernel = np.array([[0.25, 0.5, 0.25],
                                [0.5,  1.0, 0.5],
                                [0.25, 0.5, 0.25]], dtype=np.float32) / 4.0

    green_channel = convolve(green_channel, bilinear_kernel, mode='mirror')
    red_channel = convolve(red_channel, bilinear_kernel, mode='mirror')
    blue_channel = convolve(blue_channel, bilinear_kernel, mode='mirror')

    # Stack RGB channels
    rgb_image[:, :, 0] = red_channel
    rgb_image[:, :, 1] = green_channel
    rgb_image[:, :, 2] = blue_channel

    return rgb_image

def auto_tone(rgb_image: npt.NDArray[np.uint16]) -> npt.NDArray[np.uint16]:
    """Apply auto-tone adjustment to each channel based on histogram stretching."""
    max_val = 4095  # 12-bit max value
    tone_adjusted = np.zeros_like(rgb_image, dtype=np.uint16)

    for c in range(3):  # For each channel (R, G, B)
        channel = rgb_image[:, :, c]
        min_val = np.min(channel)
        max_channel_val = np.max(channel)

        # Avoid divide by zero if channel has no variation
        if max_channel_val > min_val:
            scale = max_val / (max_channel_val - min_val)
            tone_adjusted[:, :, c] = np.clip((channel - min_val) * scale, 0, max_val).astype(np.uint16)
        else:
            tone_adjusted[:, :, c] = channel  # No adjustment needed if no variation

    return tone_adjusted

def apply_white_balance(rgb_image: npt.NDArray[np.uint16]) -> npt.NDArray[np.uint16]:
    avg_r = np.mean(rgb_image[:, :, 0])
    avg_g = np.mean(rgb_image[:, :, 1])
    avg_b = np.mean(rgb_image[:, :, 2])

    scale_r = avg_g / avg_r
    scale_b = avg_g / avg_b

    rgb_image[:, :, 0] = np.clip(rgb_image[:, :, 0] * scale_r, 0, 4095)
    rgb_image[:, :, 2] = np.clip(rgb_image[:, :, 2] * scale_b, 0, 4095)

    return rgb_image


def apply_color_correction(rgb_image: npt.NDArray[np.uint16]) -> npt.NDArray[np.uint16]:
    """Apply a stronger color correction matrix for more vivid colors."""
    color_correction_matrix = np.array([
    [1.5, -0.2, -0.2],
    [-0.2, 1.5, -0.2],
    [-0.2, -0.2, 1.5]
])




    corrected_rgb_image = rgb_image.astype(np.float32)
    for i in range(rgb_image.shape[0]):
        for j in range(rgb_image.shape[1]):
            corrected_rgb_image[i, j, :] = np.dot(color_correction_matrix, rgb_image[i, j, :])

    return np.clip(corrected_rgb_image, 0, 4095).astype(np.uint16)


def adaptive_log_tone_map(rgb_image: npt.NDArray[np.uint16]) -> npt.NDArray[np.float32]:
    """Apply adaptive logarithmic tone mapping to retain vivid colors."""
    max_value = 4095  # 12-bit max value
    scaled_image = rgb_image / max_value
    tone_mapped_image = np.log1p(scaled_image * 20) / np.log1p(20)
    return np.clip(tone_mapped_image, 0, 1)

def adaptive_power_tone_map(rgb_image: npt.NDArray[np.uint16], compression_factor: float = 0.7) -> npt.NDArray[np.float32]:
    """Apply adaptive power-based tone mapping to make the image darker and preserve more contrast."""
    max_value = 4095  # 12-bit max value
    scaled_image = rgb_image / max_value
    tone_mapped_image = np.power(scaled_image, compression_factor)
    return np.clip(tone_mapped_image, 0, 1)

def gamma_correct(rgb_image: npt.NDArray[np.float32], gamma: float = 2.2) -> npt.NDArray[np.uint8]:
    """Apply gamma correction and convert to 8-bit."""
    gamma_corrected = np.clip(255 * (rgb_image ** (1 / gamma)), 0, 255)
    return gamma_corrected.astype(np.uint8)

def process_bayer_image(bayer_image_path: str, output_path: str = 'final_image18.png'):
    """
    Full image processing pipeline: demosaicing, auto-tone, white balance, color correction, tone mapping, and gamma correction.
    """
    try:
        bayer_image = load_bayer_image(bayer_image_path)
        
        rgb_image = demosaic_advanced(bayer_image)
        
        rgb_image = auto_tone(rgb_image)  # Apply auto tone adjustment
        
        rgb_image = apply_white_balance(rgb_image)
        
        rgb_image = apply_color_correction(rgb_image)
        
        
        tone_mapped_image = adaptive_power_tone_map(rgb_image, compression_factor=2)
        
        final_image = gamma_correct(tone_mapped_image, gamma=2.2)
        
        imwrite(output_path, final_image)
        print(f"Image processing completed and saved to: {output_path}")
    
    except Exception as e:
        print(f"Error in image processing: {e}")

# Example usage
if __name__ == "__main__":
    bayer_image_path = '12bits2.raw'  # Change to your raw image path
    process_bayer_image(bayer_image_path)
