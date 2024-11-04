import cv2
import numpy as np
import os

def process_bayer_image(
    file_path, gamma=2.2, white_balance=True, denoise=True, sharpen=True,
    demosaic=True, blur=5):
    
    # Load the 12-bit Bayer RAW image
    height, width = 1280, 1920  # Replace with actual dimensions if different
    raw_data = np.fromfile(file_path, dtype=np.uint16)
    raw_data = raw_data.reshape((height, width))

    # 1. Apply Demosaic with Edge-based Interpolation (5x5)
    if demosaic:
        # Assuming a GBRG Bayer pattern; adjust pattern as needed
        bayer_pattern = cv2.COLOR_BAYER_GR2BGR
        rgb_image = cv2.cvtColor(raw_data, bayer_pattern)
    else:
        # Stack Bayer image into 3 channels if demosaicing is disabled
        rgb_image = np.stack((raw_data,) * 3, axis=-1)

    # Convert 12-bit to 8-bit
    rgb_image = (rgb_image / 16).astype(np.uint8)

    # 2. Apply White Balance using the Gray World Algorithm
    if white_balance:
        avg_rgb = np.mean(rgb_image, axis=(0, 1))
        gray_world_scale = np.mean(avg_rgb) / avg_rgb
        for i in range(3):
            rgb_image[:, :, i] = np.clip(rgb_image[:, :, i] * gray_world_scale[i], 0, 255)

# Ensure the blur kernel size is positive and odd
    if denoise:
        blur = blur if blur % 2 == 1 else blur + 1
        rgb_image = cv2.GaussianBlur(rgb_image, (blur, blur), 0)


    # 4. Apply Gamma Correction (12-bit to 8-bit sRGB)
    if gamma != 1.0:
        inv_gamma = 1.0 / gamma
        lookup_table = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)]).astype('uint8')
        rgb_image = cv2.LUT(rgb_image, lookup_table)

    # 5. Apply Sharpening Filter using Unsharp Mask
    if sharpen:
        # Applying Gaussian blur to create the blurred version for unsharp mask
        blurred = cv2.GaussianBlur(rgb_image, (5, 5), 0)
        # Weighted blend to create the sharpened effect
        rgb_image = cv2.addWeighted(rgb_image, 1.5, blurred, -0.5, 0)

    # Save the processed image
    processed_image_path = os.path.join('static', 'processed_image.jpg')
    cv2.imwrite(processed_image_path, rgb_image)

    return processed_image_path
