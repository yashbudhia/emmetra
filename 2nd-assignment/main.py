import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import subprocess
import sys
print(sys.executable)

def run_ffdnet_denoising():
    """
    Read the denoised image from the FFDNet directory and return it as a color image.
    """
    ffdnet_image_path = "ffdnet-pytorch/ffdnet.png"
    
    # Read and return the denoised image as a color image
    denoised_image = cv2.imread(ffdnet_image_path, cv2.IMREAD_COLOR)
    if denoised_image is None:
        print(f"Failed to read the denoised image from {ffdnet_image_path}.")
    return denoised_image

def compute_spatial_snr(image, roi_coords, method_name=""):
    x, y, w, h = roi_coords
    roi = image[y:y+h, x:x+w]
    mean_signal = np.mean(roi)
    noise = np.std(roi)
    snr = 20 * np.log10(mean_signal / noise) if noise != 0 else float('inf')
    plt.figure(figsize=(8, 4))
    plt.subplot(121)
    plt.imshow(image, cmap='gray')
    rect = plt.Rectangle((x, y), w, h, fill=False, color='red')
    plt.gca().add_patch(rect)
    plt.title(f"{method_name}\nROI Location")
    plt.subplot(122)
    plt.imshow(roi, cmap='gray')
    plt.title(f"ROI Analysis\nSNR: {snr:.2f}dB")
    plt.show()
    return snr

def compute_edge_strength(image, method='sobel'):
    if method == 'sobel':
        gradient_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    elif method == 'laplacian':
        gradient_magnitude = np.abs(cv2.Laplacian(image, cv2.CV_64F))
    gradient_magnitude = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX)
    return gradient_magnitude.astype(np.uint8)

input_dir = "input"
output_dir = "output_images"
os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(input_dir):
    image_path = os.path.join(input_dir, filename)
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        print(f"Error loading image {filename}.")
        continue
    
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    roi_dark = (100, 100, 50, 50)
    roi_mid = (200, 200, 50, 50)
    roi_light = (300, 300, 50, 50)

    results = {
        'Original': gray_image,
        'Median': cv2.medianBlur(gray_image, 5),
        'Bilateral': cv2.bilateralFilter(gray_image, 9, 75, 75),
        'Gaussian': cv2.GaussianBlur(gray_image, (5, 5), 0)
    }

    # Retrieve the FFDNet denoised image (now directly reading from the file)
    results['FFDNet'] = run_ffdnet_denoising()

    comparison_results = {}
    for method_name, processed_image in results.items():
        print(f"\nAnalyzing {method_name} for {filename}:")
        
        snr_dark = compute_spatial_snr(processed_image, roi_dark, f"{method_name} - Dark Region")
        snr_mid = compute_spatial_snr(processed_image, roi_mid, f"{method_name} - Mid Region")
        snr_light = compute_spatial_snr(processed_image, roi_light, f"{method_name} - Light Region")
        
        edge_sobel = compute_edge_strength(processed_image, 'sobel')
        edge_laplacian = compute_edge_strength(processed_image, 'laplacian')
        
        comparison_results[method_name] = {
            'SNR_Dark': snr_dark,
            'SNR_Mid': snr_mid,
            'SNR_Light': snr_light,
            'Edge_Sobel': np.mean(edge_sobel),
            'Edge_Laplacian': np.mean(edge_laplacian)
        }
        
        cv2.imwrite(os.path.join(output_dir, f"{filename}_{method_name.lower()}_sobel_edges.png"), edge_sobel)
        cv2.imwrite(os.path.join(output_dir, f"{filename}_{method_name.lower()}_laplacian_edges.png"), edge_laplacian)

    print(f"\nComparison Results for {filename}:")
    print("\n| Method     | SNR Dark | SNR Mid  | SNR Light | Edge (Sobel) | Edge (Laplacian) |")
    print("|------------|-----------|----------|-----------|--------------|-----------------|")
    for method, results in comparison_results.items():
        print(f"| {method:<10} | {results['SNR_Dark']:9.2f} | {results['SNR_Mid']:8.2f} | {results['SNR_Light']:9.2f} | {results['Edge_Sobel']:12.2f} | {results['Edge_Laplacian']:15.2f} |")
