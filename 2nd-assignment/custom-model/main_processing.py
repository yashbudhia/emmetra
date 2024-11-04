from datetime import datetime
import os
import numpy as np
import cv2
import torch
import torch.nn as nn
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import utils, colors
from reportlab.lib.units import inch
from reportlab.platypus import Table, TableStyle
from denoising_model import DenoisingCNN

def get_device():
    """
    Determine the best available device (CUDA GPU or CPU)
    Returns: torch.device object
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("CUDA not available. Using CPU")
    return device

def run_denoising_cnn(noisy_image, model_path='model_checkpoints/best_denoising_model.pth', device=None):
    try:
        if device is None:
            device = get_device()

        model = DenoisingCNN().to(device)
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model weights file not found at {model_path}")
        
        # Load checkpoint with weights_only=True for security
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        
        # Directly load the weights from the checkpoint
        model.load_state_dict(checkpoint, strict=False)  # Use strict=False to ignore any missing keys
        model.eval()

        noisy_torch = torch.from_numpy(noisy_image).float().permute(2, 0, 1).unsqueeze(0).to(device)

        with torch.no_grad():
            if device.type == 'cuda':
                with torch.autocast(device_type="cuda"):  # Fixed autocast for CUDA
                    denoised_torch = model(noisy_torch)
            else:
                denoised_torch = model(noisy_torch)

        denoised_np = denoised_torch.cpu().squeeze().permute(1, 2, 0).numpy()
        return np.clip(denoised_np, 0, 1)

    except Exception as e:
        print(f"Error running denoising model: {e}")
        return None


def compute_spatial_snr(image, roi_coords):
    x, y, w, h = roi_coords
    roi = image[y:y+h, x:x+w]
    mean_signal = np.mean(roi)
    noise = np.std(roi)
    snr = 20 * np.log10(mean_signal / noise) if noise != 0 else float('inf')
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

def add_image_with_caption(c, img_path, caption, x, y, width=2.5*inch):
    """Add an image with a properly aligned caption underneath"""
    try:
        img = utils.ImageReader(img_path)
        img_width, img_height = img.getSize()
        aspect = img_height / float(img_width)
        height = width * aspect
        
        # Draw image
        c.drawImage(img, x, y - height, width=width, height=height)
        
        # Add caption
        c.setFont("Helvetica", 8)
        text_width = c.stringWidth(caption, "Helvetica", 8)
        caption_x = x + (width - text_width) / 2  # Center caption under image
        c.drawString(caption_x, y - height - 12, caption)
        
        return y - height - 25  # Return new y position after image and caption
    except Exception as e:
        print(f"Error adding image {img_path}: {str(e)}")
        return y - 50  # Return fallback y position if image fails

def create_pdf_report(results, output_file="report.pdf", image_dir="output_images"):
    c = canvas.Canvas(output_file, pagesize=letter)
    width, height = letter
    margin = 50  # Standard margin
    
    # Title Page
    c.setFont("Helvetica-Bold", 24)
    c.drawCentredString(width/2, height-100, "Image Analysis Report")
    c.setFont("Helvetica", 14)
    c.drawCentredString(width/2, height-130, "Denoising and Edge Detection Comparison")
    c.drawCentredString(width/2, height-150, f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    c.showPage()
    
    # Results Pages
    for method, data in results.items():
        y_position = height - margin
        
        # Method header
        c.setFont("Helvetica-Bold", 16)
        c.drawString(margin, y_position, f"Method: {method}")
        y_position -= 30
        
        # Metrics section
        c.setFont("Helvetica-Bold", 12)
        c.drawString(margin, y_position, "Analysis Metrics:")
        y_position -= 20
        
        c.setFont("Helvetica", 10)
        metrics_text = [
            f"Signal-to-Noise Ratio (SNR):",
            f"  • Dark Region: {data['SNR_Dark']:.2f} dB",
            f"  • Mid Region: {data['SNR_Mid']:.2f} dB",
            f"  • Light Region: {data['SNR_Light']:.2f} dB",
            "",
            f"Edge Detection Metrics:",
            f"  • Sobel Edge Strength: {data['Edge_Sobel']:.2f}",
            f"  • Laplacian Edge Strength: {data['Edge_Laplacian']:.2f}"
        ]
        
        for text in metrics_text:
            c.drawString(margin + 10, y_position, text)
            y_position -= 15
        
        y_position -= 20
        
        # Image section
        image_width = (width - 2*margin - 40) / 3  # Divide available width into three columns
        
        try:
            # Original/filtered image
            x_position = margin
            filtered_path = os.path.join(image_dir, f"{method.lower()}.png")
            y_position = add_image_with_caption(c, filtered_path, "Filtered Image", x_position, y_position, width=image_width)
            
            # Sobel edges
            x_position = margin + image_width + 20
            sobel_path = os.path.join(image_dir, f"{method.lower()}_sobel_edges.png")
            add_image_with_caption(c, sobel_path, "Sobel Edge Detection", x_position, y_position + 25, width=image_width)
            
            # Laplacian edges
            x_position = margin + 2*(image_width + 20)
            laplacian_path = os.path.join(image_dir, f"{method.lower()}_laplacian_edges.png")
            add_image_with_caption(c, laplacian_path, "Laplacian Edge Detection", x_position, y_position + 25, width=image_width)
        except Exception as e:
            print(f"Error adding images for {method}: {str(e)}")
        
        c.showPage()
    
    # Summary table page
    c.setFont("Helvetica-Bold", 16)
    c.drawString(margin, height - margin, "Comparative Analysis Summary")
    
    # Create summary table data
    table_data = [["Method", "SNR Dark", "SNR Mid", "SNR Light", "Edge Sobel", "Edge Laplacian"]]
    for method, values in results.items():
        row = [
            method,
            f"{values['SNR_Dark']:.2f}",
            f"{values['SNR_Mid']:.2f}",
            f"{values['SNR_Light']:.2f}",
            f"{values['Edge_Sobel']:.2f}",
            f"{values['Edge_Laplacian']:.2f}"
        ]
        table_data.append(row)
    
    # Create and style table
    table = Table(table_data, colWidths=[120, 80, 80, 80, 80, 80])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('ALIGN', (1, 1), (-1, -1), 'RIGHT'),
    ]))
    
    # Draw table
    table.wrapOn(c, width - 2*margin, height)
    table.drawOn(c, margin, height - 200)
    
    c.save()
    print(f"PDF report generated successfully: {output_file}")

def main():
    image_path = "image.png"
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    
    if image is None:
        print("Error: Unable to load image.")
        return
    
    # Process image and generate results
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
    
    # Add AI-denoised result
    noisy_np = np.random.poisson(image / 255.0 * 255) / 255.0
    noisy_np = np.clip(noisy_np, 0, 1)
    ai_denoised_image = run_denoising_cnn(noisy_np)
    results['AI-Denoised'] = (ai_denoised_image * 255).astype(np.uint8)
    
    # Process results and generate report
    comparison_results = {}
    output_dir = "output_images"
    os.makedirs(output_dir, exist_ok=True)
    
    for method_name, processed_image in results.items():
        comparison_results[method_name] = {
            'SNR_Dark': compute_spatial_snr(processed_image, roi_dark),
            'SNR_Mid': compute_spatial_snr(processed_image, roi_mid),
            'SNR_Light': compute_spatial_snr(processed_image, roi_light),
            'Edge_Sobel': np.mean(compute_edge_strength(processed_image, 'sobel')),
            'Edge_Laplacian': np.mean(compute_edge_strength(processed_image, 'laplacian'))
        }
        
        # Save images
        cv2.imwrite(os.path.join(output_dir, f"{method_name.lower()}.png"), processed_image)
        cv2.imwrite(
            os.path.join(output_dir, f"{method_name.lower()}_sobel_edges.png"),
            compute_edge_strength(processed_image, 'sobel')
        )
        cv2.imwrite(
            os.path.join(output_dir, f"{method_name.lower()}_laplacian_edges.png"),
            compute_edge_strength(processed_image, 'laplacian')
        )
    
    create_pdf_report(comparison_results)

if __name__ == "__main__":
    main()