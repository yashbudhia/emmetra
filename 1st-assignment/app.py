from flask import Flask, render_template, request, send_file, jsonify, url_for
import os
from werkzeug.utils import secure_filename
import numpy as np
from datetime import datetime
from functools import wraps
import time
from scipy.ndimage import convolve
import numpy.typing as npt
from imageio import imwrite
from typing import Dict, Any, Tuple
from scipy.ndimage import gaussian_filter

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'raw'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH
app.config['SECRET_KEY'] = 'your-secret-key-here'

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Store processing parameters and results
processing_cache: Dict[str, Any] = {}

# ISP Pipeline Functions
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
    """Apply white balance correction based on gray world assumption."""
    avg_r = np.mean(rgb_image[:, :, 0])
    avg_g = np.mean(rgb_image[:, :, 1])
    avg_b = np.mean(rgb_image[:, :, 2])

    scale_r = avg_g / avg_r
    scale_b = avg_g / avg_b

    rgb_image[:, :, 0] = np.clip(rgb_image[:, :, 0] * scale_r, 0, 4095)
    rgb_image[:, :, 2] = np.clip(rgb_image[:, :, 2] * scale_b, 0, 4095)

    return rgb_image

def apply_color_correction(rgb_image: npt.NDArray[np.uint16]) -> npt.NDArray[np.uint16]:
    """Apply a color correction matrix for more vivid colors."""
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

def adaptive_power_tone_map(rgb_image: npt.NDArray[np.uint16], compression_factor: float = 0.7) -> npt.NDArray[np.float32]:
    """Apply adaptive power-based tone mapping."""
    max_value = 4095  # 12-bit max value
    scaled_image = rgb_image / max_value
    tone_mapped_image = np.power(scaled_image, compression_factor)
    return np.clip(tone_mapped_image, 0, 1)

def gamma_correct(rgb_image: npt.NDArray[np.float32], gamma: float = 2.2) -> npt.NDArray[np.uint8]:
    """Apply gamma correction and convert to 8-bit."""
    gamma_corrected = np.clip(255 * (rgb_image ** (1 / gamma)), 0, 255)
    return gamma_corrected.astype(np.uint8)

def denoise_image(rgb_image: npt.NDArray[np.uint16], sigma: float = 1.0) -> npt.NDArray[np.uint16]:
    """Apply Gaussian denoising to the RGB image."""
    denoised_image = np.zeros_like(rgb_image)
    for c in range(3):  # For each channel (R, G, B)
        denoised_image[:, :, c] = gaussian_filter(rgb_image[:, :, c].astype(np.float32), sigma=sigma)
    return np.clip(denoised_image, 0, 4095).astype(np.uint16)

def sharpen_image(rgb_image: npt.NDArray[np.uint16]) -> npt.NDArray[np.uint16]:
    """Apply sharpening to the RGB image using an unsharp mask."""
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    
    sharpened_image = np.zeros_like(rgb_image)
    for c in range(3):  # For each channel (R, G, B)
        sharpened_image[:, :, c] = convolve(rgb_image[:, :, c].astype(np.float32), kernel, mode='reflect')
    return np.clip(sharpened_image, 0, 4095).astype(np.uint16)

def apply_gaussian_blur(rgb_image: npt.NDArray[np.uint16], sigma: float = 1.0) -> npt.NDArray[np.uint16]:
    """Apply Gaussian blur to the RGB image."""
    blurred_image = np.zeros_like(rgb_image)
    for c in range(3):  # For each channel (R, G, B)
        blurred_image[:, :, c] = gaussian_filter(rgb_image[:, :, c].astype(np.float32), sigma=sigma)
    return np.clip(blurred_image, 0, 4095).astype(np.uint16)

# Utility Functions
def get_timestamp() -> str:
    """Generate a timestamp string for unique filenames."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def allowed_file(filename: str) -> bool:
    """Check if the file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Flask Routes
@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file uploads."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        timestamp = get_timestamp()
        filename = secure_filename(f"{timestamp}_{file.filename}")
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Store initial processing parameters
        processing_cache[filename] = {
            'filepath': filepath,
            'params': {
                'gamma': 2.2,
                'compression_factor': 2.0,
                'width': 1920,
                'height': 1280
            },
            'output_path': os.path.join(app.config['UPLOAD_FOLDER'], f"processed_{filename}.png")
        }
        
        return jsonify({
            'success': True,
            'filename': filename,
            'message': 'File uploaded successfully'
        })
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/process', methods=['POST'])
def process_image():
    """Process the uploaded image with specified parameters."""
    data = request.json
    filename = data.get('filename')
    params = data.get('params', {})
    
    if not filename or filename not in processing_cache:
        return jsonify({'error': 'Invalid filename'}), 400
    
    # Update parameters
    processing_cache[filename]['params'].update(params)
    cache_entry = processing_cache[filename]
    
    try:
        # Load and process the image
        bayer_image = load_bayer_image(
            cache_entry['filepath'],
            width=cache_entry['params']['width'],
            height=cache_entry['params']['height']
        )
        
        rgb_image = demosaic_advanced(bayer_image)
        rgb_image = auto_tone(rgb_image)
        rgb_image = apply_white_balance(rgb_image)
        rgb_image = apply_color_correction(rgb_image)

        # New processing steps
        if params.get('denoise', False):
            sigma = params.get('denoise_sigma', 1.0)
            rgb_image = denoise_image(rgb_image, sigma)
        
        if params.get('sharpen', False):
            rgb_image = sharpen_image(rgb_image)
        
        if params.get('gaussian_blur', False):
            sigma = params.get('blur_sigma', 1.0)
            rgb_image = apply_gaussian_blur(rgb_image, sigma)

        tone_mapped_image = adaptive_power_tone_map(
            rgb_image,
            compression_factor=cache_entry['params']['compression_factor']
        )
        
        final_image = gamma_correct(
            tone_mapped_image,
            gamma=cache_entry['params']['gamma']
        )
        
        # Save the processed image
        imwrite(cache_entry['output_path'], final_image)
        
        return jsonify({
            'success': True,
            'message': 'Image processed successfully',
            'output_path': cache_entry['output_path']
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/download/<filename>')
def download_file(filename):
    """Handle image downloads."""
    if filename not in processing_cache:
        return jsonify({'error': 'File not found'}), 404
    
    return send_file(
        processing_cache[filename]['output_path'],
        as_attachment=True,
        download_name=f"processed_{filename}.png"
    )

# Error Handlers
@app.errorhandler(413)
def too_large(e):
    """Handle file too large error."""
    return jsonify({'error': 'File is too large'}), 413

@app.errorhandler(500)
def internal_error(e):
    """Handle internal server error."""
    return jsonify({'error': 'Internal server error'}), 500

# Main entry point
if __name__ == '__main__':
    app.run(debug=True)