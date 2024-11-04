from flask import Flask, render_template, request, jsonify, send_file
import cv2
import numpy as np
from PIL import Image, ExifTags
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def align_images_cv(src_img, dest_img, src_kp, dest_kp, matches, ransac_thresh=3.0):
    src_pts = np.float32([src_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dest_pts = np.float32([dest_kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    matrix, mask = cv2.findHomography(src_pts, dest_pts, cv2.RANSAC, ransac_thresh)
    aligned_img = cv2.warpPerspective(src_img, matrix, (dest_img.shape[1], dest_img.shape[0]))
    return aligned_img

def align_images_for_custom(image_paths):
    """Align images using OpenCV before custom HDR processing"""
    # Read images
    images = [cv2.imread(path) for path in image_paths]
    gray_images = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in images]
    
    # Feature detection
    akaze = cv2.AKAZE_create()
    keypoints_descriptors = [akaze.detectAndCompute(gray, None) for gray in gray_images]
    
    # Match features and align to the first image
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    aligned_images = [images[0]]  # First image is reference
    
    for i in range(1, len(images)):
        matches = bf.match(keypoints_descriptors[i][1], keypoints_descriptors[0][1])
        good_matches = [m for m in matches if m.distance < 50]
        
        aligned = align_images_cv(
            images[i], 
            images[0],
            keypoints_descriptors[i][0],
            keypoints_descriptors[0][0],
            good_matches
        )
        aligned_images.append(aligned)
    
    # Save aligned images and return their paths
    aligned_paths = []
    for i, img in enumerate(aligned_images):
        path = os.path.join(app.config['UPLOAD_FOLDER'], f'aligned_{i}.jpg')
        cv2.imwrite(path, img)
        aligned_paths.append(path)
    
    return aligned_paths

def get_exposure_time(image_path):
    try:
        img = Image.open(image_path)
        exif_data = img._getexif()
        
        if exif_data is not None:
            for tag, value in exif_data.items():
                tag_name = ExifTags.TAGS.get(tag, tag)
                if tag_name == "ExposureTime":
                    if isinstance(value, tuple):
                        return float(value[0]) / float(value[1])
                    else:
                        return float(value)
        return 1.0
    except:
        return 1.0

def load_image(filepath):
    """Load an image and convert to linear space with improved precision."""
    image = Image.open(filepath).convert("RGB")
    # Convert to float32 for better precision
    img_array = np.asarray(image, dtype=np.float32) / 255.0
    # Apply gamma correction to convert to linear space
    return np.power(img_array, 2.2)

def calculate_radiance(images, exposure_times):
    """Calculate HDR radiance map with improved weighting and dynamic range."""
    num_images = len(images)
    height, width, channels = images[0].shape
    hdr_radiance = np.zeros((height, width, channels), dtype=np.float32)
    
    def weight(z):
        """Enhanced weighting function with smoother falloff and better handling of extremes"""
        # Triangle weighting function with wider sweet spot
        w = np.ones_like(z)
        w[z <= 0.1] = z[z <= 0.1] * 10  # Linear ramp up
        w[z >= 0.9] = (1 - z[z >= 0.9]) * 10  # Linear ramp down
        return w
    
    # Normalize exposure times relative to the middle exposure
    mid_exposure = exposure_times[len(exposure_times)//2]
    norm_exposures = exposure_times / mid_exposure
    
    weight_sum = np.zeros((height, width, channels), dtype=np.float32)
    weighted_radiance_sum = np.zeros((height, width, channels), dtype=np.float32)
    
    for i in range(num_images):
        img = images[i]
        t = norm_exposures[i]
        
        # Calculate weights with improved consideration for exposure
        w = weight(img)
        
        # Calculate radiance with better handling of extreme values
        eps = 1e-8  # Small epsilon to avoid division by zero
        radiance = img / (t + eps)
        
        weighted_radiance_sum += w * radiance
        weight_sum += w
    
    # Avoid division by zero and normalize
    weight_sum = np.maximum(weight_sum, eps)
    hdr_radiance = weighted_radiance_sum / weight_sum
    
    # Apply bilateral filter for noise reduction while preserving edges
    for c in range(channels):
        hdr_radiance[:,:,c] = cv2.bilateralFilter(hdr_radiance[:,:,c], 9, 75, 75)
    
    return hdr_radiance

def tone_map(hdr_image, gamma=2.2, saturation_boost=1.2):
    """Enhanced tone mapping with improved contrast and color preservation"""
    # Convert to luminance domain
    luminance = 0.299 * hdr_image[:,:,0] + 0.587 * hdr_image[:,:,1] + 0.114 * hdr_image[:,:,2]
    luminance = np.maximum(luminance, 1e-8)
    
    # Apply local tone mapping
    log_luminance = np.log10(luminance)
    bilateral = cv2.bilateralFilter(log_luminance.astype(np.float32), 17, 0.3, 17)
    detail_layer = log_luminance - bilateral
    
    # Compress the range of the bilateral layer
    compressed = np.tanh(bilateral) * 0.5 + 0.5
    
    # Combine layers
    mapped_luminance = np.power(10, compressed + detail_layer * 0.5)
    
    # Normalize
    mapped_luminance = (mapped_luminance - np.min(mapped_luminance)) / (np.max(mapped_luminance) - np.min(mapped_luminance))
    
    # Color preservation and saturation enhancement
    result = np.zeros_like(hdr_image)
    for i in range(3):
        # Preserve color ratios while applying tone mapping
        ratio = np.where(luminance > 1e-8, hdr_image[:,:,i] / luminance, 0)
        result[:,:,i] = ratio * mapped_luminance
        
        # Enhance saturation
        mean_value = np.mean(result[:,:,i])
        result[:,:,i] = mean_value + (result[:,:,i] - mean_value) * saturation_boost
    
    # Final gamma correction and scaling
    result = np.clip(result, 0, 1)
    result = np.power(result, 1/gamma)
    
    return (result * 255).astype(np.uint8)

def process_hdr_custom(low_path, medium_path, high_path):
    # First align images
    aligned_paths = align_images_for_custom([low_path, medium_path, high_path])
    
    # Load and normalize aligned images
    images = [load_image(path) for path in aligned_paths]
    
    # Get exposure times from original images
    exposure_times = np.array([
        get_exposure_time(low_path),
        get_exposure_time(medium_path),
        get_exposure_time(high_path)
    ], dtype=np.float32)
    
    # Calculate HDR radiance map
    hdr_radiance = calculate_radiance(images, exposure_times)
    
    # Apply enhanced tone mapping
    ldr_image = tone_map(hdr_radiance, gamma=2.2, saturation_boost=1.2)
    
    # Save result
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'hdr_result_custom.jpg')
    Image.fromarray(ldr_image).save(output_path, quality=95)
    
    return aligned_paths, output_path

def process_hdr_opencv(low_path, medium_path, high_path):
    # [OpenCV HDR processing function remains unchanged]
    # Load images
    img_low = cv2.imread(low_path)
    img_medium = cv2.imread(medium_path)
    img_high = cv2.imread(high_path)
    
    # Convert to grayscale for alignment
    gray_low = cv2.cvtColor(img_low, cv2.COLOR_BGR2GRAY)
    gray_medium = cv2.cvtColor(img_medium, cv2.COLOR_BGR2GRAY)
    gray_high = cv2.cvtColor(img_high, cv2.COLOR_BGR2GRAY)
    
    # Feature detection and matching
    akaze = cv2.AKAZE_create()
    keypoints_low, descriptors_low = akaze.detectAndCompute(gray_low, None)
    keypoints_medium, descriptors_medium = akaze.detectAndCompute(gray_medium, None)
    keypoints_high, descriptors_high = akaze.detectAndCompute(gray_high, None)
    
    # Match features
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches_medium_low = bf.match(descriptors_medium, descriptors_low)
    matches_high_low = bf.match(descriptors_high, descriptors_low)
    
    # Filter matches
    good_matches_medium_low = [m for m in matches_medium_low if m.distance < 50]
    good_matches_high_low = [m for m in matches_high_low if m.distance < 50]
    
    # Align images
    aligned_medium = align_images_cv(img_medium, img_low, keypoints_medium, keypoints_low, good_matches_medium_low)
    aligned_high = align_images_cv(img_high, img_low, keypoints_high, keypoints_low, good_matches_high_low)
    
    # Save aligned images
    aligned_paths = []
    for i, img in enumerate([img_low, aligned_medium, aligned_high]):
        path = os.path.join(app.config['UPLOAD_FOLDER'], f'aligned_{i}.jpg')
        cv2.imwrite(path, img)
        aligned_paths.append(path)
    
    # Get exposure times
    exposure_times = np.array([
        get_exposure_time(low_path),
        get_exposure_time(medium_path),
        get_exposure_time(high_path)
    ], dtype=np.float32)
    
    # Create HDR image
    aligned_images = [img_low, aligned_medium, aligned_high]
    merge_debevec = cv2.createMergeDebevec()
    hdr_image = merge_debevec.process(aligned_images, times=exposure_times)
    
    # Tone mapping
    tonemap_reinhard = cv2.createTonemapReinhard(gamma=1.2)
    ldr_image = tonemap_reinhard.process(hdr_image)
    ldr_image_8bit = np.clip(ldr_image * 255, 0, 255).astype('uint8')
    
    # Save result
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'hdr_result_opencv.jpg')
    cv2.imwrite(output_path, ldr_image_8bit)
    
    return aligned_paths, output_path

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_images():
    if 'low' not in request.files or 'medium' not in request.files or 'high' not in request.files:
        return jsonify({'error': 'Missing files'}), 400
    
    method = request.form.get('method', 'opencv')
    
    low_file = request.files['low']
    medium_file = request.files['medium']
    high_file = request.files['high']
    
    if not all(f and allowed_file(f.filename) for f in [low_file, medium_file, high_file]):
        return jsonify({'error': 'Invalid file type'}), 400
    
    # Save uploaded files
    low_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(low_file.filename))
    medium_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(medium_file.filename))
    high_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(high_file.filename))
    
    low_file.save(low_path)
    medium_file.save(medium_path)
    high_file.save(high_path)
    
    try:
        if method == 'opencv':
            aligned_paths, result_path = process_hdr_opencv(low_path, medium_path, high_path)
        else:
            aligned_paths, result_path = process_hdr_custom(low_path, medium_path, high_path)
        
        return jsonify({
            'result': result_path.replace('static/', ''),
            'aligned': [p.replace('static/', '') for p in aligned_paths]
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)