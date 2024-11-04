from flask import Flask, render_template, request, send_file, jsonify
import os
from processing import process_bayer_image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'inputs'
app.config['PROCESSED_FOLDER'] = 'static'
app.config['PROCESSED_IMAGE'] = os.path.join(app.config['PROCESSED_FOLDER'], 'processed_image.jpg')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    # Save uploaded file for processing
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'temp_image.raw')
    file.save(file_path)
    return jsonify({'success': True}), 200

@app.route('/process-image', methods=['POST'])
def process_image():
    # Parameters for image processing
    gamma = float(request.form.get('gamma', 1.0))
    white_balance = request.form.get('white_balance') == 'true'
    denoise = request.form.get('denoise') == 'true'
    sharpen = request.form.get('sharpen') == 'true'
    demosaic = request.form.get('demosaic') == 'true'
    blur = int(request.form.get('blur', 5))

    # Process the image with current settings
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'temp_image.raw')
    processed_image_path = process_bayer_image(
        file_path, gamma, white_balance, denoise, sharpen, demosaic, blur
    )

    # Serve the processed image path
    return send_file(processed_image_path, mimetype='image/jpeg')

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)
    app.run(debug=True)
