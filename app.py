import firebase_admin
from firebase_admin import credentials, db
from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
import os
from google.cloud import storage
from model.sam_model import SAMModel
import matplotlib.pyplot as plt
from PIL import Image
import rasterio
import base64
import numpy as np
import io
from pathlib import Path
from werkzeug.exceptions import RequestEntityTooLarge

# Path kredensial GCP
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.environ.get("GCP-KEY","gcp-key.json")

# Inisialisasi client GCS
storage_client = storage.Client()
GCS_BUCKET_NAME = "solafuneteam02"  # Ganti dengan nama bucket Anda
bucket = storage_client.bucket(GCS_BUCKET_NAME)

def save_image_as_png(image, output_png_path):
    # Convert image to PNG
    pil_image = Image.fromarray(image)
    pil_image.save(output_png_path, format="PNG")
    return output_png_path

cred = credentials.Certificate(os.path.join(os.getcwd(), 'firebase-adminsdk.json'))
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://solafunef02-default-rtdb.firebaseio.com/'  
})

def register_user(first_name, last_name, email, password):
    ref = db.reference('users') 
    
    user_ref = ref.order_by_child('email').equal_to(email).get()
    if user_ref:
        return {'status': 'error', 'message': 'Email already registered'}
    
    user_data = {
        'first_name': first_name,
        'last_name': last_name,
        'email': email,
        'password': password 
    }
    
    ref.push(user_data) 
    return {'status': 'success', 'message': 'Registration successful'}

def authenticate_user(email, password):
    ref = db.reference('users')
    user_ref = ref.order_by_child('email').equal_to(email).get()
    
    if user_ref:
        user_data = next(iter(user_ref.values()))
        if user_data['password'] == password:
            return {'status': 'success', 'message': 'Login successful'}
    
    return {'status': 'error', 'message': 'Invalid email or password'}

# Inisialisasi model SAM
checkpoint_path = "models/sam_vit_b_01ec64.pth"
sam_model = SAMModel(checkpoint_path)

app = Flask(__name__)
# Batas ukuran file 100MB (ubah sesuai kebutuhan)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024
app.secret_key = 'your_secret_key'

# Folder penyimpanan gambar
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / 'static/images'
RESULTS_FOLDER = BASE_DIR / 'static/results'
JPG_FOLDER = BASE_DIR / 'static/png'

# Pastikan folder ada
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)
JPG_FOLDER.mkdir(parents=True, exist_ok=True)

@app.errorhandler(RequestEntityTooLarge)
def handle_large_request(e):
    return "File terlalu besar! Maksimum 100MB.", 413

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/index.html')
def index():
    return render_template('index.html')

@app.route('/segmentation')
def segmentation():
    return render_template('segmentation.html')

@app.route('/about-us')
def about_us():
    return render_template('about-us.html')


@app.route('/sign-in', methods=['GET', 'POST'])
def sign_in():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        
        auth_result = authenticate_user(email, password)
        
        if auth_result['status'] == 'success':
            flash(auth_result['message'], 'success')
            return redirect(url_for('segmentation'))
        else:
            flash(auth_result['message'], 'error')
            return redirect(url_for('sign_in'))
    return render_template('sign-in.html')

@app.route('/sign-up', methods=['GET', 'POST'])
def sign_up():
    if request.method == 'POST':
        first_name = request.form['first-name']
        last_name = request.form['last-name']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm-password']
        
        if password != confirm_password:
            flash('Passwords do not match', 'error')
            return redirect(url_for('sign_up'))
        
        reg_result = register_user(first_name, last_name, email, password)
        
        if reg_result['status'] == 'success':
            flash(reg_result['message'], 'success')
            return redirect(url_for('sign_in'))
        else:
            flash(reg_result['message'], 'error')
            return redirect(url_for('sign_up'))
    return render_template('sign-up.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'image' not in request.files:
        return "No file part", 400
    
    file = request.files['image']
    if file.filename == '':
        return "No selected file", 400
    
    if file and (file.filename.endswith('.tif') or file.filename.endswith('.tiff')):
        # Upload file ke GCS
        blob = bucket.blob(file.filename)
        blob.upload_from_file(file)
        
        filename = os.path.join(UPLOAD_FOLDER, file.filename)
        blob.download_to_filename(filename)
        
        # Segmentasi gambar menggunakan model SAM
        masks = sam_model.segment_image(filename)
        
        # Mendapatkan latitude dan longitude dari titik tengah gambar
        image = sam_model.read_multiband_tiff(filename)
        row, col = image.shape[0] // 2, image.shape[1] // 2
        lat, lon = sam_model.get_lat_lon(filename, row, col)

        # Menyimpan hasil segmentasi
        base_name = os.path.splitext(file.filename)[0]
        result_image_path = os.path.join(RESULTS_FOLDER, f"segmented_{base_name}.png")
        plt.imshow(masks[0], alpha=0.5, cmap='viridis')
        plt.axis('off')
        plt.savefig(result_image_path, bbox_inches='tight', pad_inches=0)
        
        # Simpan gambar asli dan hasil segmentasi dalam format PNG
        original_png_path = os.path.join(JPG_FOLDER, f"{base_name}.png")

        print(original_png_path)
        
        return render_template(
            'segmentation_results.html',
            original_image=f'static/jpg/{base_name}.png',
            result_image=f'static/results/segmented_{base_name}.png',
            latitude=lat,
            longitude=lon
        )
      
    return "Invalid file format. Only .tif or .tiff files are allowed.", 400

# Menyajikan gambar statis
@app.route('/images/<filename>')
def send_image(filename):
    return send_from_directory('static/images', filename)

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 5001)))