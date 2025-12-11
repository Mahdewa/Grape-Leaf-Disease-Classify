from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import io

app = Flask(__name__)
CORS(app)

try:
    model = tf.keras.models.load_model('model_terbaik_klasifikasi_anggur.h5')
    print("✅ Model berhasil dimuat!")
except:
    print("❌ Model tidak ditemukan.")
    model = None

# Definisi Kelas
CLASS_NAMES = [
    'Grape___Black_rot', 
    'Grape___Esca_(Black_Measles)', 
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
    'Grape___healthy'
]

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'Tidak ada file diupload'}), 400
    
    file = request.files['file']
    
    try:
        # 2. PREPROCESSING IMAGE
        # Baca gambar langsung dari memori (tanpa save ke disk dulu)
        image = Image.open(file.stream)
        
        # Resize ke 128x128 (Sesuai training di Kaggle)
        image = ImageOps.fit(image, (128, 128), Image.Resampling.LANCZOS)
        
        # Convert ke Array & Normalisasi (Sesuai training 1./255)
        img_array = np.asarray(image)
        img_array = img_array / 255.0
        
        # Tambah dimensi batch (Jadi (1, 128, 128, 3))
        img_array = np.expand_dims(img_array, axis=0)

        # 3. PREDIKSI
        if model is None:
            return jsonify({'error': 'Model belum siap'}), 500

        prediction = model.predict(img_array)
        class_index = np.argmax(prediction)
        confidence = float(np.max(prediction) * 100)
        
        result_class = CLASS_NAMES[class_index]

        # Kirim respons JSON ke React
        return jsonify({
            'class': result_class,
            'confidence': f"{confidence:.2f}",
            'status': 'success'
        })

    except Exception as e:
        print(e)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(port=5000, debug=True)