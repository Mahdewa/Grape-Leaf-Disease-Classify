from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from PIL import Image, ImageOps
import tflite_runtime.interpreter as tflite
import io
import os

app = Flask(__name__)

CORS(app)

# --- KONFIGURASI MODEL ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model_anggur.tflite")
CLASS_NAMES = [
    'Grape___Black_rot', 
    'Grape___Esca_(Black_Measles)', 
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
    'Grape___healthy'
]

# Load Model di luar request agar tidak berat (Global Load)
try:
    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print("✅ Model TFLite Loaded Successfully")
except Exception as e:
    print(f"❌ Error loading model: {e}")

@app.route('/')
def home():
    return "BotaniScan API is Running!"

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    try:
        # 1. Preprocessing Gambar
        image = Image.open(file.stream)
        # Pastikan mode RGB (kadang PNG punya 4 channel/RGBA)
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        # Resize sesuai training (128x128)
        image = ImageOps.fit(image, (128, 128), Image.Resampling.LANCZOS)
        
        # Convert ke Array & Normalisasi
        img_array = np.asarray(image, dtype=np.float32)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0) # Jadi (1, 128, 128, 3)

        # 2. Prediksi dengan TFLite
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        
        # 3. Ambil Hasil
        class_index = np.argmax(output_data)
        confidence = float(np.max(output_data) * 100)
        result_class = CLASS_NAMES[class_index]

        return jsonify({
            'class': result_class,
            'confidence': f"{confidence:.2f}",
            'message': 'Success'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Untuk Vercel, kita tidak perlu app.run() di dalam if __name__ == main
# Tapi dibiarkan juga tidak apa-apa untuk tes lokal
if __name__ == '__main__':
    app.run(debug=True)