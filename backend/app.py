import os
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.image import adjust_brightness
import tensorflow as tf
from PIL import Image

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Update this path to where your model file actually is
MODEL_PATH = os.path.join(os.path.dirname(__file__), '../model/seed_classifier_resnet50.h5')
model = load_model(MODEL_PATH)

IMG_SIZE = (224, 224)  # must match training image size

def preprocess_image(image_path):
    # Load image with PIL, resize
    img = Image.open(image_path).convert('RGB')
    img = img.resize(IMG_SIZE)
    img_array = img_to_array(img) / 255.0  # normalize to [0,1]

    # Convert to tensor for brightness adjustment
    img_tensor = tf.convert_to_tensor(img_array)
    img_tensor = adjust_brightness(img_tensor, delta=0.2)  # increase brightness by 0.2

    # Clip values to [0,1] after brightness adjust
    img_tensor = tf.clip_by_value(img_tensor, 0.0, 1.0)

    # Add batch dimension
    img_tensor = tf.expand_dims(img_tensor, axis=0)
    return img_tensor

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Save file temporarily
    temp_path = 'temp.jpg'
    file.save(temp_path)

    try:
        preprocessed_img = preprocess_image(temp_path)
        prediction = model.predict(preprocessed_img)[0][0]

        # For binary classification with sigmoid output:
        good_confidence = float(prediction)          # model output as probability for 'Good Seed'
        bad_confidence = 1.0 - good_confidence

        label = 'Good Seed' if good_confidence >= bad_confidence else 'Bad Seed'
        confidence = good_confidence if good_confidence >= bad_confidence else bad_confidence

        os.remove(temp_path)  # cleanup temp file

        return jsonify({
            'label': label,
            'confidence': round(confidence * 100, 2),      # Highest confidence %
            'good': good_confidence,                       # Good seed probability (0-1)
            'bad': bad_confidence,                         # Bad seed probability (0-1)
            'confidence_good': round(good_confidence * 100, 2),  # Good seed confidence %
            'confidence_bad': round(bad_confidence * 100, 2),    # Bad seed confidence %
            'score': round(float(prediction), 4)
        })

    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
