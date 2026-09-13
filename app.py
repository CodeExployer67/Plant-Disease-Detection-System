import os
import json
import numpy as np
from flask import Flask, request, render_template, jsonify
from PIL import Image
import tensorflow as tf

app = Flask(__name__)

# Constants
MODEL_PATH = 'models/plant_disease_model.keras'
CLASS_INDICES_PATH = 'models/class_indices.json'
IMG_HEIGHT, IMG_WIDTH = 224, 224

# Try to load the model (it might not exist yet if the user hasn't trained it)
model = None
if os.path.exists(MODEL_PATH):
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print(f"Model loaded successfully from {MODEL_PATH}")
    except Exception as e:
        print(f"Error loading model: {e}")

# Load class indices mapping
class_names = {}
if os.path.exists(CLASS_INDICES_PATH):
    try:
        with open(CLASS_INDICES_PATH, 'r') as f:
            indices = json.load(f)
            # Reverse dictionary to map index -> class_name
            class_names = {v: k for k, v in indices.items()}
    except Exception as e:
        print(f"Error loading class indices: {e}")
else:
    # Fallback default names just so the app doesn't crash
    class_names = {0: "Tomato___Healthy", 1: "Tomato___Late_blight"}

def prepare_image(img_file):
    """
    Load image using Pillow, resize, convert to Numpy array, and normalize.
    """
    # Open image
    img = Image.open(img_file)
    # Ensure image is RGB (in case of RGBA or Grayscale)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Resize image to match model expected input
    img = img.resize((IMG_WIDTH, IMG_HEIGHT))
    
    # Convert to Numpy array
    img_array = np.array(img)
    
    # Normalize pixel values to [0, 1] as done in training
    img_array = img_array / 255.0
    
    # Expand dimensions to match batch format: (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

@app.route('/', methods=['GET'])
def index():
    # Render the main frontend page
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
        
    try:
        # 1. Preprocess the uploaded image
        img_array = prepare_image(file)
        
        # 2. Check if model is loaded
        if model is None:
            # Fallback mock prediction if model isn't trained yet
            import random
            prediction_idx = random.choice(list(class_names.keys()))
            confidence = random.uniform(0.6, 0.99)
            
            result_class = class_names[prediction_idx]
            
            return jsonify({
                'disease': result_class,
                'confidence': f"{confidence*100:.2f}%",
                'mock_mode': True,
                'message': 'This is a mock prediction because no trained model was found. Please train the model first.'
            })
            
        # 3. Model Inference
        predictions = model.predict(img_array)
        
        # 4. Process Results (Get max probability index)
        predicted_class_index = np.argmax(predictions, axis=1)[0]
        confidence = float(np.max(predictions))
        
        predicted_disease_name = class_names.get(int(predicted_class_index), "Unknown Disease")
        
        # If user wants a 50% confidence threshold logic:
        if confidence < 0.50:
             return jsonify({
                'disease': 'Not Recognized or Low Confidence',
                'confidence': f"{confidence*100:.2f}%",
                'message': 'The model is not confident enough to classify this correctly.'
             })
             
        # Clean up the output string a bit
        predicted_disease_name = predicted_disease_name.replace('___', ' - ').replace('_', ' ')
        
        # 5. Return success JSON
        return jsonify({
            'disease': predicted_disease_name,
            'confidence': f"{confidence*100:.2f}%"
        })

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    # You can access this via http://127.0.0.1:5000/
    app.run(debug=True)
