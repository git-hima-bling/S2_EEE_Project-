from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf
import joblib
import os
import pandas as pd
from tensorflow.keras.models import load_model
from flask import Flask, render_template, redirect, url_for
import json

# Import the custom AttentionLayer class
from model import AttentionLayer

app = Flask(__name__)

# Load model and preprocessing objects
MODEL_PATH = 'model'

def load_model_artifacts():
    model = load_model(os.path.join(MODEL_PATH, 'final_model.keras'), 
                      custom_objects={'AttentionLayer': AttentionLayer})
    scaler = joblib.load(os.path.join(MODEL_PATH, 'scaler.pkl'))
    le = joblib.load(os.path.join(MODEL_PATH, 'label_encoder.pkl'))
    return model, scaler, le

@app.route('/login')
def login():
    return render_template('login.html')

# Route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Route for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Load model artifacts
        model, scaler, le = load_model_artifacts()
        
        # Get data from request
        data = request.get_json()
        
        # Extract feature values
        input_features = np.array(list(data.values()))
        
        # Preprocess the data
        input_scaled = scaler.transform(input_features.reshape(1, -1))
        input_reshaped = input_scaled.reshape(1, 1, input_scaled.shape[1])
        
        # Make prediction
        prediction_probs = model.predict(input_reshaped)[0]
        predicted_class_idx = np.argmax(prediction_probs)
        predicted_fault = le.inverse_transform([predicted_class_idx])[0]
        
        # Create response with probabilities for all fault types
        fault_types = le.classes_
        fault_probs = {fault: float(prediction_probs[i]) for i, fault in enumerate(fault_types)}
        
        # Create detailed response
        response = {
            'prediction': predicted_fault,
            'probabilities': fault_probs,
            'parameters': data
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Route to get available fault types
@app.route('/fault-types', methods=['GET'])
def get_fault_types():
    try:
        # Load label encoder
        le = joblib.load(os.path.join(MODEL_PATH, 'label_encoder.pkl'))
        fault_types = list(le.classes_)
        
        return jsonify({'fault_types': fault_types})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Route to train the model with new data
@app.route('/train', methods=['POST'])
def train_model():
    try:
        from model import main as train_main
        
        # Check if file was uploaded
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        
        # Save uploaded file temporarily
        temp_file_path = 'temp_dataset.xlsx'
        file.save(temp_file_path)
        
        # Train model with new data
        model, scaler, le, fault_types = train_main(file_path=temp_file_path, model_path=MODEL_PATH)
        
        # Clean up temporary file
        os.remove(temp_file_path)
        
        return jsonify({
            'message': 'Model trained successfully',
            'fault_types': list(fault_types)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Make sure the model directory exists
    os.makedirs(MODEL_PATH, exist_ok=True)
    
    # Create model file if it doesn't exist
    if not os.path.exists(os.path.join(MODEL_PATH, 'final_model.keras')):
        print("No model found. Training a new model...")
        from model import main as train_main
        train_main(model_path=MODEL_PATH)
    
    app.run(debug=True)