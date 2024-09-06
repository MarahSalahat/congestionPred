from flask import Flask, request, jsonify
import tensorflow as tf
import os
from netlify_lambda.flask_handler import handler as netlify_handler

app = Flask(__name__)

# Load the model when the app starts
model = tf.keras.models.load_model(os.path.join(os.path.dirname(__file__), 'congestion_model.h5'))

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract input data from the request
        data = request.json
        input_data = [data['temperature'], data['humidity'], data['wind_speed'], data['hour'], data['isWeekend']]
        
        # Convert input data to tensor
        input_tensor = tf.convert_to_tensor([input_data], dtype=tf.float32)
        
        # Make prediction using the loaded model
        prediction = model.predict(input_tensor)
        
        # Send back the prediction as JSON response
        return jsonify({'congestion_level': float(prediction[0][0])})
    
    except Exception as e:
        # Handle any potential errors
        return jsonify({'error': str(e)}), 400

# Netlify handler for serverless functions
def handler(event, context):
    return netlify_handler(app, event, context)
