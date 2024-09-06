# functions/predict.py
from flask import Flask, request, jsonify
import tensorflow as tf
import os

app = Flask(__name__)

# Load your model
model = tf.keras.models.load_model(os.path.join(os.path.dirname(__file__), 'congestion_model.h5'))

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    input_data = [data['temperature'], data['humidity'], data['wind_speed'], data['hour'], data['isWeekend']]
    input_tensor = tf.convert_to_tensor([input_data], dtype=tf.float32)
    prediction = model.predict(input_tensor)
    return jsonify({'congestion_level': prediction[0][0]})

# The function to be used by Netlify
def handler(event, context):
    from flask import Flask, request, jsonify
    from netlify_lambda import handler
    return handler(app, event, context)
