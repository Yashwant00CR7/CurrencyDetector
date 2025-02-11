from flask import Flask, request, jsonify
import base64
from PIL import Image
import io
import numpy as np
from flask_cors import CORS
from tensorflow.keras.models import load_model, model_from_json
from tensorflow.keras import layers
from tensorflow.keras.utils import CustomObjectScope
import tensorflow as tf
import os

app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "*"}})  # Allow CORS only on '/predict'

# Define class names
classes_name = ['10 Rupees', '100 Rupees', '20 Rupees', '200 Rupees', '50 Rupees', '500 Rupees']

class CentralFocusSpatialAttention(layers.Layer):
    def __init__(self, **kwargs):
        super(CentralFocusSpatialAttention, self).__init__(**kwargs)
        self.conv_attention = None
        self.gamma = None

    def build(self, input_shape):
        self.conv_attention = layers.Conv2D(1, (7, 7), padding='same', activation='sigmoid')
        self.gamma = self.add_weight(name='gamma', shape=(), initializer='zeros', trainable=True)
        super(CentralFocusSpatialAttention, self).build(input_shape)

    def call(self, inputs):
        avg_pool = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        max_pool = tf.reduce_max(inputs, axis=-1, keepdims=True)
        concat = tf.concat([avg_pool, max_pool], axis=-1)
        attention = self.conv_attention(concat)

        height, width = inputs.shape[1], inputs.shape[2]
        center_x, center_y = height // 2, width // 2
        sigma = tf.cast(height / 4, tf.float32)
        x = tf.range(0, height, dtype=tf.float32)
        y = tf.range(0, width, dtype=tf.float32)
        x_mask = tf.exp(-(x - center_x) ** 2 / (2 * sigma ** 2))
        y_mask = tf.exp(-(y - center_y) ** 2 / (2 * sigma ** 2))
        gaussian_mask = tf.tensordot(x_mask, y_mask, axes=0)

        gaussian_mask = tf.expand_dims(gaussian_mask, axis=-1)
        gaussian_mask = tf.expand_dims(gaussian_mask, axis=0)
        gaussian_mask = tf.cast(gaussian_mask, dtype=inputs.dtype)

        attention_weighted = attention * gaussian_mask
        return inputs * (1 + self.gamma * attention_weighted)

# ✅ Load Model with Safe Handling (Fixing Keras 3.x Issues)
model_path = "Currency_Detection_model_with_DenseNet121_and_CentralFocusSpatialAttention.h5"
json_model_path = "new_model.json"
weights_path = "new_model_weights.h5"

def rebuild_model():
    """Converts old model format to a Keras 3.x compatible model"""
    print("🔄 Rebuilding model to remove batch_shape issue...")
    old_model = load_model(model_path, compile=False)

    # Save model as JSON
    model_json = old_model.to_json()
    with open(json_model_path, "w") as json_file:
        json_file.write(model_json)

    # Save model weights separately
    old_model.save_weights(weights_path)

    # Load the model again without batch_shape
    with open(json_model_path, "r") as json_file:
        new_model_json = json_file.read()

    new_model = model_from_json(new_model_json, custom_objects={'CentralFocusSpatialAttention': CentralFocusSpatialAttention})
    new_model.load_weights(weights_path)
    new_model.save(model_path)  # Overwrite old model with new format
    print("✅ Model successfully rebuilt!")

# Check if the model exists before loading
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")

try:
    with CustomObjectScope({'CentralFocusSpatialAttention': CentralFocusSpatialAttention}):
        model = load_model(model_path, compile=False)  # Prevents compilation issues in Keras 3.x
    print("✅ Model loaded successfully!")
except ValueError as e:
    if "Unrecognized keyword arguments: ['batch_shape']" in str(e):
        rebuild_model()
        with CustomObjectScope({'CentralFocusSpatialAttention': CentralFocusSpatialAttention}):
            model = load_model(model_path, compile=False)
    else:
        raise RuntimeError(f"Error loading model: {str(e)}")

INPUT_IMAGE_SIZE = (224, 224)  # Example: Resize images to 224x224

@app.route('/predict', methods=['POST'])
def predict():
    """Handles prediction requests from clients."""
    data = request.get_json()
    images = data.get('images', [])

    if not images:
        return jsonify({'error': 'No images provided'}), 400

    try:
        predictions = []
        for img_str in images:
            try:
                img_data = base64.b64decode(img_str.split(',')[1])  # Remove 'data:image/jpeg;base64,'
                img = Image.open(io.BytesIO(img_data)).convert('RGB')
            except Exception as e:
                return jsonify({'error': f"Error decoding image: {str(e)}"}), 400

            # Preprocess image
            img = img.resize(INPUT_IMAGE_SIZE)
            img_array = np.array(img) / 255.0  # Normalize pixel values to [0, 1]
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

            # Predict using the model
            pred = model.predict(img_array)[0]  # Assuming model.predict outputs a probability array
            predictions.append(pred)

        # Calculate the average prediction
        avg_prediction = np.mean(predictions, axis=0)
        highest_class = np.argmax(avg_prediction)  # Get index of highest probability
        highest_score = avg_prediction[highest_class]  # Get highest probability score

        # Convert score to percentage
        highest_score_percentage = highest_score * 100

        # Map index to class name
        highest_class_name = classes_name[highest_class]

        return jsonify({
            'class': highest_class_name,
            'score_percentage': f"{highest_score_percentage:.2f}%",  # Format as a string with 2 decimal places
            'average_predictions': avg_prediction.tolist()  # Convert NumPy array to list for JSON serialization
        })

    except Exception as e:
        return jsonify({'error': f"Internal server error: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)  # Allows running on cloud
