import streamlit as st
import base64
from PIL import Image
import numpy as np
from ultralytics import YOLO
from tensorflow.keras.models import load_model
from tensorflow.keras import layers
from tensorflow.keras.utils import CustomObjectScope
import tensorflow as tf

# Define class names for CNN
cnn_classes = ['10 Rupees', '100 Rupees', '20 Rupees', '200 Rupees', '50 Rupees', '500 Rupees']

# Load YOLO Model
yolo_model = YOLO('runs/detect/train4/weights/best.pt')

# Define custom layer for CNN
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

# Load CNN model
with CustomObjectScope({'CentralFocusSpatialAttention': CentralFocusSpatialAttention}):
    cnn_model = load_model('Currency_Detection_model_with_DenseNet121_and_CentralFocusSpatialAttention.h5')

INPUT_IMAGE_SIZE = (224, 224)

# Streamlit UI
st.title("Currency Detection App")
st.write("Upload an image to detect currency.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Load and preprocess image
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Uploaded Image", use_column_width=True)

        # Convert image to numpy array
        img_array = np.array(img)

        # Run YOLO detection
        yolo_results = yolo_model.predict(source=img_array, save=False)
        boxes = yolo_results[0].boxes
        class_names = yolo_results[0].names

        detected_currency = False

        # Check for "Currency" in YOLO results
        for box in boxes:
            cls = int(box.cls[0])
            if class_names[cls] == "Currency":
                detected_currency = True
                break

        if detected_currency:
            st.success("Currency detected with YOLO model!")

            # Resize and normalize image for CNN
            img_resized = img.resize(INPUT_IMAGE_SIZE)
            img_array = np.array(img_resized) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Predict with CNN
            cnn_prediction = cnn_model.predict(img_array)[0]
            predicted_class_index = np.argmax(cnn_prediction)
            predicted_class_name = cnn_classes[predicted_class_index]
            confidence = cnn_prediction[predicted_class_index] * 100

            st.write(f"Predicted Currency: **{predicted_class_name}**")
            st.write(f"Confidence: **{confidence:.2f}%**")

        else:
            st.error("No currency detected by YOLO model.")

    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
