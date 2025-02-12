import streamlit as st
import numpy as np
import cv2
import base64
from io import BytesIO
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import layers
from tensorflow.keras.utils import CustomObjectScope
from ultralytics import YOLO
from gtts import gTTS
import os
import threading

# **Step 1: Define Class Names for CNN**
cnn_classes = ['10 Rupees', '100 Rupees', '20 Rupees', '200 Rupees', '50 Rupees', '500 Rupees']

# **Step 2: Load YOLO Model for Currency Detection**
yolo_model = YOLO('runs/detect/train4/weights/best.pt')

# **Step 3: Define Custom CNN Layer (For Loading Model)**
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

# **Step 4: Load CNN Model for Currency Classification**
with CustomObjectScope({'CentralFocusSpatialAttention': CentralFocusSpatialAttention}):
    cnn_model = load_model('New_Currency_Detection_Model.h5', compile=False)

# **Step 5: Helper Functions**
def encode_image(image):
    """Encodes an image to Base64 format."""
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return f"data:image/jpeg;base64,{base64.b64encode(buffered.getvalue()).decode()}"

def speak(text):
    """Converts text to speech and plays the audio."""
    try:
        tts = gTTS(text=text, lang="en")
        tts.save("speech.mp3")
        os.system("start speech.mp3" if os.name == "nt" else "mpg321 speech.mp3")
    except Exception as e:
        st.error(f"Speech synthesis error: {str(e)}")

# **Step 6: Streamlit UI**
st.title("Currency Detection App")
st.write("Upload an image to detect currency using YOLO and classify it using CNN.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # **Step 7: Load and Display Image**
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # **Step 8: Convert Image to OpenCV Format**
    img_array = np.array(image)
    
    # **Step 9: Run YOLO Model for Currency Detection**
    yolo_results = yolo_model.predict(source=img_array, save=False)
    boxes = yolo_results[0].boxes
    class_names = yolo_results[0].names

    detected_currency = False
    for box in boxes:
        cls = int(box.cls[0])
        if class_names[cls] == "Currency":
            detected_currency = True
            break

    if detected_currency:
        st.success("Currency detected with YOLO model!")

        # **Step 10: Process Image for CNN Model**
        img_resized = image.resize((224, 224))  # Resize for CNN input
        img_array = np.array(img_resized) / 255.0  # Normalize pixel values
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # **Step 11: Run CNN Model for Currency Classification**
        cnn_prediction = cnn_model.predict(img_array)[0]
        predicted_class_index = np.argmax(cnn_prediction)
        predicted_class_name = cnn_classes[predicted_class_index]
        confidence = cnn_prediction[predicted_class_index] * 100

        # **Step 12: Display Results**
        st.write(f"Predicted Currency: **{predicted_class_name}**")
        st.write(f"Confidence: **{confidence:.2f}%**")

        # **Step 13: Speak the Prediction**
        threading.Thread(target=speak, args=(f"The detected currency is {predicted_class_name} with {confidence:.2f} percent confidence.",)).start()
    else:
        st.error("No currency detected by YOLO model.")
        threading.Thread(target=speak, args=("No currency detected.",)).start()
