import subprocess
import time
import requests
import streamlit as st
import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image
import threading
import os
import sys
from gtts import gTTS  # Google Text-to-Speech

# Define class names for CNN (Ensure they match the order in your trained model)
cnn_classes = ['10 Rupees', '100 Rupees', '20 Rupees', '200 Rupees', '50 Rupees', '500 Rupees']

# Function to check if Flask API is already running
def is_api_running(url):
    try:
        response = requests.get(url, timeout=3)  # Quick check
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        return False

# Determine the Python executable (for virtual environment support)
PYTHON_EXECUTABLE = sys.executable  # Uses the same Python as the Streamlit script

# Start Flask APIs in separate terminals **only if they are not running**
def start_flask_api(script_name, port):
    """Starts Flask API in a separate terminal if it's not already running."""
    if not is_api_running(f"http://127.0.0.1:{port}"):  
        if os.name == "nt":  # Windows
            subprocess.Popen(f'start cmd /k "{PYTHON_EXECUTABLE} {script_name}"', shell=True)
        else:  # Linux/Mac
            subprocess.Popen(f'{PYTHON_EXECUTABLE} {script_name} &', shell=True)
        time.sleep(3)  # Give some time for the server to start
    else:
        st.write(f"Flask API on port {port} is already running.")

# Start YOLO and CNN APIs only if needed
start_flask_api("YOLO_Detect_api.py", 5000)
start_flask_api("CNN_Predict_api.py", 5001)

st.success("Flask APIs are running!")

# Function to convert text to speech using gTTS
def speak(text):
    try:
        tts = gTTS(text=text, lang="en")
        tts.save("speech.mp3")

        # Play the MP3 file (Compatible with Windows, Linux, Mac)
        if os.name == "nt":
            os.system("start speech.mp3")  # Windows
        else:
            os.system("mpg321 speech.mp3")  # Linux/Mac
    except Exception as e:
        st.error(f"Speech synthesis error: {str(e)}")

# Encode image to Base64
def encode_image(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return f"data:image/jpeg;base64,{base64.b64encode(buffered.getvalue()).decode()}"

# Streamlit UI
st.title("Currency Detection App")
st.write("Upload an image to detect currency using YOLO and classify it using CNN.")

# Upload Image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert uploaded image to OpenCV format
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    base64_image = encode_image(image)

    # **YOLO Model API Request (Currency Detection)**
    try:
        yolo_response = requests.post("http://127.0.0.1:5000/detect_currency", json={"images": [base64_image]})
        yolo_result = yolo_response.json()

        if yolo_result["currency_detected"]:
            st.success("Currency detected with YOLO model!")

            # **CNN Model API Request (Currency Classification)**
            cnn_response = requests.post("http://127.0.0.1:5001/predict", json={"images": [base64_image]})
            cnn_result = cnn_response.json()

            # Extract CNN prediction
            predicted_class_index = np.argmax(cnn_result["average_predictions"])
            predicted_class_name = cnn_classes[predicted_class_index]
            confidence = float(cnn_result["score_percentage"].replace("%", ""))

            st.write(f"Predicted Currency: **{predicted_class_name}**")
            st.write(f"Confidence: **{confidence:.2f}%**")

            # Speak the prediction
            threading.Thread(target=speak, args=(f"The detected currency is {predicted_class_name} with {confidence:.2f} percent confidence.",)).start()
        else:
            st.error("No currency detected by YOLO model.")
            threading.Thread(target=speak, args=("No currency detected.",)).start()

    except requests.exceptions.ConnectionError:
        st.error("Connection failed! Flask server might not be running.")
