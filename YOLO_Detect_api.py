from flask import Flask, request, jsonify
import base64
from PIL import Image
import io
import numpy as np
from ultralytics import YOLO
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load your YOLO model
model = YOLO('runs/detect/train4/weights/best.pt')

@app.route('/detect_currency', methods=['POST'])
def detect_currency():
    data = request.get_json()  # Parse JSON request
    images = data.get('images', [])

    if not images:
        return jsonify({'error': 'No images provided'}), 400

    currency_detected_count = 0  # Counter for images containing the "Currency" class

    for img_str in images:
        # Decode Base64 image
        img_data = base64.b64decode(img_str.split(',')[1])  # Remove 'data:image/jpeg;base64,'
        img = Image.open(io.BytesIO(img_data)).convert('RGB')

        # Convert PIL image to NumPy array for YOLO
        img_array = np.array(img)

        # Run YOLO prediction
        results = model.predict(source=img_array, save=False)

        # Extract predictions
        boxes = results[0].boxes  # YOLO Boxes object
        class_names = results[0].names  # Class names from the YOLO model

        # Check if "Currency" class is detected in this image
        for box in boxes:
            cls = int(box.cls[0])  # Get the class index
            if class_names[cls] == "Currency":
                currency_detected_count += 1
                break  # No need to check further boxes for this image

    # Check if "Currency" is detected in at least half the images
    if currency_detected_count >= len(images) / 2:
        return jsonify({'currency_detected': True})
    else:
        return jsonify({'currency_detected': False})

if __name__ == '__main__':
    app.run(debug=True,port=5000)
