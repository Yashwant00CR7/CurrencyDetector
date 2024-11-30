

# **Currency Detection System for Visually Impaired Users**

### **Overview**
In an effort to improve accessibility for visually impaired individuals, this project focuses on creating a **Currency Detection System** that leverages machine learning models to identify and classify currency notes in real-time. The system is designed to provide audio feedback, ensuring that visually impaired users can interact with their environment and receive clear, real-time updates about the currency they encounter.

### **Introduction**
Currency detection has long been a challenge for the visually impaired community. This project addresses that need by developing an intelligent application that detects and classifies currency using real-time camera feed. The system makes use of cutting-edge technologies like **YOLO** for object detection and **Convolutional Neural Networks (CNNs)** for classification. Once a currency note is detected, the system announces the denomination via audio to the user, facilitating greater independence and financial accuracy.

### **Problem Statement**
With a large number of different currencies and denominations across the globe, it can be difficult for visually impaired individuals to determine the value of currency notes. A system that can recognize currency in real-time and speak the value to the user will significantly improve their daily experiences.

### **Proposed Methodology**
This project uses two main models:
1. **YOLO (You Only Look Once)**: Used to detect the presence of currency in the live video feed.
2. **CNN (Convolutional Neural Network)**: Trained to classify the detected currency notes into specific denominations.

The system works as follows:
1. The user activates the system via voice input.
2. The **YOLO model** detects whether a currency note is in the frame.
3. Once detected, the **CNN model** classifies the currency note.
4. The result is then announced to the user using text-to-speech technology.

### **Model Architecture**
1. **YOLO Model**:
   - Used for detecting the presence of currency notes in the camera feed.
   - It was trained with a custom dataset that includes over 1,000 images of different currency notes from various angles, lighting, and backgrounds.
   - The model achieves an impressive **93.5% accuracy** in detecting currency.

2. **CNN Model**:
   - Based on a **DenseNet** pretrained model and fine-tuned to predict the currency denomination (e.g., 10 INR, 20 INR, etc.).
   - Trained for **13 epochs** and fine-tuned with an additional **5 epochs** to achieve high classification accuracy.

### **Data Collection and Dataset**
The dataset for this project includes images of various currency denominations and is sourced from publicly available datasets as well as custom image collection.

### **Technologies Used**
- **YOLOv5** for object detection.
- **CNN** based on the **DenseNet** architecture for classification.
- **TensorFlow** for deep learning model implementation.
- **Azure** for storing and processing the dataset.
- **Text-to-Speech (TTS)** for audio feedback.
- **Flutter** for mobile app development.

### **Results**
- The models successfully detect and classify currency with high accuracy.
- Real-time performance allows for seamless interaction with users.
- Speech feedback enhances accessibility for visually impaired individuals.

### **Dataset Link**
- [Currency Dataset](https://www.kaggle.com/datasets/yashwantk23cse/indian-currency) - A collection of images for training the detection and classification models.

### **Installation**

#### Prerequisites:
Ensure that you have the following installed:
- Python 3.x
- pip (Python's package installer)

#### Install Required Libraries:
To install the required libraries, use the following command:

```bash
pip install -r requirements.txt
```

#### Setting Up the Project:
1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/currency-detection
    ```
2. Navigate to the project folder:
    ```bash
    cd currency-detection
    ```

### **Usage**

To use the currency detection system, you need to:
1. Set up the camera feed (ensure a webcam or smartphone camera is available).
2. Run the `app.py` script to start the application:

    ```bash
    python app.py
    ```

The app will activate the camera feed and start processing the video to detect currency. Upon detecting a currency note, the system will announce the denomination.

### **Screenshots**
![Currency Detection YOLO Model](https://github.com/user-attachments/assets/87975220-b65b-4555-9567-ac2b8f6fc66c)
![Currency Detection CNN Model](https://github.com/user-attachments/assets/a572f3ce-df82-43e2-b6ae-40ee5c4a7382)

### **Built With**
- **OpenCV** - For real-time computer vision processing.
- **TensorFlow** - For machine learning model development.
- **YOLOv5** - For object detection.
- **SpeechRecognition** - For speech input functionality.
- **gTTS (Google Text-to-Speech)** - For speech output functionality.
- **Flutter** - For developing the mobile app.

### **Contributing**
Feel free to fork the repository, submit issues, or open pull requests. Any contributions, bug reports, or feature requests are welcome!

### **Conclusion**
This project demonstrates the potential of combining object detection and classification models to create an accessible solution for visually impaired users. The use of real-time detection and text-to-speech feedback provides a meaningful improvement in daily life, particularly when interacting with currency notes.

### **License**
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

