const video = document.getElementById('webcam');
const canvas = document.getElementById('canvas');
const captureBtn = document.getElementById('captureBtn');
const switchCameraBtn = document.getElementById('switchCameraBtn'); // Button to switch cameras
const status = document.getElementById('status');
const languageSelector = document.getElementById('languageSelector');

let imageDataList = [];
let selectedLanguage = 'en'; // Default language is English
let currencyDetected = false; // Initialize the currencyDetected variable as false
let isBackCamera = true; // Default to back camera

const messages = {
    en: {
        sendingImages: "Sending images...",
        errorSending: "An error occurred while sending the images.",
        prediction: "The detected currency is",
        withConfidence: "with confidence",
        skipped: "Skipping CNN model call.",
    },
    ta: {
        sendingImages: "படங்களை அனுப்புகிறேன்...",
        errorSending: "படங்களை அனுப்புவதில் பிழை ஏற்பட்டது.",
        prediction: "கண்டறியப்பட்ட நோட்டு",
        withConfidence: "விசுவாசத்துடன் உள்ளது",
        skipped: "CNN மாடல் அழைப்பை தவிர்க்கிறேன்.",
    }
};

// Function to start webcam with selected camera mode
async function startWebcam() {
    try {
        const constraints = {
            video: {
                facingMode: isBackCamera ? { exact: "environment" } : "user", // 🔄 Toggle between back and front camera
                width: { ideal: 1280 },
                height: { ideal: 720 }
            }
        };

        const stream = await navigator.mediaDevices.getUserMedia(constraints);
        if (!video) {
            console.error('Video element not found!');
            return;
        }
        video.srcObject = stream;
        console.log(`Webcam access granted. Using ${isBackCamera ? "back" : "front"} camera.`);
    } catch (error) {
        console.error('Error accessing webcam:', error.message);
        alert('Error: Could not access webcam. Please check permissions and browser settings.');
    }
}

// Function to switch camera
function switchCamera() {
    isBackCamera = !isBackCamera; // Toggle camera
    startWebcam(); // Restart webcam with new camera mode
}

// Multilingual Text-to-Speech function
function speakText(key, additionalText = '') {
    return new Promise((resolve) => {
        const text = `${messages[selectedLanguage][key]} ${additionalText ? additionalText : ''}`;
        const speech = new SpeechSynthesisUtterance(text);
        speech.lang = selectedLanguage === 'en' ? 'en-US' : 'ta-IN';
        speech.pitch = 1;
        speech.rate = 1;
        speech.volume = 1;
        speech.onend = resolve; // Resolve the promise when speech ends
        window.speechSynthesis.speak(speech);
    });
}

// Capture image from webcam and store it
async function captureImage() {
    if (!video || !canvas) {
        console.error('Video or Canvas element not found!');
        return;
    }

    const ctx = canvas.getContext('2d');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Convert canvas to Base64 string
    const imageData = canvas.toDataURL('image/jpeg');
    imageDataList.push(imageData); // Store captured image data
    console.log('Captured Image:', imageDataList);
}

// Function to send the list of images to the YOLO API for detection
async function detectCurrencyWithYOLO(images) {
    try {
        const response = await fetch('http://127.0.0.1:5000/detect_currency', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ images: images }),
        });

        const result = await response.json();
        console.log('YOLO API Response:', result);

        return result.currency_detected === true; // Explicitly check for true
    } catch (error) {
        console.error('Error during YOLO API call:', error);
        return false;
    }
}

// Function to capture 10 images every 100ms and check for "Currency"
async function captureAndCheckImages() {
    imageDataList = []; // Clear the image list

    // Capture 10 images and store them
    for (let i = 0; i < 10; i++) {
        await captureImage(); // Capture an image every 100ms
        await new Promise((resolve) => setTimeout(resolve, 100)); // Wait for 100ms
    }

    const isCurrencyDetected = await detectCurrencyWithYOLO(imageDataList);

    if (isCurrencyDetected) {
        console.log('YOLO detected sufficient Currency. Triggering CNN model.');
        currencyDetected = true;
        await speakText('sendingImages'); // Announce sending images
        await sendImagesToCNNModel();
    } else {
        console.log('YOLO detected insufficient Currency. Skipping CNN model call.');
        currencyDetected = false;
        await speakText('skipped'); // Announce skipped message
    }

    // Start next scan only after speech finishes
    await new Promise((resolve) => setTimeout(resolve, 500)); // Small delay for speech clarity
    captureAndCheckImages(); // Restart scanning process
}

// Send images to CNN API after detecting sufficient Currency instances
async function sendImagesToCNNModel() {
    if (!currencyDetected) {
        console.log("No currency detected in sufficient images. Skipping CNN model call.");
        return;
    }

    try {
        const response = await fetch('http://127.0.0.1:5001/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ images: imageDataList }),
        });

        const result = await response.json();
        console.log('CNN API Response:', result);

        const predictedClass = result.class || 'Unknown';
        const scorePercentage = result.score_percentage || 'N/A';

        if (status) {
            status.innerText = `${messages[selectedLanguage].prediction}: ${predictedClass} ${messages[selectedLanguage].withConfidence} ${scorePercentage}`;
        }
        
        await speakText('prediction', `${predictedClass} ${messages[selectedLanguage].withConfidence} ${scorePercentage}`);
    } catch (error) {
        console.error('Error sending images:', error);
        if (status) {
            status.innerText = messages[selectedLanguage].errorSending;
        }
        await speakText('errorSending');
    }
}

// Event listener for language selector dropdown
if (languageSelector) {
    languageSelector.addEventListener('change', (event) => {
        selectedLanguage = event.target.value;
    });
}

// Add event listener for capture button
if (captureBtn) {
    captureBtn.addEventListener('click', captureAndCheckImages);
}

// Add event listener for switch camera button
if (switchCameraBtn) {
    switchCameraBtn.addEventListener('click', switchCamera);
}

// Start the webcam on page load
startWebcam();
