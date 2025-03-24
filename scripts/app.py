from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from werkzeug.utils import secure_filename
from mtcnn.mtcnn import MTCNN

app = Flask(__name__, template_folder=os.path.join(os.getcwd(), "templates"))

# Allowed video formats
ALLOWED_EXTENSIONS = {'mp4', 'mov', 'avi', 'mkv'}
MODEL_PATH = "models/deefake_detector.h5"  # Update with correct model path

# Load deepfake model
def load_deepfake_model(model_path):
    try:
        model = load_model(model_path)
        print(" Model loaded successfully.")
        return model
    except Exception as e:
        print(f" Error loading model: {e}")
        return None

# Face detection: Try MTCNN first, then fallback to Haar Cascade
detector = None
try:
    detector = MTCNN()
    print(" MTCNN loaded successfully.")
except:
    print("⚠️ MTCNN failed, switching to Haar Cascade.")
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Preprocess frame for model
def preprocess_frame(frame):
    resized_frame = cv2.resize(frame, (224, 224))  # Resize to model input size
    frame = resized_frame.astype('float32') / 255.0  # Normalize
    frame = np.expand_dims(frame, axis=0)  # Add batch dimension
    return frame

# Predict deepfake score
def predict_deepfake(model, frame):
    if model is not None:
        try:
            processed_frame = preprocess_frame(frame)
            prediction = model.predict(processed_frame)
            confidence_score = float(prediction[0][0])  # Convert to float
            print(f"Confidence Score: {confidence_score:.4f}")  # Log to terminal
            return confidence_score
        except Exception as e:
            print(f"Error during prediction: {e}")
            return None
    else:
        print(" Model is not loaded.")
        return None

# Check valid file type
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Extract faces from frame
def extract_faces(frame):
    global detector
    faces = []
    
    # Try MTCNN
    if isinstance(detector, MTCNN):
        detected_faces = detector.detect_faces(frame)
        for face in detected_faces:
            x, y, w, h = face['box']
            faces.append(frame[y:y+h, x:x+w])

    # If MTCNN fails, use Haar Cascade
    else:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detected_faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        for (x, y, w, h) in detected_faces:
            faces.append(frame[y:y+h, x:x+w])
    
    return faces

# Display frame using matplotlib (fix for OpenCV GUI error)
def show_frame(frame):
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.axis('off')  # Hide axis for better visualization
    plt.show()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file and allowed_file(file.filename):
        # Save uploaded file
        filename = secure_filename(file.filename)
        file_path = os.path.join('uploads', filename)
        file.save(file_path)

        print(f"\n Processing Video: {filename}\n")

        # Process video
        video_capture = cv2.VideoCapture(file_path)
        confidence_scores = []

        frame_count = 0
        frame_skip = 3  # Experiment with 3, 4, or 5 for best accuracy

        while video_capture.isOpened():
            ret, frame = video_capture.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % frame_skip != 0:
                continue  # Skip frames for performance

            print(f"🖼️ Processing Frame {frame_count}")

            # Extract faces
            faces = extract_faces(frame)
            if faces:
                for face in faces:
                    confidence = predict_deepfake(model, face)
                    if confidence is not None:
                        confidence_scores.append(confidence)

        video_capture.release()

        if confidence_scores:
            avg_confidence = float(np.mean(confidence_scores))  # Convert to native float
            confidence_threshold = 0.6  # Fine-tune this threshold
            result = "Real" if avg_confidence < confidence_threshold else "Fake"

            print("\n Calibration Summary:")
            print(f"  ➤ Total Faces Processed: {len(confidence_scores)}")
            print(f"  ➤ Average Confidence Score: {avg_confidence:.4f}")
            print(f"  ➤ Final Classification: {result.upper()}\n")

            return jsonify({"result": result, "confidence": avg_confidence}), 200
        else:
            print(" No faces detected in the video.")
            return jsonify({"error": "No faces detected in the video."}), 400
    else:
        return jsonify({"error": "Invalid file type. Only video files are allowed."}), 400

if __name__ == "__main__":
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    model = load_deepfake_model(MODEL_PATH)  # Load model at startup
    app.run(debug=True)
