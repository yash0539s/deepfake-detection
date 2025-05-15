from flask import Flask, request, render_template, redirect, url_for, Response
import os
import numpy as np
import tensorflow as tf
import cv2
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Focal Loss
def focal_loss(alpha=0.25, gamma=2.0):
    def loss(y_true, y_pred):
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        pt = tf.where(y_true == 1, y_pred, 1 - y_pred)
        loss = alpha * (1 - pt) ** gamma * bce
        return tf.reduce_mean(loss)
    return loss

#  Load model
model = tf.keras.models.load_model(
    r"D:\deep_fake_hacthon\deepfake_detector_v8.h5",
    custom_objects={"loss": focal_loss(alpha=0.5, gamma=2.0)}
)

#  Preprocess a single frame
def preprocess_frame(frame):
    frame = cv2.resize(frame, (128, 128))
    frame = frame / 255.0
    frame = np.expand_dims(frame, axis=0)
    return frame.astype(np.float32)

#  Predict on video file
def predict_video_deepfake(video_path, frame_skip=5):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": "Could not open video file."}

    frame_count = 0
    predictions = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_skip == 0:
            processed_frame = preprocess_frame(frame)
            pred_proba = model.predict(processed_frame)[0][0]
            predictions.append(pred_proba)
        frame_count += 1

    cap.release()

    if predictions:
        avg_proba = np.mean(predictions)
        label = "Fake" if avg_proba > 0.5 else "Real"
        return {"label": label, "confidence": float(avg_proba)}
    else:
        return {"error": "No frames processed."}

#  Predict from webcam
def predict_webcam():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return {"error": "Could not access webcam."}

    predictions = []
    frame_count = 0

    while frame_count < 30:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % 5 == 0:
            processed_frame = preprocess_frame(frame)
            pred_proba = model.predict(processed_frame)[0][0]
            predictions.append(pred_proba)
        frame_count += 1

    cap.release()

    if predictions:
        avg_proba = np.mean(predictions)
        label = "Fake" if avg_proba > 0.5 else "Real"
        confidence = float(avg_proba)
        return {"label": label, "confidence": confidence}
    else:
        return {"error": "No frames processed."}

#  Video stream generator
def gen_frames():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    cap.release()


#  Routes
@app.route('/')
def home():
    return render_template('upload.html')

@app.route('/webcam')
def webcam_page():
    return render_template('webcam.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/predict', methods=['POST'])
def predict():
    mode = request.form.get('mode')
    
    if mode == 'upload':
        file = request.files['video']
        if file and file.filename != '':
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            result = predict_video_deepfake(filepath, frame_skip=2)
            return render_template('result.html', result=result)
        else:
            return render_template('result.html', result={"error": "No file selected."})

    elif mode == 'webcam':
        result = predict_webcam()
        return render_template('result.html', result=result)
    
    return render_template('result.html', result={"error": "Invalid mode selected."})

if __name__ == '__main__':
    app.run(debug=True)
