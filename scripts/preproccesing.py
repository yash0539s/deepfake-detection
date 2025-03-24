import os
import cv2
import numpy as np
from mtcnn import MTCNN

# Preprocessing function to resize, normalize, and detect faces using MTCNN
def preprocess_video(input_dir, output_dir, filename, img_size=(224, 224), frame_skip=10):
    os.makedirs(output_dir, exist_ok=True)
    video_path = os.path.join(input_dir, filename)
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Couldn't open video {filename}")
        return

    frame_count = 0
    frame_index = 0  # To track which frame we're on
    
    detector = MTCNN()  # Initialize MTCNN face detector
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Only process every 'frame_skip' frame
        if frame_index % frame_skip == 0:
            # Detect faces using MTCNN
            faces = detector.detect_faces(frame)
            
            for face in faces:
                x, y, w, h = face['box']  # Get the bounding box of the face
                keypoints = face['keypoints']  # Get the facial landmarks
                
                # Draw bounding box and landmarks on the frame (optional)
                for key, point in keypoints.items():
                    cv2.circle(frame, point, 2, (0, 255, 0), 2)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Resize and normalize the frame
            frame = cv2.resize(frame, img_size)  # Resize
            frame = frame / 255.0  # Normalize

            # Save the processed frame as an image
            output_filename = f"{filename}_frame_{frame_count}.jpg"
            cv2.imwrite(os.path.join(output_dir, output_filename), (frame * 255).astype(np.uint8))
            frame_count += 1

        frame_index += 1
    
    cap.release()

def process_videos(input_dir, output_dir, frame_skip=10):
    filenames = os.listdir(input_dir)
    for filename in filenames:
        print(f"Processing video: {filename}")  # Add debug print
        preprocess_video(input_dir, output_dir, filename, frame_skip=frame_skip)

if __name__ == "__main__":
    
    process_videos('data/raw_data/deepfake', 'data/processed/deepfake', frame_skip=10)
    process_videos('data/raw_data/real', 'data/processed/real', frame_skip=10)      