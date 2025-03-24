DeepFake Detection
Table of Contents
What is DeepFake?

Impact of DeepFake Videos

Project Objectives

Project Pipeline

Pre-processing Workflow

Prediction Workflow

Models Used and Their Architecture

Deployment

Running the Code

Technologies Used

Conclusion

Team

What is DeepFake?
DeepFake refers to manipulated videos or images in which AI is used to replace one person's face with another, often using deep learning techniques like Generative Adversarial Networks (GANs). These models can generate highly realistic yet fake videos, making it increasingly difficult to differentiate real from manipulated content.



Impact of DeepFake Videos
DeepFake technology is being misused to spread false news, create celebrity impersonations, and even generate fraudulent financial transactions.

It is a major threat to social media platforms, political integrity, and cybersecurity.

Many industries, including film, media, and security agencies, are actively working to develop detection mechanisms to counter DeepFake threats.

Project Objectives
Our primary objective is to build a robust AI model capable of detecting whether a given video is REAL or FAKE.

Key goals:  Train a deep learning model for DeepFake detection.
 Identify manipulated frames in videos using VGG16 and CNN.
 Enhance accuracy and robustness by using face landmark analysis and feature extraction.
 Deploy an AI-based solution that can be integrated into social media platforms for real-time detection.

Project Pipeline
Step	Description
Step 1	Load the dataset
Step 2	Extract videos from the dataset
Step 3	Extract frames from videos (both real & fake)
Step 4	Detect faces in each frame
Step 5	Locate facial landmarks
Step 6	Analyze variations in facial landmarks
Step 7	Classify videos as REAL or FAKE

Pre-processing Workflow
Convert video into frames.

Detect face regions in each frame.

Resize frames to 224×224 pixels.

Normalize pixel values for better training.

Extract face landmarks to detect inconsistencies.

Prediction Workflow
Extract frames from input video.

Detect faces using VGG16 feature extraction.

Pass features through a CNN model for classification.

Aggregate frame-level predictions to classify the entire video.

Models Used and Their Architecture
We implemented VGG16 for feature extraction and CNN for classification.

VGG16 (Feature Extractor)
Pretrained on ImageNet dataset.

Extracts high-level features from frames.

Freezes initial layers and fine-tunes deeper layers for our dataset.

CNN Architecture
Convolutional Layers: Extract spatial features.

Pooling Layers: Reduce dimensionality.

Fully Connected Layers: Make final classification.

sigmoid Activation: Output probabilities for REAL vs FAKE classification.

Hyperparameters Used
Optimizer: Adam (Adaptive Learning Rate)

Loss Function: Sparse Categorical Cross-Entropy

Batch Size: 32

Learning Rate: 0.0001

Epochs: 20

Test Accuracy Achieved: 87% 
Deployment
Backend: Python Flask API

Frontend: HTML, CSS, JavaScript


Processing Time: ~1 min for a 10-second, 30fps video

Running the Code

Ensure all dependencies are installed before running the application.

Step 1: Install Requirements
bash
Copy
Edit
pip install -r requirements.txt
Step 2: Run the Model
bash
Copy
Edit
python main.py

Step 3: Upload a Video for Detection
The model will process each frame.

The final output will be displayed as REAL or FAKE.

Technologies Used
Programming Languages: Python, HTML, CSS, JavaScript

Libraries: OpenCV, TensorFlow, Keras, Pandas, NumPy, Seaborn

Deeplearning Models: VGG16, CNN

Deployment: Flask,

Conclusion
We successfully developed a DeepFake Detection Model using VGG16 for feature extraction and CNN for classification.

Our model achieved an accuracy of 87%, making it reliable for real-world applications.

Further improvements could be made by fine-tuning VGG16 layers, increasing dataset diversity, and integrating LSTM for temporal analysis.