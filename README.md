# Emotion-Detection-System

Facial Emotion Detection

Overview

This project is a Facial Emotion Detection system that uses a deep learning model to classify emotions from live video feed. The system detects faces in real-time, processes them, and predicts emotions using a trained neural network model.

Project Structure

Emotion_Detection.ipynb: Jupyter Notebook containing the training process for the emotion detection model.

best_model.h5: The pre-trained deep learning model for emotion classification.

videotester.py: The script for real-time facial emotion detection using a webcam.

Requirements

To run this project, install the following dependencies:

pip install tensorflow keras opencv-python numpy matplotlib

Model Details

The deep learning model (best_model.h5) is trained on a dataset of facial expressions and predicts the following emotions:

Angry

Disgust

Fear

Happy

Sad

Surprise

Neutral

The model is loaded using Keras and expects input images of size 224x224.

How to Use

Ensure that best_model.h5 is in the project directory.

Run videotester.py to start real-time emotion detection:

python videotester.py

The webcam feed will open, and detected faces will have a bounding box with their predicted emotion displayed.

Press 'q' to quit the program.

How It Works

The system captures video frames from the webcam.

Each frame is converted to grayscale.

The haarcascade_frontalface_default.xml classifier detects faces.

Each detected face is resized to 224x224 and normalized.

The processed image is passed through the best_model.h5 model for emotion prediction.

The emotion label is displayed on the video feed in real time.

Dependencies

Python 3.x

TensorFlow & Keras

OpenCV

NumPy

Matplotlib

Notes

Make sure your webcam is properly configured.

Adjust the Haar Cascade parameters if detection is inconsistent.

The model performance depends on lighting conditions and face visibility.

Future Improvements

Train the model on a larger dataset for better accuracy.

Optimize real-time performance using GPU acceleration.

Deploy the model as a web-based or mobile application.
