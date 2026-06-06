# Emotion AI

A baseline facial emotion recognition project built with Python, OpenCV, and TensorFlow/Keras.  
The system detects a face in an image or webcam frame, preprocesses it, and predicts the most likely emotion.

## Project Goal

This project is my baseline version of an Emotion AI system.  
The current goal is not to build a large research system, but to create a clean, understandable, and working first version that can be improved step by step over time.

## What the Project Does

The project can:

- detect a face in an input image;
- predict emotion from a detected face;
- run real-time emotion recognition from a webcam;
- train a baseline CNN model on a folder-based facial emotion dataset.

## Tech Stack

- Python
- OpenCV
- TensorFlow / Keras
- NumPy
- Matplotlib

## Project Structure
```text
Emotion-AI/
├── assets/
│   ├── screenshots/
│   ├── architecture.png
│   ├── demo.gif
│   └── training_history.png
├── data/
│   ├── fer2013/
│   │   ├── train/
│   │   └── test/
│   ├── example.jpeg
│   └── README.md
├── models/
│   └── emotion_model.h5
├── notebooks/
│   └── experiments.ipynb
├── src/
│   ├── __init__.py
│   ├── model.py
│   ├── predict_image.py
│   ├── predict_webcam.py
│   ├── preprocess.py
│   ├── train.py
│   └── utils.py
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

## Dataset
This project uses a facial emotion dataset organized in folders:
```text
data/fer2013/
├── train/
│   ├── angry/
│   ├── disgust/
│   ├── fear/
│   ├── happy/
│   ├── sad/
│   ├── surprise/
│   └── neutral/
└── test/
    ├── angry/
    ├── disgust/
    ├── fear/
    ├── happy/
    ├── sad/
    ├── surprise/
    └── neutral/
```
The model is trained on grayscale face images resized to 48x48.


## Recognized Emotions
The baseline model predicts 7 emotions:
- Angry
- Disgust
- Fear
- Happy
- Sad
- Surprise
- Neutral


## Pipeline
The project follows this pipeline:

1. Input image or webcam frame
2. Face detection with OpenCV Haar Cascade
3. Face preprocessing:
 - grayscale conversion
 - resize to 48x48
 - normalization
4. CNN model inference
5. Emotion label prediction
6. Output visualization


## Installation

Clone the repository and install dependencies:
```text
git clone https://github.com/kanatKZ001/Emotion-AI.git
cd Emotion-AI
pip install -r requirements.txt
```

## How to Train

Train the baseline model:
```text
python src/train.py --epochs 20 --batch-size 64
```
For a quick test:
```text
python src/train.py --epochs 1 --batch-size 32
```
The trained model will be saved to:
```text
models/emotion_model.h5
```
The training history plot will be saved to:
```text
assets/training_history.png
```
## How to Predict from an Image

Example:
```text
python src/predict_image.py --image "data/kanat.jpeg" --output "assets/screenshots/result.jpg"
```
If you want to display the result in a window:
```text
python src/predict_image.py --image "data/kanat.jpeg" --show
```
## How to Run Webcam Emotion Detection
```text
python src/predict_webcam.py
```
Press:
- q to quit
- Esc to close the webcam window

## Current Result
This is a baseline version of the project.
At this stage, the main achievement is that the system:
- trains successfully;
- saves a model;
- predicts emotion from an image;
- runs webcam-based emotion detection.

## Limitations
This baseline project has several limitations:
- Haar Cascade face detection is simple and may fail on difficult angles;
- performance depends on lighting conditions and image quality;
- the model is still a basic CNN baseline;
- predictions may be unstable in real-world webcam conditions;
- current accuracy is limited and can be improved in future versions.

## Future Improvements
Possible future improvements include:
- stronger face detection model;
- better CNN architecture;
- data augmentation;
- improved evaluation and accuracy tracking;
- better real-time webcam performance;
- cleaner demo assets and visual results.

## Demo
Demo materials will be added in future updates:
- assets/screenshots/
- assets/demo.gif


## Author
Kanat Zhumatov
