import cv2
import numpy as np

try:
    from .preprocess import preprocess_face
    from .model import get_emotion_label
except ImportError:
    from preprocess import preprocess_face
    from model import get_emotion_label


CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"


def load_face_detector():
    """
    Load OpenCV Haar Cascade face detector.
    """
    detector = cv2.CascadeClassifier(CASCADE_PATH)

    if detector.empty():
        raise FileNotFoundError(f"Could not load Haar Cascade from: {CASCADE_PATH}")

    return detector


def detect_faces(gray_image: np.ndarray, detector):
    """
    Detect faces in a grayscale image.
    """
    faces = detector.detectMultiScale(
        gray_image,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    return faces


def detect_largest_face(gray_image: np.ndarray, detector):
    """
    Detect faces and return the largest one.
    """
    faces = detect_faces(gray_image, detector)

    if len(faces) == 0:
        return None

    return max(faces, key=lambda face: face[2] * face[3])


def predict_emotion(face_roi: np.ndarray, model):
    """
    Predict emotion label and confidence for a face ROI.
    """
    processed_face = preprocess_face(face_roi)
    prediction = model.predict(processed_face, verbose=0)[0]

    predicted_index = int(np.argmax(prediction))
    confidence = float(np.max(prediction))
    emotion_label = get_emotion_label(predicted_index)

    return emotion_label, confidence


def draw_prediction(frame: np.ndarray, face, emotion_label: str, confidence: float):
    """
    Draw bounding box and prediction text on image/frame.
    """
    x, y, w, h = face

    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    text = f"{emotion_label} ({confidence:.2f})"
    cv2.putText(
        frame,
        text,
        (x, max(y - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        cv2.LINE_AA
    )

    return frame