import cv2
import numpy as np

try:
    from .preprocess import preprocess_face
    from .model import load_emotion_model, get_emotion_label
except ImportError:
    from preprocess import preprocess_face
    from model import load_emotion_model, get_emotion_label


CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"


def load_face_detector():
    detector = cv2.CascadeClassifier(CASCADE_PATH)

    if detector.empty():
        raise FileNotFoundError(f"Could not load Haar Cascade from: {CASCADE_PATH}")

    return detector


def detect_faces(gray_frame: np.ndarray, detector):
    faces = detector.detectMultiScale(
        gray_frame,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    return faces


def predict_emotion(face_roi: np.ndarray, model):
    processed_face = preprocess_face(face_roi)
    prediction = model.predict(processed_face, verbose=0)[0]

    predicted_index = int(np.argmax(prediction))
    confidence = float(np.max(prediction))
    emotion_label = get_emotion_label(predicted_index)

    return emotion_label, confidence


def draw_prediction(frame: np.ndarray, face, emotion_label: str, confidence: float):
    x, y, w, h = face

    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    text = f"{emotion_label} ({confidence:.2f})"
    cv2.putText(
        frame,
        text,
        (x, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        cv2.LINE_AA
    )


def main():
    detector = load_face_detector()
    model = load_emotion_model()

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    print("Webcam started. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Failed to read frame from webcam.")
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detect_faces(gray_frame, detector)

        for (x, y, w, h) in faces:
            face_roi = frame[y:y + h, x:x + w]

            try:
                emotion_label, confidence = predict_emotion(face_roi, model)
                draw_prediction(frame, (x, y, w, h), emotion_label, confidence)
            except Exception as error:
                print(f"Skipping one face due to error: {error}")

        cv2.imshow("Emotion AI - Webcam", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()