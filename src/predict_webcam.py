import cv2

try:
    from .model import load_emotion_model
    from .utils import (
        load_face_detector,
        detect_faces,
        predict_emotion,
        draw_prediction,
    )
except ImportError:
    from model import load_emotion_model
    from utils import (
        load_face_detector,
        detect_faces,
        predict_emotion,
        draw_prediction,
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