import argparse
from pathlib import Path

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


def detect_largest_face(gray_image: np.ndarray, detector):
    faces = detector.detectMultiScale(
        gray_image,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    if len(faces) == 0:
        return None

    # Берём самое большое лицо
    largest_face = max(faces, key=lambda face: face[2] * face[3])
    return largest_face


def predict_emotion(face_roi: np.ndarray, model):
    processed_face = preprocess_face(face_roi)
    prediction = model.predict(processed_face, verbose=0)[0]

    predicted_index = int(np.argmax(prediction))
    confidence = float(np.max(prediction))
    emotion_label = get_emotion_label(predicted_index)

    return emotion_label, confidence


def draw_result(image: np.ndarray, face, emotion_label: str, confidence: float):
    x, y, w, h = face

    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    text = f"{emotion_label} ({confidence:.2f})"
    cv2.putText(
        image,
        text,
        (x, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        cv2.LINE_AA
    )

    return image


def parse_args():
    parser = argparse.ArgumentParser(description="Predict emotion from an image.")
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to the input image."
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save the output image."
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the output image in a window."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    image_path = Path(args.image)

    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(f"Failed to read image: {image_path}")

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    detector = load_face_detector()
    model = load_emotion_model()

    face = detect_largest_face(gray_image, detector)

    if face is None:
        print("No face detected in the image.")
        return

    x, y, w, h = face
    face_roi = image[y:y + h, x:x + w]

    emotion_label, confidence = predict_emotion(face_roi, model)
    result_image = draw_result(image.copy(), face, emotion_label, confidence)

    print(f"Predicted emotion: {emotion_label}")
    print(f"Confidence: {confidence:.4f}")

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), result_image)
        print(f"Saved result to: {output_path}")

    if args.show:
        cv2.imshow("Emotion Prediction", result_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()