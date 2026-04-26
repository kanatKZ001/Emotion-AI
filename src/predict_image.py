import argparse
from pathlib import Path

import cv2

try:
    from .model import load_emotion_model
    from .utils import (
        load_face_detector,
        detect_largest_face,
        predict_emotion,
        draw_prediction,
    )
except ImportError:
    from model import load_emotion_model
    from utils import (
        load_face_detector,
        detect_largest_face,
        predict_emotion,
        draw_prediction,
    )


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
    result_image = draw_prediction(image.copy(), face, emotion_label, confidence)

    print(f"Predicted emotion: {emotion_label}")
    print(f"Confidence: {confidence:.4f}")

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), result_image)
        print(f"Saved result to: {output_path}")

    if args.show:
        cv2.imshow("Emotion Prediction", result_image)
        print("Press any key in the image window to close.")

        while True:
            key = cv2.waitKey(1) & 0xFF
            if key != 255:
                break

            if cv2.getWindowProperty("Emotion Prediction", cv2.WND_PROP_VISIBLE) < 1:
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()