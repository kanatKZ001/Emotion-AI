from pathlib import Path

from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential, load_model


EMOTION_LABELS = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Sad",
    "Surprise",
    "Neutral",
]

INPUT_SHAPE = (48, 48, 1)
NUM_CLASSES = len(EMOTION_LABELS)

DEFAULT_MODEL_PATH = Path("models/emotion_model.h5")


def build_baseline_model(input_shape: tuple[int, int, int] = INPUT_SHAPE, num_classes: int = NUM_CLASSES):
    """
    Build a simple baseline CNN for facial emotion recognition.
    """
    model = Sequential(
        [
            Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
            MaxPooling2D((2, 2)),
            Dropout(0.25),

            Conv2D(64, (3, 3), activation="relu"),
            MaxPooling2D((2, 2)),
            Dropout(0.25),

            Conv2D(128, (3, 3), activation="relu"),
            MaxPooling2D((2, 2)),
            Dropout(0.25),

            Flatten(),
            Dense(256, activation="relu"),
            Dropout(0.5),
            Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def load_emotion_model(model_path: str | Path = DEFAULT_MODEL_PATH):
    """
    Load a trained emotion recognition model from disk.
    """
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    return load_model(model_path)


def get_emotion_label(prediction_index: int) -> str:
    """
    Return emotion label by prediction index.
    """
    if prediction_index < 0 or prediction_index >= len(EMOTION_LABELS):
        raise ValueError(f"Invalid prediction index: {prediction_index}")

    return EMOTION_LABELS[prediction_index]