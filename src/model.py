from pathlib import Path

from tensorflow.keras.models import load_model


EMOTION_LABELS = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Sad",
    "Surprise",
    "Neutral",
]


DEFAULT_MODEL_PATH = Path("models/emotion_model.h5")


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