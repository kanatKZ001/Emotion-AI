import cv2
import numpy as np


TARGET_SIZE = (48, 48)


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Convert image to grayscale if needed.
    Accepts BGR, RGB, or already grayscale image.
    """
    if image is None:
        raise ValueError("Input image is None.")

    if len(image.shape) == 2:
        return image

    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    raise ValueError(f"Unsupported image shape: {image.shape}")


def resize_face(face_image: np.ndarray, target_size: tuple[int, int] = TARGET_SIZE) -> np.ndarray:
    """
    Resize face image to the target size expected by the model.
    """
    if face_image is None or face_image.size == 0:
        raise ValueError("Face image is empty or None.")

    return cv2.resize(face_image, target_size)


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalize pixel values to [0, 1].
    """
    return image.astype("float32") / 255.0


def add_batch_and_channel_dims(image: np.ndarray) -> np.ndarray:
    """
    Convert shape from (48, 48) to (1, 48, 48, 1),
    which is commonly expected by CNN emotion models.
    """
    image = np.expand_dims(image, axis=-1)   # (48, 48, 1)
    image = np.expand_dims(image, axis=0)    # (1, 48, 48, 1)
    return image


def preprocess_face(face_image: np.ndarray, target_size: tuple[int, int] = TARGET_SIZE) -> np.ndarray:
    """
    Full preprocessing pipeline for a face ROI:
    1. Convert to grayscale
    2. Resize to 48x48
    3. Normalize
    4. Add channel and batch dimensions

    Returns:
        np.ndarray with shape (1, 48, 48, 1)
    """
    gray_face = to_grayscale(face_image)
    resized_face = resize_face(gray_face, target_size)
    normalized_face = normalize_image(resized_face)
    processed_face = add_batch_and_channel_dims(normalized_face)
    return processed_face