import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical

try:
    from .model import build_baseline_model, NUM_CLASSES
except ImportError:
    from model import build_baseline_model, NUM_CLASSES


DEFAULT_DATASET_PATH = Path("data/fer2013.csv")
DEFAULT_MODEL_OUTPUT = Path("models/emotion_model.h5")
DEFAULT_HISTORY_PLOT = Path("assets/training_history.png")


def parse_args():
    parser = argparse.ArgumentParser(description="Train baseline Emotion AI model on FER-2013 CSV dataset.")
    parser.add_argument(
        "--data",
        type=str,
        default=str(DEFAULT_DATASET_PATH),
        help="Path to FER-2013 CSV file."
    )
    parser.add_argument(
        "--model-output",
        type=str,
        default=str(DEFAULT_MODEL_OUTPUT),
        help="Path to save the trained model."
    )
    parser.add_argument(
        "--history-plot",
        type=str,
        default=str(DEFAULT_HISTORY_PLOT),
        help="Path to save the training history plot."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of training epochs."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size."
    )
    return parser.parse_args()


def parse_pixels(pixel_sequence: str) -> np.ndarray:
    """
    Convert a FER-2013 pixel string into a 48x48 numpy array.
    """
    pixels = np.fromstring(pixel_sequence, dtype=np.float32, sep=" ")

    if pixels.size != 48 * 48:
        raise ValueError(f"Expected 2304 pixels, got {pixels.size}")

    pixels = pixels.reshape((48, 48, 1))
    pixels /= 255.0
    return pixels


def load_fer2013_dataset(csv_path: str | Path):
    """
    Load FER-2013 dataset from CSV with columns:
    emotion, pixels, Usage
    """
    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    required_columns = {"emotion", "pixels", "Usage"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    x = np.stack(df["pixels"].apply(parse_pixels).to_numpy())
    y = to_categorical(df["emotion"].astype(int), num_classes=NUM_CLASSES)
    usage = df["Usage"].astype(str)

    x_train = x[usage == "Training"]
    y_train = y[usage == "Training"]

    x_val = x[usage == "PublicTest"]
    y_val = y[usage == "PublicTest"]

    x_test = x[usage == "PrivateTest"]
    y_test = y[usage == "PrivateTest"]

    if len(x_train) == 0 or len(x_val) == 0 or len(x_test) == 0:
        raise ValueError("One of the dataset splits is empty. Check the 'Usage' column values.")

    return x_train, y_train, x_val, y_val, x_test, y_test


def save_history_plot(history, output_path: str | Path):
    """
    Save training/validation accuracy and loss curves.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    history_data = history.history

    epochs = range(1, len(history_data["loss"]) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history_data["accuracy"], label="Train Accuracy")
    plt.plot(epochs, history_data["val_accuracy"], label="Validation Accuracy")
    plt.plot(epochs, history_data["loss"], label="Train Loss")
    plt.plot(epochs, history_data["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Training History")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main():
    args = parse_args()

    data_path = Path(args.data)
    model_output_path = Path(args.model_output)
    history_plot_path = Path(args.history_plot)

    model_output_path.parent.mkdir(parents=True, exist_ok=True)
    history_plot_path.parent.mkdir(parents=True, exist_ok=True)

    print("Loading dataset...")
    x_train, y_train, x_val, y_val, x_test, y_test = load_fer2013_dataset(data_path)

    print(f"Training samples:   {len(x_train)}")
    print(f"Validation samples: {len(x_val)}")
    print(f"Test samples:       {len(x_test)}")

    print("Building model...")
    model = build_baseline_model()

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=str(model_output_path),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        ),
    ]

    print("Starting training...")
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    print("Evaluating on test set...")
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)

    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")

    save_history_plot(history, history_plot_path)
    print(f"Saved training history plot to: {history_plot_path}")

    print(f"Best model saved to: {model_output_path}")


if __name__ == "__main__":
    main()