import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

try:
    from .model import build_baseline_model
except ImportError:
    from model import build_baseline_model


DEFAULT_TRAIN_DIR = Path("data/fer2013/train")
DEFAULT_TEST_DIR = Path("data/fer2013/test")
DEFAULT_MODEL_OUTPUT = Path("models/emotion_model.h5")
DEFAULT_HISTORY_PLOT = Path("assets/training_history.png")

IMAGE_SIZE = (48, 48)
CLASS_NAMES = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]


def parse_args():
    parser = argparse.ArgumentParser(description="Train baseline Emotion AI model from image folders.")
    parser.add_argument(
        "--train-dir",
        type=str,
        default=str(DEFAULT_TRAIN_DIR),
        help="Path to training directory."
    )
    parser.add_argument(
        "--test-dir",
        type=str,
        default=str(DEFAULT_TEST_DIR),
        help="Path to test directory."
    )
    parser.add_argument(
        "--model-output",
        type=str,
        default=str(DEFAULT_MODEL_OUTPUT),
        help="Path to save trained model."
    )
    parser.add_argument(
        "--history-plot",
        type=str,
        default=str(DEFAULT_HISTORY_PLOT),
        help="Path to save training history plot."
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


def save_history_plot(history, output_path: str | Path):
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

    train_dir = Path(args.train_dir)
    test_dir = Path(args.test_dir)
    model_output_path = Path(args.model_output)
    history_plot_path = Path(args.history_plot)

    if not train_dir.exists():
        raise FileNotFoundError(f"Train directory not found: {train_dir}")

    if not test_dir.exists():
        raise FileNotFoundError(f"Test directory not found: {test_dir}")

    model_output_path.parent.mkdir(parents=True, exist_ok=True)
    history_plot_path.parent.mkdir(parents=True, exist_ok=True)

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        validation_split=0.2
    )

    test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    train_generator = train_datagen.flow_from_directory(
        directory=str(train_dir),
        target_size=IMAGE_SIZE,
        color_mode="grayscale",
        classes=CLASS_NAMES,
        class_mode="categorical",
        batch_size=args.batch_size,
        shuffle=True,
        subset="training"
    )

    val_generator = train_datagen.flow_from_directory(
        directory=str(train_dir),
        target_size=IMAGE_SIZE,
        color_mode="grayscale",
        classes=CLASS_NAMES,
        class_mode="categorical",
        batch_size=args.batch_size,
        shuffle=False,
        subset="validation"
    )

    test_generator = test_datagen.flow_from_directory(
        directory=str(test_dir),
        target_size=IMAGE_SIZE,
        color_mode="grayscale",
        classes=CLASS_NAMES,
        class_mode="categorical",
        batch_size=args.batch_size,
        shuffle=False
    )

    print("Class indices:", train_generator.class_indices)

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

    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1
    )

    test_loss, test_accuracy = model.evaluate(test_generator, verbose=0)

    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")

    save_history_plot(history, history_plot_path)
    print(f"Saved training history plot to: {history_plot_path}")
    print(f"Best model saved to: {model_output_path}")


if __name__ == "__main__":
    main()