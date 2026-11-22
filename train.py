# -*- coding: utf-8 -*-
"""
Skin Disease Detection System - Training Script
Project: Skin Disease Detection System Using Deep Learning (FYP)

Trains a custom CNN on the HAM10000 dataset (+ optional healthy-skin class)
and saves the model to `models/`.

Usage:
    python train.py

Prerequisites:
    - HAM10000_images_part_1/ and HAM10000_images_part_2/ in the project root
    - HAM10000_metadata.csv in the project root
    - requirements.txt dependencies installed
"""

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)
from tensorflow.keras.layers import (
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    Input,
    MaxPooling2D,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from preprocessing import (
    LESION_TYPE_DICT,
    custom_preprocessing_generator,
    hair_removal,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

IMG_SIZE = 28           # Resize images to IMG_SIZE x IMG_SIZE
BATCH_SIZE = 32
EPOCHS = 25
LEARNING_RATE = 0.0001
VALIDATION_SPLIT = 0.2
RANDOM_STATE = 42

# HAM10000 contains these seven dermatoscopic lesion classes. Do not add a
# healthy-skin class without a suitable, independently sourced dataset.
CLASS_NAMES = [
    "Actinic keratoses",
    "Basal cell carcinoma",
    "Benign keratosis-like lesions",
    "Dermatofibroma",
    "Vascular lesions",
    "Melanocytic nevi",
    "Melanoma",
]
NUM_CLASSES = len(CLASS_NAMES)


# ---------------------------------------------------------------------------
# Data loading & balancing
# ---------------------------------------------------------------------------

def load_metadata() -> pd.DataFrame:
    """Load the HAM10000 metadata and resolve every image path.

    The full HAM10000 release is required for training.  The small images in
    ``assets/`` are UI examples, not a valid fallback training dataset.
    """
    csv_path = PROJECT_ROOT / "HAM10000_metadata.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        df["cell_type"] = df["dx"].map(LESION_TYPE_DICT)

        # Resolve image paths from both HAM10000 image directories
        img_dirs = [
            PROJECT_ROOT / "HAM10000_images_part_1",
            PROJECT_ROOT / "HAM10000_images_part_2",
        ]
        path_map: dict[str, str] = {}
        for img_dir in img_dirs:
            if img_dir.exists():
                for p in img_dir.glob("*.jpg"):
                    path_map[p.stem] = str(p)

        df["path"] = df["image_id"].map(path_map)

        df = df.dropna(subset=["path", "cell_type"])
        print(f"Loaded {len(df)} images across {df['cell_type'].nunique()} classes from metadata.")
        return df

    raise FileNotFoundError(
        f"HAM10000 metadata was not found at {csv_path}. Extract the official "
        "dataset folders and metadata CSV into the project root before training."
    )


def split_by_lesion(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create a stratified validation split with no lesion in both partitions."""
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    train_idx, val_idx = next(splitter.split(df, y=df["cell_type"], groups=df["lesion_id"]))
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)
    overlap = set(train_df["lesion_id"]) & set(val_df["lesion_id"])
    if overlap:
        raise RuntimeError("Lesion-group split failed: train/validation overlap detected.")
    if set(train_df["cell_type"]) != set(CLASS_NAMES) or set(val_df["cell_type"]) != set(CLASS_NAMES):
        raise RuntimeError("Lesion-group split did not preserve every diagnosis class.")
    return train_df, val_df


def class_weights(labels: np.ndarray) -> dict[int, float]:
    """Compute balanced training-only weights without duplicating images."""
    counts = np.bincount(labels, minlength=NUM_CLASSES)
    return {index: float(len(labels) / (NUM_CLASSES * count))
            for index, count in enumerate(counts) if count > 0}


def load_images(df: pd.DataFrame):
    """Read and preprocess all images from *df* into NumPy arrays."""
    import cv2

    images, labels = [], []
    class_to_idx = {name: idx for idx, name in enumerate(CLASS_NAMES)}

    for _, row in df.iterrows():
        img = cv2.imread(row["path"])
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = hair_removal(img)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        images.append(img)
        labels.append(class_to_idx[row["cell_type"]])

    if not images:
        raise RuntimeError("No readable images were found in the supplied metadata.")
    X = np.array(images, dtype=np.float32) / 255.0
    y = tf.keras.utils.to_categorical(labels, NUM_CLASSES)
    return X, y


# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------

def build_model() -> Sequential:
    """Build the CNN matching the shipped *skin_disease_model.h5*.

    Architecture
    ============
    Input  28 × 28 × 3
      → Conv2D(16, 3×3, relu)
      → Conv2D(32, 3×3, same, relu)
      → MaxPooling2D(2×2)
      → Conv2D(32, 3×3, same, relu)
      → Conv2D(64, 3×3, same, relu)
      → MaxPooling2D(2×2)
      → Dropout(0.4)
      → Flatten
      → Dense(128, relu)
      → Dense(64, relu)
      → Dense(32, relu)
      → Dense(8, softmax)
    """
    model = Sequential([
        Input(shape=(IMG_SIZE, IMG_SIZE, 3)),

        # Block 1
        Conv2D(16, (3, 3), activation="relu"),
        Conv2D(32, (3, 3), padding="same", activation="relu"),
        MaxPooling2D(pool_size=(2, 2)),

        # Block 2
        Conv2D(32, (3, 3), padding="same", activation="relu"),
        Conv2D(64, (3, 3), padding="same", activation="relu"),
        MaxPooling2D(pool_size=(2, 2)),

        # Regularization
        Dropout(0.4),

        # Classifier head
        Flatten(),
        Dense(128, activation="relu"),
        Dense(64, activation="relu"),
        Dense(32, activation="relu"),
        Dense(NUM_CLASSES, activation="softmax"),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ---------------------------------------------------------------------------
# Evaluation & visualisation helpers
# ---------------------------------------------------------------------------

def save_training_history(history):
    """Plot training/validation accuracy and loss curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history.history["accuracy"], label="Train")
    axes[0].plot(history.history["val_accuracy"], label="Validation")
    axes[0].set_title("Model Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()

    axes[1].plot(history.history["loss"], label="Train")
    axes[1].plot(history.history["val_loss"], label="Validation")
    axes[1].set_title("Model Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "training_history.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Training history saved to {RESULTS_DIR / 'training_history.png'}")


def save_confusion_matrix(y_true, y_pred):
    """Generate and save a confusion matrix heat-map."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
    )
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrix saved to {RESULTS_DIR / 'confusion_matrix.png'}")


def save_classification_report(y_true, y_pred):
    """Write a per-class precision/recall/F1 report to a text file."""
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES)
    path = RESULTS_DIR / "classification_report.txt"
    path.write_text(report, encoding="utf-8")
    print(f"Classification report saved to {path}")
    print(report)


def save_dataset_distribution(df: pd.DataFrame):
    """Bar chart of class counts before and after balancing."""
    counts = df["cell_type"].value_counts()
    plt.figure(figsize=(10, 6))
    counts.plot(kind="bar", color="steelblue")
    plt.title("Dataset Distribution (after balancing)")
    plt.xlabel("Skin Condition")
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "dataset_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Dataset distribution saved to {RESULTS_DIR / 'dataset_distribution.png'}")


# ---------------------------------------------------------------------------
# Main training pipeline
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  DermScan AI — Training Pipeline")
    print("=" * 60)

    tf.keras.utils.set_random_seed(RANDOM_STATE)

    # 1. Split by lesion before any augmentation or class balancing.
    df = load_metadata()
    train_df, val_df = split_by_lesion(df)
    print(f"Train images: {len(train_df)} | Validation images: {len(val_df)}")
    print(f"Train lesions: {train_df['lesion_id'].nunique()} | "
          f"Validation lesions: {val_df['lesion_id'].nunique()}")
    save_dataset_distribution(train_df)

    # 2. Load images
    print("\nLoading and preprocessing images (this may take a while)...")
    X_train, y_train = load_images(train_df)
    X_val, y_val = load_images(val_df)
    print(f"Dataset shape: X_train={X_train.shape}, X_val={X_val.shape}")
    print(f"Train: {X_train.shape[0]}  |  Validation: {X_val.shape[0]}")
    weights = class_weights(y_train.argmax(axis=1))
    print(f"Training class weights: {weights}")

    # 4. Data augmentation
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest",
    )
    datagen.fit(X_train)

    # 5. Build model
    model = build_model()
    model.summary()

    # 6. Callbacks
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
        ModelCheckpoint(
            str(MODELS_DIR / "best_model.h5"),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
    ]

    # 7. Train
    print("\nStarting training...")
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
        epochs=EPOCHS,
        validation_data=(X_val, y_val),
        class_weight=weights,
        callbacks=callbacks,
        verbose=1,
    )

    # 8. Save final model + class indices
    model.save(str(MODELS_DIR / "skin_disease_model.h5"))
    class_indices = {name: idx for idx, name in enumerate(CLASS_NAMES)}
    (MODELS_DIR / "class_indices.json").write_text(
        json.dumps(class_indices, indent=2), encoding="utf-8"
    )
    print(f"\nModel saved to {MODELS_DIR / 'skin_disease_model.h5'}")

    # 9. Evaluate
    save_training_history(history)
    y_pred = model.predict(X_val, verbose=0).argmax(axis=1)
    y_true = y_val.argmax(axis=1)
    save_confusion_matrix(y_true, y_pred)
    save_classification_report(y_true, y_pred)

    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"\nFinal validation accuracy: {val_acc:.4f}")
    print(f"Final validation loss:     {val_loss:.4f}")
    print("\n✅ Training complete!")


if __name__ == "__main__":
    main()
