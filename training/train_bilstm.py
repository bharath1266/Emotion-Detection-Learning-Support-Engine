"""
training/train_bilstm.py

Beginner-friendly BiLSTM training script for Emotion Detection & Learning Support Engine.

What it does:
- Loads `data/train.csv` and `data/test.csv` (expects `text` and `emotion` columns)
- Tokenizes and pads text using TensorFlow's Tokenizer
- Encodes emotion labels into integers (Happy=0, Sad=1, Angry=2, Fear=3, Neutral=4)
- Builds a Bidirectional LSTM model using Keras
- Trains for a small number of epochs (default 5) and evaluates on the test set
- Saves the trained model and tokenizer to `models/bltsm/`

Run from repository root:
    python training/train_bilstm.py

This script includes helpful error messages if TensorFlow is not installed.
"""

import os
import sys
import logging
from pathlib import Path
import pickle
import json

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import resample
from sklearn.utils.class_weight import compute_class_weight

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import TensorFlow with a helpful error message if missing
try:
    import tensorflow as tf
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout, BatchNormalization
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
except Exception as e:
    logger.error("TensorFlow is required to run this script. Install it with: pip install tensorflow")
    raise

from utils.config import DATA_DIR, BILSTM_MODEL_PATH, BILSTM_TOKENIZER_PATH, LABEL_MAP, BILSTM_MAX_LEN, LABELS

BILSTM_DIR = BILSTM_MODEL_PATH.parent
BILSTM_DIR.mkdir(parents=True, exist_ok=True)

# Training hyperparameters (beginner-friendly defaults)
VOCAB_SIZE = 30000
EMBED_DIM = 200
MAX_LEN = int(BILSTM_MAX_LEN)
LSTM_UNITS = 128
BATCH_SIZE = 64
EPOCHS = 15
RANDOM_STATE = 42
VALIDATION_SPLIT = 0.15
PATIENCE = 4
TARGET_BALANCED_COUNT = 20000


def load_data(train_path: Path, test_path: Path):
    """Load train and test CSVs into pandas DataFrames."""
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"Missing train/test CSVs at {train_path} or {test_path}")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df


def balance_training_data(df: pd.DataFrame) -> pd.DataFrame:
    """Resample the training data so minority classes are represented more evenly."""
    df = df.copy()
    df["emotion"] = df["emotion"].astype(str).str.strip().str.lower()
    # Determine a sensible per-class target: use the median class size capped by TARGET_BALANCED_COUNT
    counts = df["emotion"].value_counts()
    target = int(min(TARGET_BALANCED_COUNT, counts.median()))
    if target < 1:
        target = int(counts.max())

    balanced_frames = []
    for label, group in df.groupby("emotion", sort=False):
        if len(group) < target:
            balanced_frames.append(resample(group, replace=True, n_samples=target, random_state=RANDOM_STATE))
        else:
            balanced_frames.append(group.sample(n=target, random_state=RANDOM_STATE))
    return pd.concat(balanced_frames, ignore_index=True)


def preprocess_texts(train_texts, test_texts):
    """Tokenize and pad texts. Returns (padded_train, padded_test, tokenizer)."""
    tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
    tokenizer.fit_on_texts(train_texts)

    train_seq = tokenizer.texts_to_sequences(train_texts)
    test_seq = tokenizer.texts_to_sequences(test_texts)

    x_train = pad_sequences(train_seq, maxlen=MAX_LEN, padding="post", truncating="post")
    x_test = pad_sequences(test_seq, maxlen=MAX_LEN, padding="post", truncating="post")

    return x_train, x_test, tokenizer


def encode_labels(series: pd.Series):
    """Map emotion strings to integer labels using `LABEL_MAP` from config.

    This keeps label ordering consistent with the rest of the project.
    """
    mapping = LABEL_MAP
    labels = series.astype(str).str.strip().str.lower().map(mapping)
    if labels.isnull().any():
        raise ValueError("Found labels that could not be mapped to known classes. See utils/config.py")
    return labels.values


def build_model(vocab_size: int, embed_dim: int, max_len: int, lstm_units: int):
    """Constructs and returns a compiled Keras BiLSTM model."""
    model = Sequential([
        # Reserve +1 for OOV / padding index
        Embedding(input_dim=vocab_size + 1, output_dim=embed_dim),
        Bidirectional(LSTM(lstm_units, return_sequences=True)),
        Dropout(0.3),
        Bidirectional(LSTM(lstm_units // 2)),
        Dropout(0.3),
        Dense(128, activation="relu"),
        BatchNormalization(),
        Dropout(0.4),
        Dense(64, activation="relu"),
        Dropout(0.3),
        Dense(5, activation="softmax"),
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    logger.info("Loading data from: %s", DATA_DIR)
    train_csv = DATA_DIR / "train.csv"
    test_csv = DATA_DIR / "test.csv"

    train_df, test_df = load_data(train_csv, test_csv)

    # Basic sanity checks
    for col in ["text", "emotion"]:
        if col not in train_df.columns or col not in test_df.columns:
            raise KeyError(f"Required column '{col}' not present in train/test CSVs")

    # Drop any rows with missing values in required columns
    train_df = train_df.dropna(subset=["text", "emotion"]).reset_index(drop=True)
    test_df = test_df.dropna(subset=["text", "emotion"]).reset_index(drop=True)

    from utils.preprocessing import clean_text

    train_df = balance_training_data(train_df)
    # Record per-class target after balancing for metadata
    try:
        per_class_target = int(train_df["emotion"].value_counts().iloc[0])
    except Exception:
        per_class_target = None
    x_train_texts = [clean_text(text) for text in train_df["text"].astype(str).tolist()]
    x_test_texts = [clean_text(text) for text in test_df["text"].astype(str).tolist()]

    x_train, x_test, tokenizer = preprocess_texts(x_train_texts, x_test_texts)

    # Encode labels
    y_train = encode_labels(train_df["emotion"])
    y_test = encode_labels(test_df["emotion"])

    # Compute class weights to help with class imbalance during training
    classes = np.unique(y_train)
    class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weight_dict = {int(c): float(w) for c, w in zip(classes, class_weights)}

    # Set random seeds for reproducibility
    np.random.seed(RANDOM_STATE)
    tf.keras.utils.set_random_seed(RANDOM_STATE)

    # Build model
    model = build_model(VOCAB_SIZE, EMBED_DIM, MAX_LEN, LSTM_UNITS)
    model.summary()

    # Callbacks: early stopping, reduce LR on plateau, and save best model
    checkpoint_path = BILSTM_MODEL_PATH
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1),
        ModelCheckpoint(str(checkpoint_path), monitor="val_loss", save_best_only=True, verbose=1),
    ]

    # Train the model with class weights
    history = model.fit(
        x_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
        callbacks=callbacks,
        class_weight=class_weight_dict,
        verbose=1,
    )

    train_acc = history.history.get("accuracy")[-1]
    logger.info("Final training accuracy: %.4f", train_acc)

    # Evaluate on test set
    loss, test_acc = model.evaluate(x_test, y_test, verbose=1)
    logger.info("Test accuracy: %.4f", test_acc)

    y_pred = np.argmax(model.predict(x_test, verbose=0), axis=1)
    logger.info("Classification report:\n%s", classification_report(y_test, y_pred, target_names=LABELS))
    logger.info("Confusion matrix:\n%s", confusion_matrix(y_test, y_pred))

    # Per-class accuracy
    per_class_accuracy = {}
    for i, label in enumerate(LABELS):
        mask = (y_test == i)
        if mask.sum() == 0:
            per_class_accuracy[label] = None
        else:
            per_class_accuracy[label] = float((y_pred[mask] == i).sum()) / int(mask.sum())
    logger.info("Per-class accuracy: %s", per_class_accuracy)

    # Save tokenizer and metadata
    tokenizer_path = BILSTM_TOKENIZER_PATH
    with open(tokenizer_path, "wb") as f:
        pickle.dump(tokenizer, f)

    # Save label mapping and training metadata for reproducibility
    metadata = {
        "labels": LABEL_MAP,
        "vocab_size": VOCAB_SIZE,
        "embed_dim": EMBED_DIM,
        "max_len": MAX_LEN,
        "lstm_units": LSTM_UNITS,
        "batch_size": BATCH_SIZE,
        "epochs_trained": len(history.history.get("loss", [])),
        "per_class_target": per_class_target,
        "tokenizer_num_words": getattr(tokenizer, "num_words", None),
        "training_date": datetime.utcnow().isoformat(),
    }
    with open(BILSTM_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Saved tokenizer to: %s", tokenizer_path)
    logger.info("Best model saved to: %s", checkpoint_path)


if __name__ == "__main__":
    main()
