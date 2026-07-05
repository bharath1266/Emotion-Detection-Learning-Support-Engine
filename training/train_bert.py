"""
training/train_bert.py

Fine-tune a BERT classifier for 5-class emotion detection.

This script loads `data/train.csv` and `data/test.csv`, tokenizes the texts
with `bert-base-uncased`, fine-tunes `BertForSequenceClassification`, evaluates
accuracy, and saves the model and tokenizer to `models/bert_emotion_model_final/`.

Run from the project root:
    python training/train_bert.py

Dependencies:
    pip install transformers torch pandas
"""

import os
import sys
import logging
from pathlib import Path
import json

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from torch.utils.data import DataLoader, Dataset
from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast

BASE_BERT_DIR = Path(r"C:\Users\bablu\.cache\huggingface\hub\models--bert-base-uncased\snapshots\86b5e0934494bd15c9632b12f734a8a67f723594")

logging.basicConfig(level=logging.INFO, format='%(message)s', stream=sys.stdout)
logger = logging.getLogger(__name__)

print("[BERT Training] Script started", flush=True)

from utils.config import DATA_DIR, BERT_MODEL_DIR, LABEL_MAP, BERT_MAX_LEN, LABELS
print(f"[BERT Training] Loaded config: DATA_DIR={DATA_DIR}, BERT_MODEL_DIR={BERT_MODEL_DIR}", flush=True)

MODEL_DIR = BERT_MODEL_DIR
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MAX_LENGTH = int(BERT_MAX_LEN)
BATCH_SIZE = 8
EPOCHS = 3
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
TARGET_BALANCED_COUNT = 300
MAX_TRAIN_SAMPLES = 1600
VAL_FRACTION = 0.15


def get_class_weights(labels: np.ndarray) -> torch.Tensor:
    """Return inverse-frequency weights so minority classes receive more weight."""
    counts = np.bincount(labels.astype(int), minlength=len(LABELS))
    counts = counts.astype(np.float32)
    weights = 1.0 / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


class EmotionDataset(Dataset):
    """Simple PyTorch dataset for tokenized emotion text."""

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def load_csv(path: Path):
    """Load a CSV file and return a pandas DataFrame."""
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path)


def balance_training_data(df: pd.DataFrame) -> pd.DataFrame:
    """Resample the training set so minority classes are represented more evenly."""
    df = df.copy()
    df["emotion"] = df["emotion"].astype(str).str.strip().str.lower()
    # Use a more conservative per-class target: median class size capped by TARGET_BALANCED_COUNT
    counts = df["emotion"].value_counts()
    target = int(min(TARGET_BALANCED_COUNT, counts.median()))
    if target < 1:
        target = int(counts.max())

    balanced_frames = []
    for _, group in df.groupby("emotion", sort=False):
        if len(group) < target:
            balanced_frames.append(resample(group, replace=True, n_samples=target, random_state=42))
        else:
            balanced_frames.append(group.sample(n=target, random_state=42))
    return pd.concat(balanced_frames, ignore_index=True)


def augment_training_data_with_seed_examples(df: pd.DataFrame) -> pd.DataFrame:
    """Add a small set of emotionally clear seed examples to improve the classifier."""
    seed_rows = [
        ("I got selected for my dream internship.", "happy"),
        ("I feel excited and grateful for this opportunity.", "happy"),
        ("I failed my exam and feel disappointed.", "sad"),
        ("I feel lonely and discouraged today.", "sad"),
        ("Nobody listens to my ideas.", "angry"),
        ("I am frustrated because my concerns are ignored.", "angry"),
        ("I'm scared I might fail tomorrow's exam.", "fear"),
        ("I feel anxious about the upcoming test.", "fear"),
        ("Today I attended classes and completed my assignments.", "neutral"),
        ("I had a normal day and finished my tasks.", "neutral"),
    ]
    seed_df = pd.DataFrame(seed_rows, columns=["text", "emotion"])
    repeated_seed_df = pd.concat([seed_df] * 4, ignore_index=True)
    return pd.concat([df, repeated_seed_df], ignore_index=True)


def map_labels(series: pd.Series):
    """Convert emotion strings to numeric labels using LABEL_MAP."""
    mapped = series.astype(str).str.strip().str.lower().map(LABEL_MAP)
    if mapped.isnull().any():
        invalid = sorted(set(series.iloc[mapped.isnull()].astype(str).tolist()))
        raise ValueError(f"Found invalid emotion labels: {invalid}")
    return mapped.to_numpy(dtype=np.int64)


def prepare_dataset(tokenizer, texts, labels):
    """Tokenize a list of texts and create a PyTorch dataset."""
    encodings = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    return EmotionDataset(encodings, labels)


def train_model(model, train_loader, val_loader, class_weights, device):
    """Fine-tune the BERT classifier and keep the best validation checkpoint."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))

    best_state = None
    best_metrics = None

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits.view(-1, model.config.num_labels), labels.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.item())

        avg_loss = total_loss / max(1, len(train_loader))
        val_metrics, _, _ = evaluate_model(model, val_loader, device)
        logger.info(
            "Epoch %d/%d finished; avg_loss=%.4f; val_accuracy=%.4f; val_f1=%.4f",
            epoch + 1,
            EPOCHS,
            avg_loss,
            val_metrics["accuracy"],
            val_metrics["f1"],
        )

        if best_metrics is None or val_metrics["accuracy"] > best_metrics["accuracy"] + 1e-6:
            best_metrics = val_metrics
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()


def evaluate_model(model, test_loader, device):
    """Run inference on the validation/test set and return metrics."""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].cpu().tolist()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = outputs.logits.argmax(dim=-1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels)

    preds = np.asarray(all_preds, dtype=int)
    labels = np.asarray(all_labels, dtype=int)
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, average="weighted", zero_division=0),
        "recall": recall_score(labels, preds, average="weighted", zero_division=0),
        "f1": f1_score(labels, preds, average="weighted", zero_division=0),
    }, preds, labels


def main():
    print("[BERT Training] main() started", flush=True)
    logger.info("Loading data from: %s", DATA_DIR)
    train_path = DATA_DIR / "train.csv"
    test_path = DATA_DIR / "test.csv"

    train_df = load_csv(train_path)
    test_df = load_csv(test_path)

    required_columns = ["text", "emotion"]
    for col in required_columns:
        if col not in train_df.columns or col not in test_df.columns:
            raise KeyError(f"Missing required column '{col}' in train/test CSV")

    train_df = train_df.dropna(subset=required_columns).reset_index(drop=True)
    test_df = test_df.dropna(subset=required_columns).reset_index(drop=True)

    from utils.preprocessing import preprocess_text

    train_df = augment_training_data_with_seed_examples(train_df)
    train_df = balance_training_data(train_df)
    if len(train_df) > MAX_TRAIN_SAMPLES:
        train_df = train_df.sample(n=MAX_TRAIN_SAMPLES, random_state=42).reset_index(drop=True)
    train_df["emotion"] = train_df["emotion"].astype(str).str.strip().str.lower()
    train_df = pd.concat([
        train_df[train_df["emotion"] == label.lower()].sample(
            n=min(len(train_df[train_df["emotion"] == label.lower()]), TARGET_BALANCED_COUNT),
            random_state=42,
        )
        for label in LABELS
    ], ignore_index=True)
    train_df["text"] = train_df["text"].astype(str).apply(preprocess_text)
    test_df["text"] = test_df["text"].astype(str).apply(preprocess_text)

    train_df, val_df = train_test_split(
        train_df,
        test_size=VAL_FRACTION,
        random_state=42,
        stratify=train_df["emotion"],
    )
    val_df = val_df.reset_index(drop=True)
    train_df = train_df.reset_index(drop=True)

    logger.info("Mapping labels to integers...")
    y_train = map_labels(train_df["emotion"])
    y_val = map_labels(val_df["emotion"])
    y_test = map_labels(test_df["emotion"])
    class_weights = get_class_weights(y_train)

    logger.info("Loading BERT tokenizer...")
    tokenizer = BertTokenizerFast.from_pretrained(str(BASE_BERT_DIR), local_files_only=True)

    logger.info("Tokenizing training data...")
    train_dataset = prepare_dataset(tokenizer, train_df["text"].astype(str).tolist(), y_train)
    logger.info("Tokenizing validation data...")
    val_dataset = prepare_dataset(tokenizer, val_df["text"].astype(str).tolist(), y_val)
    logger.info("Tokenizing test data...")
    test_dataset = prepare_dataset(tokenizer, test_df["text"].astype(str).tolist(), y_test)

    logger.info("Loading BERT model for sequence classification...")
    model_config = BertConfig.from_pretrained(str(BASE_BERT_DIR), local_files_only=True, num_labels=len(LABELS))
    model_config.num_labels = len(LABELS)
    model_config.problem_type = "single_label_classification"
    model_config.id2label = {i: label for i, label in enumerate(LABELS)}
    model_config.label2id = {label: i for i, label in enumerate(LABELS)}
    model = BertForSequenceClassification.from_pretrained(
        str(BASE_BERT_DIR),
        local_files_only=True,
        config=model_config,
    )
    if getattr(model.classifier, "out_features", None) != len(LABELS):
        raise ValueError("BERT classifier head shape mismatch; retraining aborted.")

    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    logger.info("Starting fine-tuning...")
    train_model(model, train_loader, val_loader, class_weights, device)

    logger.info("Evaluating on test set...")
    metrics, pred_ids, y_true = evaluate_model(model, test_loader, device)
    logger.info("Eval metrics: %s", metrics)
    logger.info("Classification report:\n%s", classification_report(y_true, pred_ids, target_names=LABELS))
    logger.info("Confusion matrix:\n%s", confusion_matrix(y_true, pred_ids))

    logger.info("Saving the final model and tokenizer...")
    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)

    # Save label mapping and metadata
    # Save label mapping and metadata
    try:
        per_class_target = int(train_df["emotion"].value_counts().iloc[0])
    except Exception:
        per_class_target = None

    metadata = {
        "labels": LABEL_MAP,
        "max_length": MAX_LENGTH,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "per_class_target": per_class_target,
        "training_date": pd.Timestamp.utcnow().isoformat(),
    }
    with open(MODEL_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Saved model and tokenizer to: %s", MODEL_DIR)


if __name__ == "__main__":
    main()
