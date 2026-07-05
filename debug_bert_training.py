"""
Diagnostic script to verify BERT training pipeline.
Shows data stats, trains model, compares weights before/after, and tests inference.
"""
import sys
import os
from pathlib import Path

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

print("[TRAIN DEBUG] Starting diagnostic...", flush=True)

import numpy as np
import pandas as pd
import torch
import logging
from collections import Counter
from transformers import BertForSequenceClassification, BertTokenizerFast, Trainer, TrainingArguments, EarlyStoppingCallback
from torch.utils.data import Dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from utils.config import DATA_DIR, BERT_MODEL_DIR, LABEL_MAP, BERT_MAX_LEN, LABELS
from utils.preprocessing import clean_text

print(f"[TRAIN DEBUG] LABEL_MAP: {LABEL_MAP}", flush=True)
print(f"[TRAIN DEBUG] LABELS: {LABELS}", flush=True)

# Configuration
MODEL_DIR = BERT_MODEL_DIR
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MAX_LENGTH = int(BERT_MAX_LEN)
BATCH_SIZE = 8  # Reduced for testing
EPOCHS = 2  # Reduced for quick test
LEARNING_RATE = 2e-5

class EmotionDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

# Load data
print("\n[TRAIN DEBUG] Loading training data...", flush=True)
train_path = DATA_DIR / "train.csv"
test_path = DATA_DIR / "test.csv"

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

print(f"[TRAIN DEBUG] Train CSV shape: {train_df.shape}", flush=True)
print(f"[TRAIN DEBUG] Test CSV shape: {test_df.shape}", flush=True)
print(f"[TRAIN DEBUG] Train columns: {train_df.columns.tolist()}", flush=True)

# Clean and prepare data
train_df = train_df.dropna(subset=["text", "emotion"]).reset_index(drop=True)
test_df = test_df.dropna(subset=["text", "emotion"]).reset_index(drop=True)

print(f"[TRAIN DEBUG] After dropping NaN - Train: {len(train_df)}, Test: {len(test_df)}", flush=True)

train_df["text"] = train_df["text"].astype(str).apply(clean_text)
test_df["text"] = test_df["text"].astype(str).apply(clean_text)

# Map labels
train_df["emotion"] = train_df["emotion"].astype(str).str.strip().str.lower()
test_df["emotion"] = test_df["emotion"].astype(str).str.strip().str.lower()

y_train = train_df["emotion"].map(LABEL_MAP).to_numpy(dtype=np.int64)
y_test = test_df["emotion"].map(LABEL_MAP).to_numpy(dtype=np.int64)

print(f"\n[TRAIN DEBUG] Training samples: {len(y_train)}", flush=True)
print(f"[TRAIN DEBUG] Test samples: {len(y_test)}", flush=True)

# Label distribution
train_counts = Counter(y_train)
print(f"\n[TRAIN DEBUG] Training label distribution:", flush=True)
for label_id in sorted(train_counts.keys()):
    label_name = LABELS[label_id]
    count = train_counts[label_id]
    pct = 100 * count / len(y_train)
    print(f"  {label_name}: {count} ({pct:.1f}%)", flush=True)

test_counts = Counter(y_test)
print(f"\n[TRAIN DEBUG] Test label distribution:", flush=True)
for label_id in sorted(test_counts.keys()):
    label_name = LABELS[label_id]
    count = test_counts[label_id]
    pct = 100 * count / len(y_test)
    print(f"  {label_name}: {count} ({pct:.1f}%)", flush=True)

# Load tokenizer
print("\n[TRAIN DEBUG] Loading tokenizer...", flush=True)
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

# Tokenize
print("[TRAIN DEBUG] Tokenizing data...", flush=True)
train_encodings = tokenizer(
    train_df["text"].astype(str).tolist(),
    truncation=True,
    padding="max_length",
    max_length=MAX_LENGTH,
    return_tensors="pt"
)
test_encodings = tokenizer(
    test_df["text"].astype(str).tolist(),
    truncation=True,
    padding="max_length",
    max_length=MAX_LENGTH,
    return_tensors="pt"
)

train_dataset = EmotionDataset(train_encodings, y_train)
test_dataset = EmotionDataset(test_encodings, y_test)

# Load model
print("\n[TRAIN DEBUG] Loading BERT model...", flush=True)
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(LABEL_MAP))

# Save weights BEFORE training
print("[TRAIN DEBUG] Capturing model weights before training...", flush=True)
weights_before = {}
for name, param in model.named_parameters():
    if "classifier" in name:  # Capture classifier weights (newly initialized)
        weights_before[name] = param.data.clone()
        print(f"  {name}: shape={param.shape}, mean={param.data.mean():.6f}, std={param.data.std():.6f}", flush=True)

# Training
print(f"\n[TRAIN DEBUG] Starting training for {EPOCHS} epochs...", flush=True)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = (predictions == labels).mean()
    return {"accuracy": accuracy}

training_args = TrainingArguments(
    output_dir=str(MODEL_DIR),
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="steps",
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    save_total_limit=1,
    seed=42,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
    processing_class=tokenizer,
)
trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=2))

# Train
print("[TRAIN DEBUG] Calling trainer.train()...", flush=True)
train_result = trainer.train()
print(f"[TRAIN DEBUG] Training result: {train_result}", flush=True)

# Evaluate
print("\n[TRAIN DEBUG] Evaluating on test set...", flush=True)
eval_result = trainer.evaluate()
print(f"[TRAIN DEBUG] Eval metrics: {eval_result}", flush=True)

# Save model
print(f"\n[TRAIN DEBUG] Saving model to {MODEL_DIR}...", flush=True)
trainer.save_model(MODEL_DIR)
tokenizer.save_pretrained(MODEL_DIR)
print("[TRAIN DEBUG] Model and tokenizer saved", flush=True)

# Compare weights AFTER training
print("\n[TRAIN DEBUG] Comparing model weights after training...", flush=True)
weights_changed = False
for name, param in model.named_parameters():
    if name in weights_before:
        weights_after = param.data
        diff = (weights_after - weights_before[name]).abs().mean()
        print(f"  {name}: mean_change={diff:.6f}", flush=True)
        if diff > 1e-6:
            weights_changed = True

if weights_changed:
    print("[TRAIN DEBUG] ✓ Weights CHANGED during training", flush=True)
else:
    print("[TRAIN DEBUG] ✗ ERROR: Weights did NOT change! Training may have failed.", flush=True)

# Test inference on saved model
print("\n[TRAIN DEBUG] Testing inference on saved model...", flush=True)
print("[TRAIN DEBUG] Loading model from disk...", flush=True)

loaded_model = BertForSequenceClassification.from_pretrained(MODEL_DIR)
loaded_tokenizer = BertTokenizerFast.from_pretrained(MODEL_DIR)
loaded_model.eval()

test_sentences = [
    "I am very happy today",
    "I feel sad",
    "I am angry",
    "I am scared",
    "Today is a normal day"
]

print("\n[TRAIN DEBUG] Inference results:\n", flush=True)
with torch.no_grad():
    for sentence in test_sentences:
        inputs = loaded_tokenizer(
            sentence,
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )
        outputs = loaded_model(**inputs)
        logits = outputs.logits
        pred_id = torch.argmax(logits, dim=-1).item()
        pred_label = LABELS[pred_id]
        confidence = torch.softmax(logits, dim=-1)[0][pred_id].item()
        print(f"'{sentence}'", flush=True)
        print(f"  → {pred_label} ({confidence:.4f})\n", flush=True)

print("[TRAIN DEBUG] Diagnostic complete!", flush=True)
