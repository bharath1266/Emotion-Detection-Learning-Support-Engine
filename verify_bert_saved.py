"""
Verify that the saved BERT model can be loaded and used for inference.
"""
import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

print("[BERT Verify] Starting verification of saved model...", flush=True)

import torch
from transformers import BertForSequenceClassification, BertTokenizerFast
from pathlib import Path

# Configuration
LABELS = ["Sadness", "Anger", "Love", "Surprise", "Fear", "Happy", "Neutral"]
MODEL_DIR = Path(PROJECT_ROOT) / "models" / "bert_emotion_model_final"
MAX_LENGTH = 128

print(f"[BERT Verify] Loading model from {MODEL_DIR}", flush=True)

try:
    print("[BERT Verify] Loading saved tokenizer...", flush=True)
    tokenizer = BertTokenizerFast.from_pretrained(MODEL_DIR)
    print("[BERT Verify] Tokenizer loaded successfully", flush=True)

    print("[BERT Verify] Loading saved BERT model...", flush=True)
    model = BertForSequenceClassification.from_pretrained(MODEL_DIR)
    print("[BERT Verify] Model loaded successfully", flush=True)

    # Set model to evaluation mode
    model.eval()

    print("\n[BERT Verify] Running inference on test sentences:\n", flush=True)

    test_sentences = [
        "I am very happy today",
        "I feel sad",
        "I am angry",
        "I am scared",
        "Today is a normal day",
    ]

    with torch.no_grad():
        for sentence in test_sentences:
            inputs = tokenizer(
                sentence,
                truncation=True,
                padding="max_length",
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )
            outputs = model(**inputs)
            logits = outputs.logits
            pred_id = torch.argmax(logits, dim=-1).item()
            pred_label = LABELS[pred_id]
            confidence = torch.softmax(logits, dim=-1)[0][pred_id].item()
            print(f"Sentence: {sentence}", flush=True)
            # Use plain ASCII arrows to avoid encoding issues on some Windows consoles
            print(f"  -> Predicted Label: {pred_label}", flush=True)
            print(f"  -> Confidence: {confidence:.4f}\n", flush=True)

    print("[BERT Verify] SUCCESS: Saved model verification passed!", flush=True)

except Exception as e:
    print(f"[BERT Verify] ERROR: {type(e).__name__}: {e}", flush=True)
    import traceback

    traceback.print_exc()
    sys.exit(1)

