"""
Quick BERT loading test
"""
import sys
import os
os.chdir(r'c:\Users\bablu\OneDrive\Documents\bharath')
sys.path.insert(0, '.')

import torch
import numpy as np
from transformers import BertForSequenceClassification, BertTokenizerFast
from pathlib import Path

MODEL_DIR = Path(r'c:\Users\bablu\OneDrive\Documents\bharath\models\bert_emotion_model_final')
LABELS = ["Happy", "Sad", "Angry", "Fear", "Neutral"]

print(f"Model dir exists: {MODEL_DIR.exists()}")
print(f"Files: {list(MODEL_DIR.glob('*'))}")

# Load the model
model = BertForSequenceClassification.from_pretrained(MODEL_DIR, local_files_only=True)
tokenizer = BertTokenizerFast.from_pretrained(MODEL_DIR)

print(f"Model loaded. Num labels: {model.config.num_labels}")
print(f"Classifier out_features: {model.classifier.out_features}")
print(f"Classifier weights shape: {model.classifier.weight.shape}")
print(f"Classifier bias: {model.classifier.bias[:5]}")

# Test inference
text = "I got selected for my dream internship."
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
with torch.inference_mode():
    outputs = model(**inputs)
    logits = outputs.logits
    probs = torch.softmax(logits, dim=-1).numpy()[0]

print(f"\nText: {text}")
print(f"Probabilities: {dict(zip(LABELS, probs))}")
print(f"Prediction: {LABELS[probs.argmax()]} ({probs.max():.2%})")

    print("[BERT Test] Loading tokenizer from bert-base-uncased...", flush=True)
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    print("[BERT Test] Tokenizer loaded successfully", flush=True)
    
    print("[BERT Test] Loading BERT model...", flush=True)
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(LABEL_MAP))
    print("[BERT Test] BERT model loaded successfully", flush=True)
    
    print("[BERT Test] Saving model and tokenizer...", flush=True)
    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)
    print(f"[BERT Test] Model and tokenizer saved to {MODEL_DIR}", flush=True)
    
    # List saved files
    print("[BERT Test] Files saved:", flush=True)
    for f in sorted(MODEL_DIR.glob("*")):
        print(f"  - {f.name}", flush=True)
    
    print("[BERT Test] Testing inference...", flush=True)
    test_sentences = [
        "I am very happy today",
        "I feel sad",
        "I am angry",
        "I am scared",
        "Today is a normal day"
    ]
    
    # Set model to evaluation mode
    model.eval()
    
    with torch.no_grad():
        for sentence in test_sentences:
            print(f"\n[BERT Test] Sentence: {sentence}", flush=True)
            inputs = tokenizer(
                sentence,
                truncation=True,
                padding="max_length",
                max_length=MAX_LENGTH,
                return_tensors="pt"
            )
            outputs = model(**inputs)
            logits = outputs.logits
            pred_id = torch.argmax(logits, dim=-1).item()
            pred_label = LABELS[pred_id]
            confidence = torch.softmax(logits, dim=-1)[0][pred_id].item()
            print(f"  -> Predicted: {pred_label} (confidence: {confidence:.4f})", flush=True)
    
    print("\n[BERT Test] SUCCESS: All tests passed!", flush=True)
    
except Exception as e:
    print(f"[BERT Test] ERROR: {type(e).__name__}: {e}", flush=True)
    import traceback
    traceback.print_exc()
    sys.exit(1)
