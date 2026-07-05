"""training/verify_models.py

Small utilities to verify that required model artifacts exist and are loadable.

Usage:
    python training/verify_models.py

This script checks for:
 - BiLSTM model file and tokenizer
 - BERT model directory and tokenizer
"""
from pathlib import Path
import traceback
import pickle

from utils.config import BILSTM_MODEL_PATH, BILSTM_TOKENIZER_PATH, BERT_MODEL_DIR


def verify_bilstm():
    print("Verifying BiLSTM artifacts...")
    ok = True
    if not BILSTM_MODEL_PATH.exists():
        print(f"Missing BiLSTM model: {BILSTM_MODEL_PATH}")
        ok = False
    if not BILSTM_TOKENIZER_PATH.exists():
        print(f"Missing BiLSTM tokenizer: {BILSTM_TOKENIZER_PATH}")
        ok = False
    else:
        try:
            with open(BILSTM_TOKENIZER_PATH, "rb") as f:
                _ = pickle.load(f)
        except Exception:
            print("Failed to load BiLSTM tokenizer")
            traceback.print_exc()
            ok = False

    print("BiLSTM OK" if ok else "BiLSTM NOT OK")
    return ok


def verify_bert():
    print("Verifying BERT artifacts...")
    ok = True
    if not BERT_MODEL_DIR.exists():
        print(f"Missing BERT model dir: {BERT_MODEL_DIR}")
        ok = False
    else:
        # Check for tokenizer and config files
        expected = ["config.json", "pytorch_model.bin", "tokenizer.json"]
        found = any((BERT_MODEL_DIR / p).exists() for p in expected)
        if not found:
            print("BERT model directory missing expected files (config/tokenizer/bin)")
            ok = False

    print("BERT OK" if ok else "BERT NOT OK")
    return ok


def main():
    b1 = verify_bilstm()
    b2 = verify_bert()
    if b1 and b2:
        print("All models verified.")
    else:
        print("Some model artifacts are missing or invalid. See messages above.")


if __name__ == "__main__":
    main()
