#!/usr/bin/env python
"""Verify BERT and BiLSTM predictions with benchmark sentences."""

import sys
sys.path.insert(0, '.')

from utils.bert_predict import predict_proba as bert_predict
from utils.bilstm_predict import predict_proba as bilstm_predict
from utils.unified_predictor import predict as unified_predict
from utils.config import LABELS

test_cases = [
    ("I got selected for my dream internship.", "Happy"),
    ("I failed my exam and feel disappointed.", "Sad"),
    ("Nobody listens to my ideas.", "Angry"),
    ("I'm scared I might fail tomorrow's exam.", "Fear"),
    ("Today I attended classes and completed my assignments.", "Neutral"),
]

print("\n" + "="*80)
print("BERT PREDICTIONS")
print("="*80)
for text, expected in test_cases:
    probs = bert_predict(text)
    if probs is not None:
        pred_idx = probs.argmax()
        pred_label = LABELS[pred_idx]
        confidence = probs[pred_idx]
        match = "[PASS]" if pred_label == expected else "[FAIL]"
        print(f"{match} Text: {text[:50]}...")
        print(f"   Expected: {expected}, Got: {pred_label}, Confidence: {confidence:.2%}")
    else:
        print(f"[FAIL] BERT MODEL NOT AVAILABLE")
    print()

print("\n" + "="*80)
print("BILSTM PREDICTIONS")
print("="*80)
for text, expected in test_cases:
    probs = bilstm_predict(text)
    if probs is not None:
        pred_idx = probs.argmax()
        pred_label = LABELS[pred_idx]
        confidence = probs[pred_idx]
        match = "[PASS]" if pred_label == expected else "[FAIL]"
        print(f"{match} Text: {text[:50]}...")
        print(f"   Expected: {expected}, Got: {pred_label}, Confidence: {confidence:.2%}")
    else:
        print(f"[FAIL] BILSTM MODEL NOT AVAILABLE")
    print()

print("\n" + "="*80)
print("UNIFIED PREDICTIONS (Ensemble)")
print("="*80)
for text, expected in test_cases:
    result = unified_predict(text)
    if result:
        pred_label = result.get("final_prediction")
        confidence = result.get("confidence", 0)
        if pred_label:
            match = "[PASS]" if pred_label == expected else "[FAIL]"
            print(f"{match} Text: {text[:50]}...")
            print(f"   Expected: {expected}, Got: {pred_label}, Confidence: {confidence:.2%}")
        else:
            print(f"[FAIL] No final prediction")
    else:
        print(f"[FAIL] UNIFIED PREDICTION FAILED")
    print()
