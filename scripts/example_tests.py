"""Run example input tests against the unified predictor and print PASS/FAIL.

Usage: python scripts/example_tests.py
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.unified_predictor import predict
from utils.config import LABELS

TESTS = [
    ("I am very happy today", "Happy"),
    ("I feel extremely sad.", "Sad"),
    ("I am angry with everyone.", "Angry"),
    ("I am scared about tomorrow.", "Fear"),
    ("Today is a normal day.", "Neutral"),
]


def run_tests():
    passed = 0
    for text, expected in TESTS:
        res = predict(text)
        pred = res.get("final_prediction")
        conf = res.get("confidence")
        ok = str(pred).lower() == str(expected).lower()
        status = "PASS" if ok else "FAIL"
        if ok:
            passed += 1
        print(f"Input: {text}")
        print(f"Expected: {expected}")
        print(f"Predicted: {pred}")
        print(f"Confidence: {conf}")
        print(f"{status}\n")

    print(f"Summary: {passed}/{len(TESTS)} passed")


if __name__ == '__main__':
    run_tests()
