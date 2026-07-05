from utils.mixed_emotion import get_mixed_emotions
import numpy as np


def test_mixed_none():
    probs = np.array([0.9, 0.05, 0.02, 0.02, 0.01])
    out = get_mixed_emotions(probs, labels=["A","B","C","D","E"], threshold=0.15)
    assert out["mixed"] is False


def test_mixed_true():
    probs = np.array([0.6, 0.25, 0.05, 0.05, 0.05])
    out = get_mixed_emotions(probs, labels=["A","B","C","D","E"], threshold=0.15)
    assert out["mixed"] is True
    assert out["primary"] == "A"
    assert out["secondary"] == "B"
