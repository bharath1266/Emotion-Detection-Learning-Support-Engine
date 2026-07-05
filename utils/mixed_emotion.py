"""Utilities to detect mixed emotions from a probability vector."""
from typing import List, Optional, Dict, Any
import numpy as np


def get_mixed_emotions(probs: List[float] | np.ndarray, labels: Optional[List[str]] = None, threshold: float = 0.15) -> Dict[str, Any]:
    """Return primary and secondary emotion information.

    Args:
        probs: 1-D list or numpy array of class probabilities (length = n_classes).
        labels: optional list of label names corresponding to indices.
        threshold: minimum probability for secondary emotion to be considered mixed.

    Returns a dict with keys:
        - primary: label or index of primary emotion
        - primary_prob: probability of primary emotion
        - secondary: label or index of secondary emotion, or None
        - secondary_prob: probability of secondary emotion, or None
        - mixed: bool indicating whether secondary >= threshold
    """
    probs_arr = np.asarray(probs, dtype=float)
    if probs_arr.ndim != 1:
        raise ValueError("probs must be a 1-D array of class probabilities")

    # Sort indices from highest to lowest probability
    idx = np.argsort(probs_arr)[::-1]
    primary_idx = int(idx[0])
    secondary_idx = int(idx[1])

    primary_prob = float(probs_arr[primary_idx])
    secondary_prob = float(probs_arr[secondary_idx])

    primary = labels[primary_idx] if labels is not None else primary_idx
    secondary = labels[secondary_idx] if labels is not None else secondary_idx

    mixed_flag = secondary_prob >= threshold

    return {
        "primary": primary,
        "primary_prob": primary_prob,
        "secondary": secondary if mixed_flag else None,
        "secondary_prob": secondary_prob if mixed_flag else None,
        "mixed": mixed_flag,
    }
