"""
utils/unified_predictor.py

Combine predictions from BiLSTM and BERT into a unified result.
"""
from typing import Any, Dict, Optional

import numpy as np

from utils.bilstm_predict import predict_proba as bilstm_proba
from utils.bert_predict import predict_proba as bert_proba
from utils.mixed_emotion import get_mixed_emotions
from utils.config import LABELS, MODEL_WEIGHTS, MIXED_THRESHOLD
from utils.logger import get_logger

logger = get_logger(__name__)


def normalize_probabilities(probs: np.ndarray) -> np.ndarray:
    """Convert a probability vector to a normalized, non-negative distribution."""
    probs_arr = np.asarray(probs, dtype=float).reshape(-1)
    if probs_arr.size == 0:
        return np.full(len(LABELS), 1.0 / len(LABELS), dtype=float)

    if probs_arr.size < len(LABELS):
        padded = np.zeros(len(LABELS), dtype=float)
        padded[: probs_arr.size] = probs_arr
        probs_arr = padded
    elif probs_arr.size > len(LABELS):
        probs_arr = probs_arr[: len(LABELS)]

    probs_arr = np.clip(probs_arr, 0.0, None)
    probs_arr = np.nan_to_num(probs_arr, nan=0.0, posinf=0.0, neginf=0.0)
    total = float(probs_arr.sum())
    if not np.isfinite(total) or total <= 0:
        return np.full(len(LABELS), 1.0 / len(LABELS), dtype=float)
    return probs_arr / total


def predict(text: str, model_choice: str = "both") -> Dict[str, Any]:
    """Combine BiLSTM and BERT predictions into a unified result.

    Strategy:
      - Obtain probability vectors from both models when available.
      - Normalize each vector before combining.
      - Use weighted averaging with configurable weights.
      - Return both raw model probabilities, model-level predictions, a
        combined final prediction, combined confidence, and mixed-emotion info.
    """
    choice = model_choice.lower() if isinstance(model_choice, str) else "both"
    b_probs = None
    t_probs = None
    try:
        if choice in ("both", "bilstm"):
            b_probs = bilstm_proba(text)
    except Exception as e:
        logger.exception("BiLSTM prediction error: %s", e)

    try:
        if choice in ("both", "bert"):
            t_probs = bert_proba(text)
    except Exception as e:
        logger.exception("BERT prediction error: %s", e)

    result: Dict[str, Optional[Any]] = {
        "bilstm_probs": None,
        "bert_probs": None,
        "bilstm_pred": None,
        "bert_pred": None,
        "final_prediction": None,
        "confidence": None,
        "mixed": {"mixed": False},
    }

    if b_probs is not None:
        b_probs = normalize_probabilities(b_probs)
        result["bilstm_probs"] = b_probs.tolist()
        if b_probs.size > 0:
            result["bilstm_pred"] = LABELS[int(b_probs.argmax())]

    if t_probs is not None:
        t_probs = normalize_probabilities(t_probs)
        result["bert_probs"] = t_probs.tolist()
        if t_probs.size > 0:
            result["bert_pred"] = LABELS[int(t_probs.argmax())]

    print(f"[unified_predictor] text={text}")
    print(f"[unified_predictor] bilstm_probs={b_probs}")
    print(f"[unified_predictor] bert_probs={t_probs}")

    combined_probs = None
    if t_probs is not None and b_probs is not None:
        # Defensive weight normalization
        w_b = float(MODEL_WEIGHTS.get("bilstm", 0.5))
        w_t = float(MODEL_WEIGHTS.get("bert", 0.5))
        if not (w_b >= 0 and w_t >= 0):
            w_b, w_t = 0.5, 0.5
        total = float(w_b + w_t)
        if total <= 0:
            w_b, w_t = 0.5, 0.5
            total = 1.0
        w_b /= total
        w_t /= total
        combined_probs = w_b * b_probs + w_t * t_probs
    elif t_probs is not None:
        combined_probs = t_probs
    elif b_probs is not None:
        combined_probs = b_probs

    if combined_probs is not None:
        combined_probs = normalize_probabilities(combined_probs)
        print(f"[unified_predictor] combined_probs={combined_probs}")
        if combined_probs.size == 0:
            return result
        final_idx = int(combined_probs.argmax())
        final_conf = float(combined_probs[final_idx])

        # Prefer a strong non-neutral emotional prediction over a low-confidence neutral one.
        if b_probs is not None and t_probs is not None:
            b_idx = int(np.argmax(b_probs))
            t_idx = int(np.argmax(t_probs))
            b_conf = float(b_probs[b_idx])
            t_conf = float(t_probs[t_idx])
            if LABELS[b_idx] == "Neutral" and b_conf < 0.45 and LABELS[t_idx] != "Neutral" and t_conf > b_conf + 0.15:
                final_idx = t_idx
                final_conf = t_conf
            elif LABELS[t_idx] == "Neutral" and t_conf < 0.45 and LABELS[b_idx] != "Neutral" and b_conf > t_conf + 0.15:
                final_idx = b_idx
                final_conf = b_conf
            else:
                sorted_idx = list(np.argsort(combined_probs)[::-1])
                top1, top2 = sorted_idx[0], sorted_idx[1] if len(sorted_idx) > 1 else None
                tie_delta = 0.05
                if top2 is not None and (combined_probs[top1] - combined_probs[top2]) < tie_delta:
                    prefer = None
                    try:
                        b_conf_top1 = float(b_probs[top1]) if b_probs is not None else -1.0
                        t_conf_top1 = float(t_probs[top1]) if t_probs is not None else -1.0
                        if b_conf_top1 > t_conf_top1 + 1e-6:
                            prefer = "bilstm"
                        elif t_conf_top1 > b_conf_top1 + 1e-6:
                            prefer = "bert"
                    except Exception:
                        prefer = None

                    if prefer == "bilstm" and b_probs is not None:
                        final_idx = int(np.argmax(b_probs))
                        final_conf = float(b_probs[final_idx])
                    elif prefer == "bert" and t_probs is not None:
                        final_idx = int(np.argmax(t_probs))
                        final_conf = float(t_probs[final_idx])

        print(f"[unified_predictor] selected_label={LABELS[final_idx] if 0 <= final_idx < len(LABELS) else LABELS[0]}")
        if 0 <= final_idx < len(LABELS):
            result["final_prediction"] = LABELS[final_idx]
        else:
            result["final_prediction"] = LABELS[0]
        result["confidence"] = final_conf
        mixed = get_mixed_emotions(combined_probs, labels=LABELS, threshold=MIXED_THRESHOLD)
        result["mixed"] = mixed

    return result
