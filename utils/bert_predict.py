"""utils/bert_predict.py

Provides a cached loader and a `predict_proba` function for a fine-tuned
HuggingFace BERT sequence classification model.
"""
from typing import List, Optional, Tuple, Union
import numpy as np
import torch

from utils.config import BERT_MODEL_DIR, BERT_MAX_LEN, LABELS
from utils.preprocessing import preprocess_text

try:
    from transformers import BertConfig, BertTokenizerFast, BertForSequenceClassification
except Exception:  # pragma: no cover - environment dependent
    BertConfig = None  # type: ignore[assignment]
    BertTokenizerFast = None  # type: ignore[assignment]
    BertForSequenceClassification = None  # type: ignore[assignment]

_MODEL = None
_TOKENIZER = None


def _load_bert() -> Tuple[Optional[torch.nn.Module], Optional[BertTokenizerFast]]:
    """Load and cache the BERT model and tokenizer."""
    global _MODEL, _TOKENIZER
    if _MODEL is not None and _TOKENIZER is not None:
        return _MODEL, _TOKENIZER

    try:
        if not BERT_MODEL_DIR.exists():
            return None, None
        if BertTokenizerFast is None or BertForSequenceClassification is None or BertConfig is None:
            return None, None

        _TOKENIZER = BertTokenizerFast.from_pretrained(BERT_MODEL_DIR, local_files_only=True)
        model_config = BertConfig.from_pretrained(BERT_MODEL_DIR, local_files_only=True)
        model_config.num_labels = len(LABELS)
        model_config.problem_type = "single_label_classification"
        model_config.id2label = {i: label for i, label in enumerate(LABELS)}
        model_config.label2id = {label: i for i, label in enumerate(LABELS)}
        _MODEL = BertForSequenceClassification.from_pretrained(
            BERT_MODEL_DIR,
            local_files_only=True,
            config=model_config,
        )
        _MODEL.config.num_labels = len(LABELS)
        _MODEL.config.id2label = {i: label for i, label in enumerate(LABELS)}
        _MODEL.config.label2id = {label: i for i, label in enumerate(LABELS)}
        if getattr(_MODEL.classifier, "out_features", None) != len(LABELS):
            raise ValueError("BERT classifier head mismatch; checkpoint is invalid for inference.")
        _MODEL.eval()
        # Move model to appropriate device for inference
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            _MODEL.to(device)
        except Exception:
            pass
        return _MODEL, _TOKENIZER
    except Exception:
        _MODEL, _TOKENIZER = None, None
        return None, None


def _align_probabilities(probs: np.ndarray) -> np.ndarray:
    probs_arr = np.asarray(probs, dtype=float)
    if probs_arr.ndim == 1:
        if probs_arr.size == len(LABELS):
            return probs_arr
        if probs_arr.size < len(LABELS):
            padded = np.zeros(len(LABELS), dtype=float)
            padded[: probs_arr.size] = probs_arr
            probs_arr = padded
        else:
            probs_arr = probs_arr[: len(LABELS)]
        total = float(probs_arr.sum())
        if not np.isfinite(total) or total <= 0:
            return np.full(len(LABELS), 1.0 / len(LABELS), dtype=float)
        return probs_arr / total

    if probs_arr.ndim == 2:
        aligned = []
        for row in probs_arr:
            aligned.append(_align_probabilities(row))
        return np.vstack(aligned)

    return probs_arr


def predict_proba(texts: Union[str, List[str]]) -> Optional[np.ndarray]:
    """Predict probabilities for a single text or list of texts."""
    model, tokenizer = _load_bert()
    if model is None or tokenizer is None:
        return None

    single = False
    if isinstance(texts, str):
        texts = [texts]
        single = True

    cleaned_texts = [preprocess_text(t) for t in texts]
    enc = tokenizer(
        cleaned_texts,
        truncation=True,
        padding="max_length",
        max_length=int(BERT_MAX_LEN),
        return_tensors="pt",
    )
    # Move inputs to same device as model
    device = next(model.parameters()).device if hasattr(model, "parameters") else None
    if device is not None:
        enc = {k: v.to(device) for k, v in enc.items()}

    with torch.inference_mode():
        outputs = model(**enc)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()

    probs = _align_probabilities(probs[0] if single else probs)
    return probs
