"""utils/bilstm_predict.py

Provides a cached loader for a trained Keras BiLSTM model and tokenizer,
and a simple `predict_proba` function that returns class probabilities.
"""
from typing import List, Optional, Tuple, Union
import pickle
import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from utils.config import BILSTM_MODEL_PATH, BILSTM_TOKENIZER_PATH, BILSTM_MAX_LEN
from utils.preprocessing import clean_text
from utils.logger import get_logger

logger = get_logger(__name__)

_MODEL = None
_TOKENIZER = None


def _load_bilstm() -> Tuple[Optional[object], Optional[object]]:
    """Load and return the BiLSTM model and tokenizer."""
    global _MODEL, _TOKENIZER
    if _MODEL is not None and _TOKENIZER is not None:
        return _MODEL, _TOKENIZER

    try:
        if not BILSTM_MODEL_PATH.exists() or not BILSTM_TOKENIZER_PATH.exists():
            logger.info("BiLSTM model or tokenizer not found at configured paths.")
            return None, None

        _MODEL = load_model(BILSTM_MODEL_PATH)
        with open(BILSTM_TOKENIZER_PATH, "rb") as f:
            _TOKENIZER = pickle.load(f)
        logger.info("Loaded BiLSTM model and tokenizer.")
        return _MODEL, _TOKENIZER
    except Exception as e:
        logger.exception("Failed to load BiLSTM model/tokenizer: %s", e)
        _MODEL, _TOKENIZER = None, None
        return None, None


def predict_proba(texts: Union[str, List[str]]) -> Optional[np.ndarray]:
    """Predict class probabilities for input text(s)."""
    model, tokenizer = _load_bilstm()
    if model is None or tokenizer is None:
        return None

    single = False
    if isinstance(texts, str):
        texts = [texts]
        single = True

    cleaned_texts = [clean_text(t) for t in texts]
    seqs = tokenizer.texts_to_sequences(cleaned_texts)
    x = pad_sequences(seqs, maxlen=int(BILSTM_MAX_LEN), padding="post", truncating="post")

    try:
        probs = model.predict(x, verbose=0)
    except Exception as e:
        logger.exception("Error during BiLSTM prediction: %s", e)
        return None
    probs = np.asarray(probs)

    return probs[0] if single else probs
