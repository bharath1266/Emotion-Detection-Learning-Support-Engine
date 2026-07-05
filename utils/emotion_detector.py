"""
utils/emotion_detector.py

Provides a compatibility `detect_emotion(text)` function. If trained models
are available, it uses the unified predictor to return a model-based result.
Otherwise it falls back to the simple rule-based keyword detector for quick
responses.
"""
from typing import Dict

def _rule_based(text: str) -> str:
    text = text.lower()

    emotions = {
        "Happy": ["happy", "good", "great", "excited", "joy", "awesome", "amazing", "love"],
        "Sad": ["sad", "lonely", "depressed", "cry", "down", "upset", "hurt"],
        "Angry": ["angry", "mad", "frustrated", "annoyed", "irritated"],
        "Fear": ["stress", "stressed", "fear", "anxious", "worried", "nervous"],
    }

    scores = {k: sum(w in text for w in v) for k, v in emotions.items()}
    max_emotion = max(scores, key=scores.get)
    if scores[max_emotion] == 0:
        return "Neutral"
    return max_emotion


def detect_emotion(text: str) -> Dict:
    """Return a dictionary with model and rule-based predictions.

    If the unified predictor is available it will be used. Otherwise the
    function returns a simple rule-based result for compatibility.

    Returns a dict with keys:
      - final_prediction: str
      - confidence: float or None
      - bilstm_pred, bert_pred: optional
      - mixed: optional
    """
    try:
        from utils.unified_predictor import predict
        res = predict(text)
        return res
    except Exception:
        # Fallback to rule-based single emotion
        label = _rule_based(text)
        return {
            "final_prediction": label,
            "confidence": None,
            "bilstm_pred": None,
            "bert_pred": None,
            "mixed": {"mixed": False},
        }