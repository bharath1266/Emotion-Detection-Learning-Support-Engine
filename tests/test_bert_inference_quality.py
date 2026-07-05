import numpy as np

from utils.bert_predict import predict_proba
from utils.config import LABELS


def test_bert_predictions_are_not_uniform_for_clear_sentences():
    samples = {
        "happy": "I got selected for my dream internship.",
        "sad": "I failed my exam and feel disappointed.",
        "angry": "Nobody listens to my ideas.",
        "fear": "I'm scared I might fail tomorrow's exam.",
        "neutral": "Today I attended classes and completed my assignments.",
    }

    for expected, text in samples.items():
        probs = predict_proba(text)
        assert probs is not None
        assert probs.shape == (len(LABELS),)
        assert not np.allclose(probs, np.full(len(LABELS), 1.0 / len(LABELS)))

        pred = LABELS[int(np.argmax(probs))]
        if expected != "neutral":
            assert pred != "Neutral"
        else:
            assert pred == "Neutral"
