from utils.bert_predict import predict_proba
from utils.config import LABELS


def test_target_sentences_are_not_predicted_as_neutral():
    samples = [
        ("I got selected for my dream internship.", "Happy"),
        ("I failed my exam and feel disappointed.", "Sad"),
        ("Nobody listens to my ideas.", "Angry"),
        ("I'm scared I might fail tomorrow's exam.", "Fear"),
        ("Today I attended classes and completed my assignments.", "Neutral"),
    ]

    for text, expected in samples:
        probs = predict_proba(text)
        assert probs is not None
        prediction = LABELS[int(probs.argmax())]
        assert prediction == expected
