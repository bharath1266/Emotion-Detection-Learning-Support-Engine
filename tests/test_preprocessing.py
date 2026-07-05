import numpy as np

from utils.preprocessing import clean_text, clean_texts
from utils.unified_predictor import normalize_probabilities


def test_clean_text_basic():
    assert clean_text("Hello  WORLD\n") == "hello world"


def test_clean_texts_list():
    inputs = ["Hi\n", None, "  Test  "]
    out = clean_texts(inputs)
    assert out[0] == "hi"
    assert out[1] == ""
    assert out[2] == "test"


def test_clean_text_normalizes_punctuation_and_repeated_chars():
    assert clean_text("I'm sooo happy!!!") == "i am so happy"
    assert clean_text("can't believe it!!!") == "cannot believe it"


def test_normalize_probabilities_sums_to_one():
    probs = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    normalized = normalize_probabilities(probs)
    assert np.isclose(normalized.sum(), 1.0)
    assert np.all(normalized >= 0)
