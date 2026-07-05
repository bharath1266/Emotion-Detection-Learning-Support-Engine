"""
utils/preprocessing.py

Reusable text preprocessing helpers used by training and inference.
"""
from typing import List
import re

_CONTRACTION_MAP = {
    "can't": "cannot",
    "cant": "cannot",
    "won't": "will not",
    "wont": "will not",
    "don't": "do not",
    "dont": "do not",
    "isn't": "is not",
    "isnt": "is not",
    "aren't": "are not",
    "arent": "are not",
    "wasn't": "was not",
    "wasnt": "was not",
    "didn't": "did not",
    "didnt": "did not",
    "i'm": "i am",
    "im": "i am",
    "it's": "it is",
    "its": "it is",
    "that's": "that is",
    "thats": "that is",
    "there's": "there is",
    "theres": "there is",
    "he's": "he is",
    "hes": "he is",
    "she's": "she is",
    "shes": "she is",
}


def clean_text(text: str) -> str:
    """Normalize text for training and inference.

    The function lowercases, removes punctuation, expands common contractions,
    collapses repeated characters, and normalizes whitespace.
    """
    if text is None:
        return ""

    s = str(text).lower()
    s = s.replace("\u2019", "'").replace("\u2018", "'")
    s = s.replace("\n", " ").replace("\t", " ")

    for token, replacement in _CONTRACTION_MAP.items():
        s = re.sub(rf"\b{re.escape(token)}\b", replacement, s)

    s = re.sub(r"[^a-z0-9\s']", " ", s)
    # Reduce long character repetitions (3+) to single char
    # e.g. 'sooo' -> 'so', 'aaaa' -> 'a'
    s = re.sub(r"(\w)\1{2,}", r"\1", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def clean_texts(texts: List[str]) -> List[str]:
    """Apply `clean_text` to a list of strings."""
    return [clean_text(t) for t in texts]


def preprocess_text(text: str) -> str:
    """Shared preprocessing helper used by training and inference."""
    return clean_text(text)
