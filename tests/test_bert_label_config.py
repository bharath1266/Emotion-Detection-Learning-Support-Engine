import json
from pathlib import Path

from utils.config import LABELS


def test_saved_bert_config_matches_expected_emotion_labels():
    config_path = Path("models/bert_emotion_model_final/config.json")
    assert config_path.exists(), "BERT model config should exist"

    config = json.loads(config_path.read_text(encoding="utf-8"))

    num_labels = config.get("num_labels", len(LABELS))
    assert num_labels == len(LABELS)
    assert config.get("id2label", {}) == {str(i): label for i, label in enumerate(LABELS)}
    assert config.get("label2id", {}) == {label: i for i, label in enumerate(LABELS)}
