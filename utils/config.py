from pathlib import Path
from typing import Dict, List

# Central configuration for labels, paths and model weights

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR: Path = ROOT / "data"
MODELS_DIR: Path = ROOT / "models"

LABELS: List[str] = ["Happy", "Sad", "Angry", "Fear", "Neutral"]
LABEL_MAP: Dict[str, int] = {"happy": 0, "sad": 1, "angry": 2, "fear": 3, "neutral": 4}

# Model file locations
BILSTM_MODEL_PATH: Path = MODELS_DIR / "bltsm" / "bilstm_emotion_model.h5"
BILSTM_TOKENIZER_PATH: Path = MODELS_DIR / "bltsm" / "tokenizer.pkl"

BERT_MODEL_DIR: Path = MODELS_DIR / "bert_emotion_model_final"

# Inference settings
BILSTM_MAX_LEN: int = 100
BERT_MAX_LEN: int = 128

# Weights to combine model probabilities. Sum should be 1.0 when both models available.
MODEL_WEIGHTS: Dict[str, float] = {"bilstm": 0.4, "bert": 0.6}

# Mixed emotion threshold (15% as requirement)
MIXED_THRESHOLD: float = 0.15
