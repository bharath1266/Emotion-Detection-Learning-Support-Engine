import os
import sys
import traceback
import logging

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.unified_predictor import predict
from utils.bilstm_predict import predict_proba as bilstm_predict
from utils.bert_predict import predict_proba as bert_predict
from utils.gemini_helper import get_support_response
from utils.history_logger import load_history

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_checks():
    text = "I'm really anxious about the exam and can't concentrate."
    checks = []
    try:
        res = predict(text)
        checks.append(("unified predictor", True, res))
    except Exception as exc:
        checks.append(("unified predictor", False, str(exc)))

    try:
        bilstm_predict(text)
        checks.append(("BiLSTM predictor", True, "ok"))
    except Exception as exc:
        checks.append(("BiLSTM predictor", False, str(exc)))

    try:
        bert_predict(text)
        checks.append(("BERT predictor", True, "ok"))
    except Exception as exc:
        checks.append(("BERT predictor", False, str(exc)))

    try:
        advice = get_support_response("sad", None, 0.8, text, field="education")
        checks.append(("Gemini helper", True, advice[:60]))
    except Exception as exc:
        checks.append(("Gemini helper", False, str(exc)))

    try:
        history = load_history()
        checks.append(("history logger", True, len(history)))
    except Exception as exc:
        checks.append(("history logger", False, str(exc)))

    for name, ok, detail in checks:
        status = "PASS" if ok else "FAIL"
        logger.info("%s: %s - %s", name, status, detail)


if __name__ == "__main__":
    run_checks()
