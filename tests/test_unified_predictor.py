import numpy as np
import utils.unified_predictor as up


def test_predict_combination(monkeypatch):
    # Mock bilstm_proba and bert_proba
    monkeypatch.setattr(up, "bilstm_proba", lambda text: np.array([0.2,0.3,0.1,0.1,0.3]))
    monkeypatch.setattr(up, "bert_proba", lambda text: np.array([0.1,0.6,0.1,0.1,0.1]))

    res = up.predict("sample", model_choice="both")
    assert "final_prediction" in res
    assert res["final_prediction"] in up.LABELS
    assert isinstance(res["confidence"], float)
