"""training/evaluate_models.py

Evaluate trained models on `data/test.csv` and produce classification
metrics: accuracy, precision, recall, F1, confusion matrix, and a
classification report.

Exports:
- models/<model>/evaluation.json (per-model)
- results/compare_models.csv (both-model comparison)
- results/compare_models.json (both-model comparison)
- results/confusion_matrix_bilstm.png
- results/confusion_matrix_bert.png

Usage:
    python training/evaluate_models.py --model bilstm
    python training/evaluate_models.py --model bert
    python training/evaluate_models.py --model both
"""
import argparse
import gc
import json
import pickle
from pathlib import Path
import sys
import os

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from tqdm import tqdm

from utils.config import DATA_DIR, BILSTM_MODEL_PATH, BILSTM_TOKENIZER_PATH, BERT_MODEL_DIR, BERT_MAX_LEN, LABEL_MAP, LABELS



RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def _ensure_results_dir() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _plot_confusion_matrix(cm: np.ndarray, labels: list[str], out_path: Path) -> None:
    """Plot and save a labeled confusion matrix image."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        ylabel="True label",
        xlabel="Predicted label",
        title=out_path.stem.replace("_", " ").title(),
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Annotate cells
    cm_int = cm.astype(int)
    thresh = cm_int.max() / 2.0 if cm_int.size else 0
    for i in range(cm_int.shape[0]):
        for j in range(cm_int.shape[1]):
            ax.text(
                j,
                i,
                str(cm_int[i, j]),
                ha="center",
                va="center",
                color="white" if cm_int[i, j] > thresh else "black",
            )

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _normalize_bert_probabilities(probs: np.ndarray) -> np.ndarray:
    probs_arr = np.asarray(probs, dtype=float)
    if probs_arr.ndim == 1:
        if probs_arr.size == len(LABELS):
            return probs_arr
        if probs_arr.size < len(LABELS):
            padded = np.zeros(len(LABELS), dtype=float)
            padded[: probs_arr.size] = probs_arr
            probs_arr = padded
        else:
            probs_arr = probs_arr[: len(LABELS)]
        total = float(probs_arr.sum())
        if not np.isfinite(total) or total <= 0:
            return np.full(len(LABELS), 1.0 / len(LABELS), dtype=float)
        return probs_arr / total

    if probs_arr.ndim == 2:
        aligned = []
        for row in probs_arr:
            aligned.append(_normalize_bert_probabilities(row))
        return np.vstack(aligned)

    return probs_arr


def _sanitize_label_predictions(y_true: np.ndarray, preds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y_true_arr = np.asarray(y_true, dtype=int)
    preds_arr = np.asarray(preds, dtype=int)
    max_label = max(len(LABELS) - 1, 0)
    y_true_arr = np.clip(y_true_arr, 0, max_label).astype(int)
    preds_arr = np.clip(preds_arr, 0, max_label).astype(int)
    return y_true_arr, preds_arr


def _compute_metrics(y_true: np.ndarray, preds: np.ndarray) -> dict:
    y_true_arr, preds_arr = _sanitize_label_predictions(y_true, preds)
    labels = np.arange(len(LABELS), dtype=int)
    label_names = list(LABELS)

    acc = accuracy_score(y_true_arr, preds_arr)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true_arr, preds_arr, average="weighted", zero_division=0)
    cm = confusion_matrix(y_true_arr, preds_arr, labels=labels)
    report = classification_report(
        y_true_arr,
        preds_arr,
        labels=labels,
        target_names=label_names,
        output_dict=True,
        zero_division=0,
    )
    return {
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }


def _write_classification_report(y_true: np.ndarray, preds: np.ndarray, out_path: Path) -> None:
    y_true_arr, preds_arr = _sanitize_label_predictions(y_true, preds)
    labels = np.arange(len(LABELS), dtype=int)
    label_names = list(LABELS)

    report_text = classification_report(
        y_true_arr,
        preds_arr,
        labels=labels,
        target_names=label_names,
        output_dict=False,
        zero_division=0,
    )
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report_text)


def _infer_bert_batches(model, tokenizer, texts: list[str], batch_size: int = 2):
    device = next(model.parameters()).device
    loader = DataLoader(list(range(len(texts))), batch_size=batch_size, shuffle=False)
    probs_list = []

    for batch_indices in tqdm(loader, desc="BERT evaluation", unit="batch"):
        batch_texts = [texts[i] for i in batch_indices.tolist()]
        enc = tokenizer(
            batch_texts,
            truncation=True,
            padding=True,
            max_length=BERT_MAX_LEN,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.inference_mode():
            outputs = model(**enc)
            probs = torch.softmax(outputs.logits, dim=-1).detach().cpu().numpy()

        probs = _normalize_bert_probabilities(probs)
        probs_list.append(probs)
        del outputs
        del enc
        del probs
        gc.collect()

    if not probs_list:
        return np.empty((0, len(LABEL_MAP)))
    return np.vstack(probs_list)


def load_test():

    test_path = DATA_DIR / "test.csv"

    if not test_path.exists():
        raise FileNotFoundError("data/test.csv not found. Run training/data_preprocessing.py first.")
    df = pd.read_csv(test_path).dropna(subset=["text", "emotion"]).reset_index(drop=True)
    return df


def _iter_test_batches(batch_size: int = 2, chunk_size: int = 64, max_samples: int | None = None):
    test_path = DATA_DIR / "test.csv"
    if not test_path.exists():
        raise FileNotFoundError("data/test.csv not found. Run training/data_preprocessing.py first.")

    total_yielded = 0
    for chunk in pd.read_csv(test_path, chunksize=chunk_size):
        chunk = chunk.dropna(subset=["text", "emotion"]).reset_index(drop=True)
        if chunk.empty:
            continue

        texts = chunk["text"].astype(str).tolist()
        labels = chunk["emotion"].astype(str).str.strip().str.lower().map(LABEL_MAP).values
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]
            batch_labels = labels[start : start + batch_size]
            if max_samples is not None and total_yielded + len(batch_texts) > max_samples:
                remaining = max_samples - total_yielded
                if remaining <= 0:
                    return
                batch_texts = batch_texts[:remaining]
                batch_labels = batch_labels[:remaining]
            yield batch_texts, batch_labels
            total_yielded += len(batch_texts)
            if max_samples is not None and total_yielded >= max_samples:
                return


def eval_bilstm():
    import tensorflow as tf
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    print("Loading BiLSTM model and tokenizer...")
    if not BILSTM_MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing model: {BILSTM_MODEL_PATH}")
    if not BILSTM_TOKENIZER_PATH.exists():
        raise FileNotFoundError(f"Missing tokenizer: {BILSTM_TOKENIZER_PATH}")

    model = tf.keras.models.load_model(str(BILSTM_MODEL_PATH))
    with open(BILSTM_TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)

    df = load_test()
    texts = df["text"].astype(str).tolist()
    seqs = tokenizer.texts_to_sequences(texts)
    x = pad_sequences(seqs, maxlen=model.input_shape[1], padding="post", truncating="post")

    probs = model.predict(x, batch_size=32)
    preds = probs.argmax(axis=1)

    y_true = df["emotion"].astype(str).str.strip().str.lower().map(LABEL_MAP).values

    results = _compute_metrics(y_true, preds)

    out = BILSTM_MODEL_PATH.parent / "evaluation.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved evaluation to: {out}")

    cm = np.array(results["confusion_matrix"], dtype=int)
    _plot_confusion_matrix(cm, LABELS, RESULTS_DIR / "confusion_matrix_bilstm.png")

    _write_classification_report(
        y_true,
        preds,
        RESULTS_DIR / "classification_report_bilstm.txt",
    )

    return results



def eval_bert(max_samples: int | None = None):
    import traceback
    from transformers import BertConfig, BertTokenizerFast, BertForSequenceClassification
    import torch

    try:
        print("Loading BERT model and tokenizer...")

        if not BERT_MODEL_DIR.exists():
            raise FileNotFoundError(f"Missing BERT dir: {BERT_MODEL_DIR}")

        tokenizer = BertTokenizerFast.from_pretrained(str(BERT_MODEL_DIR), local_files_only=True)

        model_config = BertConfig.from_pretrained(str(BERT_MODEL_DIR), local_files_only=True)
        model_config.num_labels = len(LABELS)
        model_config.problem_type = "single_label_classification"
        model_config.id2label = {i: label for i, label in enumerate(LABELS)}
        model_config.label2id = {label: i for i, label in enumerate(LABELS)}
        model = BertForSequenceClassification.from_pretrained(
            str(BERT_MODEL_DIR),
            local_files_only=True,
            config=model_config,
        )
        model.config.num_labels = len(LABELS)
        model.config.id2label = {i: label for i, label in enumerate(LABELS)}
        model.config.label2id = {label: i for i, label in enumerate(LABELS)}
        if getattr(model.classifier, "out_features", None) != len(LABELS):
            raise ValueError("BERT classifier head mismatch during evaluation.")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()

        all_preds = []
        all_y_true = []
        batch_size = 8 if device.type == "cpu" else 16

        for texts, y_true_batch in _iter_test_batches(batch_size=batch_size, chunk_size=128, max_samples=max_samples):
            probs = _infer_bert_batches(model, tokenizer, texts, batch_size=batch_size)
            preds = probs.argmax(axis=1)
            all_preds.extend(preds.tolist())
            all_y_true.extend(y_true_batch.tolist())

        preds = np.asarray(all_preds, dtype=int)
        y_true = np.asarray(all_y_true, dtype=int)

        results = _compute_metrics(y_true, preds)
        cm_arr = np.array(results["confusion_matrix"], dtype=int)
        cm = cm_arr.tolist()

        results["confusion_matrix"] = cm

        # Per-model artifacts
        out = BERT_MODEL_DIR / "evaluation.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Saved evaluation to: {out}")

        _plot_confusion_matrix(
            cm_arr.astype(int),
            LABELS,
            RESULTS_DIR / "confusion_matrix_bert.png",
        )

        _write_classification_report(
            y_true,
            preds,
            RESULTS_DIR / "classification_report_bert.txt",
        )

        return results
    except Exception as e:
        error_out = RESULTS_DIR / "bert_eval_error.json"
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        payload = {
            "error_type": type(e).__name__,
            "error_message": str(e),
            "traceback": traceback.format_exc(),
        }
        with open(error_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"BERT evaluation failed. Error written to: {error_out}")
        raise




def _save_comparison(bilstm_results: dict, bert_results: dict) -> None:
    _ensure_results_dir()

    rows = []
    for name, res in (("BiLSTM", bilstm_results), ("BERT", bert_results)):
        rows.append(
            {
                "model": name,
                "accuracy": res["accuracy"],
                "precision": res["precision"],
                "recall": res["recall"],
                "f1": res["f1"],
            }
        )

    compare_df = pd.DataFrame(rows)
    compare_csv = RESULTS_DIR / "model_comparison.csv"
    compare_json = RESULTS_DIR / "model_comparison.json"
    legacy_compare_csv = RESULTS_DIR / "compare_models.csv"
    legacy_compare_json = RESULTS_DIR / "compare_models.json"

    compare_df.to_csv(compare_csv, index=False)
    compare_df.to_csv(legacy_compare_csv, index=False)
    with open(compare_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "rows": rows,
                "best_model": str(compare_df.sort_values("f1", ascending=False).iloc[0]["model"]),
            },
            f,
            indent=2,
        )
    with open(legacy_compare_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "rows": rows,
                "best_model": str(compare_df.sort_values("f1", ascending=False).iloc[0]["model"]),
            },
            f,
            indent=2,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["bilstm", "bert", "both"], default="both")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional limit for evaluating a smaller sample")
    args = parser.parse_args()

    _ensure_results_dir()

    bilstm_results = None
    bert_results = None

    if args.model in ("bilstm", "both"):
        bilstm_results = eval_bilstm()

    if args.model in ("bert", "both"):
        bert_results = eval_bert(max_samples=args.max_samples)

    if args.model == "both" and bilstm_results is not None and bert_results is not None:
        _save_comparison(bilstm_results, bert_results)


if __name__ == "__main__":
    main()

