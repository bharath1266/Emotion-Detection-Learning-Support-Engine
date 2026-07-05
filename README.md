# Student Emotion Support (BiLSTM + BERT) — Portfolio Project

This project detects student emotions from text using **two classifiers** (BiLSTM + BERT), combines their probability outputs via a **unified predictor**, and generates **supportive learning guidance**. Results are presented through a **Streamlit** UI with prediction history + analytics and model-evaluation comparison.

---

## Features

- **Preprocessing + dataset split** (see `training/` and `training/data_preprocessing.py`)
- **BiLSTM** model training and inference pipeline
- **BERT** model fine-tuning and inference pipeline
- **Unified inference** combining BiLSTM + BERT probabilities (configurable weights)
- **Mixed-emotion detection** (secondary emotion threshold = **15%**)
- **Streamlit UI**:
  - Model selector: **Both / BERT / BiLSTM**
  - Guidance generation via Gemini (with local fallback when API key is missing)
  - Prediction history stored in `data/history.csv`
  - Analytics: emotion distribution, weekly trends, pie chart
- **Evaluation artifacts viewer** in the app:
  - Metrics table + confusion matrix images (from `results/`)

---

## Architecture

**Data flow**

1. Streamlit UI (`app.py`) accepts a student message.
2. UI calls unified inference (`utils/unified_predictor.py`).
3. Unified predictor loads and calls:
   - BiLSTM (`utils/bilstm_predict.py`)
   - BERT (`utils/bert_predict.py`)
4. Unified predictor:
   - normalizes probabilities
   - combines them using `utils/config.py::MODEL_WEIGHTS`
   - computes final emotion + confidence
   - computes mixed emotions using `utils/mixed_emotion.py`
5. UI logs the interaction to history (`utils/history_logger.py`) and renders analytics.
6. UI requests guidance (`utils/gemini_helper.py`):
   - uses Gemini when `GEMINI_API_KEY` is configured
   - otherwise returns a safe fallback template.

**Evaluation artifacts**

- `training/evaluate_models.py --model both` produces:
  - `results/compare_models.csv`
  - `results/compare_models.json`
  - `results/confusion_matrix_bilstm.png`
  - `results/confusion_matrix_bert.png`
- Streamlit UI loads and displays these artifacts under **“Model Evaluation (BiLSTM vs BERT)”**.

---

## Dataset

The app and evaluation scripts expect CSV files under `data/`:

- `data/train.csv`
- `data/test.csv`

Each dataset CSV is expected to contain at least:

- `text`: input student text
- `emotion`: emotion label

> Note: The repository’s `data/` and `models/` directories are intended to store large artifacts and are typically gitignored.

---

## Model Training

### BiLSTM

Train BiLSTM and save model artifacts under `models/`:

```powershell
python training/train_bilstm.py
```

### BERT

Fine-tune BERT and save model artifacts under `models/`:

```powershell
python training/train_bert.py
```

---

## Evaluation Results (Model Comparison)

To evaluate both models on the same test set and generate comparison artifacts:

```powershell
python training/evaluate_models.py --model both
```

Artifacts are written to:

- `results/compare_models.csv`
- `results/compare_models.json`
- `results/confusion_matrix_bilstm.png`
- `results/confusion_matrix_bert.png`

The Streamlit UI will automatically display them when present.

---

## Installation

### 1) Create a virtual environment

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 2) Install dependencies

```powershell
pip install -r requirements.txt
```

### 3) Gemini (optional)

To enable Gemini guidance generation, configure:

- environment variable: `GEMINI_API_KEY`

See `.env.example` (if present) and `utils/gemini_helper.py`.

---

## Running the Streamlit App

```powershell
streamlit run app.py
```

Open the URL shown in the terminal.

### What you can do in the UI

- Analyze a student message with **BiLSTM / BERT / Both**
- View:
  - final predicted emotion + confidence
  - mixed-emotion detection (when applicable)
  - per-model probability progress bars
- Generate supportive learning guidance:
  - **Gemini** when API key is configured
  - **fallback template** otherwise
- Review **Prediction History & Analytics**:
  - stored in `data/history.csv`
  - searchable and downloadable as CSV

---

## Project Structure

- `app.py` — Streamlit UI
- `training/` — training scripts + evaluation
- `utils/` — prediction utilities, preprocessing, logging, config, Gemini helper
- `tests/` — unit tests
- `data/` — dataset artifacts (typically gitignored)
- `models/` — trained model artifacts (typically gitignored)
- `results/` — evaluation outputs (comparison CSV/JSON + confusion matrices)

---

## Future Improvements

- Add richer analytics filters (date range, label filtering UI)
- Persist per-message guidance + model probabilities in history (if desired)
- Add benchmark report page inside Streamlit for evaluation artifacts
- Add UI for uploading a CSV and running batch predictions

---

## Screenshots

Add screenshots of:
- Streamlit live inference with guidance
- Prediction history + analytics
- Model evaluation comparison section

(If you have generated images under `results/`, they can be referenced here.)

