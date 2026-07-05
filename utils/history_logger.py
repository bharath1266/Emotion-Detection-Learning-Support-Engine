"""
utils/history_logger.py

Save prediction results to data/history.csv in a schema that is easy to use
for analytics and backward compatible with older files.
"""
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import csv
from datetime import datetime

import pandas as pd

from utils.config import DATA_DIR

logger = logging.getLogger(__name__)

DATA_PATH = DATA_DIR / "history.csv"
LATEST_COLUMNS = [
    "timestamp",
    "text",
    "bilstm_prediction",
    "bert_prediction",
    "final_prediction",
    "confidence",
    "response",
]
LEGACY_COLUMN_MAP = {
    "input_text": "text",
    "bilstm_pred": "bilstm_prediction",
    "bert_pred": "bert_prediction",
    "final_pred": "final_prediction",
    "emotion": "final_prediction",
    "ai_response": "response",
}


def _write_history_rows(rows: List[Dict[str, Any]]) -> None:
    """Write rows to the history CSV using the latest schema."""
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(DATA_PATH, mode="w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=LATEST_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in LATEST_COLUMNS})


def _normalize_history_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert older history rows into the latest schema and fill missing values."""
    if not rows:
        return []

    df = pd.DataFrame(rows)

    # Map legacy column names from older history files to the latest schema.
    for old_name, new_name in LEGACY_COLUMN_MAP.items():
        if old_name in df.columns and new_name not in df.columns:
            df[new_name] = df[old_name]

    # Ensure every expected column exists before analytics uses it.
    for col in LATEST_COLUMNS:
        if col not in df.columns:
            if col == "timestamp":
                df[col] = pd.NaT
            elif col == "confidence":
                df[col] = None
            else:
                df[col] = ""

    # Fill missing timestamps with the current time so the history remains usable.
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    if df["timestamp"].isna().any():
        df.loc[df["timestamp"].isna(), "timestamp"] = pd.Timestamp.utcnow().tz_convert("UTC")

    # Keep text-like fields safe for display and search.
    for col in ["text", "bilstm_prediction", "bert_prediction", "final_prediction", "response"]:
        df[col] = df[col].fillna("")

    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")

    normalized_rows: List[Dict[str, Any]] = []
    for _, row in df[LATEST_COLUMNS].iterrows():
        normalized_row: Dict[str, Any] = {}
        for col in LATEST_COLUMNS:
            value = row[col]
            if pd.isna(value):
                if col == "timestamp":
                    value = pd.Timestamp.utcnow()
                elif col == "confidence":
                    value = ""
                else:
                    value = ""
            if isinstance(value, pd.Timestamp):
                value = value.isoformat()
            normalized_row[col] = value
        normalized_rows.append(normalized_row)

    return normalized_rows


def _migrate_history_file_if_needed() -> bool:
    """Rewrite legacy history files in the latest schema when needed."""
    if not DATA_PATH.exists():
        return False

    try:
        with open(DATA_PATH, mode="r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
            fieldnames = reader.fieldnames or []
    except Exception:
        _write_history_rows([])
        return True

    if not rows and not fieldnames:
        return False

    # If the file uses old names or is missing any of the new columns, migrate it.
    legacy_fields_present = any(name in fieldnames for name in LEGACY_COLUMN_MAP)
    missing_columns = [col for col in LATEST_COLUMNS if col not in fieldnames]
    needs_migration = legacy_fields_present or bool(missing_columns)

    if needs_migration:
        _write_history_rows(_normalize_history_rows(rows))
        return True

    return False


def log_entry(
    input_text: str,
    bilstm_pred: Optional[str],
    bert_pred: Optional[str],
    final_pred: Optional[str],
    confidence: Optional[float],
    response: Optional[str] = None,
) -> None:
    """Append a prediction result to the history CSV with a UTC timestamp.

    The CSV will be created with a header row if it does not exist.
    Older CSV files are migrated automatically.
    """
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    if not DATA_PATH.exists():
        _write_history_rows([])
    else:
        _migrate_history_file_if_needed()

    row = {
        "timestamp": datetime.utcnow().isoformat(),
        "text": input_text or "",
        "bilstm_prediction": bilstm_pred or "",
        "bert_prediction": bert_pred or "",
        "final_prediction": final_pred or "",
        "confidence": confidence if confidence is not None else "",
        "response": response or "",
    }

    with open(DATA_PATH, mode="a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=LATEST_COLUMNS)
        if DATA_PATH.stat().st_size == 0:
            writer.writeheader()
        writer.writerow(row)


def load_history() -> List[Dict[str, str]]:
    """Load the history CSV and return a list of rows as dictionaries.

    Returns an empty list if the file does not exist.
    Older CSV files are migrated automatically.
    """
    if not DATA_PATH.exists():
        return []

    try:
        with open(DATA_PATH, mode="r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
    except Exception as exc:
        logger.exception("Failed to read history file: %s", exc)
        _write_history_rows([])
        return []

    if not rows:
        return []

    normalized_rows = _normalize_history_rows(rows)
    if _migrate_history_file_if_needed():
        return _normalize_history_rows(normalized_rows)
    return normalized_rows


def load_history_dataframe() -> pd.DataFrame:
    """Load history as a dataframe with a safe, normalized schema."""
    try:
        rows = load_history()
        if not rows:
            return pd.DataFrame(columns=LATEST_COLUMNS)
        df = pd.DataFrame(rows)
        for col in LATEST_COLUMNS:
            if col not in df.columns:
                if col == "timestamp":
                    df[col] = pd.NaT
                elif col == "confidence":
                    df[col] = None
                else:
                    df[col] = ""
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        if df["timestamp"].isna().any():
            df.loc[df["timestamp"].isna(), "timestamp"] = pd.Timestamp.utcnow().tz_convert("UTC")
        for col in ["text", "bilstm_prediction", "bert_prediction", "final_prediction", "response"]:
            df[col] = df[col].fillna("")
        df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")
        return df[LATEST_COLUMNS]
    except Exception as exc:
        logger.exception("Failed to build history dataframe: %s", exc)
        return pd.DataFrame(columns=LATEST_COLUMNS)
