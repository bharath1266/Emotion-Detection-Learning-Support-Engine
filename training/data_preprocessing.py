"""
training/data_preprocessing.py

Beginner-friendly dataset loading and preprocessing for Emotion Detection & Learning Support Engine.

This script loads CSV files from the project's `data/` folder, inspects them,
cleans text and labels, maps emotions to five target classes, splits into
train/test sets, and writes cleaned CSVs back to `data/train.csv` and
`data/test.csv`.

Usage (from repository root):
    python training/data_preprocessing.py

Requirements:
    - pandas
    - scikit-learn

The script assumes source CSV(s) have at least the columns: `text`, `emotion`.
"""

from pathlib import Path
from typing import List, Dict

import pandas as pd
from sklearn.model_selection import train_test_split


DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def find_csv_files(data_dir: Path) -> List[Path]:
    """Return a list of CSV files in `data_dir` (recursive).

    Excludes the canonical output files `train.csv` and `test.csv` to avoid
    re-loading previously saved splits.
    """
    return sorted([p for p in data_dir.rglob("*.csv") if p.name not in {"train.csv", "test.csv"}])


def load_csv_files(paths: List[Path]) -> pd.DataFrame:
    """Load multiple CSVs and concatenate them into a single DataFrame.

    Only columns that exist across files are preserved. If no CSVs are
    found, an empty DataFrame is returned.
    """
    if not paths:
        return pd.DataFrame()

    standardized_rows = []

    # Known single-label text column candidates and emotion column candidates
    text_candidates = ["text", "user input", "user_input", "message", "content", "input"]
    emotion_candidates = ["emotion", "label", "labels", "sentiment", "emotion_label"]

    # Known GoEmotions emotion names (one-hot columns)
    go_emotions = {"admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion", "curiosity", "desire", "disappointment", "disapproval", "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief", "joy", "love", "nervousness", "optimism", "pride", "realization", "relief", "remorse", "sadness", "surprise", "neutral"}

    for p in paths:
        try:
            df = pd.read_csv(p)
            df.rename(columns={c: c.strip() for c in df.columns}, inplace=True)
            cols_lower = {c.lower().strip(): c for c in df.columns}

            # Attempt single-label extraction
            text_col = next((cols_lower[k] for k in text_candidates if k in cols_lower), None)
            emotion_col = next((cols_lower[k] for k in emotion_candidates if k in cols_lower), None)

            extracted = []
            if text_col and emotion_col:
                tmp = df.loc[:, [text_col, emotion_col]].dropna()
                tmp = tmp.rename(columns={text_col: "text", emotion_col: "emotion"})
                extracted = tmp[["text", "emotion"]].to_dict(orient="records")
                print(f"Extracted {len(extracted)} rows from {p.relative_to(DATA_DIR.parent)} using ({text_col}, {emotion_col})")
            elif "text" in cols_lower and go_emotions.intersection(set(k for k in cols_lower)):
                # Handle GoEmotions one-hot style: find which emotion columns have a truthy value
                text_col_actual = cols_lower["text"]
                emotion_cols = [cols_lower[k] for k in cols_lower if k in go_emotions]
                count = 0
                for _, row in df.iterrows():
                    text_val = row.get(text_col_actual)
                    if pd.isna(text_val):
                        continue
                    # find first emotion column marked as positive
                    found = None
                    for ec in emotion_cols:
                        val = row.get(ec)
                        if (isinstance(val, (int, float)) and val > 0) or str(val).strip().lower() in {"1", "true", "t", "yes"}:
                            found = ec
                            break
                    if found:
                        extracted.append({"text": text_val, "emotion": found})
                        count += 1
                print(f"Extracted {count} rows from {p.relative_to(DATA_DIR.parent)} using one-hot emotion columns")
            else:
                print(f"Skipping {p.relative_to(DATA_DIR.parent)} — could not find usable text/emotion columns")

            standardized_rows.extend(extracted)

        except Exception as e:
            print(f"Warning: failed to process {p}: {e}")

    if not standardized_rows:
        return pd.DataFrame()

    return pd.DataFrame(standardized_rows)


def inspect_dataset(df: pd.DataFrame) -> None:
    """Prints basic dataset information to the console."""
    if df.empty:
        print("No data available to inspect.")
        return

    print("\n=== Dataset Inspection ===")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print("First 5 rows:")
    # show a small, readable preview
    with pd.option_context("display.max_colwidth", 200):
        print(df.head(5))
    print("\nMissing values per column:")
    print(df.isnull().sum())


def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the DataFrame:
    - Keep only `text` and `emotion` columns if present
    - Drop rows with null `text` or `emotion`
    - Drop duplicate rows
    - Lowercase text
    """
    if df.empty:
        return df

    # Detect common column name variants for text and emotion (case-insensitive)
    if df.empty:
        return df

    col_map = {c.lower(): c for c in df.columns}

    text_candidates = ["text", "user input", "user_input", "message", "content", "input"]
    emotion_candidates = ["emotion", "label", "labels", "sentiment", "emotion_label", "emotionlabels"]

    text_col = next((col_map[k] for k in text_candidates if k in col_map), None)
    emotion_col = next((col_map[k] for k in emotion_candidates if k in col_map), None)

    if text_col is None or emotion_col is None:
        # If either is missing, report and return empty DataFrame so downstream
        # steps don't silently operate on the wrong columns.
        print(f"Could not detect required columns. Found: {list(df.columns)}")
        return pd.DataFrame()

    # Keep only the detected columns and normalize their names
    df = df.loc[:, [text_col, emotion_col]].copy()
    df.rename(columns={text_col: "text", emotion_col: "emotion"}, inplace=True)

    # Drop rows where either column is missing
    df.dropna(subset=["text", "emotion"], inplace=True)

    # Drop exact duplicates
    df.drop_duplicates(inplace=True)

    # Ensure text is string and lowercase it
    df["text"] = df["text"].astype(str).str.strip().str.lower()

    # Also normalize emotion entries to strings
    df["emotion"] = df["emotion"].astype(str).str.strip()

    return df


def map_labels_to_five(df: pd.DataFrame) -> pd.DataFrame:
    """Map existing emotion labels to 5 target classes.

    The mapping strategy is intentionally simple and beginner-friendly. It
    tries to match keywords in the original `emotion` label/value. Any
    unmapped labels are assigned `neutral`.
    """
    if df.empty or "emotion" not in df.columns:
        return df

    # Lowercase emotion column for consistent mapping
    df["emotion"] = df["emotion"].astype(str).str.strip().str.lower()

    mapping: Dict[str, List[str]] = {
        "happy": ["happy", "joy", "joyous", "excited", "love", "surprise"],
        "sad": ["sad", "sadness", "depressed", "down", "unhappy"],
        "angry": ["angry", "anger", "annoyed", "frustrated", "hate"],
        "fear": ["fear", "afraid", "scared", "anxious", "anxiety"],
        "neutral": ["neutral", "no_emotion", "none", "other", "neutral emotion"],
    }

    def map_one(label: str) -> str:
        for target, keywords in mapping.items():
            for kw in keywords:
                if kw in label:
                    return target
        # Fallback: try to map some common full-label matches
        if label in ["happiness", "joyful"]:
            return "happy"
        return "neutral"

    df["emotion"] = df["emotion"].apply(map_one)
    return df


def split_and_save(df: pd.DataFrame, data_dir: Path, test_size: float = 0.2, random_state: int = 42) -> None:
    """Split DataFrame into train/test and save CSVs at `data_dir/train.csv` and `data_dir/test.csv`."""
    if df.empty:
        print("No data to split or save.")
        return

    # If there are too few classes or samples for stratification, skip stratify
    stratify_col = df["emotion"] if df["emotion"].nunique() > 1 and df.shape[0] >= 2 * df["emotion"].nunique() else None
    if stratify_col is None:
        print("Warning: skipping stratified split due to small dataset or class imbalance")

    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state, stratify=stratify_col)

    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Saved cleaned train to: {train_path} (shape={train_df.shape})")
    print(f"Saved cleaned test to:  {test_path} (shape={test_df.shape})")


def main() -> None:
    print(f"Data directory: {DATA_DIR}")

    # Ensure data directory exists
    if not DATA_DIR.exists():
        raise SystemExit(f"Data directory not found: {DATA_DIR}")

    csv_files = find_csv_files(DATA_DIR)
    print(f"Found CSV files: {[p.name for p in csv_files]}")

    df = load_csv_files(csv_files)

    inspect_dataset(df)

    df = basic_cleaning(df)
    print("\nAfter basic cleaning:")
    inspect_dataset(df)

    df = map_labels_to_five(df)
    print("\nAfter label mapping to five classes:")
    print(df["emotion"].value_counts())

    split_and_save(df, DATA_DIR)


if __name__ == "__main__":
    main()
