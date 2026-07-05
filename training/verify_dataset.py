from pathlib import Path
import pandas as pd

root = Path(__file__).resolve().parents[1]
for name in ["train.csv", "test.csv"]:
    path = root / "data" / name
    print("FILE:", path)
    df = pd.read_csv(path)
    print("shape", df.shape)
    print("columns", list(df.columns))
    print("missing", df[["text", "emotion"]].isnull().sum().to_dict())
    labels = sorted(df["emotion"].astype(str).str.strip().str.lower().unique())
    print("unique labels", labels)
    print("value counts:\n", df["emotion"].astype(str).str.strip().value_counts().head(20))
    print("---")
