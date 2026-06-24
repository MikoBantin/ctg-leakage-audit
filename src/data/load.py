"""Load CTG dataset and run basic integrity checks (duplicates, NaNs, class counts)."""

from pathlib import Path

import pandas as pd

DATA_PATH = Path(__file__).parents[2] / "data" / "raw" / "fetal_health.csv"

CLASS_NAMES = {1.0: "Normal", 2.0: "Suspect", 3.0: "Pathological"}


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the fetal health CTG dataset from a CSV file."""
    return pd.read_csv(path)


def integrity_report(df: pd.DataFrame) -> dict:
    """Print and return an integrity summary: shape, NaNs, duplicates, class counts."""
    n_rows, n_cols = df.shape
    n_features = n_cols - 1
    n_missing = int(df.isna().sum().sum())
    n_duplicates = int(df.duplicated().sum())

    class_counts = (
        df["fetal_health"]
        .value_counts()
        .sort_index()
        .rename(index=CLASS_NAMES)
    )

    print(f"Shape            : {n_rows} rows × {n_features} features + 1 target")
    print(f"Missing values   : {n_missing}")
    print(f"Exact duplicates : {n_duplicates}")
    print("\nClass distribution:")
    for label, count in class_counts.items():
        print(f"  {label:>15s}: {count:>5d}  ({count / n_rows * 100:.1f}%)")

    return {
        "n_rows": n_rows,
        "n_features": n_features,
        "n_missing": n_missing,
        "n_duplicates": n_duplicates,
        "class_counts": class_counts.to_dict(),
    }


if __name__ == "__main__":
    integrity_report(load_data())
