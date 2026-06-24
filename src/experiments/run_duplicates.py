"""Quantify duplicate and near-duplicate row overlap across train/test splits (L3)."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.data.load import load_data

RANDOM_STATE = 42
TEST_SIZE = 0.2
NEAR_DUP_THRESHOLD = 0.05   # normalised Euclidean distance
TABLES_DIR = Path("results/tables")


def run_duplicates():
    df = load_data()
    feature_cols = [c for c in df.columns if c != "fetal_health"]

    # ── exact duplicates in the full dataset ──────────────────────────────────
    n_exact = int(df.duplicated().sum())
    print(f"Dataset shape         : {df.shape}")
    print(f"Exact duplicates      : {n_exact}  ({n_exact / len(df) * 100:.2f}%)")

    # ── train/test overlap (exact) ────────────────────────────────────────────
    X = df[feature_cols]
    y = df["fetal_health"]
    X_train, X_test, _, _ = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    train_keys = X_train.apply(tuple, axis=1)
    test_keys  = X_test.apply(tuple, axis=1)
    exact_overlap = int(train_keys.isin(test_keys).sum())
    print(f"\nStratified 80/20 split:")
    print(f"  Train rows          : {len(X_train)}")
    print(f"  Test rows           : {len(X_test)}")
    print(f"  Exact overlap       : {exact_overlap} train rows match a test row")

    # ── near-duplicate overlap (scaled Euclidean distance) ────────────────────
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    dists = pairwise_distances(X_train_s, X_test_s, metric="euclidean")
    near_dup_pairs = int((dists < NEAR_DUP_THRESHOLD).sum())
    near_dup_train = int((dists < NEAR_DUP_THRESHOLD).any(axis=1).sum())

    print(f"  Near-dup pairs      : {near_dup_pairs}  (distance < {NEAR_DUP_THRESHOLD})")
    print(f"  Near-dup train rows : {near_dup_train} train rows near a test row")

    # ── save table ────────────────────────────────────────────────────────────
    result = pd.DataFrame([{
        "total_rows": len(df),
        "exact_duplicates_in_dataset": n_exact,
        "duplicate_pct": round(n_exact / len(df) * 100, 4),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "exact_train_test_overlap": exact_overlap,
        "near_dup_threshold": NEAR_DUP_THRESHOLD,
        "near_dup_pairs": near_dup_pairs,
        "near_dup_train_rows": near_dup_train,
    }])

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(TABLES_DIR / "duplicates.csv", index=False)
    print(f"\nSaved: {TABLES_DIR / 'duplicates.csv'}")

    return result


if __name__ == "__main__":
    run_duplicates()
