"""Leaky baseline pipeline: SMOTE + scaling applied before train/test split."""

import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.data.load import load_data

RANDOM_STATE = 42
TEST_SIZE = 0.2
N_SPLITS = 5


def _make_classifier(model: str, random_state: int):
    if model == "rf":
        return RandomForestClassifier(n_estimators=100, random_state=random_state)
    if model == "xgb":
        return XGBClassifier(
            n_estimators=100,
            random_state=random_state,
            eval_metric="mlogloss",
            verbosity=0,
        )
    raise ValueError(f"Unknown model {model!r}. Choose 'rf' or 'xgb'.")


def run_naive(
    df: pd.DataFrame | None = None,
    model: str = "rf",
    random_state: int = RANDOM_STATE,
    test_size: float = TEST_SIZE,
) -> dict:
    """
    Leaky pipeline reproducing common published CTG approaches.

    Leakage sources applied:
      L1 — SMOTE before train/test split
      L2 — CV on already-resampled data (SMOTE outside folds)
      L4 — StandardScaler fit on full dataset before splitting
    """
    if df is None:
        df = load_data()

    X = df.drop(columns=["fetal_health"]).values
    y = (df["fetal_health"].values - 1).astype(int)  # XGBoost requires 0-indexed labels

    # L4: scaler fit on the full dataset — test statistics bleed into training
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # L1: SMOTE on the full dataset before any split
    X_res, y_res = SMOTE(random_state=random_state).fit_resample(X_scaled, y)

    # L2: CV run on the already-resampled pool (SMOTE outside folds)
    clf = _make_classifier(model, random_state)
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=random_state)
    cv_scores = cross_val_score(clf, X_res, y_res, cv=cv, scoring="accuracy")

    # Split after resampling (the wrong order), then train and evaluate
    X_train, X_test, y_train, y_test = train_test_split(
        X_res, y_res, test_size=test_size, random_state=random_state
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    test_acc = float(accuracy_score(y_test, y_pred))

    print(f"[NAIVE / {model.upper()}]")
    print(f"  CV accuracy (leaky)   : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"  Test accuracy (leaky) : {test_acc:.4f}")
    print()
    print(classification_report(
        y_test, y_pred,
        target_names=["Normal", "Suspect", "Pathological"],
    ))

    return {
        "model": model,
        "cv_mean": float(cv_scores.mean()),
        "cv_std": float(cv_scores.std()),
        "test_accuracy": test_acc,
        "classifier": clf,
        "scaler": scaler,
    }


if __name__ == "__main__":
    run_naive(model="rf")
    run_naive(model="xgb")
