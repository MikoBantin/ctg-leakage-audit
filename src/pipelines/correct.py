"""Leakage-free pipeline: stratified split first; SMOTE + scaling inside CV folds via imblearn Pipeline."""

import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.data.load import load_data

RANDOM_STATE = 42
TEST_SIZE = 0.2
N_SPLITS = 5


def _make_pipeline(model: str, random_state: int) -> Pipeline:
    if model == "rf":
        clf = RandomForestClassifier(n_estimators=100, random_state=random_state)
    elif model == "xgb":
        clf = XGBClassifier(
            n_estimators=100,
            random_state=random_state,
            eval_metric="mlogloss",
            verbosity=0,
        )
    else:
        raise ValueError(f"Unknown model {model!r}. Choose 'rf' or 'xgb'.")

    return Pipeline([
        ("scaler", StandardScaler()),
        ("smote", SMOTE(random_state=random_state)),
        ("clf", clf),
    ])


def run_correct(
    df: pd.DataFrame | None = None,
    model: str = "rf",
    random_state: int = RANDOM_STATE,
    test_size: float = TEST_SIZE,
) -> dict:
    """
    Leakage-free pipeline.

    Fixes applied vs the naive approach:
      - Stratified split before any preprocessing or resampling
      - Scaler + SMOTE wrapped in an imblearn Pipeline so they run inside each CV fold
      - Final evaluation on the untouched held-out test set only
    """
    if df is None:
        df = load_data()

    X = df.drop(columns=["fetal_health"]).values
    y = (df["fetal_health"].values - 1).astype(int)

    # Split first — test set never touched during training or resampling
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    pipe = _make_pipeline(model, random_state)

    # CV on training data only; SMOTE runs inside each fold via the Pipeline
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=random_state)
    cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="accuracy")

    # Fit on full training set, evaluate on untouched test set
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    test_acc = float(accuracy_score(y_test, y_pred))

    print(f"[CORRECT / {model.upper()}]")
    print(f"  CV accuracy (clean)   : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"  Test accuracy (clean) : {test_acc:.4f}")
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
        "pipeline": pipe,
    }


if __name__ == "__main__":
    run_correct(model="rf")
    run_correct(model="xgb")
