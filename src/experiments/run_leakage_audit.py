"""Run naive vs corrected pipelines across L1-L5 leakage sources; emit results tables."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

from src.data.load import load_data
from src.evaluation.metrics import evaluate, plot_calibration

RANDOM_STATE = 42
TEST_SIZE = 0.2
N_SPLITS = 5
TABLES_DIR = Path("results/tables")
FIGURES_DIR = Path("results/figures")


def _rf():
    return RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)


def _cv(estimator, X, y):
    kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(estimator, X, y, cv=kf, scoring="accuracy")
    return float(scores.mean()), float(scores.std())


# ── configuration A: fully leakage-free ───────────────────────────────────────
def config_correct(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    pipe = ImbPipeline([
        ("scaler", StandardScaler()),
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        ("clf", _rf()),
    ])
    cv_mean, cv_std = _cv(pipe, X_train, y_train)
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)
    return cv_mean, cv_std, evaluate(y_test, y_pred, y_proba), y_test, y_proba


# ── configuration B: add L4 (scaler fit on full dataset) ─────────────────────
def config_l4(X, y):
    X_scaled = StandardScaler().fit_transform(X)          # L4 applied here
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    pipe = ImbPipeline([
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        ("clf", _rf()),
    ])
    cv_mean, cv_std = _cv(pipe, X_train, y_train)
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)
    return cv_mean, cv_std, evaluate(y_test, y_pred, y_proba)


# ── configuration C: add L1+L4 (SMOTE + scaler before split) ─────────────────
def config_l1_l4(X, y):
    X_scaled = StandardScaler().fit_transform(X)          # L4
    X_res, y_res = SMOTE(random_state=RANDOM_STATE).fit_resample(X_scaled, y)  # L1
    X_train, X_test, y_train, y_test = train_test_split(
        X_res, y_res, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    clf = _rf()
    cv_mean, cv_std = _cv(clf, X_train, y_train)          # CV on train fold only
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)
    return cv_mean, cv_std, evaluate(y_test, y_pred, y_proba)


# ── configuration D: full naive (L1+L2+L4) ───────────────────────────────────
def config_naive(X, y):
    X_scaled = StandardScaler().fit_transform(X)          # L4
    X_res, y_res = SMOTE(random_state=RANDOM_STATE).fit_resample(X_scaled, y)  # L1
    clf = _rf()
    cv_mean, cv_std = _cv(clf, X_res, y_res)              # L2: CV on full resampled pool
    X_train, X_test, y_train, y_test = train_test_split(
        X_res, y_res, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)
    return cv_mean, cv_std, evaluate(y_test, y_pred, y_proba), y_test, y_proba


def run_audit():
    df = load_data()
    X = df.drop(columns=["fetal_health"]).values
    y = (df["fetal_health"].values - 1).astype(int)

    print("Running leakage audit...\n")

    cv_a, std_a, m_a, y_test_c, proba_c = config_correct(X, y)
    cv_b, std_b, m_b                    = config_l4(X, y)
    cv_c, std_c, m_c                    = config_l1_l4(X, y)
    cv_d, std_d, m_d, y_test_n, proba_n = config_naive(X, y)

    configs = {
        "A — Correct (no leakage)":    (cv_a, std_a, m_a),
        "B — +L4 (scaler before split)": (cv_b, std_b, m_b),
        "C — +L1+L4 (SMOTE+scaler before split)": (cv_c, std_c, m_c),
        "D — +L1+L2+L4 (full naive)":  (cv_d, std_d, m_d),
    }

    for name, (cv, std, m) in configs.items():
        print(
            f"  {name}\n"
            f"    CV={cv:.4f}±{std:.4f}  Test={m['accuracy']:.4f}  "
            f"BalAcc={m['balanced_accuracy']:.4f}  MacroF1={m['macro_f1']:.4f}\n"
            f"    Recall  Normal={m['per_class']['Normal']['recall']:.3f}  "
            f"Suspect={m['per_class']['Suspect']['recall']:.3f}  "
            f"Pathological={m['per_class']['Pathological']['recall']:.3f}\n"
        )

    # ── results table ──────────────────────────────────────────────────────────
    rows = []
    for name, (cv, std, m) in configs.items():
        rows.append({
            "Config": name,
            "CV Accuracy": round(cv, 4),
            "CV Std": round(std, 4),
            "Test Accuracy": round(m["accuracy"], 4),
            "Balanced Accuracy": round(m["balanced_accuracy"], 4),
            "Macro F1": round(m["macro_f1"], 4),
            "ROC-AUC OvR": round(m["roc_auc_ovr"], 4),
            "ECE": round(m["ece"], 4),
            "Recall Normal": round(m["per_class"]["Normal"]["recall"], 4),
            "Recall Suspect": round(m["per_class"]["Suspect"]["recall"], 4),
            "Recall Pathological": round(m["per_class"]["Pathological"]["recall"], 4),
        })

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    table.to_csv(TABLES_DIR / "leakage_audit.csv", index=False)
    print(f"Saved: {TABLES_DIR / 'leakage_audit.csv'}")

    # ── figure 1: CV accuracy bar chart ───────────────────────────────────────
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    labels = list(configs)
    cv_vals = [configs[c][0] for c in labels]
    cv_errs = [configs[c][1] for c in labels]
    colors = ["#2ca02c", "#17becf", "#ff7f0e", "#d62728"]

    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.barh(labels, cv_vals, xerr=cv_errs, capsize=4, color=colors, height=0.5)
    ax.axvline(cv_vals[0], color="#2ca02c", linestyle="--", linewidth=1, alpha=0.6, label="Correct baseline")
    ax.set_xlabel("CV Accuracy")
    ax.set_title("Leakage attribution — CV accuracy by configuration (RF, n=100, 5-fold)")
    ax.set_xlim(0.88, 1.01)
    for bar, val in zip(bars, cv_vals):
        ax.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=9)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "leakage_attribution.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {FIGURES_DIR / 'leakage_attribution.png'}")

    # ── figure 2: calibration curves ──────────────────────────────────────────
    fig = plot_calibration(y_test_c, proba_c, title="Reliability diagram — Correct pipeline")
    fig.savefig(FIGURES_DIR / "calibration_correct.png", dpi=150)
    plt.close(fig)

    fig = plot_calibration(y_test_n, proba_n, title="Reliability diagram — Naive pipeline")
    fig.savefig(FIGURES_DIR / "calibration_naive.png", dpi=150)
    plt.close(fig)
    print(f"Saved calibration curves to {FIGURES_DIR}/")

    # ── figure 3: per-class recall ─────────────────────────────────────────────
    classes = ["Normal", "Suspect", "Pathological"]
    x = np.arange(len(classes))
    width = 0.35
    rec_correct = [m_a["per_class"][c]["recall"] for c in classes]
    rec_naive   = [m_d["per_class"][c]["recall"] for c in classes]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - width / 2, rec_correct, width, label="Correct", color="#2ca02c")
    ax.bar(x + width / 2, rec_naive,   width, label="Naive",   color="#d62728")
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.set_ylabel("Recall")
    ax.set_ylim(0, 1.12)
    ax.set_title("Per-class recall: correct vs naive pipeline")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "per_class_recall.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {FIGURES_DIR / 'per_class_recall.png'}")

    return table


if __name__ == "__main__":
    run_audit()
