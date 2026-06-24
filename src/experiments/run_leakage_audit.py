"""Run naive vs corrected pipelines across L1-L5 leakage sources; emit results tables."""

from pathlib import Path

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from scipy.stats import wilcoxon
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=FutureWarning)

from src.data.load import load_data
from src.evaluation.metrics import evaluate, plot_calibration

SEEDS = list(range(10))
TEST_SIZE = 0.2
N_SPLITS = 5
TABLES_DIR = Path("results/tables")
FIGURES_DIR = Path("results/figures")

MODELS = {
    "RF":  lambda seed: RandomForestClassifier(n_estimators=100, random_state=seed),
    "XGB": lambda seed: XGBClassifier(
        n_estimators=100, random_state=seed, eval_metric="mlogloss", verbosity=0
    ),
    "SVM": lambda seed: CalibratedClassifierCV(
        SVC(kernel="rbf", random_state=seed), ensemble=False
    ),
}

CONFIG_COLORS = ["#2ca02c", "#17becf", "#ff7f0e", "#d62728"]


def _cv(estimator, X, y, random_state):
    kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=random_state)
    return float(
        cross_val_score(estimator, X, y, cv=kf, scoring="accuracy", n_jobs=-1).mean()
    )


# ── four pipeline configurations ──────────────────────────────────────────────

def config_correct(X, y, make_clf, rs):
    """A: fully leakage-free."""
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=rs
    )
    pipe = ImbPipeline([
        ("scaler", StandardScaler()),
        ("smote",  SMOTE(random_state=rs)),
        ("clf",    make_clf(rs)),
    ])
    cv = _cv(pipe, X_tr, y_tr, rs)
    pipe.fit(X_tr, y_tr)
    y_pred  = pipe.predict(X_te)
    y_proba = pipe.predict_proba(X_te)
    return cv, evaluate(y_te, y_pred, y_proba), y_te, y_proba


def config_l4(X, y, make_clf, rs):
    """B: +L4 — scaler fit on full dataset before split."""
    X_sc = StandardScaler().fit_transform(X)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_sc, y, test_size=TEST_SIZE, stratify=y, random_state=rs
    )
    pipe = ImbPipeline([("smote", SMOTE(random_state=rs)), ("clf", make_clf(rs))])
    cv = _cv(pipe, X_tr, y_tr, rs)
    pipe.fit(X_tr, y_tr)
    y_pred  = pipe.predict(X_te)
    y_proba = pipe.predict_proba(X_te)
    return cv, evaluate(y_te, y_pred, y_proba), y_te, y_proba


def config_l1_l4(X, y, make_clf, rs):
    """C: +L1+L4 — SMOTE and scaler applied before split."""
    X_sc = StandardScaler().fit_transform(X)
    X_res, y_res = SMOTE(random_state=rs).fit_resample(X_sc, y)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_res, y_res, test_size=TEST_SIZE, random_state=rs
    )
    clf = make_clf(rs)
    cv = _cv(clf, X_tr, y_tr, rs)
    clf.fit(X_tr, y_tr)
    y_pred  = clf.predict(X_te)
    y_proba = clf.predict_proba(X_te)
    return cv, evaluate(y_te, y_pred, y_proba), y_te, y_proba


def config_naive(X, y, make_clf, rs):
    """D: +L1+L2+L4 — full naive; CV on the full resampled pool."""
    X_sc = StandardScaler().fit_transform(X)
    X_res, y_res = SMOTE(random_state=rs).fit_resample(X_sc, y)
    clf = make_clf(rs)
    cv = _cv(clf, X_res, y_res, rs)          # L2: CV on full resampled pool
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_res, y_res, test_size=TEST_SIZE, random_state=rs
    )
    clf.fit(X_tr, y_tr)
    y_pred  = clf.predict(X_te)
    y_proba = clf.predict_proba(X_te)
    return cv, evaluate(y_te, y_pred, y_proba), y_te, y_proba


CONFIGS = {
    "A — Correct":    config_correct,
    "B — +L4":        config_l4,
    "C — +L1+L4":     config_l1_l4,
    "D — Full naive": config_naive,
}


def run_audit():
    df = load_data()
    X = df.drop(columns=["fetal_health"]).values
    y = (df["fetal_health"].values - 1).astype(int)

    records = []
    cv_by = {m: {c: [] for c in CONFIGS} for m in MODELS}
    total = len(MODELS) * len(SEEDS)
    step  = 0

    for model_name, make_clf in MODELS.items():
        print(f"\n[{model_name}]")
        for seed in SEEDS:
            step += 1
            print(f"  seed {seed:2d} ({step:3d}/{total})  ", end="", flush=True)
            for config_name, config_fn in CONFIGS.items():
                cv, m, _, _ = config_fn(X, y, make_clf, seed)
                cv_by[model_name][config_name].append(cv)
                records.append({
                    "model":              model_name,
                    "config":             config_name,
                    "seed":               seed,
                    "cv_accuracy":        cv,
                    "test_accuracy":      m["accuracy"],
                    "balanced_accuracy":  m["balanced_accuracy"],
                    "macro_f1":           m["macro_f1"],
                    "roc_auc_ovr":        m.get("roc_auc_ovr", np.nan),
                    "ece":                m.get("ece", np.nan),
                    "recall_normal":      m["per_class"]["Normal"]["recall"],
                    "recall_suspect":     m["per_class"]["Suspect"]["recall"],
                    "recall_pathological":m["per_class"]["Pathological"]["recall"],
                })
                print(".", end="", flush=True)
            print()

    raw = pd.DataFrame(records)

    # ── aggregated table (mean ± std across seeds) ─────────────────────────────
    agg = (
        raw.groupby(["model", "config"])
        .agg(
            cv_mean=           ("cv_accuracy",        "mean"),
            cv_std=            ("cv_accuracy",        "std"),
            test_mean=         ("test_accuracy",      "mean"),
            test_std=          ("test_accuracy",      "std"),
            bal_acc_mean=      ("balanced_accuracy",  "mean"),
            bal_acc_std=       ("balanced_accuracy",  "std"),
            macro_f1_mean=     ("macro_f1",           "mean"),
            macro_f1_std=      ("macro_f1",           "std"),
            recall_normal_mean=("recall_normal",      "mean"),
            recall_normal_std= ("recall_normal",      "std"),
            recall_suspect_mean=("recall_suspect",    "mean"),
            recall_suspect_std= ("recall_suspect",    "std"),
            recall_patho_mean= ("recall_pathological","mean"),
            recall_patho_std=  ("recall_pathological","std"),
        )
        .round(4)
        .reset_index()
    )

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    agg.to_csv(TABLES_DIR / "leakage_audit.csv",     index=False)
    raw.to_csv(TABLES_DIR / "leakage_audit_raw.csv", index=False)
    print(f"\nSaved: {TABLES_DIR / 'leakage_audit.csv'}")
    print(f"Saved: {TABLES_DIR / 'leakage_audit_raw.csv'}")

    # ── Wilcoxon signed-rank test (naive CV > correct CV) ─────────────────────
    wrows = []
    for model_name in MODELS:
        correct = cv_by[model_name]["A — Correct"]
        naive   = cv_by[model_name]["D — Full naive"]
        stat, p = wilcoxon(naive, correct, alternative="greater")
        wrows.append({
            "model":           model_name,
            "correct_cv_mean": round(np.mean(correct), 4),
            "naive_cv_mean":   round(np.mean(naive),   4),
            "gap_pp":          round((np.mean(naive) - np.mean(correct)) * 100, 2),
            "wilcoxon_stat":   round(stat, 3),
            "p_value":         round(p, 4),
            "significant":     p < 0.05,
        })
    wilcoxon_df = pd.DataFrame(wrows)
    wilcoxon_df.to_csv(TABLES_DIR / "wilcoxon.csv", index=False)
    print(f"Saved: {TABLES_DIR / 'wilcoxon.csv'}")

    print("\nWilcoxon (naive CV > correct CV, 10 seeds):")
    for _, row in wilcoxon_df.iterrows():
        sig = "p<0.05" if row["significant"] else "ns"
        print(f"  {row['model']:4s}  gap={row['gap_pp']:+.2f}pp  "
              f"stat={row['wilcoxon_stat']}  p={row['p_value']:.4f}  {sig}")

    # ── figure 1: leakage attribution (3-panel, error bars across seeds) ───────
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    model_names  = list(MODELS.keys())
    config_order = list(CONFIGS.keys())

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for i, (ax, model_name) in enumerate(zip(axes, model_names)):
        m_agg = agg[agg["model"] == model_name].copy()
        m_agg["_ord"] = m_agg["config"].map({c: j for j, c in enumerate(config_order)})
        m_agg = m_agg.sort_values("_ord")          # A at bottom → D at top

        bars = ax.barh(
            m_agg["config"], m_agg["cv_mean"],
            xerr=m_agg["cv_std"], capsize=4,
            color=CONFIG_COLORS, height=0.5,
        )
        correct_cv = m_agg.loc[m_agg["config"] == "A — Correct", "cv_mean"].values[0]
        ax.axvline(correct_cv, color="#2ca02c", linestyle="--", linewidth=1, alpha=0.7)
        for bar, val in zip(bars, m_agg["cv_mean"]):
            ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", fontsize=8)
        ax.set_xlabel("CV Accuracy")
        ax.set_title(model_name, fontsize=11)
        ax.set_xlim(0.85, 1.04)
        if i > 0:
            ax.tick_params(labelleft=False)

    fig.suptitle(
        "Leakage attribution — CV accuracy by config (mean ± std, 10 seeds, 5-fold)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "leakage_attribution.png", dpi=150)
    plt.close(fig)
    print(f"\nSaved: {FIGURES_DIR / 'leakage_attribution.png'}")

    # ── figure 2: per-class recall, correct vs naive (3-panel) ────────────────
    col_map = {
        "Normal":       "recall_normal",
        "Suspect":      "recall_suspect",
        "Pathological": "recall_pathological",
    }
    classes = list(col_map.keys())

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    for i, (ax, model_name) in enumerate(zip(axes, model_names)):
        sub = raw[raw["model"] == model_name]
        rec_c = [sub[sub["config"] == "A — Correct"]   [col_map[c]].mean() for c in classes]
        rec_n = [sub[sub["config"] == "D — Full naive"] [col_map[c]].mean() for c in classes]
        err_c = [sub[sub["config"] == "A — Correct"]   [col_map[c]].std()  for c in classes]
        err_n = [sub[sub["config"] == "D — Full naive"] [col_map[c]].std()  for c in classes]

        x = np.arange(len(classes))
        w = 0.35
        ax.bar(x - w / 2, rec_c, w, yerr=err_c, capsize=3, label="Correct", color="#2ca02c")
        ax.bar(x + w / 2, rec_n, w, yerr=err_n, capsize=3, label="Naive",   color="#d62728")
        ax.set_xticks(x)
        ax.set_xticklabels(classes)
        ax.set_ylim(0, 1.15)
        ax.set_title(model_name, fontsize=11)
        ax.legend(fontsize=8)
        if i == 0:
            ax.set_ylabel("Recall")

    fig.suptitle("Per-class recall: correct vs naive (mean ± std, 10 seeds)", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "per_class_recall.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {FIGURES_DIR / 'per_class_recall.png'}")

    # ── calibration curves (RF, seed 0, for illustration) ─────────────────────
    _, _, y_te_c, pr_c = config_correct(X, y, MODELS["RF"], 0)
    _, _, y_te_n, pr_n = config_naive(X, y,   MODELS["RF"], 0)

    fig = plot_calibration(y_te_c, pr_c, title="Reliability diagram — Correct (RF, seed 0)")
    fig.savefig(FIGURES_DIR / "calibration_correct.png", dpi=150)
    plt.close(fig)

    fig = plot_calibration(y_te_n, pr_n, title="Reliability diagram — Naive (RF, seed 0)")
    fig.savefig(FIGURES_DIR / "calibration_naive.png", dpi=150)
    plt.close(fig)
    print(f"Saved calibration curves to {FIGURES_DIR}/")

    return agg, wilcoxon_df


if __name__ == "__main__":
    run_audit()
