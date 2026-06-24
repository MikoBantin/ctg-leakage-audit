"""Honest evaluation: balanced accuracy, per-class P/R/F1, macro-F1, OvR ROC-AUC, ECE."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

CLASS_NAMES = ["Normal", "Suspect", "Pathological"]
N_BINS = 10


def expected_calibration_error(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = N_BINS,
) -> float:
    """Mean ECE across classes using one-vs-rest binning."""
    n_classes = y_proba.shape[1]
    ece_per_class = []
    bin_edges = np.linspace(0, 1, n_bins + 1)

    for c in range(n_classes):
        y_bin = (y_true == c).astype(int)
        probs = y_proba[:, c]
        bin_idx = np.digitize(probs, bin_edges[1:-1])
        ece = 0.0
        for b in range(n_bins):
            mask = bin_idx == b
            if mask.sum() == 0:
                continue
            avg_conf = probs[mask].mean()
            avg_acc = y_bin[mask].mean()
            ece += mask.sum() / len(y_true) * abs(avg_conf - avg_acc)
        ece_per_class.append(ece)

    return float(np.mean(ece_per_class))


def evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
    labels: list[str] = CLASS_NAMES,
) -> dict:
    """
    Full honest evaluation: accuracy, balanced accuracy, per-class P/R/F1,
    macro-F1, OvR ROC-AUC, and ECE (when probabilities are supplied).
    """
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(labels))), zero_division=0
    )

    per_class = {
        label: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i, label in enumerate(labels)
    }

    result = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "per_class": per_class,
    }

    if y_proba is not None:
        result["roc_auc_ovr"] = float(
            roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
        )
        result["ece"] = expected_calibration_error(y_true, y_proba)

    return result


def print_report(metrics: dict, title: str = "") -> None:
    """Pretty-print an evaluate() result dict."""
    if title:
        print(f"\n{'=' * 50}")
        print(f"  {title}")
        print("=" * 50)
    print(f"  Accuracy          : {metrics['accuracy']:.4f}")
    print(f"  Balanced accuracy : {metrics['balanced_accuracy']:.4f}")
    print(f"  Macro F1          : {metrics['macro_f1']:.4f}")
    if "roc_auc_ovr" in metrics:
        print(f"  ROC-AUC (OvR)     : {metrics['roc_auc_ovr']:.4f}")
    if "ece" in metrics:
        print(f"  ECE               : {metrics['ece']:.4f}")
    print()
    print(f"  {'Class':<15} {'Prec':>6} {'Rec':>6} {'F1':>6} {'N':>6}")
    print(f"  {'-' * 42}")
    for label, vals in metrics["per_class"].items():
        print(
            f"  {label:<15} {vals['precision']:>6.3f} {vals['recall']:>6.3f} "
            f"{vals['f1']:>6.3f} {vals['support']:>6}"
        )


def plot_calibration(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    labels: list[str] = CLASS_NAMES,
    title: str = "Reliability diagram",
    save_path: Path | None = None,
) -> plt.Figure:
    """Plot per-class reliability curves and optionally save the figure."""
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")

    for i, label in enumerate(labels):
        y_bin = (y_true == i).astype(int)
        prob_true, prob_pred = calibration_curve(y_bin, y_proba[:, i], n_bins=N_BINS)
        ax.plot(prob_pred, prob_true, marker="o", label=label)

    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150)

    return fig
