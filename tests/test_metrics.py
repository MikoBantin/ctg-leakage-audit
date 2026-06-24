"""Tests for evaluation metrics module."""

import numpy as np
import pytest

from src.evaluation.metrics import evaluate, expected_calibration_error

Y_TRUE = np.array([0, 0, 1, 1, 2, 2])
Y_PRED_PERFECT = np.array([0, 0, 1, 1, 2, 2])
Y_PRED_WRONG = np.array([1, 1, 0, 0, 0, 0])

PERFECT_PROBA = np.array([
    [1.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [0.0, 0.0, 1.0],
])


def test_perfect_accuracy():
    result = evaluate(Y_TRUE, Y_PRED_PERFECT)
    assert result["accuracy"] == 1.0
    assert result["balanced_accuracy"] == 1.0
    assert result["macro_f1"] == 1.0


def test_per_class_keys_present():
    result = evaluate(Y_TRUE, Y_PRED_PERFECT)
    for label in ["Normal", "Suspect", "Pathological"]:
        assert label in result["per_class"]
        assert set(result["per_class"][label]) == {"precision", "recall", "f1", "support"}


def test_roc_auc_and_ece_present_with_proba():
    result = evaluate(Y_TRUE, Y_PRED_PERFECT, y_proba=PERFECT_PROBA)
    assert "roc_auc_ovr" in result
    assert "ece" in result
    assert result["roc_auc_ovr"] == pytest.approx(1.0, abs=1e-6)


def test_roc_auc_absent_without_proba():
    result = evaluate(Y_TRUE, Y_PRED_PERFECT)
    assert "roc_auc_ovr" not in result
    assert "ece" not in result


def test_ece_zero_for_perfect_calibration():
    y_true = np.array([0, 1])
    y_proba = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert expected_calibration_error(y_true, y_proba) == pytest.approx(0.0, abs=1e-6)


def test_ece_positive_for_miscalibrated():
    y_true = np.array([0, 0, 1, 1])
    # Confidently wrong on half the samples
    y_proba = np.array([
        [0.9, 0.1],
        [0.1, 0.9],
        [0.9, 0.1],
        [0.1, 0.9],
    ])
    assert expected_calibration_error(y_true, y_proba) > 0.0


def test_balanced_accuracy_lower_than_accuracy_on_skewed_preds():
    # All predictions are majority class — accuracy looks OK but balanced is worse
    y_true = np.array([0, 0, 0, 0, 1, 2])
    y_pred = np.array([0, 0, 0, 0, 0, 0])
    result = evaluate(y_true, y_pred)
    assert result["balanced_accuracy"] < result["accuracy"]


def test_pathological_recall_in_per_class():
    result = evaluate(Y_TRUE, Y_PRED_PERFECT)
    assert result["per_class"]["Pathological"]["recall"] == pytest.approx(1.0)
