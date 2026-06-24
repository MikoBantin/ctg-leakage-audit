"""Core leakage proof: naive CV is inflated; correct pipeline CV and test are consistent."""

import numpy as np
import pandas as pd
import pytest

from src.data.load import DATA_PATH
from src.pipelines.correct import run_correct
from src.pipelines.naive import run_naive

# Smallest observed gap across models on the real dataset (~3 pp for XGB)
MIN_LEAKAGE_GAP = 0.02

FEATURE_COLS = [
    "baseline value", "accelerations", "fetal_movement", "uterine_contractions",
    "light_decelerations", "severe_decelerations", "prolongued_decelerations",
    "abnormal_short_term_variability", "mean_value_of_short_term_variability",
    "percentage_of_time_with_abnormal_long_term_variability",
    "mean_value_of_long_term_variability", "histogram_width", "histogram_min",
    "histogram_max", "histogram_number_of_peaks", "histogram_number_of_zeroes",
    "histogram_mode", "histogram_mean", "histogram_median", "histogram_variance",
    "histogram_tendency",
]


def make_synthetic_ctg(n_samples: int = 600, random_state: int = 0) -> pd.DataFrame:
    """Imbalanced 3-class dataset shaped like the real CTG data for offline testing."""
    rng = np.random.default_rng(random_state)
    n_normal = int(n_samples * 0.70)
    n_suspect = int(n_samples * 0.20)
    n_patho = n_samples - n_normal - n_suspect

    X_normal = rng.standard_normal((n_normal, 21))
    X_suspect = rng.standard_normal((n_suspect, 21)) + 1.5
    X_patho = rng.standard_normal((n_patho, 21)) + 3.0

    X = np.vstack([X_normal, X_suspect, X_patho])
    y = np.array(
        [1.0] * n_normal + [2.0] * n_suspect + [3.0] * n_patho
    )

    df = pd.DataFrame(X, columns=FEATURE_COLS)
    df["fetal_health"] = y
    return df


@pytest.fixture(scope="module")
def synthetic_df():
    return make_synthetic_ctg()


def test_naive_cv_higher_than_correct_cv(synthetic_df):
    """Naive CV must exceed correct CV — leakage inflates the training signal."""
    naive = run_naive(df=synthetic_df, model="rf")
    correct = run_correct(df=synthetic_df, model="rf")
    assert naive["cv_mean"] > correct["cv_mean"], (
        f"naive CV {naive['cv_mean']:.4f} should exceed correct CV {correct['cv_mean']:.4f}"
    )


def test_correct_cv_and_test_consistent(synthetic_df):
    """Correct pipeline CV and test accuracy should be within 5 pp — no leakage."""
    result = run_correct(df=synthetic_df, model="rf")
    gap = abs(result["cv_mean"] - result["test_accuracy"])
    assert gap < 0.05, (
        f"CV/test gap {gap:.4f} is too large for a leakage-free pipeline"
    )


@pytest.mark.skipif(not DATA_PATH.exists(), reason="raw CSV not present")
def test_naive_cv_above_95_on_real_data():
    """Naive CV must exceed 95% on the real dataset — reproducing published inflation."""
    result = run_naive(model="rf")
    assert result["cv_mean"] > 0.95, (
        f"Naive CV {result['cv_mean']:.4f} not high enough to match published reports"
    )


@pytest.mark.skipif(not DATA_PATH.exists(), reason="raw CSV not present")
def test_leakage_gap_on_real_data():
    """Naive CV must beat correct CV by at least {MIN_LEAKAGE_GAP*100:.0f} pp on the real dataset."""
    naive = run_naive(model="rf")
    correct = run_correct(model="rf")
    gap = naive["cv_mean"] - correct["cv_mean"]
    assert gap >= MIN_LEAKAGE_GAP, (
        f"Leakage gap {gap:.4f} smaller than expected minimum {MIN_LEAKAGE_GAP}"
    )
