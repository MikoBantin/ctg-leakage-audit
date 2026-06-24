"""Tests for data loader and integrity checks."""

from pathlib import Path

import pandas as pd
import pytest

from src.data.load import DATA_PATH, CLASS_NAMES, integrity_report, load_data

# 21 feature columns matching the public fetal-health CTG dataset
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


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Synthetic CTG dataframe: 4 rows, 1 exact duplicate, all classes present."""
    rows = [
        {**dict.fromkeys(FEATURE_COLS, 0.0), "fetal_health": 1.0},  # Normal
        {**dict.fromkeys(FEATURE_COLS, 0.0), "fetal_health": 1.0},  # Normal — duplicate
        {**dict.fromkeys(FEATURE_COLS, 1.0), "fetal_health": 2.0},  # Suspect
        {**dict.fromkeys(FEATURE_COLS, 2.0), "fetal_health": 3.0},  # Pathological
    ]
    return pd.DataFrame(rows)


def test_integrity_report_shape(sample_df):
    report = integrity_report(sample_df)
    assert report["n_rows"] == 4
    assert report["n_features"] == 21


def test_integrity_report_no_missing(sample_df):
    report = integrity_report(sample_df)
    assert report["n_missing"] == 0


def test_integrity_report_detects_duplicates(sample_df):
    report = integrity_report(sample_df)
    assert report["n_duplicates"] == 1


def test_integrity_report_class_counts(sample_df):
    counts = integrity_report(sample_df)["class_counts"]
    assert counts["Normal"] == 2
    assert counts["Suspect"] == 1
    assert counts["Pathological"] == 1


def test_integrity_report_missing_values():
    rows = [{**dict.fromkeys(FEATURE_COLS, 0.0), "fetal_health": 1.0}]
    df = pd.DataFrame(rows)
    df.loc[0, "baseline value"] = float("nan")
    report = integrity_report(df)
    assert report["n_missing"] == 1


def test_load_data_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_data(Path("data/raw/does_not_exist.csv"))


@pytest.mark.skipif(not DATA_PATH.exists(), reason="raw CSV not present")
def test_load_data_real_file():
    df = load_data()
    report = integrity_report(df)
    assert report["n_rows"] == 2126
    assert report["n_features"] == 21
    assert report["n_missing"] == 0
    assert set(report["class_counts"]) == {"Normal", "Suspect", "Pathological"}
