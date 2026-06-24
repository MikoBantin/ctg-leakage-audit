# CTG Leakage Audit

> **Reproducibility and data-leakage audit of fetal-health cardiotocography (CTG) machine learning**
> Preprint target: arXiv · Journal target: JMIR / PLOS Digital Health / BMC Medical Informatics

## What this repo proves

Published CTG classifiers routinely report ~96–99% accuracy on the public Kaggle/UCI
fetal-health dataset (2,126 records, 3 classes). This audit quantifies how much of that
is artefact of data leakage — chiefly SMOTE applied before the train/test split and
improper cross-validation — and provides a leakage-free corrected benchmark alongside a
reusable reporting checklist.

## Leakage sources audited

| ID | Source |
|----|--------|
| L1 | SMOTE applied before train/test split |
| L2 | Resampling outside CV folds |
| L3 | Duplicate / near-duplicate rows across splits |
| L4 | Scaler / feature-selection fit on full dataset |
| L5 | Metric inflation from class imbalance (plain accuracy on 78/14/8 split) |

## Reproduce

```bash
# install dev tools
pip install -e ".[dev]"

# run tests
pytest

# run the audit (generates results/tables/ and results/figures/)
python src/experiments/run_leakage_audit.py
python src/experiments/run_duplicates.py
```

## Dataset

Fetal Health Classification — public on Kaggle and UCI.
Download the CSV and place it at `data/raw/fetal_health.csv` (never committed to git).

## Structure

```
src/pipelines/naive.py    — leaky baseline (reproduces ~99% accuracy)
src/pipelines/correct.py  — leakage-free corrected pipeline
src/experiments/          — runs both pipelines, emits result tables
src/evaluation/metrics.py — balanced accuracy, per-class F1, ROC-AUC, ECE
checklist/                — reusable CTG ML reporting checklist
paper/                    — section-by-section paper outline
```

## Citation

*To be added at submission.*
