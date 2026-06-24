# CTG Leakage Audit

> **Reproducibility and data-leakage audit of fetal-health cardiotocography (CTG) machine learning**
> Preprint target: arXiv · Journal target: JMIR / PLOS Digital Health / BMC Medical Informatics

## What this repo proves

Published CTG classifiers routinely report 96–99% accuracy on the public Kaggle/UCI
fetal-health dataset (2,126 records, 3 classes). This audit shows that most of that
is artefact — chiefly SMOTE applied to the full dataset before splitting — and provides
a leakage-free corrected benchmark alongside a reusable reporting checklist.

**Headline result (Random Forest, 5-fold CV):**

| Pipeline | CV Accuracy | Balanced Accuracy | Suspect Recall |
|---|---|---|---|
| Naive (leaky) | **98.0%** | 97.1% | 97.8% |
| Correct (no leakage) | **94.2%** | 86.2% | **74.6%** |
| Gap | −3.8 pp | −10.9 pp | −23.2 pp |

The Suspect class (n = 295) is the most affected. Plain accuracy hides a 23 pp recall
collapse — on the clinically important minority class.

## Leakage sources audited

| ID | Source | CV inflation |
|---|---|---|
| L1 | SMOTE applied before train/test split | +3.3 pp |
| L2 | CV on full resampled pool (SMOTE outside folds) | +0.1 pp |
| L3 | Duplicate / near-duplicate rows across splits | 5 rows |
| L4 | Scaler fit on full dataset before splitting | +0.3 pp |
| L5 | Plain accuracy on 78/14/8 imbalanced test set | hides −23 pp Suspect recall |

## Reproduce

```bash
# 1. clone and create a virtual environment (Python 3.13)
git clone https://github.com/MikoBantin/ctg-leakage-audit.git
cd ctg-leakage-audit
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2. install dependencies
pip install -r requirements.txt
pip install -e ".[dev]"          # adds ruff, black, pytest

# 3. download the dataset
# Fetal Health Classification — public on Kaggle and UCI
# Place the CSV at: data/raw/fetal_health.csv

# 4. run tests
pytest

# 5. run the audit (writes to results/tables/ and results/figures/)
python src/experiments/run_leakage_audit.py
python src/experiments/run_duplicates.py
```

## Results

Generated outputs (committed to the repo):

| File | Contents |
|---|---|
| `results/tables/leakage_audit.csv` | CV and test metrics for all 4 pipeline configurations |
| `results/tables/duplicates.csv` | Exact and near-duplicate counts |
| `results/figures/leakage_attribution.png` | Bar chart: CV accuracy by leakage configuration |
| `results/figures/per_class_recall.png` | Per-class recall: correct vs naive |
| `results/figures/calibration_correct.png` | Reliability diagram — correct pipeline |
| `results/figures/calibration_naive.png` | Reliability diagram — naive pipeline |

## Repo structure

```
src/
  data/load.py              — load CSV, integrity checks (NaNs, duplicates, class counts)
  pipelines/
    naive.py                — leaky baseline: SMOTE+scaler before split (reproduces ~98%)
    correct.py              — corrected pipeline: imblearn Pipeline, split-first
  experiments/
    run_leakage_audit.py    — runs 4 configs A–D, saves tables and figures
    run_duplicates.py       — quantifies L3 (duplicate leakage)
  evaluation/metrics.py     — balanced accuracy, per-class F1, OvR ROC-AUC, ECE
tests/                      — pytest suite (19 tests; real-data tests skip if CSV absent)
checklist/                  — reusable CTG ML reporting checklist (filled in for this audit)
paper/                      — section-by-section paper outline with claims and backing evidence
results/                    — generated tables and figures (committed)
```

## Checklist

`checklist/ctg_model_info_sheet.md` is a standalone deliverable: a reusable reporting
checklist for CTG ML studies, filled in with this audit's numbers. Copy the "Your study"
column and attach it to your next submission.

## Citation

*To be added at preprint submission.*
