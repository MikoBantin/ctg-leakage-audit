# Paper Outline

**Working title:** "Data Leakage in Fetal Health CTG Classification: A Reproducibility
Audit and Corrected Benchmark"

**Venues:** arXiv preprint first, then JMIR / PLOS Digital Health / BMC Medical
Informatics & Decision Making.

**Framing throughout:** constructive. "Here is a protocol for doing this correctly" —
not naming or shaming specific papers.

---

## Abstract

**Claims to make (≤ 250 words):**
- CTG ML papers on the public fetal-health dataset routinely report 96–99% accuracy.
- We identify five data-leakage sources (L1–L5) and quantify each source's
  contribution to the inflation.
- Headline number: naive pipeline CV accuracy **98.0%** → correct pipeline **94.2%**
  (3.8 pp gap); balanced accuracy gap is **10.9 pp** (97.1% vs 86.2%).
- The most clinically important finding: Suspect-class recall collapses from **97.8%**
  to **74.6%** once leakage is removed — a class that plain accuracy hides entirely.
- Contributions: (1) leakage taxonomy with per-source attribution, (2) leakage-free
  corrected benchmark, (3) reusable reporting checklist.

**Backing evidence:** `results/tables/leakage_audit.csv` (configs A and D).

---

## 1. Introduction

**Purpose:** motivate the audit; state contributions clearly.

**Narrative arc:**
1. CTG monitoring is routine in perinatal care; automated classification of fetal
   wellbeing is an active ML research area.
2. The public Kaggle/UCI fetal-health dataset (2,126 records, 3 classes) is the
   de facto benchmark — dozens of papers use it.
3. Reported accuracies cluster at 96–99%, which is implausibly high for a 3-class
   problem on a small, imbalanced dataset.
4. Kapoor & Narayanan (2022) showed that data leakage is endemic in ML-based science;
   this paper applies that lens to CTG classification specifically.

**Contributions (numbered list in the paper):**
- C1: First systematic audit of data-leakage sources in CTG ML, with per-source
  attribution (Table 1).
- C2: A leakage-free corrected benchmark that future work can compare against.
- C3: A reusable CTG ML reporting checklist.

---

## 2. Related Work

**Subsections:**
- *Reproducibility in ML-based medical research* — Kapoor & Narayanan (2022) as the
  main reference; Schulam & Saria (2019) on leakage in clinical prediction.
- *Imbalanced learning and SMOTE* — Chawla et al. (2002) original SMOTE; Blagus &
  Lusa (2013) on SMOTE pitfalls; note that imblearn Pipeline is the established fix.
- *CTG classification literature* — brief, non-attributive survey: note the accuracy
  range, common models (RF, XGBoost, SVM), and the near-universal absence of
  calibration reporting.

**Tone:** descriptive, not accusatory. No specific paper names in the leakage
discussion.

---

## 3. Dataset

**Claims:**
- 2,126 records; 21 numeric CTG features; target: fetal_health (1/2/3).
- Class distribution: 1,655 Normal (77.9%), 295 Suspect (13.9%), 176 Pathological
  (8.3%) — a 9:1.7:1 ratio.
- Zero missing values.
- 13 exact-duplicate rows (0.61%) — identified by `src/data/load.py`.
- The severe imbalance is the proximate cause of SMOTE adoption and therefore of L1/L2
  leakage.

**Backing evidence:** `src/data/load.py` integrity report; `results/tables/duplicates.csv`.

---

## 4. Methods

**4.1 Leakage taxonomy**

| ID | Source | Correct fix |
|---|---|---|
| L1 | SMOTE before train/test split | Split first; SMOTE inside Pipeline |
| L2 | CV on full resampled pool | Wrap resampler in imblearn Pipeline |
| L3 | Duplicate rows across splits | Deduplicate or check overlap |
| L4 | Scaler fit on full dataset | Fit scaler on training data only |
| L5 | Plain accuracy on imbalanced test | Report balanced accuracy, macro F1 |

**4.2 Naive pipeline** (`src/pipelines/naive.py`)

Standard scaler → SMOTE → train/test split → RF (n=100) → CV on full resampled pool.
Intentionally reproduces the common published approach.

**4.3 Corrected pipeline** (`src/pipelines/correct.py`)

Stratified split first → `imblearn.Pipeline(StandardScaler, SMOTE, RF)` → 5-fold
stratified CV on training set only → evaluate on untouched test set.

**4.4 Leakage attribution experiment** (`src/experiments/run_leakage_audit.py`)

Four configurations (A–D), each adding one leakage source incrementally:
- A: fully correct (baseline)
- B: +L4 (scaler before split)
- C: +L1+L4 (SMOTE before split)
- D: +L1+L2+L4 (full naive; CV on resampled pool)

**4.5 Evaluation protocol**

All reported metrics: accuracy, balanced accuracy, macro F1, per-class precision/recall/
F1, OvR ROC-AUC, ECE. Implementation: `src/evaluation/metrics.py`.
Calibration curves: `results/figures/calibration_*.png`.

---

## 5. Results

**5.1 Leakage attribution (Table 1)**

| Config | CV Acc | Balanced Acc | Suspect Recall | Patho Recall |
|---|---|---|---|---|
| A — Correct | 94.2% | 86.2% | 74.6% | 88.6% |
| B — +L4 | 94.5% | 87.5% | 78.0% | 88.6% |
| C — +L1+L4 | 97.9% | 97.1% | 97.8% | 99.7% |
| D — Full naive | 98.0% | 97.1% | 97.8% | 99.7% |

**Key claims:**
- L1 (SMOTE before split) accounts for **3.3 of the 3.8 pp** CV accuracy gap — by far
  the dominant source.
- L4 adds 0.3 pp; L2 adds 0.1 pp.
- Total balanced accuracy inflation: **10.9 pp**.
- Suspect recall is the most affected metric: **−23.2 pp** (97.8% → 74.6%).

**Backing evidence:** `results/tables/leakage_audit.csv`; `results/figures/leakage_attribution.png`.

**5.2 Per-class recall (Figure 2)**

Bar chart comparing correct vs naive pipeline recall for all three classes.
Key visual: the near-perfect Suspect recall under the naive pipeline collapses to 74.6%
under the correct evaluation. File: `results/figures/per_class_recall.png`.

**5.3 Duplicate analysis**

13 exact duplicates in the full dataset (0.61%); 5 cross the 80/20 train/test
boundary. Effect on accuracy: minor, but flagged as L3 leakage.
File: `results/tables/duplicates.csv`.

**5.4 Calibration (Figure 3)**

Reliability diagrams for naive and correct pipelines.
The naive model, trained on synthetic SMOTE data, shows overconfidence — probability
outputs are not well-calibrated despite high accuracy.
Files: `results/figures/calibration_naive.png`, `results/figures/calibration_correct.png`.

---

## 6. Discussion

**Main points:**
1. **L1 dominates.** SMOTE-before-split is responsible for nearly all the accuracy
   inflation. Practitioners who switch to `imblearn.Pipeline` will recover most of the
   gap immediately.
2. **Accuracy is the wrong metric.** Plain accuracy on this dataset is a majority-class
   proxy; the Pathological and Suspect classes are clinically relevant and must be
   reported separately.
3. **Calibration matters in clinical contexts.** The naive model is overconfident —
   probability outputs cannot be trusted as risk scores.
4. **The corrected benchmark.** 94.2% CV / 92.0% test accuracy (RF, 5-fold, no
   leakage) with 74.6% Suspect recall is the honest reference point future work
   should beat.

**Limitations:**
- Single dataset (n = 2,126); results may not generalise to other CTG datasets.
- Retrospective study; no prospective clinical validation.
- Scope limited to public-benchmark CTG ML, not clinical deployment.
- Models not compared against clinician performance.

---

## 7. Checklist

Short section pointing to `checklist/ctg_model_info_sheet.md`.
Describe each of the 8 sections briefly; reproduce the leakage taxonomy table (same as
Methods Table 1). Emphasise that the "Your study" column makes it directly reusable.

---

## 8. Conclusion

- The 98% accuracy reported in CTG classification literature is substantially an
  artefact of data leakage, primarily from SMOTE applied before splitting.
- The corrected benchmark (94.2% CV / 74.6% Suspect recall) is the honest baseline.
- The reporting checklist is the reusable contribution: fill it in, attach it to
  submissions, ask reviewers to verify it.
- One-sentence call to action: use `imblearn.Pipeline`.
