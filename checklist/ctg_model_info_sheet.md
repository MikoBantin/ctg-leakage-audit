# CTG ML Reporting Checklist

A reusable model info sheet for fetal-health cardiotocography (CTG) machine learning,
in the spirit of Kapoor & Narayanan (2022). Designed to surface the specific failure
modes — data leakage, metric inflation, calibration omission — that recur in the CTG
classification literature.

Copy this sheet, fill in your own values, and attach it as a supplementary file or
appendix. Reviewers can verify each item independently.

---

## 1. Dataset

| Field | This audit | Your study |
|---|---|---|
| Dataset name | Fetal Health Classification | |
| Source | Kaggle / UCI | |
| Version / retrieval date | UCI v1, 2020 | |
| Records | 2,126 | |
| Features | 21 numeric CTG features | |
| Target | `fetal_health` (1 Normal, 2 Suspect, 3 Pathological) | |
| Class distribution | 1655 / 295 / 176 (78% / 14% / 8%) | |
| Missing values | 0 | |
| Exact duplicates | 13 rows (0.61%) | |

**Why it matters:** Class distribution and duplicate counts must be stated. A 78/14/8
split means a majority-class predictor achieves ~78% accuracy — a baseline any claimed
improvement must clearly exceed.

---

## 2. Splitting protocol

- [x] Train/test split performed **before** any resampling or preprocessing
- [x] Split is stratified on the target label
- [x] Test set is held out and never used during model selection or hyperparameter tuning
- [x] Random seed fixed and reported

| Field | This audit | Your study |
|---|---|---|
| Split ratio | 80 / 20 | |
| Stratified | Yes | |
| Random seed | 42 | |
| Test set size | 426 rows | |

**Common failure (L1):** Many published CTG papers apply SMOTE to the full dataset
*before* splitting. Synthetic samples derived from test-set rows then appear in training,
inflating CV accuracy by 3.7–5.7 pp across RF, XGBoost, and SVM
(Wilcoxon p=0.001, 10/10 seeds naive > correct for all three models).

---

## 3. Resampling

- [x] Oversampling (SMOTE or equivalent) applied **inside** each training fold only
- [x] Resampler wrapped in an `imblearn.Pipeline` — it cannot see validation or test data
- [x] Resampling is NOT applied to the test set

| Field | This audit | Your study |
|---|---|---|
| Method | SMOTE (k=5 neighbours, default) | |
| Applied to | Training folds only, via Pipeline | |
| Library | imbalanced-learn 0.14.2 | |

**Common failure (L2):** Running SMOTE once on the whole training set and then
performing cross-validation on the resampled pool leaks synthetic points across folds.
Use `imblearn.pipeline.Pipeline` to prevent this.

---

## 4. Preprocessing

- [x] StandardScaler fit on training data only
- [x] Scaler transform applied to validation/test using training-set parameters
- [x] No feature selection or dimensionality reduction fit on the full dataset

**Common failure (L4):** Fitting a scaler on the full dataset (train + test) before
splitting causes the model to use test-set statistics during training. Effect on this
dataset: ≤0.1 pp CV accuracy inflation across all three models tested.

---

## 5. Duplicate and near-duplicate check

| Metric | This audit | Your study |
|---|---|---|
| Exact duplicates in full dataset | 13 (0.61%) | |
| Exact train/test overlap | 5 rows | |
| Near-duplicate pairs (dist < 0.05) | 5 pairs | |

**Why it matters (L3):** Duplicate rows that appear in both train and test sets
constitute direct leakage. Near-duplicates have a similar effect. Check before
reporting held-out accuracy.

---

## 6. Evaluation metrics

- [x] Plain accuracy reported (with caveat about class imbalance)
- [x] Balanced accuracy reported
- [x] Per-class precision, recall, and F1 reported for all three classes
- [x] Macro-averaged F1 reported
- [x] One-vs-rest ROC-AUC reported
- [x] Expected calibration error (ECE) computed
- [x] Reliability diagram (calibration curve) included
- [x] **Pathological-class recall explicitly foregrounded** (clinically critical class)

| Metric | Correct pipeline | Naive pipeline | Delta |
|---|---|---|---|
| CV Accuracy — RF | 93.7% ± 0.4% | 97.8% ± 0.1% | −4.1 pp |
| CV Accuracy — XGB | 94.5% ± 0.4% | 98.2% ± 0.1% | −3.7 pp |
| CV Accuracy — SVM | 88.2% ± 0.5% | 93.9% ± 0.2% | −5.7 pp |
| Balanced Accuracy (RF) | 90.5% | 97.7% | −7.2 pp |
| Macro F1 (RF) | 90.3% | 97.7% | −7.4 pp |
| Recall — Suspect (RF/XGB) | **~81%** | **~97%** | **~−16 pp** |
| Recall — Pathological (RF) | 93.7% | 99.8% | −6.1 pp |
| Wilcoxon p (all models) | — | — | **0.001** |

**Common failure (L5):** Reporting only plain accuracy on a 78/14/8 split hides the
collapse of Suspect-class recall. The naive pipeline appears to achieve ~99% accuracy
while missing ~13–19% of Suspect cases on the corrected evaluation (model-dependent).

---

## 7. Reproducibility

- [x] All random seeds pinned and reported; results averaged across 10 seeds (0–9)
- [x] Code publicly released with open licence
- [x] `requirements.txt` with pinned package versions provided
- [x] Python version stated (3.13.3)
- [x] Dataset retrieval instructions documented (data not bundled — public source only)
- [ ] Zenodo DOI assigned (at submission)

| Field | This audit | Your study |
|---|---|---|
| Python | 3.13.3 | |
| scikit-learn | 1.9.0 | |
| imbalanced-learn | 0.14.2 | |
| xgboost | 3.3.0 | |
| Random seed | 0–9 (10-seed sweep; mean ± std reported) | |
| Code repository | github.com/MikoBantin/ctg-leakage-audit | |

---

## 8. Scope and limitations

State these explicitly to bound your claims:

- [x] Results are specific to this one public dataset (n = 2,126)
- [x] Retrospective study — no prospective clinical validation
- [x] Claims limited to *public-benchmark CTG ML*, not clinical deployment
- [x] Models not compared against clinician performance

---

## Quick-reference: leakage taxonomy

| ID | Source | Effect on this dataset |
|---|---|---|
| L1 | SMOTE before train/test split | +3.3–5.3 pp CV accuracy (RF/XGB/SVM) |
| L2 | CV on full resampled pool | <0.3 pp CV accuracy |
| L3 | Duplicate rows across splits | 5 rows; minor |
| L4 | Scaler fit on full dataset | ≤0.1 pp CV accuracy |
| L5 | Plain accuracy on imbalanced test | Hides ~13–19 pp Suspect recall gap |
