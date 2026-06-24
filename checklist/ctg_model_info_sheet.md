# CTG ML Reporting Checklist

*Reusable model info sheet for fetal-health CTG machine learning — filled in during Phase 6.*

## Data provenance
- [ ] Dataset name, version, and public URL stated
- [ ] Record count and feature count confirmed
- [ ] Class distribution reported (not assumed balanced)

## Splitting
- [ ] Train/test split performed **before** any resampling or preprocessing
- [ ] Stratification used given class imbalance
- [ ] Random seed fixed and reported

## Resampling
- [ ] SMOTE (or equivalent) applied **inside** training folds only
- [ ] Resampling wrapped in a Pipeline so it never sees test data

## Preprocessing
- [ ] Scaler / feature selector fit on training data only
- [ ] No test statistics used during preprocessing

## Duplicate check
- [ ] Exact-duplicate rows identified and counted
- [ ] Near-duplicate overlap across splits checked

## Evaluation
- [ ] Per-class precision, recall, F1 reported (not just accuracy)
- [ ] Macro-F1 and balanced accuracy reported
- [ ] OvR ROC-AUC reported
- [ ] Calibration assessed (ECE + reliability curve)
- [ ] Minority-class (Pathological) recall foregrounded

## Reproducibility
- [ ] All random seeds pinned
- [ ] Code publicly released (GitHub + Zenodo DOI)
- [ ] Environment specified (Python version, package versions)
