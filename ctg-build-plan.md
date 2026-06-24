# CTG Reproducibility & Data-Leakage Audit — Build Plan

A build spec for a publishable reproducibility/leakage audit of fetal-health
cardiotocography (CTG) machine learning. Written to be handed to Claude Code phase
by phase. Target: arXiv preprint first, then a mid-tier journal (JMIR / PLOS Digital
Health / BMC Medical Informatics & Decision Making).

---

## 0. What this paper actually is (read first)

This is NOT "I built a CTG classifier." It is a **methods/reproducibility paper**
with three deliverables that must all be present, or it won't publish:

1. **The audit** — reproduce the common CTG ML pipelines, then isolate and *quantify*
   how much of the reported ~99% accuracy is inflated by data leakage (chiefly: SMOTE
   applied before the train/test split, and improper cross-validation).
2. **The corrected benchmark** — a clean, leakage-free reference pipeline with honest
   evaluation (calibration + per-class metrics, not just accuracy) that future work
   can compare against.
3. **The checklist / model info sheet** — a short, reusable "CTG ML reporting
   checklist" (in the Kapoor & Narayanan "model info sheet" spirit) others can apply.

The contribution is the *gap between naive and correct* numbers, plus the reusable
artifacts. The honest framing throughout is constructive ("here is how to do it
right"), never naming-and-shaming.

## 1. The dataset

Fetal-health Cardiotocography: **2,126 records, 21 numeric features, 3 classes**
(Normal = 1,655, Suspect = 295, Pathological = 176). Public on Kaggle ("Fetal Health
Classification") and UCI. Heavily class-imbalanced — which is exactly *why* so many
papers reach for SMOTE and exactly where the leakage happens.

## 2. The specific leakage sources to audit

These are the concrete, defensible targets (each becomes an experiment):

- **L1 — SMOTE-before-split.** The big one. Many papers oversample the *entire*
  dataset and *then* split into train/test. Synthetic points derived from test
  samples leak into training. Correct method: SMOTE inside the training fold only.
- **L2 — Resampling inside cross-validation done wrong.** SMOTE applied once before
  CV instead of inside each fold via a Pipeline.
- **L3 — Duplicate / near-duplicate rows** across train and test.
- **L4 — Preprocessing leakage.** Scaler/feature-selection fit on the full dataset
  before splitting (test statistics leak into training).
- **L5 — Metric inflation from imbalance.** Reporting plain accuracy on a 78/14/8
  split, where a naive majority bias looks impressive.

## 3. Tech stack (already installed)

Python 3.13; scikit-learn, xgboost, imbalanced-learn (SMOTE + Pipeline), pandas,
numpy, matplotlib, seaborn, shap, jupyter. Plus ruff + black + pytest for hygiene.

## 4. Repo structure

```
ctg-leakage-audit/
├── README.md
├── requirements.txt
├── pyproject.toml
├── .gitignore
├── data/
│   ├── raw/                  # the downloaded CTG csv
│   └── processed/
├── src/
│   ├── data/
│   │   └── load.py           # load + basic integrity checks (dupes, NaNs)
│   ├── pipelines/
│   │   ├── naive.py          # the LEAKY pipeline (SMOTE-before-split, etc.)
│   │   └── correct.py        # the leakage-free pipeline (SMOTE-in-fold via Pipeline)
│   ├── experiments/
│   │   ├── run_leakage_audit.py   # runs naive vs correct across leakage sources
│   │   └── run_duplicates.py      # quantifies duplicate/near-duplicate leakage
│   └── evaluation/
│       └── metrics.py        # accuracy, per-class P/R/F1, macro-F1, ROC-AUC (OvR),
│                             # balanced accuracy, ECE/calibration
├── notebooks/
│   └── eda.ipynb             # exploration only
├── results/
│   ├── tables/               # generated result tables (naive vs corrected)
│   └── figures/              # calibration curves, score-drop charts
├── checklist/
│   └── ctg_model_info_sheet.md   # the reusable reporting checklist (a deliverable)
├── paper/
│   └── outline.md            # section-by-section paper outline + claims
└── tests/
    ├── test_load.py
    ├── test_leakage.py       # asserts naive leaks and correct doesn't (the core proof)
    └── test_metrics.py
```

The key design point: `naive.py` and `correct.py` are *deliberately* two pipelines
so the paper can run them head-to-head and report the gap. The whole result is that
difference.

---

## 5. Phased task breakdown (hand to Claude Code one phase at a time)

After EACH phase: review, then **commit and push** (small commits keep the GitHub
graph active and the history readable — see section 7).

### Phase 0 — Scaffold + repo + GitHub
- Create the structure above; set up pyproject (ruff/black/pytest), .gitignore
  (ignore .venv, data/raw, __pycache__, .claude), README stub.
- `git init`, first commit.
- Create the GitHub repo and push (we'll do this together, step by step).

### Phase 1 — Data + integrity checks
- `src/data/load.py`: load the CTG csv, report shape, class counts, missing values,
  and **exact-duplicate count** (this feeds L3).
- `tests/test_load.py`.
- *Commit:* "add data loader and integrity checks".

### Phase 2 — The naive (leaky) pipeline
- `src/pipelines/naive.py`: reproduce the common published approach — SMOTE the whole
  dataset, scale on the whole dataset, then split, then train (RF / XGBoost), report
  accuracy. This intentionally reproduces the inflated ~99% result.
- *Commit:* "add naive leaky baseline pipeline".

### Phase 3 — The correct (leakage-free) pipeline
- `src/pipelines/correct.py`: stratified split FIRST; then an imblearn `Pipeline` that
  applies scaling + SMOTE **inside cross-validation folds only**; evaluate on the
  untouched test set.
- `tests/test_leakage.py`: the core proof — assert the naive pipeline's CV score is
  optimistically inflated vs a held-out test, and the correct one is consistent.
- *Commit:* "add leakage-free corrected pipeline + leakage test".

### Phase 4 — Evaluation harness (honest metrics)
- `src/evaluation/metrics.py`: accuracy, balanced accuracy, per-class precision/recall/
  F1, macro-F1, one-vs-rest ROC-AUC, and **calibration (ECE + reliability curve)**.
- Why: the paper's secondary message is that accuracy alone hides the failure on the
  minority Pathological class — the clinically important one.
- *Commit:* "add honest evaluation metrics incl. calibration".

### Phase 5 — Run the audit experiments
- `src/experiments/run_leakage_audit.py`: run naive vs correct across L1, L2, L4, L5
  and emit a results table (the headline: naive ~99% vs corrected realistic number,
  with the per-source contribution).
- `run_duplicates.py`: quantify L3.
- Save tables to `results/tables/`, figures (calibration curves, the score-drop bar
  chart) to `results/figures/`.
- *Commit:* "add leakage audit experiments and result tables".

### Phase 6 — The reusable checklist
- `checklist/ctg_model_info_sheet.md`: a short, practical reporting checklist (data
  provenance, split-before-resample, in-fold resampling, duplicate check, report
  per-class + calibration, fixed seeds). This is a *named deliverable* reviewers value.
- *Commit:* "add CTG model info sheet / reporting checklist".

### Phase 7 — Paper outline + README write-up
- `paper/outline.md`: section-by-section (Abstract, Intro, Related Work, Methods,
  Experiments, Results, Discussion/Limitations, Checklist, Conclusion) with the exact
  claims each section makes and which result table backs it.
- Flesh out README: what the repo proves, how to reproduce, link to results.
- *Commit:* "add paper outline and finalize README".

---

## 6. Things to get right (so it publishes, not just runs)

1. **Reproduce the inflated number for real.** The audit is only convincing if the
   naive pipeline genuinely hits the ~96–99% that published papers report. Match a
   real common setup.
2. **Quantify per-source.** Don't just say "leakage inflates results" — attribute how
   many points come from SMOTE-before-split vs preprocessing leakage vs duplicates.
   The decomposition is the novelty.
3. **Report the minority class.** The Pathological class (n=176) is the one that
   matters clinically and the one accuracy hides. Foreground per-class recall.
4. **Calibration.** Show the naive model is also mis-calibrated, not just over-scored.
5. **Constructive tone.** Frame as "a reproducible protocol + checklist for CTG ML,"
   not "these authors were wrong." Reviewers reward the former.
6. **Fixed seeds + released code.** Everything reproducible, seeds pinned, repo public
   on GitHub (and archived to Zenodo for a DOI at submission).
7. **Honest scope.** One dataset, retrospective. State it plainly. Claims limited to
   "public-benchmark CTG ML," not clinical deployment.

## 7. Git rhythm (keep the contribution graph active)

- Commit at every checkpoint above, and at smaller steps within a phase whenever a
  discrete piece works (e.g. "add ECE calculation", then "add reliability plot").
- Small, meaningful commit messages — describe the actual change.
- Claude Code can run `git add` / `git commit` / `git push` for you in its terminal.
- Push after each commit so the green squares land daily as you work.

## 8. First message to Claude Code (Phase 0)

> I'm building a reproducibility and data-leakage audit of fetal-health CTG machine
> learning, for an arXiv preprint then a journal. Read ctg-build-plan.md for the full
> plan. Start with Phase 0 only: scaffold the repo structure from section 4, set up
> tooling (ruff, black, pytest in pyproject.toml), a .gitignore that ignores .venv,
> data/raw/, __pycache__ and .claude, and a README stub describing the audit. Then run
> git init and make the first commit. Use Python 3.13; packages are already installed
> in .venv. Show me the planned structure before creating anything.

Then go phase by phase, reviewing and committing after each.
