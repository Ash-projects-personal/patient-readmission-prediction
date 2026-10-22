# Clean Readmission Re-Evaluation: Results

All numbers below were produced by the runs in this directory on 2026-10-22.
Nothing is copied from the repo's advertised figures. Where a run's output
differs from what the original project claimed, the run's number is the one
reported.

Environment: Python 3.12, numpy 2.5.4, pandas 3.0.6, scikit-learn 1.9.1,
xgboost 2.1.4, imbalanced-learn 0.14.2, shap 0.54.1 (full freeze in
`environment.txt`). Compute: 2 vCPU VM.

---

## Part A: Leakage ablation on the shipped synthetic pipeline

Source: `repo/train_model.py` (github.com/Ash-projects-personal/
patient-readmission-prediction, cloned 2026-10-22). The shipped script
generates 50,000 synthetic rows (seed 42), builds the label by thresholding a
hand-written `risk_score` at its 85th percentile, adds that same score back
as the feature `risk_index_composite`, splits 80/20 (random_state 42, no
stratify), and trains a fixed XGBoost configuration.

### A1. Shipped pipeline, reproduced as is

The shipped script's own output: Accuracy 0.9982, AUC 1.0000 (4 dp).
Full-precision measurement of the identical pipeline (`part_a/measure_part_a.py`,
metrics in `part_a/metrics_as_is.json`):

| Metric (held out test, n = 10,000) | Value |
|---|---|
| Accuracy | 0.9982 |
| ROC-AUC | 0.99999 (0.9999886) |
| PR-AUC | 0.99994 |

Note: reproduction lands above the repo's advertised 0.96 accuracy /
0.99 AUC. Same code, same seeds; the small difference is attributable to
library versions (see run_log.md). The mechanism is identical: the model is
handed the exact variable the label was thresholded from.

### A2. Ablation: only `risk_index_composite` removed

The ablation copy (`part_a/ablation/train_model.py`) differs from the shipped
file by exactly one line: the assignment `df['risk_index_composite'] =
risk_score` is removed. Data generation, the split, and the model
configuration are byte-identical otherwise, and because the removed line is a
pure column assignment, every random draw is unchanged. The inference-time
twin in `repo/app.py` (lines 68-77, which recompute the same score for the
demo UI) does not affect training and was not part of the ablation.

| Metric (held out test, n = 10,000) | With leaked feature | Without it | Delta |
|---|---|---|---|
| Accuracy | 0.9982 | 0.9250 | -0.0732 |
| ROC-AUC | 0.99999 | 0.96094 | -0.03905 |
| PR-AUC | 0.99994 | 0.83147 | -0.16847 |

Reading: removing the leaked feature costs 3.9 points of ROC-AUC and 16.8
points of PR-AUC. The residual 0.961 ROC-AUC is the genuine signal in the
synthetic generator's observable features: those features are the actual
inputs to the hand-written risk score (minus its noise term), so a high
honest score is expected on this synthetic data. It says nothing about real
patients.

---

## Part B: Clean pipeline on the UCI Diabetes 130-US Hospitals dataset

Dataset: "Diabetes 130-US hospitals for years 1999-2008", UCI Machine
Learning Repository (dataset 296, Strack et al. 2014).
Download URL used: https://archive.ics.uci.edu/ml/machine-learning-databases/00296/dataset_diabetes.zip
(file `diabetic_data.csv`). This is the re-evaluation cohort chosen by the
pack. It is not, and must never be described as, the hackathon's data.

Label: `readmitted == "<30"` is positive; `">30"` and `"NO"` are negative.

### Cohort flow

| Step | Rows remaining |
|---|---|
| Raw rows in diabetic_data.csv | 101,766 |
| First encounter per patient (sorted by patient, then encounter id) | 71,518 |
| Excluding expired / hospice discharge dispositions (ids 11, 13, 14, 19, 20, 21: patients who cannot be readmitted) | 69,973 |
| Excluding gender "Unknown/Invalid" (3 rows) | 69,970 |
| Final cohort | 69,970 |
| Positives (readmitted < 30 days) | 6,277 |
| Prevalence | 8.97% |

Patient-level split, seed 42, 80/20, performed before any resampling or
fitting (after first-encounter filtering each patient contributes exactly
one row; the split was made explicitly on patient_nbr and disjointness was
asserted in code): train 55,976 (5,017 positives), test 13,994
(1,260 positives).

Protocol, as run: all preprocessing fit on training data only; SMOTE inside
the training pipeline only, so cross-validation folds resample correctly;
hyperparameter search confined to 3-fold cross-validation on the training
set; exactly one `predict_proba` evaluation pass on the held out test set
per model, from which all test metrics below are computed.

Features: demographics (age band midpoint, race, gender), encounter
features (time in hospital, lab/procedure/medication counts, prior
outpatient/emergency/inpatient utilization, diagnosis count), admission
type, discharge disposition, admission source, payer code, medical
specialty, primary/secondary/tertiary diagnoses grouped into the standard
ICD-9 chapter groups, A1C and glucose results, medication change and
diabetes-medication flags, and the 23 per-medication columns. The `weight`
column (about 97% missing) was dropped. One-hot encoding dropped categories
seen fewer than 100 times in the training data.

### Test set results (single evaluation per model)

| Model | ROC-AUC | 95% CI (bootstrap, 1,000 resamples) | PR-AUC | Accuracy @ 0.5 |
|---|---|---|---|---|
| Logistic regression (baseline) | 0.6326 | 0.6165 to 0.6480 | 0.1553 | 0.6224 |
| Gradient boosting, untuned (HistGradientBoosting, defaults) | 0.6497 | 0.6329 to 0.6652 | 0.1774 | 0.9103 |
| XGBoost, tuned (final model) | 0.6517 | 0.6354 to 0.6678 | 0.1717 | 0.9097 |

Tuned XGBoost hyperparameters, selected by training-set CV only:
max_depth 3, learning_rate 0.05, n_estimators 500, min_child_weight 1
(grid: depth {3, 4} x learning rate {0.05, 0.1} x n_estimators {300, 500}
x min_child_weight {1, 3}). Best CV ROC-AUC: 0.6523.

Honest reading: on real data under a clean protocol, all three models sit in
the mid 0.60s, exactly where the published literature on this cohort sits.
Tuning bought the XGBoost model 0.002 ROC-AUC over the untuned gradient
boosting baseline, and its PR-AUC (0.1717) is slightly below the untuned
baseline's (0.1774). The accuracy figures for the tree models are close to
the 91.03% majority-class rate (predicting "not readmitted" for everyone)
and should not be quoted without that context; ROC-AUC and PR-AUC are the
informative numbers.

### Calibration of the final model (same single test pass)

| Quantity | Value |
|---|---|
| Calibration slope | 1.0247 |
| Calibration intercept | -0.0787 |
| Brier score | 0.0795 |

A slope near 1 and an intercept near 0 mean the SMOTE-trained model's
probabilities are, on this test set, close to calibrated despite the
resampled training prevalence. Full predicted-vs-observed deciles are in
`part_b/results_part_b.json`.

### SHAP: top features of the final model (mean |SHAP|, 2,000 test rows)

| Rank | Feature (as encoded) | Mean |SHAP| |
|---|---|---|
| 1 | discharge_disposition_id = 1 (discharged to home) | 0.2612 |
| 2 | gender = Male | 0.2399 |
| 3 | gender = Female | 0.2217 |
| 4 | number_inpatient (prior inpatient visits) | 0.1587 |
| 5 | diabetesMed = Yes | 0.1300 |
| 6 | diag_3 group = Other | 0.1290 |
| 7 | medical_specialty = Missing | 0.1251 |
| 8 | diag_3 group = Circulatory | 0.1170 |
| 9 | payer_code = Missing | 0.1116 |
| 10 | diag_2 group = Circulatory | 0.0968 |
| 11 | change = No | 0.0929 |
| 12 | insulin = Steady | 0.0902 |
| 13 | diag_2 group = Other | 0.0875 |
| 14 | diag_1 group = Circulatory | 0.0830 |
| 15 | metformin = No | 0.0808 |

Prior inpatient utilization (rank 4) is the strongest clinical signal, in
line with the readmission literature. The two gender indicators ranking 2-3
is a property of this encoding and model, reported as produced; it should
be examined, not celebrated, before any interpretive claim is made about
it in the paper.

### Seed sensitivity (final pipeline, 3 further seeds)

Split, SMOTE and model seeds varied; hyperparameters held at the tuned
values above (declared deviation, see run_log.md).

| Seed | Test ROC-AUC | Test PR-AUC |
|---|---|---|
| 101 | 0.6623 | 0.1734 |
| 202 | 0.6590 | 0.1761 |
| 303 | 0.6539 | 0.1784 |

Spread: min 0.6539, max 0.6623, mean 0.6584, SD 0.0035. Seed-to-seed
variation (about 0.008 across the range) is larger than the margin tuning
added over the untuned baseline (0.002), which is itself a finding worth
stating in the paper.

---

## Files

- `repo/` : the original project, cloned unmodified.
- `part_a/as_is/` : shipped `train_model.py`, run as is (`run_as_is.log`).
- `part_a/ablation/` : the one-line ablation copy (`run_ablation.log`).
- `part_a/measure_part_a.py`, `part_a/metrics_*.json` : full-precision metrics.
- `part_b/clean_pipeline.py` : the clean pipeline, runnable end to end.
- `part_b/results_part_b.json` : every Part B number, including calibration
  deciles and per-seed PR-AUCs.
- `data/` : the downloaded UCI archive and extracted CSVs.
- `environment.txt` : exact package versions used.
