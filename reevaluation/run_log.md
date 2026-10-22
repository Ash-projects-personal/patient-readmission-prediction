# Run Log: deviations, failures, and fixes

2026-10-22. Everything that did not go exactly by the book, in the order it
happened. No protocol step was changed to improve a score; the changes below
are environment, compute-layout, and one declared interpretation choice.

## The pack file is truncated

`~/workspace/your_files/readmission-preprint-pack.md` ends mid-sentence in
Section 4: the file's own bytes terminate with a literal truncation marker
just after "Diabetes 130-US Hospitals for Years 1999". Section 4's numbered
item list and any later sections are not in the file. The protocol followed
here is the one stated in the task brief, which agrees with everything the
pack does contain (the abstract in Section 2 and the outline in Section 3):
patient-level split before resampling, SMOTE inside training folds only,
tuning confined to cross-validation, one held out test evaluation,
calibration and SHAP on the final model. The parent agent may want to
regenerate the pack's missing tail.

## Environment and installation

- PyPI was reachable only intermittently through the egress proxy; large
  wheels repeatedly died mid-download ("incomplete-download",
  connection resets). Packages were installed one at a time with resume
  retries until each landed.
- One early background install attempt died with "No space left on
  device" during tempfile cleanup. Disk checks afterwards showed ample
  free space on all mounts; the error did not recur.
- XGBoost was installed with `--no-deps`. Every xgboost >= 2.0 wheel
  declares `nvidia-nccl-cu12` (a 351 MB CUDA wheel) as a dependency on this
  platform; that download failed on every attempt. XGBoost is used
  CPU-only here and neither needs nor imports nccl; numpy and scipy (its
  actual runtime dependencies) were installed normally.
- Installed versions are newer than the repo's `requirements.txt` pins for
  numpy (2.5.4 vs <2.0), pandas (3.0.6 vs <3.0) and scikit-learn (1.9.1 vs
  <2.0): the pinned set could not be fetched intact through the proxy.
  XGBoost 2.1.4 and SHAP 0.54.1 are inside the pinned ranges. Full freeze
  in `environment.txt`. Consequence: Part A's reproduction lands slightly
  above the repo's advertised metrics (accuracy 0.9982, ROC-AUC 0.99999,
  vs the advertised 0.96 / 0.99). Same code, same seeds; direction and
  mechanism are identical, and the ablation delta is the quantity the
  paper relies on.

## Part A notes

- The shipped `train_model.py` was run byte-for-byte as cloned, in a copied
  directory, and printed Accuracy 0.9982 / AUC 1.0000 itself; the
  full-precision numbers in results.md come from a measurement driver that
  calls the shipped `generate_synthetic_data` and repeats the shipped
  split and model configuration verbatim (it changes nothing, it only
  prints more decimals and adds PR-AUC).
- The ablation is a one-line removal (verified by diff): the assignment
  that copies `risk_score` into `risk_index_composite`. Because the line
  consumes no random numbers, the synthetic data, split and training are
  otherwise identical.
- `app.py` recomputes the same score at inference time (lines 68-77) and
  appends it to the model input. That twin was documented, not ablated:
  it plays no role in training metrics. Note for the repo fix: a model
  trained without the leaked feature cannot be served by `app.py` as
  written, because the app always injects a twelfth feature the ablated
  model never saw.

## Part B notes

- Dataset URL: the current UCI static-archive URL for dataset 296
  returned "NOT FOUND"; the file was obtained from the UCI Machine
  Learning Repository's classic archive path
  (https://archive.ics.uci.edu/ml/machine-learning-databases/00296/dataset_diabetes.zip),
  which is the same repository's distribution of the same dataset. The zip
  in `data/` contains `diabetic_data.csv` and `IDs_mapping.csv`; the
  discharge-disposition exclusions were read off that mapping file
  (11 Expired; 13 Hospice/home; 14 Hospice/medical facility; 19, 20, 21
  Expired, Medicaid hospice variants).
- First Part B run killed at ~29 CPU-minutes. Cause: the VM has 2 vCPUs
  and the grid search ran its 48 fits sequentially, each XGBoost fit
  asking for 4 threads. The rerun changed compute layout only:
  GridSearchCV n_jobs 1 -> 2, per-fit XGBoost n_jobs 4 -> 1 inside the
  search (seed runs keep n_jobs 2). Grid, folds, fold seed, SMOTE seeds
  and model seeds are unchanged, so the selection and the metrics are the
  ones the original layout would have produced, just sooner. With the
  parallel layout the full Part B run took about 27 minutes of wall time
  on the shared box.
- Declared interpretation choice, seed sensitivity: the three seed repeats
  vary the patient split, the SMOTE seed and the model seed, with
  hyperparameters held at the values tuned on the primary run's training
  set, rather than re-running the full grid search per seed. This is the
  one place where "repeat the final pipeline" was read as repeating the
  final (already tuned) pipeline. It is declared here and in results.md;
  if the paper wants full re-tuning per seed, budget roughly 15 minutes
  of compute per seed on this box.
- pandas 3.0 removed `to_numeric(errors="ignore")`; the first-encounter
  ordering uses a numeric helper column for encounter id instead. Pure
  ordering mechanics, no analytic effect.
- Not computed, deliberately: the pack's Section 3 mentions subgroup
  slices for the results section. The task brief for this run did not ask
  for them, and no subgroup numbers appear in results.md. They remain
  future work for the paper, not a silently dropped step.
