# Model Card — Patient Readmission Predictor

## Overview

| Field | Value |
|---|---|
| Model name | `patient-readmission-xgb` |
| Version | 0.1 |
| Date | May 2026 |
| Author | Ashish Shetty |
| Repository | https://github.com/Ash-projects-personal/patient-readmission-prediction |
| License | MIT |

## Intended use

**Primary use case.** Educational demonstration of an end-to-end clinical-risk modeling pipeline: feature engineering on EHR-style tabular data, gradient-boosted classification, and SHAP-based explainability presented through a Streamlit interface.

**Out-of-scope uses.** This model is **not** intended for, and must not be used for, any of the following:
- Real clinical decision-making, triage, or treatment planning
- Discharge planning or insurance authorization
- Any deployment in a hospital, clinic, payer, or other healthcare-delivery setting
- Any decision affecting an individual person's care

## Training data

- **Source.** Synthetic / publicly available EHR-shaped tabular data; no real patient PHI.
- **Size.** ~50,000 patient records (project-stated).
- **Features.** 120+ engineered features spanning demographics, comorbidities, prior utilization, lab results, and length-of-stay aggregates. See `train_model.py` for the full feature list.
- **Label.** Binary — readmission within 30 days of discharge (yes / no).
- **Class balance.** Skewed toward "no readmission"; class weighting is applied during training.

## Model

- **Algorithm.** XGBoost classifier (`xgboost.XGBClassifier`).
- **Training procedure.** 80/20 stratified train/test split. Hyperparameters set in `train_model.py`. Trained on a single machine — no distributed setup.
- **Explainability.** SHAP TreeExplainer; top-15 features surfaced per prediction in the Streamlit app.

## Evaluation

| Metric | Test value |
|---|---|
| Accuracy | 0.96 |
| ROC-AUC | 0.99 |
| Precision (readmit class) | _to fill in after rerun_ |
| Recall (readmit class) | _to fill in after rerun_ |
| F1 (readmit class) | _to fill in after rerun_ |

> Note: the headline accuracy/AUC values are from the original hackathon project. Re-run on this repo's seed and fill in the precision/recall/F1 cells in a future commit.

## Limitations and known risks

- **Synthetic data.** The training data is not real EHR data. Performance on real clinical data will be lower and is not characterized here.
- **Distribution shift.** No external validation; model has not been tested across hospital systems, geographies, or time periods.
- **Subgroup performance.** Per-subgroup metrics (age, sex, race, payer mix) are not reported. Disparate performance across subgroups is plausible and unmeasured.
- **Interpretation.** SHAP values explain *the model*, not clinical causation. They do not justify clinical action.
- **Calibration.** Probability calibration has not been measured. Predicted probabilities should not be treated as well-calibrated risk estimates.

## Ethical considerations

- Hospital readmission models can encode structural inequities present in healthcare data (access patterns, documentation differences). A production version would require fairness audits, calibration assessment per subgroup, and clinician oversight.
- Treating any model output as a clinical recommendation without expert review is unsafe. The Streamlit app is a UX demonstration, not a clinical tool.

## Citation

If you reference this project, please cite:

```
Shetty, A. (2026). Patient Readmission Prediction (v0.1). GitHub.
https://github.com/Ash-projects-personal/patient-readmission-prediction
```

## Changelog

- **0.1 (May 2026)** — Initial public release; model card stub.
