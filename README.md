# patient-readmission-prediction

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-006400)](https://xgboost.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B)](https://streamlit.io/)
[![Hackathon](https://img.shields.io/badge/UNT_Gradinho_2025-1st_Place-gold)](#)

Built this during the UNT Gradinho Hackathon 2025, where my team of 4 won 1st place and I led the team. We built it in 24 hours. It is an XGBoost model with SHAP explanations and a Streamlit dashboard that scores 30 day hospital readmission risk.

## An honest correction (October 2026)

I rechecked this project end to end and found a serious problem in the original version. The training script generated synthetic data, built the label by thresholding a hand written risk score, and then fed that same score back in as a feature called risk_index_composite. That is target leakage, and it is why this repo used to claim 0.96 accuracy and 0.99 AUC. Those numbers came from the leak, not from a real measurement. I have removed the leaked feature, removed the claims, and replaced them with two measured evaluations:

1. The demo pipeline in this repo now trains on 50,000 synthetic rows with the leaked feature removed. Measured on its held out test split: accuracy 0.9250, ROC AUC 0.96094. That is a synthetic demo. It says nothing about real patients.
2. A clean evaluation on real data lives in the `reevaluation/` folder. On the UCI Diabetes 130 US Hospitals cohort (69,970 patients after cohort filtering), a tuned XGBoost model reached ROC AUC 0.6517, against 0.6326 for a logistic regression baseline and in line with published results on this dataset. The full protocol, cohort flow, calibration, and seed checks are in `reevaluation/results.md`.

The UCI cohort is the dataset for the clean evaluation only. It was not the hackathon's data. The hackathon demo used synthetic data, and this README says so wherever it matters.

## What is in here

- `train_model.py`: the synthetic demo pipeline (XGBoost with SHAP), leaked feature removed.
- `app.py`: the Streamlit dashboard from the hackathon. It still loads the original model file, so treat its scores as a UI demo only.
- `MODEL_CARD.md`: intended use, training data, honest evaluation, and limitations.
- `reevaluation/`: Part A is the leakage ablation on the shipped pipeline. Part B is the clean pipeline on the real UCI cohort. Code, metrics, and the full run log are included.

## Run it

```bash
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

Run train_model.py first to generate the model file, then launch the dashboard.

## Model card

A summary of intended use, training data, performance, and limitations is in [`MODEL_CARD.md`](MODEL_CARD.md).

## Disclaimer

This project is for educational and research purposes only. It is **not** a medical device and has not been validated for clinical use. The demo training data is synthetic and does not contain real patient PHI. Do not use this model to make actual healthcare decisions.

## License

Released under the [MIT License](LICENSE).
