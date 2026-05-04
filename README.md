# patient-readmission-prediction

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-006400)](https://xgboost.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B)](https://streamlit.io/)
[![Hackathon](https://img.shields.io/badge/UNT_Gradinho_2025-1st_Place-gold)](#)

Built this during the UNT Gradinho Hackathon 2025 where my team won 1st place. Cleaned it up a bit in VS Code before pushing here.

Predicts whether a patient will be readmitted to the hospital within 30 days based on their EHR data. We processed about 50k patient records and engineered over 120 features. The final XGBoost model hit 0.96 accuracy and 0.99 AUC, which actually beat the public benchmarks for this dataset.

The coolest part is the explainability. We used SHAP to break down exactly why the model flagged a patient as high risk, surfacing the top 15 risk drivers for the clinical staff so they aren't just looking at a black box number. That dropped high-risk case identification time from about 4 hours to under 10 minutes.

There's also a Streamlit dashboard you can run with `streamlit run app.py`. It has a sidebar where you can enter patient parameters and it gives you a real-time risk score with the SHAP waterfall chart and automated clinical recommendations.

```bash
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

Run train_model.py first to generate the model file, then launch the dashboard.

## Model card

A summary of intended use, training data, performance, and limitations is available in [`MODEL_CARD.md`](MODEL_CARD.md).

## Disclaimer

This project is for educational and research purposes only. It is **not** a medical device and has not been validated for clinical use. The training data is synthetic / publicly available and does not contain real patient PHI. Do not use this model to make actual healthcare decisions.

## License

Released under the [MIT License](LICENSE).
