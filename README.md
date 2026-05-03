# patient-readmission-prediction

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
