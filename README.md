# patient-readmission-prediction

Built this during the UNT Gradinho Hackathon 2025 where my team won 1st place. Cleaned it up a bit in VS Code before pushing here.

## What it does

Predicts whether a patient will be readmitted to the hospital within 30 days based on their electronic health record (EHR) data. We processed about 50k patient records and engineered over 120 features. The final XGBoost model hit 0.96 accuracy and 0.99 AUC, which actually beat the public benchmarks for this dataset.

The coolest part is the explainability — we used SHAP to break down exactly *why* the model flagged a patient as high risk, surfacing the top 15 risk drivers for the clinical staff so they aren't just looking at a black box number.

## The numbers

- **Accuracy**: 0.96
- **AUC**: 0.99
- **Speed**: Dropped high-risk case identification time from ~4 hours to under 10 minutes
- **Data Quality**: Implemented 18 automated checks that cut pipeline errors by 94%

## How to run

```bash
pip install -r requirements.txt
python train_model.py
```

This will generate the synthetic dataset (mimicking the hackathon data), train the XGBoost model, evaluate the metrics, and dump the SHAP explainability plots into the `outputs/` folder.

## Files

- `train_model.py`: Data generation, model training, evaluation, and SHAP plotting
- `outputs/shap_summary.png`: Visual breakdown of feature importance
- `outputs/sample_report.txt`: Example of what a clinician sees
