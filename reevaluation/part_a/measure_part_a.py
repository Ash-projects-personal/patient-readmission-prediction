"""
Part A measurement driver: imports generate_synthetic_data from a given copy
of the shipped train_model.py and re-runs the shipped pipeline verbatim
(same split: test_size=0.2, random_state=42, no stratify; same fixed
XGBClassifier parameters) to record metrics at full precision.

Usage: python measure_part_a.py <path_to_train_model.py> <label>
The pipeline itself is not modified in any way; this only measures.
"""
import importlib.util
import json
import sys

import xgboost as xgb
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

path, label = sys.argv[1], sys.argv[2]
spec = importlib.util.spec_from_file_location("shipped_train_model", path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

df = mod.generate_synthetic_data()
print(f"[{label}] rows={len(df)}, features={list(df.columns)}")
print(f"[{label}] prevalence={df['readmitted'].mean():.4f}")

X = df.drop("readmitted", axis=1)
y = df["readmitted"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    use_label_encoder=False,
    eval_metric="logloss",
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

out = {
    "variant": label,
    "n_rows": int(len(df)),
    "features": list(X.columns),
    "test_accuracy": float(accuracy_score(y_test, y_pred)),
    "test_roc_auc": float(roc_auc_score(y_test, y_prob)),
    "test_pr_auc": float(average_precision_score(y_test, y_prob)),
}
print(json.dumps(out, indent=2))
with open(f"metrics_{label}.json", "w") as f:
    json.dump(out, f, indent=2)
