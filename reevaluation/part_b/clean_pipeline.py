"""
Part B: clean re-evaluation on the UCI Diabetes 130-US Hospitals dataset
(UCI Machine Learning Repository, dataset 296: "Diabetes 130-US hospitals
for years 1999-2008", Strack et al. 2014).

Protocol (from readmission-preprint-pack.md, no deviations):
  - Cohort: first encounter per patient; exclude expired / hospice discharges
    (patients who cannot be readmitted); exclude unknown/invalid gender.
  - Label: readmitted == "<30" -> 1; ">30" or "NO" -> 0.
  - Patient-level train/test split BEFORE any resampling or fitting
    (after first-encounter filtering each patient contributes exactly one
    row; the split is performed explicitly on patient_nbr).
  - All preprocessing fit on training data only (imblearn Pipeline).
  - SMOTE applied inside training folds only (it is a step of the pipeline
    used by GridSearchCV, so cross-validation folds resample correctly).
  - Hyperparameter search confined to cross-validation on the training set.
  - Exactly ONE evaluation pass on the held-out test set per model: a single
    predict_proba on the test set, from which all test metrics are computed.
  - Seed sensitivity: the final (tuned XGBoost) pipeline is repeated with 3
    additional split/training seeds, hyperparameters held at the values tuned
    in the primary run. This is documented in results.md.

Outputs: part_b/results_part_b.json plus console narration.
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
    accuracy_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.calibration import calibration_curve

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

import xgboost as xgb
import shap

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data" / "uci_diabetes" / "dataset_diabetes" / "diabetic_data.csv"
PRIMARY_SEED = 42
TEST_SIZE = 0.20

# ----------------------------------------------------------------------------
# 1. Load and build the cohort, recording the count after every step.
# ----------------------------------------------------------------------------

cohort_log = []

raw = pd.read_csv(DATA, dtype=str, keep_default_na=False, na_values=["?"])
cohort_log.append(("raw rows in diabetic_data.csv", len(raw)))

# First encounter per patient: the file is ordered chronologically within
# patient in the UCI release; sort by patient and encounter id to make the
# "first" deterministic, then keep the first row per patient_nbr.
raw["_enc_num"] = pd.to_numeric(raw["encounter_id"], errors="coerce")
raw_sorted = raw.sort_values(["patient_nbr", "_enc_num"], kind="mergesort")
first = raw_sorted.drop_duplicates(subset="patient_nbr", keep="first").copy()
cohort_log.append(("after keeping first encounter per patient", len(first)))

# Exclude discharges that make readmission impossible: expired or hospice.
# Discharge disposition ids per IDs_mapping.csv in the same UCI release:
#   11 Expired, 13 Hospice / home, 14 Hospice / medical facility,
#   19 Expired at home (Medicaid only), 20 Expired in a medical facility,
#   21 Expired, place unknown.
EXCLUDED_DISCHARGE = {"11", "13", "14", "19", "20", "21"}
coh = first[~first["discharge_disposition_id"].isin(EXCLUDED_DISCHARGE)].copy()
cohort_log.append(
    ("after excluding expired/hospice discharge dispositions "
     "(ids 11, 13, 14, 19, 20, 21)", len(coh))
)

# Exclude the handful of unknown/invalid gender rows.
coh = coh[coh["gender"].isin(["Male", "Female"])].copy()
cohort_log.append(("after excluding gender == 'Unknown/Invalid'", len(coh)))

# Label: readmitted within 30 days is the positive class.
coh["target"] = (coh["readmitted"] == "<30").astype(int)
n_final = len(coh)
n_pos = int(coh["target"].sum())
prevalence = float(coh["target"].mean())
cohort_log.append(("final cohort", n_final))
cohort_log.append(("positive (readmitted <30d)", n_pos))

print("Cohort flow:")
for step, n in cohort_log:
    print(f"  {step}: {n}")
print(f"  prevalence: {prevalence:.4f}")

# ----------------------------------------------------------------------------
# 2. Feature engineering (deterministic mappings only; nothing is fit here).
# ----------------------------------------------------------------------------

AGE_MIDPOINT = {
    "[0-10)": 5, "[10-20)": 15, "[20-30)": 25, "[30-40)": 35, "[40-50)": 45,
    "[50-60)": 55, "[60-70)": 65, "[70-80)": 75, "[80-90)": 85, "[90-100)": 95,
}


def diag_group(code):
    """Map an ICD-9 code to the standard Strack et al. diagnosis groups."""
    if code is None or (isinstance(code, float) and np.isnan(code)):
        return "Missing"
    code = str(code)
    if code.startswith("250"):
        return "Diabetes"
    try:
        v = float(code)
    except ValueError:
        return "Other"  # V / E codes
    if 390 <= v <= 459 or v == 785:
        return "Circulatory"
    if 460 <= v <= 519 or v == 786:
        return "Respiratory"
    if 520 <= v <= 579 or v == 787:
        return "Digestive"
    if 580 <= v <= 629 or v == 788:
        return "Genitourinary"
    if 710 <= v <= 739:
        return "Musculoskeletal"
    if 140 <= v <= 239:
        return "Neoplasms"
    if 800 <= v <= 999:
        return "Injury"
    return "Other"


MED_COLS = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide",
    "glimepiride", "acetohexamide", "glipizide", "glyburide", "tolbutamide",
    "pioglitazone", "rosiglitazone", "acarbose", "miglitol", "troglitazone",
    "tolazamide", "examide", "citoglipton", "insulin",
    "glyburide-metformin", "glipizide-metformin", "glimepiride-pioglitazone",
    "metformin-rosiglitazone", "metformin-pioglitazone",
]

NUMERIC_FEATURES = [
    "age_years", "time_in_hospital", "num_lab_procedures", "num_procedures",
    "num_medications", "number_outpatient", "number_emergency",
    "number_inpatient", "number_diagnoses",
]

CATEGORICAL_FEATURES = [
    "race", "gender", "admission_type_id", "discharge_disposition_id",
    "admission_source_id", "payer_code", "medical_specialty",
    "diag_1_group", "diag_2_group", "diag_3_group",
    "max_glu_serum", "A1Cresult", "change", "diabetesMed",
] + MED_COLS


def build_features(df):
    out = pd.DataFrame(index=df.index)
    out["patient_nbr"] = df["patient_nbr"]
    out["age_years"] = df["age"].map(AGE_MIDPOINT)
    for c in NUMERIC_FEATURES[1:]:
        out[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ["race", "gender", "admission_type_id", "discharge_disposition_id",
              "admission_source_id", "payer_code", "medical_specialty",
              "max_glu_serum", "A1Cresult", "change", "diabetesMed"]:
        out[c] = df[c]
    for d in ["diag_1", "diag_2", "diag_3"]:
        out[f"{d}_group"] = df[d].map(diag_group)
    for c in MED_COLS:
        out[c] = df[c]
    out["target"] = df["target"].values
    return out


feat = build_features(coh)
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# ----------------------------------------------------------------------------
# 3. Patient-level split BEFORE any resampling or fitting.
# ----------------------------------------------------------------------------

patient_ids = feat["patient_nbr"].unique()
train_ids, test_ids = train_test_split(
    patient_ids, test_size=TEST_SIZE, random_state=PRIMARY_SEED
)
train_df = feat[feat["patient_nbr"].isin(set(train_ids))]
test_df = feat[feat["patient_nbr"].isin(set(test_ids))]
assert not set(train_df["patient_nbr"]) & set(test_df["patient_nbr"])

X_train, y_train = train_df[FEATURES], train_df["target"].astype(int)
X_test, y_test = test_df[FEATURES], test_df["target"].astype(int)
print(f"train rows: {len(X_train)} (pos {int(y_train.sum())}), "
      f"test rows: {len(X_test)} (pos {int(y_test.sum())})")

# ----------------------------------------------------------------------------
# 4. Pipelines. Preprocessing is fit on training data only; SMOTE lives
#    inside the pipeline, so it only ever touches training folds.
# ----------------------------------------------------------------------------

preprocess = ColumnTransformer(
    transformers=[
        ("num", SimpleImputer(strategy="median"), NUMERIC_FEATURES),
        (
            "cat",
            SkPipeline(
                steps=[
                    ("impute", SimpleImputer(strategy="constant",
                                             fill_value="Missing")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore",
                                             min_frequency=100)),
                ]
            ),
            CATEGORICAL_FEATURES,
        ),
    ],
    sparse_threshold=0.0,  # dense output so SMOTE and SHAP see plain arrays
)


def make_pipeline(model, scale_for_lr=False):
    steps = [("prep", preprocess)]
    if scale_for_lr:
        steps.append(("scale", StandardScaler(with_mean=False)))
    steps.append(("smote", SMOTE(random_state=PRIMARY_SEED)))
    steps.append(("model", model))
    return ImbPipeline(steps)


def bootstrap_auc_ci(y_true, p, n_boot=1000, seed=123):
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    p = np.asarray(p)
    aucs = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[idx], p[idx]))
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    return float(lo), float(hi)


results = {"cohort_log": cohort_log, "n_final": int(n_final),
           "n_positive": n_pos, "prevalence": prevalence,
           "n_train": int(len(X_train)), "n_test": int(len(X_test)),
           "primary_seed": PRIMARY_SEED, "models": {}}


def evaluate_once(name, fitted_pipeline):
    """The single held-out test evaluation for a fitted pipeline."""
    p = fitted_pipeline.predict_proba(X_test)[:, 1]
    auc = float(roc_auc_score(y_test, p))
    lo, hi = bootstrap_auc_ci(y_test, p)
    entry = {
        "roc_auc": auc,
        "roc_auc_ci95": [lo, hi],
        "pr_auc": float(average_precision_score(y_test, p)),
        "accuracy_at_0.5": float(accuracy_score(y_test, p >= 0.5)),
    }
    results["models"][name] = entry
    print(f"{name}: ROC-AUC {auc:.4f} (95% CI {lo:.4f}-{hi:.4f}), "
          f"PR-AUC {entry['pr_auc']:.4f}, acc {entry['accuracy_at_0.5']:.4f}")
    return p


# --- Model 1: logistic regression baseline ---------------------------------
lr_pipe = make_pipeline(
    LogisticRegression(max_iter=1000, C=1.0, random_state=PRIMARY_SEED),
    scale_for_lr=True,
)
lr_pipe.fit(X_train, y_train)
evaluate_once("logistic_regression", lr_pipe)

# --- Model 2: untuned gradient boosting baseline ---------------------------
gb_pipe = make_pipeline(
    HistGradientBoostingClassifier(random_state=PRIMARY_SEED)
)
gb_pipe.fit(X_train, y_train)
evaluate_once("gradient_boosting_untuned", gb_pipe)

# --- Model 3: tuned XGBoost (search confined to CV on the training set) ---
xgb_base = xgb.XGBClassifier(
    objective="binary:logistic", eval_metric="logloss",
    tree_method="hist", random_state=PRIMARY_SEED,
    n_jobs=1,  # fits are parallelised by GridSearchCV across the 2 vCPUs
)
xgb_pipe = make_pipeline(xgb_base)
param_grid = {
    "model__max_depth": [3, 4],
    "model__learning_rate": [0.05, 0.1],
    "model__n_estimators": [300, 500],
    "model__min_child_weight": [1, 3],
}
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=PRIMARY_SEED)
search = GridSearchCV(xgb_pipe, param_grid, scoring="roc_auc", cv=cv, n_jobs=2)
search.fit(X_train, y_train)
print("best CV ROC-AUC:", round(float(search.best_score_), 4))
print("best params:", search.best_params_)
results["xgb_best_params"] = {
    k.replace("model__", ""): v for k, v in search.best_params_.items()
}
results["xgb_cv_roc_auc"] = float(search.best_score_)
p_final = evaluate_once("xgboost_tuned", search.best_estimator_)

# --- Calibration of the final model (from the same single test pass) ------
eps = 1e-15
p_clip = np.clip(p_final, eps, 1 - eps)
logit = np.log(p_clip / (1 - p_clip)).reshape(-1, 1)
cal_fit = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
cal_fit.fit(logit, y_test)
cal_slope = float(cal_fit.coef_[0][0])
cal_intercept = float(cal_fit.intercept_[0])
brier = float(brier_score_loss(y_test, p_final))
frac_pos, mean_pred = calibration_curve(y_test, p_final, n_bins=10)
results["calibration"] = {
    "slope": cal_slope, "intercept": cal_intercept, "brier": brier,
    "deciles": {"mean_predicted": [float(x) for x in mean_pred],
                "fraction_positive": [float(x) for x in frac_pos]},
}
print(f"calibration: slope {cal_slope:.3f}, intercept {cal_intercept:.3f}, "
      f"Brier {brier:.4f}")

# --- SHAP on the final model ------------------------------------------------
best = search.best_estimator_
X_test_enc = best.named_steps["prep"].transform(X_test)
feature_names = list(best.named_steps["prep"].get_feature_names_out())
final_model = best.named_steps["model"]
sample_n = min(2000, X_test_enc.shape[0])
rng = np.random.default_rng(PRIMARY_SEED)
sample_idx = rng.choice(X_test_enc.shape[0], sample_n, replace=False)
explainer = shap.TreeExplainer(final_model)
shap_values = explainer.shap_values(X_test_enc[sample_idx])
if isinstance(shap_values, list):
    shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
shap_values = np.asarray(shap_values)
if shap_values.ndim == 3:
    shap_values = shap_values[:, :, -1]
mean_abs = np.abs(shap_values).mean(axis=0)
top_idx = np.argsort(mean_abs)[::-1][:15]
results["shap_top15"] = [
    {"feature": feature_names[i], "mean_abs_shap": float(mean_abs[i])}
    for i in top_idx
]
print("SHAP top 15:")
for row in results["shap_top15"]:
    print(f"  {row['feature']}: {row['mean_abs_shap']:.4f}")

# ----------------------------------------------------------------------------
# 5. Seed sensitivity: final pipeline, 3 more seeds. Split, SMOTE and model
#    seeds vary; hyperparameters stay at the values tuned above (documented).
# ----------------------------------------------------------------------------

best_params = results["xgb_best_params"]
seed_runs = []
for seed in [101, 202, 303]:
    tr_ids, te_ids = train_test_split(
        patient_ids, test_size=TEST_SIZE, random_state=seed
    )
    tr = feat[feat["patient_nbr"].isin(set(tr_ids))]
    te = feat[feat["patient_nbr"].isin(set(te_ids))]
    Xtr, ytr = tr[FEATURES], tr["target"].astype(int)
    Xte, yte = te[FEATURES], te["target"].astype(int)
    pipe = ImbPipeline(
        steps=[
            ("prep", preprocess),
            ("smote", SMOTE(random_state=seed)),
            ("model", xgb.XGBClassifier(
                objective="binary:logistic", eval_metric="logloss",
                tree_method="hist", random_state=seed, n_jobs=2,
                **best_params)),
        ]
    )
    pipe.fit(Xtr, ytr)
    p = pipe.predict_proba(Xte)[:, 1]
    auc = float(roc_auc_score(yte, p))
    seed_runs.append({"seed": seed, "roc_auc": auc,
                      "pr_auc": float(average_precision_score(yte, p)),
                      "n_test": int(len(Xte))})
    print(f"seed {seed}: test ROC-AUC {auc:.4f}")

aucs = [r["roc_auc"] for r in seed_runs]
results["seed_sensitivity"] = {
    "runs": seed_runs,
    "roc_auc_min": float(np.min(aucs)), "roc_auc_max": float(np.max(aucs)),
    "roc_auc_mean": float(np.mean(aucs)), "roc_auc_sd": float(np.std(aucs)),
    "note": ("Split, SMOTE and model seeds varied; hyperparameters fixed at "
             "the values tuned in the primary run."),
}

with open(HERE / "results_part_b.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nWrote", HERE / "results_part_b.json")
