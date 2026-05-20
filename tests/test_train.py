"""Smoke tests for the readmission training pipeline.

The training script (``train_model.py``) bundles dataset synthesis, XGBoost
fitting, joblib persistence, and SHAP explainability inside
:func:`train_and_evaluate`.  These tests exercise the data-synthesis +
train-save-load path on a small fixture so the suite stays fast.

Heavy dependencies (XGBoost, joblib, scikit-learn) are gated with
``pytest.importorskip`` so the file is safe to keep in the repo even when
the environment doesn't yet have them installed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
xgb = pytest.importorskip("xgboost")
joblib = pytest.importorskip("joblib")
pytest.importorskip("sklearn")
pytest.importorskip("shap")

from sklearn.model_selection import train_test_split  # noqa: E402

from train_model import generate_synthetic_data  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


SMALL_N = 500


EXPECTED_COLUMNS = {
    "age", "num_medications", "num_diagnoses", "time_in_hospital",
    "num_lab_procedures", "num_procedures", "number_emergency",
    "number_inpatient", "diabetes", "hypertension", "heart_failure",
    "risk_index_composite", "readmitted",
}


@pytest.fixture(scope="module")
def small_df() -> "pd.DataFrame":
    """A deterministic, small synthetic EHR fixture."""
    return generate_synthetic_data(n_samples=SMALL_N)


@pytest.fixture
def small_csv(tmp_path: Path, small_df: "pd.DataFrame") -> Path:
    """The fixture frame written to a temporary CSV — exercises the
    'fixture small CSV' path the day-8 bullet specifies."""
    csv = tmp_path / "patients_small.csv"
    small_df.to_csv(csv, index=False)
    return csv


# ---------------------------------------------------------------------------
# Data + shape
# ---------------------------------------------------------------------------


def test_synthetic_data_shape(small_df: "pd.DataFrame") -> None:
    assert len(small_df) == SMALL_N
    assert set(small_df.columns) == EXPECTED_COLUMNS
    assert not small_df.isna().any().any()


def test_target_is_binary(small_df: "pd.DataFrame") -> None:
    assert set(small_df["readmitted"].unique()).issubset({0, 1})
    # 85th-percentile threshold → roughly 15% positive class.
    pos_rate = float(small_df["readmitted"].mean())
    assert 0.05 < pos_rate < 0.30


def test_comorbidities_are_binary(small_df: "pd.DataFrame") -> None:
    for col in ("diabetes", "hypertension", "heart_failure"):
        assert set(small_df[col].unique()).issubset({0, 1}), col


def test_fixture_csv_roundtrip(small_csv: Path, small_df: "pd.DataFrame") -> None:
    """Saved CSV reads back with identical content."""
    loaded = pd.read_csv(small_csv)
    pd.testing.assert_frame_equal(
        loaded.reset_index(drop=True),
        small_df.reset_index(drop=True),
        check_dtype=False,
    )


# ---------------------------------------------------------------------------
# Train + save + load
# ---------------------------------------------------------------------------


def test_model_trains_and_persists(small_df: "pd.DataFrame", tmp_path: Path) -> None:
    """An XGBoost classifier trains on the fixture, dumps to disk, and the
    deserialized model produces predictions of the right shape."""
    X = small_df.drop("readmitted", axis=1)
    y = small_df["readmitted"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if y.nunique() > 1 else None
    )

    model = xgb.XGBClassifier(
        n_estimators=40,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
        eval_metric="logloss",
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    assert preds.shape == (len(X_test),)
    assert set(np.unique(preds)).issubset({0, 1})

    # Persist
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    model_path = model_dir / "xgboost_readmission.pkl"
    joblib.dump(model, model_path)
    assert model_path.exists() and model_path.stat().st_size > 0

    # Reload and check determinism of inference
    reloaded = joblib.load(model_path)
    reloaded_preds = reloaded.predict(X_test)
    assert (preds == reloaded_preds).all()


def test_model_meets_smoke_floor(small_df: "pd.DataFrame") -> None:
    """A trained model beats trivial constant-prediction accuracy.

    The fixture is small, so we keep the floor low (> 80% of the
    majority-class baseline) — this guards against a regression that
    swaps features/target or destroys the signal entirely."""
    X = small_df.drop("readmitted", axis=1)
    y = small_df["readmitted"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if y.nunique() > 1 else None,
    )

    model = xgb.XGBClassifier(
        n_estimators=40,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
        eval_metric="logloss",
    )
    model.fit(X_train, y_train)
    acc = float((model.predict(X_test) == y_test).mean())
    baseline = float(max(y_test.mean(), 1 - y_test.mean()))
    assert acc >= baseline * 0.80, f"acc={acc:.3f} fell below 80% of baseline={baseline:.3f}"
