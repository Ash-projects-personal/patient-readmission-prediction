"""Streamlit smoke test for the clinical dashboard (``app.py``).

We use :class:`streamlit.testing.v1.AppTest` (Streamlit's official
in-process test harness) to load the script and assert it renders without
raising — the bedside-flow code paths that don't require a trained model
on disk are exercised here.

Streamlit AppTest is gated behind ``pytest.importorskip`` so the test
file is safe to keep checked in even when the environment lacks
Streamlit/XGBoost/SHAP.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
pytest.importorskip("shap")
pytest.importorskip("xgboost")
pytest.importorskip("joblib")
pytest.importorskip("pandas")
pytest.importorskip("numpy")

# ``streamlit.testing.v1`` ships in Streamlit >= 1.28.  The repo's
# requirements pin ``streamlit>=1.30,<2.0`` so this should always be
# present in CI, but we still guard against older local installs.
try:
    from streamlit.testing.v1 import AppTest  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - older streamlit
    pytest.skip("streamlit.testing.v1 unavailable", allow_module_level=True)


APP_PATH = Path(__file__).resolve().parent.parent / "app.py"


@pytest.fixture
def app() -> "AppTest":
    """Return a freshly-loaded AppTest for the dashboard."""
    return AppTest.from_file(str(APP_PATH), default_timeout=30)


def test_app_loads_without_exception(app: "AppTest") -> None:
    """The script imports + renders cleanly with default sidebar values."""
    app.run()
    assert not app.exception, [str(e) for e in app.exception]


def test_app_renders_title(app: "AppTest") -> None:
    """The big H1 makes it onto the page."""
    app.run()
    titles = [t.value for t in app.title]
    assert any("Clinical Readmission Risk Dashboard" in t for t in titles), titles


def test_app_has_expected_sidebar_inputs(app: "AppTest") -> None:
    """All clinical sliders and number inputs the README claims are present."""
    app.run()
    slider_labels = [s.label for s in app.sidebar.slider]
    number_labels = [n.label for n in app.sidebar.number_input]
    checkbox_labels = [c.label for c in app.sidebar.checkbox]

    for label in ("Age", "Days in Hospital", "Surgical Procedures",
                  "Emergency Visits (past yr)", "Inpatient Visits (past yr)"):
        assert label in slider_labels, f"slider {label!r} missing; have {slider_labels}"

    for label in ("Number of Medications", "Number of Diagnoses", "Lab Procedures"):
        assert label in number_labels, f"number_input {label!r} missing; have {number_labels}"

    for label in ("Diabetes", "Hypertension", "Heart Failure"):
        assert label in checkbox_labels, f"checkbox {label!r} missing; have {checkbox_labels}"


def test_app_shows_missing_model_warning_when_clicked(app: "AppTest") -> None:
    """Without ``models/xgboost_readmission.pkl`` on disk, clicking the
    primary action surfaces the 'run train_model.py first' error and
    does NOT raise."""
    app.run()
    # Find and click the "Calculate" button.
    buttons = [b for b in app.button if "Calculate Readmission Risk" in (b.label or "")]
    assert buttons, "Primary action button not rendered"
    buttons[0].click().run()
    assert not app.exception, [str(e) for e in app.exception]
    # The model load was cached at import time before any train run, so the
    # button branch should hit the 'model is None' guard and emit an error
    # element rather than crashing.
    error_messages = [e.value for e in app.error]
    assert any("train_model.py" in m for m in error_messages), error_messages
