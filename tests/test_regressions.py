"""Regression tests for bugs found during the 2026-11 audit, so they cannot silently return."""
import numpy as np
import pytest

from src.data.loader import load_raw_dataset
from src.data.cleaner import clean_dataset
from src.data.preprocessor import preprocess_dataset
from src.models.trainer import train_model
from src.models.evaluator import evaluate_predictive_performance
from src.fairness.metrics import compute_fairness_metrics
from src.fairness.bootstrap import compute_bootstrap_confidence_intervals
from src.fairness.mitigators import train_exponentiated_gradient_model, train_threshold_optimizer_model
from src.benchmark_store import uses_precomputed_benchmark


@pytest.fixture(scope="module")
def compas_data():
    # COMPAS gives a *mixture* of EG predictors, which is where randomized predict() bit us.
    df = clean_dataset(load_raw_dataset("compas"), "compas")
    return preprocess_dataset(df, target_col="two_year_recid", protected_col="race", random_state=42)


# --- Bug: EG / ThresholdOptimizer predictions changed on every call -----------------------------

def test_exponentiated_gradient_predictions_are_reproducible(compas_data):
    eg = train_exponentiated_gradient_model(compas_data, model_type="random_forest", eps=0.05)
    first = eg.predict(compas_data.X_test)
    for _ in range(3):
        np.testing.assert_array_equal(first, eg.predict(compas_data.X_test))

    # Accuracy reported by the evaluator must come from the same predictions used for fairness.
    acc = evaluate_predictive_performance(eg, compas_data.X_test, compas_data.y_test)["accuracy"]
    assert acc == pytest.approx(float((first == compas_data.y_test.values).mean()), abs=1e-4)


def test_exponentiated_gradient_exposes_probabilities(compas_data):
    # Without predict_proba the evaluator silently reported ROC-AUC = 0.5.
    eg = train_exponentiated_gradient_model(compas_data, model_type="random_forest", eps=0.05)
    proba = eg.predict_proba(compas_data.X_test)
    assert proba.shape == (len(compas_data.X_test), 2)
    assert np.all((proba >= 0) & (proba <= 1))
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)
    auc = evaluate_predictive_performance(eg, compas_data.X_test, compas_data.y_test)["roc_auc"]
    assert auc != 0.5


def test_threshold_optimizer_seeded_predictions_are_reproducible(compas_data):
    base, _ = train_model(compas_data, model_type="random_forest", dataset_key="compas")
    to = train_threshold_optimizer_model(base, compas_data, model_type="random_forest")
    kw = dict(sensitive_features=compas_data.s_test, random_state=42)
    np.testing.assert_array_equal(to.predict(compas_data.X_test, **kw), to.predict(compas_data.X_test, **kw))


# --- Bug: reported fairness value was the (upward-biased) bootstrap mean ------------------------

def _synthetic(gap, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    s = np.where(rng.random(n) < 0.5, "A", "B")
    p = np.where(s == "A", 0.5 + gap / 2, 0.5 - gap / 2)
    y_pred = (rng.random(n) < p).astype(int)
    y_true = rng.integers(0, 2, n)
    return y_true, y_pred, s


def test_bootstrap_value_is_full_sample_point_estimate():
    y_true, y_pred, s = _synthetic(gap=0.0)
    res = compute_bootstrap_confidence_intervals(y_true, y_pred, s, n_bootstraps=200,
                                                 privileged_group="A", unprivileged_group="B")
    point = compute_fairness_metrics(y_true, y_pred, s, "A", "B")
    for m in ["demographic_parity_diff", "equalized_odds_diff", "equal_opportunity_diff", "predictive_parity_diff"]:
        assert res[m]["val"] == pytest.approx(point[m], abs=1e-4)


# --- Bug: "CI excludes zero" was tested on |gap|, which almost never contains 0 -----------------

def test_no_true_gap_is_not_significant():
    y_true, y_pred, s = _synthetic(gap=0.0, seed=1)
    dp = compute_bootstrap_confidence_intervals(y_true, y_pred, s, n_bootstraps=300,
                                                privileged_group="A", unprivileged_group="B")["demographic_parity_diff"]
    assert dp["ci_low"] > 0                      # the absolute-gap CI excludes 0 even with no real gap...
    assert dp["signed_ci_low"] <= 0 <= dp["signed_ci_high"]
    assert dp["is_significant"] is False         # ...so significance must come from the signed CI


def test_large_true_gap_is_significant():
    y_true, y_pred, s = _synthetic(gap=0.3, seed=2)
    dp = compute_bootstrap_confidence_intervals(y_true, y_pred, s, n_bootstraps=300,
                                                privileged_group="A", unprivileged_group="B")["demographic_parity_diff"]
    assert dp["is_significant"] is True


# --- Bug: dashboard showed the benchmark model's metrics for any selected model -----------------

BENCH = {
    "selected_baseline_models": {"adult_income": "xgboost"},
    "experiments": {"adult_income": {"baseline_fairness": {}, "mitigations": {}}},
}


@pytest.mark.parametrize("model_type,protected,expected", [
    ("xgboost", "sex", True),                    # exactly the benchmarked configuration
    ("logistic_regression", "sex", False),       # different model -> must compute live
    ("random_forest", "sex", False),
    ("xgboost", "race", False),                  # different protected attribute -> live
])
def test_dashboard_uses_benchmark_only_for_benchmarked_model(model_type, protected, expected):
    assert uses_precomputed_benchmark(BENCH, "adult_income", protected, model_type, "sex") is expected
