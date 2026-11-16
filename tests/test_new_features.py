"""Tests for outcome labels, intersectional fairness, calibration, frontier and model card."""
import numpy as np
import pandas as pd
import pytest

from src.config import get_outcome_labels
from src.data.loader import load_raw_dataset
from src.data.cleaner import clean_dataset
from src.data.preprocessor import preprocess_dataset
from src.analysis.eda import get_dataset_kpis
from src.models.trainer import train_model, raw_feature_model
from src.fairness.intersectional import compute_intersectional_metrics
from src.fairness.calibration import compute_group_calibration
from src.fairness.frontier import pareto_frontier, sweep_points
from src.reporting.model_card import generate_model_card


# --- Outcome labels come from config -------------------------------------------------------------

def test_compas_labels_treat_recidivism_as_unfavorable():
    lab = get_outcome_labels("compas")
    assert lab["favorable_value"] == 0 and lab["unfavorable_value"] == 1
    assert lab["by_value"][1] == "High risk"
    assert "REJECTED" not in " ".join(map(str, lab.values())).upper()


def test_german_labels_and_custom_default():
    assert get_outcome_labels("german_credit")["favorable_value"] == 1
    assert get_outcome_labels("custom")["favorable_value"] == 1   # unknown dataset -> sensible default


def test_compas_favorable_rate_is_no_recidivism_rate():
    df = clean_dataset(load_raw_dataset("compas"), "compas")
    recid_rate = df["target"].mean()
    kpis = get_dataset_kpis(df, "target", "race", favorable_value=0)
    assert kpis["overall_positive_rate"] == pytest.approx(1 - recid_rate)


# --- Logistic regression explained on raw features ------------------------------------------------

def test_raw_feature_model_matches_scaled_logistic_regression():
    df = clean_dataset(load_raw_dataset("german_credit"), "german_credit")
    data = preprocess_dataset(df, target_col="credit_risk", protected_col="sex")
    lr, _ = train_model(data, model_type="logistic_regression")
    wrapped = raw_feature_model(lr, "logistic_regression", data.scaler)
    np.testing.assert_array_equal(wrapped.predict(data.X_test), lr.predict(data.X_test_scaled))
    rf, _ = train_model(data, model_type="random_forest")
    assert raw_feature_model(rf, "random_forest", data.scaler) is rf


# --- Intersectional --------------------------------------------------------------------------------

def test_intersectional_finds_gap_hidden_from_single_attributes():
    # Selection depends on the XOR of A and B: each attribute alone shows no gap, intersections do.
    n = 400
    a = np.array(["a0", "a1"] * (n // 2))
    b = np.repeat(["b0", "b1"], n // 2)
    y_pred = ((a == "a1") ^ (b == "b1")).astype(int)
    y_true = y_pred.copy()
    res = compute_intersectional_metrics(y_true, y_pred, pd.Series(a), pd.Series(b))
    assert res["single_attribute_dp"]["attr_a"] == pytest.approx(0.0)
    assert res["single_attribute_dp"]["attr_b"] == pytest.approx(0.0)
    assert res["gaps"]["demographic_parity_diff"] == pytest.approx(1.0)
    assert res["n_included"] == 4


def test_intersectional_excludes_small_groups():
    a = pd.Series(["x"] * 100 + ["y"] * 100 + ["z"] * 5)
    b = pd.Series(["m"] * 205)
    y_pred = np.r_[np.ones(100), np.zeros(100), np.ones(5)].astype(int)
    res = compute_intersectional_metrics(y_pred, y_pred, a, b, min_count=30)
    assert res["n_excluded"] == 1
    assert not res["table"].set_index("group").loc["z × m", "included"]


# --- Calibration -----------------------------------------------------------------------------------

def test_calibration_well_vs_badly_calibrated_group():
    rng = np.random.default_rng(0)
    n = 4000
    p = rng.random(n)
    s = np.where(rng.random(n) < 0.5, "good", "bad")
    y = (rng.random(n) < p).astype(int)                 # outcomes drawn from p -> calibrated
    y_prob = np.where(s == "good", p, np.clip(p + 0.3, 0, 1))  # 'bad' group is over-predicted
    res = compute_group_calibration(y, y_prob, s)
    summ = res["summary"].set_index("group")
    assert summ.loc["good", "ece"] < 0.05
    assert summ.loc["bad", "ece"] > 0.2
    assert summ.loc["bad", "brier_score"] > summ.loc["good", "brier_score"]
    assert set(res["curves"]["group"]) == {"good", "bad"}


# --- Frontier --------------------------------------------------------------------------------------

def test_pareto_frontier_drops_dominated_points():
    df = pd.DataFrame({"dp_diff": [0.01, 0.02, 0.05, 0.03, 0.10],
                       "accuracy": [0.80, 0.85, 0.84, 0.86, 0.90]})
    front = pareto_frontier(df)
    # (0.05, 0.84) is dominated by (0.02, 0.85) and (0.03, 0.86)
    assert list(zip(front["dp_diff"], front["accuracy"])) == [(0.01, 0.80), (0.02, 0.85), (0.03, 0.86), (0.10, 0.90)]


def test_pareto_frontier_handles_duplicate_index_labels():
    # Regression: concat() of sweep + other mitigations produced duplicate labels, and .loc
    # pulled a dominated point back onto the frontier.
    a = pd.DataFrame({"dp_diff": [0.0208, 0.0082], "accuracy": [0.705, 0.695]})
    b = pd.DataFrame({"dp_diff": [0.0147], "accuracy": [0.715]})
    front = pareto_frontier(pd.concat([a, b]))
    assert (0.0208, 0.705) not in list(zip(front["dp_diff"], front["accuracy"]))
    assert len(front) == 2


def test_sweep_points_marks_selected_eps():
    bench = {"selected_epsilons": {"d": 0.01}, "eg_sweep_results": {"d": {
        "demographic_parity": [{"eps": 0.01, "accuracy_raw": 0.7, "dp_diff_raw": 0.02, "eo_diff": "0.05"},
                               {"eps": 0.1, "accuracy_raw": 0.72, "dp_diff_raw": 0.04, "eo_diff": "0.06"}],
        "equalized_odds": [{"eps": 0.01, "accuracy_raw": 0.71, "dp_diff_raw": 0.03, "eo_diff": "0.02"}]}}}
    pts = sweep_points(bench, "d")
    assert len(pts) == 3
    assert pts["selected"].sum() == 1
    assert pts.loc[pts["selected"], "constraint"].item() == "demographic_parity"


# --- Model card ------------------------------------------------------------------------------------

def test_model_card_contains_required_sections():
    ci = lambda v: {"val": v, "ci_low": v / 2, "ci_high": v * 2, "signed_ci_low": -0.01, "signed_ci_high": v * 2, "is_significant": False}
    card = generate_model_card(
        dataset_key="compas", dataset_name="COMPAS", model_label="Random Forest Classifier", protected_col="race",
        kpis={"total_records": 6172, "overall_positive_rate": 0.545, "base_rate_disparity": 0.12, "group_rates": {"Caucasian": 0.6}},
        baseline_pred_metrics={"accuracy": 0.68, "balanced_accuracy": 0.67, "f1_score": 0.63, "roc_auc": 0.73},
        baseline_fairness={m: ci(0.2) for m in ["demographic_parity_diff", "equalized_odds_diff", "equal_opportunity_diff", "predictive_parity_diff"]},
        mitigations={"Reweighing": {"pred_metrics": {"accuracy": 0.67}, "metrics": {"demographic_parity_diff": {"val": 0.19}},
                                    "significance": {"mean_delta": 0.01, "ci_low": -0.01, "ci_high": 0.03, "is_significant": False}}},
        split_sizes={"train": 3700, "val": 1235, "test": 1235}, feature_count=14, selected_eps=0.01,
    )
    for section in ["# Model Card", "## 1. Model Details", "## 2. Intended Use", "## 3. Datasheet",
                    "## 4. Predictive Performance", "## 5. Fairness Evaluation", "## 6. Mitigations", "## 7. Ethical"]:
        assert section in card
    assert "High risk" in card
    assert "decile_score" in card          # COMPAS leakage note
    assert "ε = 0.01" in card
