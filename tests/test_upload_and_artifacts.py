"""Regression tests for bugs found in the pre-submission audit: uploaded-CSV handling and
benchmark model files being overwritten by live runs."""
import numpy as np
import pandas as pd
import pytest

import src.models.trainer as trainer
from src.data.cleaner import clean_dataset
from src.data.loader import binarize_target, validate_custom_csv, load_raw_dataset
from src.data.preprocessor import preprocess_dataset
from src.models.trainer import train_model, model_matches_features
from src.pipeline import run_fairlens_pipeline
from src.analysis.proxy import compute_proxy_leakage


def _upload(n=400, seed=0):
    rng = np.random.default_rng(seed)
    g = rng.choice(["F", "M"], n)
    x = rng.normal(size=n)
    y = ((x + (g == "M") * 0.5 + rng.normal(size=n)) > 0.3).astype(int)
    # 'income' and 'start' are ordinary features here, even though they are label/leakage names in the benchmarks
    return pd.DataFrame({"income": x, "start": rng.integers(0, 9, n), "city": rng.choice(["A", "B"], n),
                         "gender": g, "approved": y})


# --- Live runs must not overwrite the benchmark model the dashboard loads -------------------

def test_live_pipeline_does_not_write_benchmark_model(tmp_path):
    before = set(tmp_path.iterdir())          # conftest points MODELS_DIR at a temp dir
    run_fairlens_pipeline("german_credit", protected_attribute="foreign_worker", model_type="random_forest")
    written = {p.name for p in set(trainer.MODELS_DIR.iterdir())}
    assert "german_credit_random_forest_baseline.joblib" not in written


def test_stale_model_is_detected():
    df = clean_dataset(load_raw_dataset("german_credit"), "german_credit")
    by_sex = preprocess_dataset(df, target_col="credit_risk", protected_col="sex")
    by_fw = preprocess_dataset(df, target_col="credit_risk", protected_col="foreign_worker")
    model, _ = train_model(by_fw, model_type="random_forest", save=False)
    assert model_matches_features(model, by_fw.feature_names)
    assert not model_matches_features(model, by_sex.feature_names)


# --- Target binarization ---------------------------------------------------------------------

@pytest.mark.parametrize("values,positive", [
    ([0, 1, 1, 0], 1),
    ([True, False, True], True),
    (["no", "yes", "yes"], "yes"),
    (["Rejected", "Approved"], "Approved"),
    ([1, 2, 2, 1], 2),
    (["b", "a", "b"], "b"),
])
def test_binarize_target(values, positive):
    mapped, pos = binarize_target(pd.Series(values))
    assert pos == positive
    assert set(mapped.dropna().unique()) <= {0, 1}
    assert (mapped == 1).sum() == sum(v == positive for v in values)


def test_binarize_target_respects_explicit_choice():
    mapped, pos = binarize_target(pd.Series([1, 2, 2, 1]), positive=1)   # e.g. UCI German Credit: 1 = good
    assert pos == 1 and mapped.tolist() == [1, 0, 0, 1]
    with pytest.raises(ValueError):
        binarize_target(pd.Series(["a", "b"]), positive="c")


def test_pipeline_accepts_text_and_one_two_targets():
    for transform in (lambda s: s.map({1: "yes", 0: "no"}), lambda s: s + 1):
        df = _upload()
        df["approved"] = transform(df["approved"])
        r = run_fairlens_pipeline("custom", custom_df=df, custom_target_col="approved",
                                  custom_protected_col="gender", model_type="xgboost")
        assert 0 <= r["baseline_fairness"]["demographic_parity_diff"]["val"] <= 1


# --- Uploads keep their own columns; benchmark drop lists do not apply ------------------------

def test_upload_keeps_columns_named_like_benchmark_labels():
    df = clean_dataset(_upload(), "custom")
    d = preprocess_dataset(df, target_col="approved", protected_col="gender", drop_known_leakage=False)
    assert "income" in d.feature_names and "start" in d.feature_names
    assert "approved" not in d.feature_names and not any(c.startswith("gender") for c in d.feature_names)


def test_benchmarks_still_drop_known_leakage():
    df = clean_dataset(load_raw_dataset("compas"), "compas")
    d = preprocess_dataset(df, target_col="two_year_recid", protected_col="race")
    assert not any(c.startswith(("decile_score", "score_text", "two_year_recid")) for c in d.feature_names)


# --- Missing values are imputed from the training split only --------------------------------

def test_missing_numeric_values_imputed_with_train_median():
    df = _upload()
    df.loc[df.index[::7], "income"] = np.nan
    d = preprocess_dataset(clean_dataset(df, "custom"), target_col="approved", protected_col="gender",
                           drop_known_leakage=False)
    assert not d.X_train.isna().any().any() and not d.X_test.isna().any().any()
    r = run_fairlens_pipeline("custom", custom_df=df, custom_target_col="approved",
                              custom_protected_col="gender", model_type="logistic_regression")
    assert len(r["mitigations"]) == 3


# --- Validator rejects inputs the pipeline cannot audit meaningfully -------------------------

def test_validator_rejects_continuous_attribute_and_tiny_data():
    df = _upload()
    df["age"] = np.random.default_rng(1).integers(18, 70, len(df))
    ok, msg = validate_custom_csv(df, "approved", "age")
    assert not ok and "continuous" in msg
    ok, msg = validate_custom_csv(_upload().head(30), "approved", "gender")
    assert not ok and "rows" in msg
    assert validate_custom_csv(_upload(), "approved", "gender")[0]


def test_foreign_worker_uses_configured_group_names():
    df = clean_dataset(load_raw_dataset("german_credit"), "german_credit")
    assert set(df["foreign_worker"].unique()) == {"yes", "no"}


def test_proxy_analysis_handles_gaps_and_keeps_upload_columns():
    df = _upload()
    df.loc[df.index[::9], "start"] = np.nan          # blank cells must not crash mutual information
    res = compute_proxy_leakage(df, protected_col="gender", target_col="approved", drop_known_leakage=False)
    assert {"income", "start", "city"} <= set(res["feature"])
    assert res["mutual_information"].notna().all()
    bench = compute_proxy_leakage(df.dropna(), protected_col="gender", target_col="approved")
    assert "income" not in set(bench["feature"])     # benchmark mode still drops the label alias
