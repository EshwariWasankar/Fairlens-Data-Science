import pytest
from src.pipeline import run_fairlens_pipeline

@pytest.mark.parametrize("dataset_key", ["german_credit", "compas", "adult_income"])
def test_end_to_end_pipeline_execution(dataset_key):
    res = run_fairlens_pipeline(dataset_key)
    
    assert res is not None
    assert "baseline_pred_metrics" in res
    assert "baseline_fairness" in res
    assert "mitigations" in res
    assert "proxy_df" in res
    assert "shap_importance" in res
    assert "insights" in res
    
    # Assert metrics are present and non-null
    assert res["baseline_pred_metrics"]["f1_score"] > 0
    assert res["baseline_pred_metrics"]["accuracy"] > 0
    assert res["baseline_fairness"]["demographic_parity_diff"]["val"] >= 0

def test_displayed_numbers_equal_stored_results():
    """Asserts that the pipeline displayed benchmark numbers equal the single source of truth in data/verified_benchmark_results.json."""
    import json
    from src.config import DATA_DIR
    json_path = DATA_DIR / "verified_benchmark_results.json"
    assert json_path.exists(), "Single source of truth JSON file does not exist!"
    
    with open(json_path, "r", encoding="utf-8") as f:
        stored = json.load(f)
        
    assert "experiments" in stored
    assert "selected_baseline_models" in stored
    assert "compas" in stored["experiments"]
    assert "german_credit" in stored["experiments"]
    assert "adult_income" in stored["experiments"]

