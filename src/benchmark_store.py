from typing import Any, Dict


def uses_precomputed_benchmark(
    benchmarks: Dict[str, Any],
    dataset_key: str,
    protected_col: str,
    model_type: str,
    default_protected_col: str,
) -> bool:
    """True only when the dashboard selection is exactly the configuration that master_eval.py
    benchmarked (default protected attribute AND the selected baseline model for this dataset).

    Any other selection must be computed live; otherwise the dashboard would show one model's
    metrics under another model's name.
    """
    exp = benchmarks.get("experiments", {}).get(dataset_key, {})
    return (
        protected_col == default_protected_col
        and model_type == benchmarks.get("selected_baseline_models", {}).get(dataset_key)
        and "baseline_fairness" in exp
        and "mitigations" in exp
    )
