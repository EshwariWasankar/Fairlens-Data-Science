"""Fairness-accuracy frontier over the Exponentiated Gradient epsilon sweep."""
from typing import Any, Dict, List

import pandas as pd


def pareto_frontier(df: pd.DataFrame, x: str = "dp_diff", y: str = "accuracy") -> pd.DataFrame:
    """Rows not dominated by any other row (lower ``x`` is better, higher ``y`` is better),
    sorted by ``x``. A point is dominated if another is at least as good on both axes and strictly
    better on one."""
    # Positional indexing: callers often concat frames, which leaves duplicate index labels.
    pts = df.reset_index(drop=True).sort_values([x, y], ascending=[True, False])
    keep, best_y = [], float("-inf")
    for pos, row in pts.iterrows():
        if row[y] > best_y:
            keep.append(pos)
            best_y = row[y]
    return pts.loc[keep].sort_values(x).reset_index(drop=True)


def sweep_points(benchmarks: Dict[str, Any], dataset_key: str) -> pd.DataFrame:
    """Flatten the epsilon sweep stored in verified_benchmark_results.json into plot-ready rows."""
    rows: List[Dict[str, Any]] = []
    sweep = benchmarks.get("eg_sweep_results", {}).get(dataset_key, {})
    selected = benchmarks.get("selected_epsilons", {}).get(dataset_key)
    for constraint, entries in sweep.items():
        for r in entries:
            rows.append({
                "constraint": constraint,
                "eps": r["eps"],
                "accuracy": float(r["accuracy_raw"]),
                "dp_diff": float(r["dp_diff_raw"]),
                "eo_diff": float(r["eo_diff"]),
                "selected": constraint == "demographic_parity" and r["eps"] == selected,
            })
    return pd.DataFrame(rows)
