"""Intersectional fairness: metrics over combinations of two protected attributes (e.g. race x sex).

Single-attribute audits can hide disparities that only appear for intersections (e.g. young Black
women). Subgroups smaller than ``min_count`` are reported but excluded from the gap statistics,
because rates estimated from a handful of people are mostly noise.
"""
from typing import Any, Dict

import numpy as np
import pandas as pd

from src.fairness.metrics import compute_group_rates


def compute_intersectional_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    attr_a: pd.Series,
    attr_b: pd.Series,
    min_count: int = 30,
) -> Dict[str, Any]:
    """Per-intersection selection rate / TPR / FPR / precision, plus max-min gaps across
    intersections with at least ``min_count`` members."""
    a = np.asarray(attr_a).astype(str)
    b = np.asarray(attr_b).astype(str)
    combo = np.char.add(np.char.add(a, " × "), b)

    table = compute_group_rates(np.asarray(y_true), np.asarray(y_pred), combo)
    parts = table["group"].str.split(" × ", n=1, expand=True)
    table.insert(1, "attr_a", parts[0])
    table.insert(2, "attr_b", parts[1])
    table["included"] = table["count"] >= min_count
    table = table.sort_values("count", ascending=False).reset_index(drop=True)

    valid = table[table["included"]]
    if len(valid) >= 2:
        gaps = {
            "demographic_parity_diff": float(valid["selection_rate"].max() - valid["selection_rate"].min()),
            "equal_opportunity_diff": float(valid["tpr"].max() - valid["tpr"].min()),
            "fpr_diff": float(valid["fpr"].max() - valid["fpr"].min()),
        }
        gaps["equalized_odds_diff"] = max(gaps["equal_opportunity_diff"], gaps["fpr_diff"])
        hi = valid.loc[valid["selection_rate"].idxmax()]
        lo = valid.loc[valid["selection_rate"].idxmin()]
        extremes = {"highest_selection_group": hi["group"], "lowest_selection_group": lo["group"]}
    else:
        gaps = {k: float("nan") for k in ["demographic_parity_diff", "equal_opportunity_diff", "fpr_diff", "equalized_odds_diff"]}
        extremes = {"highest_selection_group": None, "lowest_selection_group": None}

    # Max-min gap for each attribute alone over the same rows, so the intersectional gap can be
    # compared with what a single-attribute audit would report.
    def _single_gap(groups: np.ndarray) -> float:
        rates = compute_group_rates(np.asarray(y_true), np.asarray(y_pred), groups)
        rates = rates[rates["count"] >= min_count]
        return float(rates["selection_rate"].max() - rates["selection_rate"].min()) if len(rates) >= 2 else float("nan")

    return {
        "table": table,
        "gaps": gaps,
        **extremes,
        "single_attribute_dp": {"attr_a": _single_gap(a), "attr_b": _single_gap(b)},
        "n_included": int(table["included"].sum()),
        "n_excluded": int((~table["included"]).sum()),
        "min_count": min_count,
    }
