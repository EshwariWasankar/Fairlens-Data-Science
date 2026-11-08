"""Calibration by group: reliability curves, Brier score and expected calibration error (ECE).

This is the centre of the COMPAS debate: a score can be equally calibrated across groups
(a "7" means the same risk for everyone) while still producing unequal error rates — the two
fairness notions cannot generally hold at once when base rates differ.
"""
from typing import Any, Dict

import numpy as np
import pandas as pd


def _ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int) -> float:
    """Expected calibration error with equal-width bins (weighted mean |observed - predicted|)."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(y_prob, edges[1:-1]), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(y_true[m].mean() - y_prob[m].mean())
    return float(ece)


def _reliability_curve(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int) -> pd.DataFrame:
    """Quantile-binned reliability curve (each point ~ equal number of people)."""
    qs = np.unique(np.quantile(y_prob, np.linspace(0, 1, n_bins + 1)))
    if len(qs) < 2:
        return pd.DataFrame({"mean_predicted": [float(y_prob.mean())], "observed_rate": [float(y_true.mean())], "count": [len(y_true)]})
    idx = np.clip(np.digitize(y_prob, qs[1:-1]), 0, len(qs) - 2)
    rows = []
    for b in range(len(qs) - 1):
        m = idx == b
        if m.any():
            rows.append({"mean_predicted": float(y_prob[m].mean()), "observed_rate": float(y_true[m].mean()), "count": int(m.sum())})
    return pd.DataFrame(rows)


def compute_group_calibration(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    s: np.ndarray,
    n_bins: int = 10,
    min_count: int = 30,
) -> Dict[str, Any]:
    """Per-group Brier score, ECE and reliability curve for P(label = 1).

    Groups with fewer than ``min_count`` members are skipped (their curves are noise).
    """
    y_true = np.asarray(y_true).astype(float)
    y_prob = np.asarray(y_prob).astype(float)
    s = np.asarray(s).astype(str)

    summary, curves = [], []
    for g in pd.unique(s):
        m = s == g
        n = int(m.sum())
        if n < min_count:
            continue
        yt, yp = y_true[m], y_prob[m]
        summary.append({
            "group": g,
            "count": n,
            "brier_score": float(np.mean((yp - yt) ** 2)),
            "ece": _ece(yt, yp, n_bins),
            "mean_predicted": float(yp.mean()),
            "observed_rate": float(yt.mean()),
        })
        curve = _reliability_curve(yt, yp, n_bins)
        curve.insert(0, "group", g)
        curves.append(curve)

    summary_df = pd.DataFrame(summary).sort_values("count", ascending=False).reset_index(drop=True) if summary else pd.DataFrame()
    curves_df = pd.concat(curves, ignore_index=True) if curves else pd.DataFrame()
    overall = {"brier_score": float(np.mean((y_prob - y_true) ** 2)), "ece": _ece(y_true, y_prob, n_bins)}
    gaps = {}
    if len(summary_df) >= 2:
        gaps = {
            "brier_gap": float(summary_df["brier_score"].max() - summary_df["brier_score"].min()),
            "ece_gap": float(summary_df["ece"].max() - summary_df["ece"].min()),
        }
    return {"summary": summary_df, "curves": curves_df, "overall": overall, "gaps": gaps}
