import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional
from src.fairness.metrics import compute_fairness_metrics

def compute_bootstrap_confidence_intervals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    s: np.ndarray,
    n_bootstraps: int = 500,
    ci_level: float = 0.95,
    privileged_group: Optional[str] = None,
    unprivileged_group: Optional[str] = None,
    random_state: int = 42
) -> Dict[str, Dict[str, Any]]:
    """Runs bootstrap resampling on test set predictions to generate 95% Confidence Intervals for all 4 fairness metrics."""
    np.random.seed(random_state)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    s = np.asarray(s)

    if len(y_true) > 3000:
        sub_idx = np.random.choice(len(y_true), size=3000, replace=False)
        y_true = y_true[sub_idx]
        y_pred = y_pred[sub_idx]
        s = s[sub_idx]

    n_samples = len(y_true)
    
    if n_samples < 10:
        base_m = compute_fairness_metrics(y_true, y_pred, s, privileged_group, unprivileged_group)
        return {
            metric: {
                "val": base_m[metric],
                "ci_low": base_m[metric],
                "ci_high": base_m[metric],
                "bootstrap_distribution": [base_m[metric]]
            }
            for metric in ["demographic_parity_diff", "equalized_odds_diff", "equal_opportunity_diff", "predictive_parity_diff"]
        }

    dp_list, eo_list, eopp_list, pp_list = [], [], [], []

    for _ in range(n_bootstraps):
        boot_idx = np.random.choice(n_samples, size=n_samples, replace=True)
        y_t_boot = y_true[boot_idx]
        y_p_boot = y_pred[boot_idx]
        s_boot = s[boot_idx]
        
        m = compute_fairness_metrics(y_t_boot, y_p_boot, s_boot, privileged_group, unprivileged_group)
        dp_list.append(m["demographic_parity_diff"])
        eo_list.append(m["equalized_odds_diff"])
        eopp_list.append(m["equal_opportunity_diff"])
        pp_list.append(m["predictive_parity_diff"])

    alpha = (1.0 - ci_level) / 2.0
    lower_pct = alpha * 100.0
    upper_pct = (1.0 - alpha) * 100.0

    results = {}
    metric_map = {
        "demographic_parity_diff": dp_list,
        "equalized_odds_diff": eo_list,
        "equal_opportunity_diff": eopp_list,
        "predictive_parity_diff": pp_list
    }

    for metric_name, dist in metric_map.items():
        ci_low = float(np.percentile(dist, lower_pct))
        ci_high = float(np.percentile(dist, upper_pct))
        mean_val = float(np.mean(dist))
        results[metric_name] = {
            "val": np.round(mean_val, 4),
            "ci_low": np.round(ci_low, 4),
            "ci_high": np.round(ci_high, 4),
            "bootstrap_distribution": dist
        }

    return results

def compute_paired_bootstrap_difference(
    y_true: np.ndarray,
    y_pred_base: np.ndarray,
    y_pred_mit: np.ndarray,
    s: np.ndarray,
    n_bootstraps: int = 500,
    ci_level: float = 0.95,
    privileged_group: Optional[str] = None,
    unprivileged_group: Optional[str] = None,
    random_state: int = 42
) -> Dict[str, Dict[str, Any]]:
    """
    Performs paired bootstrap resampling of the difference (baseline - mitigated) on identical 
    resampled rows. Reports mean delta, 95% CI of delta, and marks is_significant=True 
    ONLY IF the 95% CI excludes 0 (specifically ci_low > 0 for a positive disparity reduction).
    """
    np.random.seed(random_state)
    y_true = np.asarray(y_true)
    y_pred_base = np.asarray(y_pred_base)
    y_pred_mit = np.asarray(y_pred_mit)
    s = np.asarray(s)

    if len(y_true) > 3000:
        sub_idx = np.random.choice(len(y_true), size=3000, replace=False)
        y_true = y_true[sub_idx]
        y_pred_base = y_pred_base[sub_idx]
        y_pred_mit = y_pred_mit[sub_idx]
        s = s[sub_idx]

    n_samples = len(y_true)
    metric_names = ["demographic_parity_diff", "equalized_odds_diff", "equal_opportunity_diff", "predictive_parity_diff"]
    
    deltas = {m: [] for m in metric_names}

    for _ in range(n_bootstraps):
        boot_idx = np.random.choice(n_samples, size=n_samples, replace=True)
        y_t_b = y_true[boot_idx]
        s_b = s[boot_idx]
        
        m_base = compute_fairness_metrics(y_t_b, y_pred_base[boot_idx], s_b, privileged_group, unprivileged_group)
        m_mit = compute_fairness_metrics(y_t_b, y_pred_mit[boot_idx], s_b, privileged_group, unprivileged_group)

        for m in metric_names:
            # delta = baseline disparity - mitigated disparity
            delta = m_base[m] - m_mit[m]
            deltas[m].append(delta)

    alpha = (1.0 - ci_level) / 2.0
    lower_pct = alpha * 100.0
    upper_pct = (1.0 - alpha) * 100.0

    results = {}
    for m in metric_names:
        dist = deltas[m]
        mean_delta = float(np.mean(dist))
        ci_low = float(np.percentile(dist, lower_pct))
        ci_high = float(np.percentile(dist, upper_pct))
        
        # Marked significant ONLY IF the 95% CI excludes 0 (ci_low > 0 for disparity reduction)
        is_significant = bool(ci_low > 0)
        
        # Empirical p-value: proportion of bootstrap deltas <= 0
        p_val = float(np.mean(np.array(dist) <= 0))

        results[m] = {
            "mean_delta": np.round(mean_delta, 4),
            "ci_low": np.round(ci_low, 4),
            "ci_high": np.round(ci_high, 4),
            "is_significant": is_significant,
            "p_value": np.round(p_val, 4),
            "delta_distribution": dist
        }

    return results
