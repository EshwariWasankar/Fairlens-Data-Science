import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional
from src.fairness.metrics import compute_fairness_metrics

# Group-rate column behind each "simple" gap metric (EO is a max of two gaps, so it has no single sign).
_SIGNED_RATE_COLS = {
    "demographic_parity_diff": "selection_rate",
    "equal_opportunity_diff": "tpr",
    "predictive_parity_diff": "precision",
}


def _rates_by_group(group_rates) -> Dict[str, Dict[str, float]]:
    return {str(r["group"]): r for r in group_rates}


def _reference_pair(point_m: Dict[str, Any], rate_col: str, privileged_group, unprivileged_group, min_group_count: int = 30):
    """Fixes the two groups whose gap is tested, oriented so the full-sample gap is >= 0.

    Mirrors compute_fairness_metrics: explicit privileged/unprivileged pair, else privileged vs the
    largest other group, else the highest- vs lowest-rate group (n >= min_group_count).
    """
    rates = _rates_by_group(point_m["group_rates"])
    if len(rates) < 2:
        return None
    valid = {g: r for g, r in rates.items() if r["count"] >= min_group_count}
    if len(valid) < 2:
        valid = rates
    p, u = (str(privileged_group) if privileged_group is not None else None,
            str(unprivileged_group) if unprivileged_group is not None else None)
    if p in rates and u in rates:
        a, b = p, u
    elif p in valid:
        others = [g for g in valid if g != p]
        a, b = p, max(others, key=lambda g: valid[g]["count"])
    else:
        a = max(valid, key=lambda g: valid[g][rate_col])
        b = min(valid, key=lambda g: valid[g][rate_col])
    if rates[a][rate_col] < rates[b][rate_col]:
        a, b = b, a
    return a, b

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

    # Point estimate on the full (non-resampled) test set. The bootstrap mean of an
    # absolute difference is biased upward, so it must not be reported as the metric value.
    point_m = compute_fairness_metrics(y_true, y_pred, s, privileged_group, unprivileged_group)

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
    pairs = {m: _reference_pair(point_m, col, privileged_group, unprivileged_group) for m, col in _SIGNED_RATE_COLS.items()}
    signed = {m: [] for m in _SIGNED_RATE_COLS}

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
        boot_rates = _rates_by_group(m["group_rates"])
        for metric_name, col in _SIGNED_RATE_COLS.items():
            pair = pairs[metric_name]
            if pair and pair[0] in boot_rates and pair[1] in boot_rates:
                signed[metric_name].append(boot_rates[pair[0]][col] - boot_rates[pair[1]][col])

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
            "val": np.round(float(point_m[metric_name]), 4),
            "bootstrap_mean": np.round(mean_val, 4),
            "ci_low": np.round(ci_low, 4),
            "ci_high": np.round(ci_high, 4),
            "bootstrap_distribution": dist
        }
        # Significance must use the SIGNED gap: a CI on |gap| essentially never contains 0,
        # so "CI excludes zero" on the absolute metric is not a valid test.
        sdist = signed.get(metric_name)
        if sdist:
            s_low = float(np.percentile(sdist, lower_pct))
            s_high = float(np.percentile(sdist, upper_pct))
            results[metric_name].update({
                "signed_ci_low": np.round(s_low, 4),
                "signed_ci_high": np.round(s_high, 4),
                "reference_groups": list(pairs[metric_name]),
                "is_significant": bool(s_low > 0),
            })

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
