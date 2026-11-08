import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional

def compute_group_rates(y_true: np.ndarray, y_pred: np.ndarray, s: np.ndarray) -> pd.DataFrame:
    """Computes selection rate, TPR, FPR, and Precision for each subgroup in protected attribute vector s."""
    df = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred, 's': s})
    groups = df['s'].unique()
    
    group_stats = []
    for g in groups:
        sub = df[df['s'] == g]
        n = len(sub)
        if n == 0:
            continue
            
        pos_pred = (sub['y_pred'] == 1).sum()
        sel_rate = pos_pred / n
        
        pos_true = (sub['y_true'] == 1).sum()
        tpr = (sub[(sub['y_pred'] == 1) & (sub['y_true'] == 1)]).shape[0] / pos_true if pos_true > 0 else 0.0
        
        neg_true = (sub['y_true'] == 0).sum()
        fpr = (sub[(sub['y_pred'] == 1) & (sub['y_true'] == 0)]).shape[0] / neg_true if neg_true > 0 else 0.0
        
        prec = (sub[(sub['y_pred'] == 1) & (sub['y_true'] == 1)]).shape[0] / pos_pred if pos_pred > 0 else 0.0
        
        group_stats.append({
            'group': str(g),
            'count': n,
            'selection_rate': float(sel_rate),
            'tpr': float(tpr),
            'fpr': float(fpr),
            'precision': float(prec)
        })
        
    return pd.DataFrame(group_stats)

def compute_fairness_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    s: np.ndarray,
    privileged_group: Optional[str] = None,
    unprivileged_group: Optional[str] = None,
    min_group_count: int = 30
) -> Dict[str, Any]:
    """Computes Demographic Parity Difference, Equalized Odds Difference, Equal Opportunity Difference, and Predictive Parity Difference."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    s = np.asarray(s)
    
    group_df = compute_group_rates(y_true, y_pred, s)
    if group_df.empty or len(group_df) < 2:
        return {
            "demographic_parity_diff": 0.0,
            "equalized_odds_diff": 0.0,
            "equal_opportunity_diff": 0.0,
            "predictive_parity_diff": 0.0,
            "group_rates": group_df.to_dict(orient="records")
        }

    # Filter out tiny subgroups (count < min_group_count) for robust disparity calculations
    valid_group_df = group_df[group_df['count'] >= min_group_count].copy()
    if valid_group_df.empty or len(valid_group_df) < 2:
        valid_group_df = group_df

    # Case 1: Direct Pairwise Comparison (Privileged vs Unprivileged)
    if privileged_group is not None and unprivileged_group is not None \
       and str(privileged_group) in group_df['group'].values \
       and str(unprivileged_group) in group_df['group'].values:
        
        priv_row = group_df[group_df['group'] == str(privileged_group)].iloc[0]
        unread_row = group_df[group_df['group'] == str(unprivileged_group)].iloc[0]
        
        dp_diff = float(np.abs(priv_row['selection_rate'] - unread_row['selection_rate']))
        tpr_diff = float(np.abs(priv_row['tpr'] - unread_row['tpr']))
        fpr_diff = float(np.abs(priv_row['fpr'] - unread_row['fpr']))
        eo_diff = float(max(tpr_diff, fpr_diff))
        eopp_diff = tpr_diff
        pp_diff = float(np.abs(priv_row['precision'] - unread_row['precision']))

    # Case 2: Privileged Group vs Largest Unprivileged Subgroup
    elif privileged_group is not None and str(privileged_group) in valid_group_df['group'].values:
        priv_row = valid_group_df[valid_group_df['group'] == str(privileged_group)].iloc[0]
        unread_rows = valid_group_df[valid_group_df['group'] != str(privileged_group)]
        
        # Compare against the largest unprivileged group
        largest_unread = unread_rows.sort_values(by="count", ascending=False).iloc[0]
        
        dp_diff = float(np.abs(priv_row['selection_rate'] - largest_unread['selection_rate']))
        tpr_diff = float(np.abs(priv_row['tpr'] - largest_unread['tpr']))
        fpr_diff = float(np.abs(priv_row['fpr'] - largest_unread['fpr']))
        eo_diff = float(max(tpr_diff, fpr_diff))
        eopp_diff = tpr_diff
        pp_diff = float(np.abs(priv_row['precision'] - largest_unread['precision']))

    # Case 3: Max - Min difference across valid substantial subgroups (n >= 10)
    else:
        dp_diff = float(valid_group_df['selection_rate'].max() - valid_group_df['selection_rate'].min())
        tpr_diff = float(valid_group_df['tpr'].max() - valid_group_df['tpr'].min())
        fpr_diff = float(valid_group_df['fpr'].max() - valid_group_df['fpr'].min())
        eo_diff = float(max(tpr_diff, fpr_diff))
        eopp_diff = tpr_diff
        pp_diff = float(valid_group_df['precision'].max() - valid_group_df['precision'].min())

    return {
        "demographic_parity_diff": np.round(dp_diff, 4),
        "equalized_odds_diff": np.round(eo_diff, 4),
        "equal_opportunity_diff": np.round(eopp_diff, 4),
        "predictive_parity_diff": np.round(pp_diff, 4),
        "group_rates": group_df.to_dict(orient="records")
    }
