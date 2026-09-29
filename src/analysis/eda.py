import pandas as pd
import numpy as np
from typing import Dict, Any, List

def get_dataset_kpis(df: pd.DataFrame, target_col: str, protected_col: str) -> Dict[str, Any]:
    """Computes high-level dataset KPIs including overall positive rate and protected group disparity."""
    total_records = len(df)
    
    if target_col not in df.columns or protected_col not in df.columns:
        return {
            "total_records": total_records,
            "overall_positive_rate": 0.0,
            "protected_groups_count": 0,
            "base_rate_disparity": 0.0,
            "group_disparities": {}
        }
        
    y = df[target_col].astype(int)
    overall_pos_rate = y.mean()
    
    group_stats = df.groupby(protected_col)[target_col].agg(['count', 'mean']).reset_index()
    group_stats.columns = [protected_col, 'sample_count', 'favorable_rate']
    
    rates = group_stats.set_index(protected_col)['favorable_rate'].to_dict()
    
    if len(rates) >= 2:
        max_rate = max(rates.values())
        min_rate = min(rates.values())
        base_disparity = max_rate - min_rate
    else:
        base_disparity = 0.0
        
    return {
        "total_records": total_records,
        "overall_positive_rate": overall_pos_rate,
        "protected_groups_count": len(rates),
        "base_rate_disparity": base_disparity,
        "group_rates": rates,
        "group_stats": group_stats.to_dict(orient="records")
    }

def get_group_disparity_table(df: pd.DataFrame, target_col: str, protected_col: str) -> pd.DataFrame:
    """Returns detailed breakdown of approval/favorable rates per protected subgroup."""
    if target_col not in df.columns or protected_col not in df.columns:
        return pd.DataFrame()
        
    grouped = df.groupby(protected_col)[target_col].agg(
        Total_Applicants='count',
        Favorable_Outcomes='sum',
        Favorable_Rate=lambda x: (x.sum() / len(x)) * 100
    ).reset_index()
    
    grouped['Unfavorable_Rate'] = 100.0 - grouped['Favorable_Rate']
    grouped.rename(columns={protected_col: 'Protected_Group'}, inplace=True)
    return grouped

def get_numeric_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Computes pairwise Pearson correlation matrix for numerical features, excluding outlier flags."""
    num_df = df.select_dtypes(include=[np.number]).copy()
    flag_cols = [c for c in num_df.columns if c.endswith('_outlier_flag')]
    if flag_cols:
        num_df.drop(columns=flag_cols, inplace=True, errors='ignore')
    if num_df.empty:
        return pd.DataFrame()
    return num_df.corr().round(3)
