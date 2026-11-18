import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional

def _resolve_columns(df: pd.DataFrame, target_col: str, protected_col: str) -> Tuple[Optional[str], Optional[str]]:
    """Resolves target and protected column names from dataframe columns or common aliases."""
    resolved_target = target_col if target_col in df.columns else None
    if resolved_target is None:
        if 'target' in df.columns:
            resolved_target = 'target'
        else:
            for alias in ['credit_risk', 'two_year_recid', 'income', 'class', 'Default', 'default']:
                if alias in df.columns:
                    resolved_target = alias
                    break

    resolved_protected = protected_col if protected_col in df.columns else None
    if resolved_protected is None:
        for alias in ['sex', 'race', 'age_group', 'age', 'foreign_worker']:
            if alias in df.columns:
                resolved_protected = alias
                break

    return resolved_target, resolved_protected

def _extract_binary_target(series: pd.Series) -> pd.Series:
    """Extracts binary 0/1 target series handling strings, 1/2 UCI encoding, and numerics."""
    if series.dtype == 'object' or str(series.dtype) == 'string':
        return series.apply(lambda x: 1 if str(x).strip().lower() in ['1', 'good', '>50k', 'positive'] else 0)
    
    u_vals = set(series.dropna().unique())
    if u_vals == {1, 2} or u_vals == {1.0, 2.0}:
        return series.apply(lambda x: 1 if x == 1 else 0)
        
    return series.astype(int)

def get_dataset_kpis(df: pd.DataFrame, target_col: str, protected_col: str, favorable_value: int = 1) -> Dict[str, Any]:
    """Computes high-level dataset KPIs. 'Positive rate' is the FAVORABLE-outcome rate, where the
    favorable label comes from the dataset config (e.g. 0 = no recidivism for COMPAS)."""
    total_records = len(df)
    col_target, col_protected = _resolve_columns(df, target_col, protected_col)
    
    if col_target is None or col_protected is None:
        return {
            "total_records": total_records,
            "overall_positive_rate": 0.0,
            "protected_groups_count": 0,
            "base_rate_disparity": 0.0,
            "group_disparities": {}
        }
        
    y = (_extract_binary_target(df[col_target]) == favorable_value).astype(int)
    overall_pos_rate = float(y.mean())
    
    temp_df = pd.DataFrame({
        col_protected: df[col_protected],
        '_y_eval': y
    })
    group_stats = temp_df.groupby(col_protected)['_y_eval'].agg(['count', 'mean']).reset_index()
    group_stats.columns = [col_protected, 'sample_count', 'favorable_rate']
    
    rates = {str(k): float(v) for k, v in group_stats.set_index(col_protected)['favorable_rate'].to_dict().items()}
    
    if len(rates) >= 2:
        max_rate = max(rates.values())
        min_rate = min(rates.values())
        base_disparity = float(max_rate - min_rate)
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

def get_group_disparity_table(df: pd.DataFrame, target_col: str, protected_col: str, favorable_value: int = 1) -> pd.DataFrame:
    """Returns detailed breakdown of approval/favorable rates per protected subgroup."""
    col_target, col_protected = _resolve_columns(df, target_col, protected_col)
    
    if col_target is None or col_protected is None:
        return pd.DataFrame()
        
    y = (_extract_binary_target(df[col_target]) == favorable_value).astype(int)
    temp_df = pd.DataFrame({
        'Protected_Group': df[col_protected].astype(str),
        '_y_eval': y
    })
    
    grouped = temp_df.groupby('Protected_Group')['_y_eval'].agg(
        Total_Applicants='count',
        Favorable_Outcomes='sum',
        Favorable_Rate=lambda x: (x.sum() / len(x)) * 100.0
    ).reset_index()
    
    grouped['Unfavorable_Rate'] = 100.0 - grouped['Favorable_Rate']
    return grouped

def get_numeric_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Computes pairwise Pearson correlation matrix for numerical features, excluding outlier flags."""
    num_df = df.select_dtypes(include=[np.number]).copy()
    flag_cols = [c for c in num_df.columns if c.endswith('_outlier_flag')]
    if flag_cols:
        num_df = num_df.drop(columns=flag_cols, errors='ignore')
    if num_df.empty:
        return pd.DataFrame()
    return num_df.corr().round(3)
