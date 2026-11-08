import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from src.config import DATASET_CONFIGS

def clean_dataset(df: pd.DataFrame, dataset_key: str) -> pd.DataFrame:
    """Cleans a raw dataset according to dataset-specific configuration and standard data quality rules."""
    cleaned_df = df.copy()
    
    # 1. Standardize string whitespace and missing token handling (e.g., '?')
    for col in cleaned_df.select_dtypes(include=['object', 'string']).columns:
        cleaned_df[col] = cleaned_df[col].astype(str).str.strip()
        cleaned_df[col] = cleaned_df[col].replace({'?': np.nan, 'Unknown': np.nan, 'unknown': np.nan})

    # 2. Exact row deduplication
    initial_len = len(cleaned_df)
    cleaned_df = cleaned_df.drop_duplicates().reset_index(drop=True)
    dedup_count = initial_len - len(cleaned_df)
    if dedup_count > 0:
        print(f"[{dataset_key}] Removed {dedup_count} exact duplicate rows.")

    # 3. Dataset-specific Cleaning & Target Mapping
    if dataset_key == "compas":
        relevant_cols = [
            'age', 'juv_fel_count', 'juv_misd_count', 'juv_other_count',
            'priors_count', 'c_charge_degree', 'days_b_screening_arrest',
            'sex', 'race', 'age_cat', 'two_year_recid'
        ]
        available_cols = [c for c in relevant_cols if c in cleaned_df.columns]
        cleaned_df = cleaned_df[available_cols].copy()
        
        # Filter outliers in days_b_screening_arrest standard in COMPAS literature (-30 to 30 days)
        if 'days_b_screening_arrest' in cleaned_df.columns:
            cleaned_df = cleaned_df[(cleaned_df['days_b_screening_arrest'] >= -30) & (cleaned_df['days_b_screening_arrest'] <= 30)]

        # Target: 0 = No Recidivism (Favorable), 1 = Recidivism (Unfavorable)
        if 'two_year_recid' in cleaned_df.columns:
            cleaned_df['target'] = cleaned_df['two_year_recid'].astype(int)
            cleaned_df.drop(columns=['two_year_recid'], inplace=True, errors='ignore')

    elif dataset_key == "german_credit":
        target_col = DATASET_CONFIGS["german_credit"]["target_column"]
        if target_col in cleaned_df.columns:
            u_vals = cleaned_df[target_col].unique()
            if 2 in u_vals or '2' in u_vals:
                mapped_target = cleaned_df[target_col].apply(lambda x: 1 if str(x) in ['1', 'good'] else 0)
            else:
                mapped_target = cleaned_df[target_col].astype(int)
            cleaned_df['target'] = mapped_target
            if target_col != 'target':
                cleaned_df.drop(columns=[target_col], inplace=True, errors='ignore')

        if 'age' in cleaned_df.columns and 'age_group' not in cleaned_df.columns:
            cleaned_df['age_group'] = cleaned_df['age'].apply(lambda a: "older (>25)" if a > 25 else "young (<=25)")

        # Ensure 'sex' is extracted from personal_status_sex if not already present
        if 'sex' not in cleaned_df.columns and 'personal_status_sex' in cleaned_df.columns:
            cleaned_df['sex'] = cleaned_df['personal_status_sex'].apply(
                lambda val: 'female' if str(val).strip() in ['A92', 'A95', 'female'] else 'male'
            )

        # Explicitly drop personal_status_sex to eliminate 1.0/-1.0 proxy leakage with sex
        if 'personal_status_sex' in cleaned_df.columns:
            cleaned_df.drop(columns=['personal_status_sex'], inplace=True, errors='ignore')

    elif dataset_key == "adult_income":
        target_col = DATASET_CONFIGS["adult_income"]["target_column"]
        if target_col in cleaned_df.columns:
            mapped_target = cleaned_df[target_col].apply(
                lambda x: 1 if '>50K' in str(x) else 0
            )
            cleaned_df['target'] = mapped_target
            if target_col != 'target':
                cleaned_df.drop(columns=[target_col], inplace=True, errors='ignore')

        for col in ['workclass', 'occupation', 'native_country']:
            if col in cleaned_df.columns and cleaned_df[col].isna().sum() > 0:
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mode()[0])

        if 'age' in cleaned_df.columns and 'age_group' not in cleaned_df.columns:
            cleaned_df['age_group'] = cleaned_df['age'].apply(
                lambda a: "prime_age (25-60)" if 25 <= a <= 60 else "young (<25)" if a < 25 else "senior (>60)"
            )

    else:
        # Custom dataset cleaning
        pass

    # 4. Outlier Flagging via IQR (Non-destructive)
    numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
    for ncol in numeric_cols:
        if ncol in ['target', 'two_year_recid', 'credit_risk', 'income']:
            continue
        q1 = cleaned_df[ncol].quantile(0.25)
        q3 = cleaned_df[ncol].quantile(0.75)
        iqr = q3 - q1
        lower_b = q1 - 1.5 * iqr
        upper_b = q3 + 1.5 * iqr
        cleaned_df[f"{ncol}_outlier_flag"] = ((cleaned_df[ncol] < lower_b) | (cleaned_df[ncol] > upper_b)).astype(int)

    return cleaned_df.reset_index(drop=True)
