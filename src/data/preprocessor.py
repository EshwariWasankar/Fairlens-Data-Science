import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

@dataclass
class PreprocessedData:
    X_train: pd.DataFrame
    y_train: pd.Series
    s_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    s_val: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    s_test: pd.Series
    X_train_scaled: np.ndarray
    X_val_scaled: np.ndarray
    X_test_scaled: np.ndarray
    feature_names: List[str]
    scaler: StandardScaler
    protected_col: str
    target_col: str

def preprocess_dataset(
    df: pd.DataFrame,
    target_col: str = "target",
    protected_col: str = "sex",
    train_size: float = 0.6,
    val_size: float = 0.2,
    test_size: float = 0.2,
    random_state: int = 42,
    drop_known_leakage: bool = True
) -> PreprocessedData:
    """Preprocesses a cleaned dataframe with joint stratification, feature encoding, and leakage prevention."""
    df_proc = df.copy()
    
    # Ensure target column is named 'target'
    if target_col not in df_proc.columns and 'target' in df_proc.columns:
        target_col = 'target'

    # Extract target and protected attribute
    y = df_proc[target_col]
    s = df_proc[protected_col]

    # Create feature set X by dropping target and protected attribute (and any outlier flags or ID columns)
    drop_cols = [target_col, protected_col, 'target']
    
    # Drop all target aliases
    # Benchmark-specific label aliases and leakage fields. Off for user-uploaded data, where a column
    # called 'income', 'id' or 'start' is an ordinary feature, not a label or a COMPAS output.
    known_aliases = ['two_year_recid', 'credit_risk', 'income', 'class', 'Default', 'default'] if drop_known_leakage else []
    for c in known_aliases:
        if c in df_proc.columns:
            drop_cols.append(c)
            
    # Drop all COMPAS score outputs, future recidivism labels, and metadata fields
    compas_leakage = [
        'decile_score', 'decile_score.1', 'score_text', 'v_decile_score', 'v_score_text',
        'is_recid', 'violent_recid', 'is_violent_recid', 'r_case_number', 'r_charge_degree',
        'r_days_from_arrest', 'r_offense_date', 'r_charge_desc', 'r_jail_in', 'r_jail_out',
        'vr_case_number', 'vr_charge_degree', 'vr_offense_date', 'vr_charge_desc',
        'in_custody', 'out_custody', 'start', 'end', 'event', 'first', 'last',
        'compas_screening_date', 'dob', 'c_jail_in', 'c_jail_out', 'c_case_number',
        'c_offense_date', 'c_arrest_date', 'c_days_from_compas', 'c_charge_desc',
        'type_of_assessment', 'screening_date', 'v_type_of_assessment', 'v_screening_date',
        'fnlwgt', 'personal_status_sex', 'id', 'name'
    ]
    for c in (compas_leakage if drop_known_leakage else []):
        if c in df_proc.columns and c != protected_col:
            drop_cols.append(c)
    
    # Drop outlier flag columns from training features to avoid noise
    drop_cols.extend([c for c in df_proc.columns if c.endswith('_outlier_flag')])

    # Dynamic proxy column dropping based on protected attribute
    if protected_col == 'age_group':
        drop_cols.append('age')
    if protected_col == 'sex':
        drop_cols.append('personal_status_sex')

    X_raw = df_proc.drop(columns=list(set(drop_cols)), errors='ignore')

    # One-hot encode categoricals in X_raw
    X_encoded = pd.get_dummies(X_raw, drop_first=True)
    X_encoded.columns = [
        str(c).replace('[', '_').replace(']', '_').replace('<', 'lt_').replace('>', 'gt_').replace('=', 'eq_')
        for c in X_encoded.columns
    ]

    # Combined strata for joint stratification
    strata = y.astype(str) + "_" + s.astype(str)

    # Handle rare strata if needed (replace singletons)
    strata_counts = strata.value_counts()
    rare_strata = strata_counts[strata_counts < 2].index
    if len(rare_strata) > 0:
        strata = y.astype(str)

    # 1st split: Train (60%) vs Temp (40%)
    X_train_df, X_temp_df, y_train, y_temp, s_train, s_temp = train_test_split(
        X_encoded, y, s,
        train_size=train_size,
        stratify=strata,
        random_state=random_state
    )

    # 2nd split: Val (20%) vs Test (20%) from Temp (50/50 of temp)
    temp_strata = y_temp.astype(str) + "_" + s_temp.astype(str)
    temp_strata_counts = temp_strata.value_counts()
    if (temp_strata_counts < 2).any():
        temp_strata = y_temp.astype(str)

    X_val_df, X_test_df, y_val, y_test, s_val, s_test = train_test_split(
        X_temp_df, y_temp, s_temp,
        test_size=0.5,
        stratify=temp_strata,
        random_state=random_state
    )

    # Align columns across splits
    X_train_df, X_val_df = X_train_df.align(X_val_df, join='left', axis=1, fill_value=0)
    X_train_df, X_test_df = X_train_df.align(X_test_df, join='left', axis=1, fill_value=0)

    # Impute missing numeric values with TRAINING-split medians only (no information from val/test).
    # Categorical gaps are already handled: get_dummies encodes a missing category as all zeros.
    train_medians = X_train_df.median(numeric_only=True)
    X_train_df = X_train_df.fillna(train_medians).fillna(0)
    X_val_df = X_val_df.fillna(train_medians).fillna(0)
    X_test_df = X_test_df.fillna(train_medians).fillna(0)

    # Scale continuous features using StandardScaler fit ONLY on X_train
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_df)
    X_val_scaled = scaler.transform(X_val_df)
    X_test_scaled = scaler.transform(X_test_df)

    feature_names = list(X_train_df.columns)

    return PreprocessedData(
        X_train=X_train_df,
        y_train=y_train,
        s_train=s_train,
        X_val=X_val_df,
        y_val=y_val,
        s_val=s_val,
        X_test=X_test_df,
        y_test=y_test,
        s_test=s_test,
        X_train_scaled=X_train_scaled,
        X_val_scaled=X_val_scaled,
        X_test_scaled=X_test_scaled,
        feature_names=feature_names,
        scaler=scaler,
        protected_col=protected_col,
        target_col=target_col
    )
