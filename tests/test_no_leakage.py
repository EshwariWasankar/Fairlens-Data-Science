import pytest
import pandas as pd
from src.data.loader import load_raw_dataset
from src.data.cleaner import clean_dataset
from src.data.preprocessor import preprocess_dataset

LEAKAGE_COLUMNS = [
    'target', 'two_year_recid', 'credit_risk', 'income', 'class', 'Default', 'default',
    'decile_score', 'decile_score.1', 'score_text', 'v_decile_score', 'v_score_text',
    'is_recid', 'violent_recid', 'is_violent_recid', 'r_case_number', 'fnlwgt'
]

@pytest.mark.parametrize("dataset_key,target_col,protected_col", [
    ("german_credit", "credit_risk", "sex"),
    ("compas", "two_year_recid", "race"),
    ("adult_income", "income", "sex")
])
def test_no_leakage_in_features(dataset_key, target_col, protected_col):
    df_raw = load_raw_dataset(dataset_key)
    df_clean = clean_dataset(df_raw, dataset_key)
    preproc = preprocess_dataset(df_clean, target_col=target_col, protected_col=protected_col)
    
    train_cols = preproc.X_train.columns
    test_cols = preproc.X_test.columns
    
    for forbidden in LEAKAGE_COLUMNS:
        assert forbidden not in train_cols, f"Data leakage detected! '{forbidden}' found in X_train for {dataset_key}"
        assert forbidden not in test_cols, f"Data leakage detected! '{forbidden}' found in X_test for {dataset_key}"

def test_toggling_protected_attribute_drops_derived_columns():
    """Asserts that selecting any protected attribute properly excludes it and all derived proxy columns from feature matrix X."""
    df_raw = load_raw_dataset("german_credit")
    df_clean = clean_dataset(df_raw, "german_credit")
    
    # Toggle to sex
    p_sex = preprocess_dataset(df_clean, target_col="credit_risk", protected_col="sex")
    assert "sex" not in p_sex.X_train.columns
    assert "personal_status_sex" not in p_sex.X_train.columns

    # Toggle to age_group
    p_age = preprocess_dataset(df_clean, target_col="credit_risk", protected_col="age_group")
    assert "age_group" not in p_age.X_train.columns

