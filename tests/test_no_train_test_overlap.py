import pytest
import pandas as pd
from src.data.loader import load_raw_dataset
from src.data.cleaner import clean_dataset
from src.data.preprocessor import preprocess_dataset

@pytest.mark.parametrize("dataset_key,target_col,protected_col", [
    ("german_credit", "credit_risk", "sex"),
    ("compas", "two_year_recid", "race"),
    ("adult_income", "income", "sex")
])
def test_no_train_test_overlap(dataset_key, target_col, protected_col):
    df_raw = load_raw_dataset(dataset_key)
    df_clean = clean_dataset(df_raw, dataset_key)
    preproc = preprocess_dataset(df_clean, target_col=target_col, protected_col=protected_col)
    
    train_idx = set(preproc.X_train.index)
    val_idx = set(preproc.X_val.index)
    test_idx = set(preproc.X_test.index)
    
    assert len(train_idx.intersection(test_idx)) == 0, f"Found index overlap between train and test splits for {dataset_key}"
    assert len(train_idx.intersection(val_idx)) == 0, f"Found index overlap between train and val splits for {dataset_key}"
    assert len(val_idx.intersection(test_idx)) == 0, f"Found index overlap between val and test splits for {dataset_key}"
