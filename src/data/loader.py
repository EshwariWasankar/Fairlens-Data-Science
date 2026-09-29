import os
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
from src.config import RAW_DATA_DIR, DATASET_CONFIGS

def load_raw_dataset(dataset_key: str) -> pd.DataFrame:
    """Loads a bundled dataset by key from raw data directory."""
    if dataset_key not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset key: {dataset_key}. Expected one of {list(DATASET_CONFIGS.keys())}")
    
    cfg = DATASET_CONFIGS[dataset_key]
    file_path = RAW_DATA_DIR / cfg["file_name"]
    
    if not file_path.exists():
        from src.data.downloader import download_all_datasets
        download_all_datasets()

    df = pd.read_csv(file_path)
    return df

def validate_custom_csv(df: pd.DataFrame, target_col: str, protected_col: str) -> Tuple[bool, str]:
    """Validates an uploaded custom dataset for compatibility with FairLens."""
    if df.empty:
        return False, "Uploaded file is empty."
    
    if target_col not in df.columns:
        return False, f"Target column '{target_col}' not found in uploaded CSV."
        
    if protected_col not in df.columns:
        return False, f"Protected attribute column '{protected_col}' not found in uploaded CSV."
        
    # Check binary target
    unique_targets = df[target_col].dropna().unique()
    if len(unique_targets) != 2:
        return False, f"Target column '{target_col}' must be binary (found {len(unique_targets)} unique values: {unique_targets})."
        
    # Check protected attribute unique values
    unique_protected = df[protected_col].dropna().unique()
    if len(unique_protected) < 2:
        return False, f"Protected attribute '{protected_col}' must have at least 2 distinct groups."
        
    return True, "Dataset validated successfully."
