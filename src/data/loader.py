import os
import numpy as np
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

_POSITIVE_TOKENS = {"1", "yes", "y", "true", "t", "good", "approved", "approve", "accept", "accepted",
                    "granted", "positive", "pos", "success", "high", ">50k", "1.0"}

def binarize_target(series: pd.Series, positive: Any = None) -> Tuple[pd.Series, Any]:
    """Maps a two-valued target to 0/1 and returns (mapped series, value treated as 1).

    0/1 and True/False keep their meaning; a value that reads as positive ('yes', 'approved', ...)
    becomes 1; for two numbers (e.g. 1/2) the larger becomes 1; otherwise the second value in
    sorted order. Rows with a missing target stay missing. Pass `positive` to choose explicitly
    (the dashboard asks the user, pre-filled with this guess).
    """
    values = list(pd.unique(series.dropna()))
    if len(values) != 2:
        raise ValueError(f"target must have exactly 2 values, found {len(values)}")
    as_str = {v: str(v).strip().lower() for v in values}
    is_number = lambda v: isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, (bool, np.bool_))
    if positive is not None:
        if positive not in values:
            raise ValueError(f"favorable value {positive!r} is not one of {values}")
    elif set(as_str.values()) in ({"0", "1"}, {"false", "true"}, {"0.0", "1.0"}):
        positive = next(v for v in values if as_str[v] in ("1", "true", "1.0"))
    elif all(is_number(v) for v in values):
        positive = max(values)
    elif sum(s in _POSITIVE_TOKENS for s in as_str.values()) == 1:
        positive = next(v for v in values if as_str[v] in _POSITIVE_TOKENS)
    else:
        positive = sorted(values, key=str)[1]
    mapped = series.map(lambda v: v if pd.isna(v) else int(v == positive))
    return mapped, positive

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
    if pd.api.types.is_numeric_dtype(df[protected_col]) and len(unique_protected) > 10:
        return False, (f"Protected attribute '{protected_col}' looks continuous ({len(unique_protected)} distinct values). "
                       "Bin it into a few groups first (e.g. age bands).")
    if len(unique_protected) > 20:
        return False, f"Protected attribute '{protected_col}' has {len(unique_protected)} groups; FairLens supports up to 20."
    if len(df) < 50:
        return False, f"The dataset has only {len(df)} rows; at least 50 are needed for a train/validation/test split."
        
    return True, "Dataset validated successfully."
