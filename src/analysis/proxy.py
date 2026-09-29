import pandas as pd
import numpy as np
from typing import Dict, Any, List
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

def compute_proxy_leakage(
    df: pd.DataFrame,
    protected_col: str,
    target_col: str = "target",
    random_state: int = 42
) -> pd.DataFrame:
    """Quantifies proxy leakage by computing Mutual Information between each non-protected feature and the protected attribute."""
    if protected_col not in df.columns:
        raise ValueError(f"Protected column '{protected_col}' not present in dataframe.")

    df_calc = df.copy()
    
    # Target and protected attribute
    s_raw = df_calc[protected_col]
    
    # Drop target and protected attribute (and any ID columns)
    drop_cols = [protected_col, target_col, 'two_year_recid', 'credit_risk', 'income', 'id', 'name', 'fnlwgt']
    drop_cols.extend([c for c in df_calc.columns if c.endswith('_outlier_flag')])
    
    features_df = df_calc.drop(columns=list(set(drop_cols)), errors='ignore')
    
    # Convert categorical features to numeric codes for MI calculation
    encoded_features = pd.get_dummies(features_df, drop_first=False)
    
    # Convert protected attribute to integer codes if categorical
    if s_raw.dtype == 'object' or isinstance(s_raw.dtype, pd.CategoricalDtype):
        s_encoded = s_raw.astype('category').cat.codes
    else:
        s_encoded = s_raw.astype(int)

    # Sample max 5,000 rows for MI computation if dataset is large for snappy execution
    if len(encoded_features) > 5000:
        sample_idx = encoded_features.sample(n=5000, random_state=random_state).index
        mi_features = encoded_features.loc[sample_idx]
        mi_s = s_encoded.loc[sample_idx]
    else:
        mi_features = encoded_features
        mi_s = s_encoded

    # Compute Mutual Information
    mi_scores = mutual_info_classif(
        mi_features, mi_s, discrete_features='auto', random_state=random_state
    )

    # Calculate correlation where applicable
    correlations = []
    for col in encoded_features.columns:
        try:
            corr = np.corrcoef(encoded_features[col], s_encoded)[0, 1]
            if np.isnan(corr):
                corr = 0.0
        except Exception:
            corr = 0.0
        correlations.append(corr)

    results = pd.DataFrame({
        "feature": encoded_features.columns,
        "mutual_information": np.round(mi_scores, 4),
        "correlation": np.round(correlations, 4)
    })

    # Group one-hot encoded features back to original feature names for cleaner visualization
    orig_features = []
    for col in encoded_features.columns:
        matched = False
        for orig in features_df.columns:
            if col == orig or col.startswith(orig + "_"):
                orig_features.append(orig)
                matched = True
                break
        if not matched:
            orig_features.append(col)
            
    results["base_feature"] = orig_features

    # Aggregate max MI per base feature
    agg_results = results.groupby("base_feature").agg({
        "mutual_information": "max",
        "correlation": lambda x: max(x, key=abs)
    }).reset_index()

    agg_results.rename(columns={"base_feature": "feature"}, inplace=True)
    agg_results = agg_results.sort_values(by="mutual_information", ascending=False).reset_index(drop=True)

    # Categorize Risk Level
    def categorize_risk(mi: float) -> str:
        if mi >= 0.10:
            return "High Proxy Risk"
        elif mi >= 0.03:
            return "Moderate Proxy Risk"
        else:
            return "Low Proxy Risk"

    agg_results["risk_level"] = agg_results["mutual_information"].apply(categorize_risk)
    return agg_results
