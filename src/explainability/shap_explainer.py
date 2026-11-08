import shap
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from sklearn.pipeline import Pipeline


def _unwrap(model: Any, X: pd.DataFrame) -> Tuple[Any, Any]:
    """For a (scaler -> estimator) Pipeline, return the estimator and transformed X."""
    if isinstance(model, Pipeline):
        X_t = model[:-1].transform(X)
        if hasattr(X, "columns"):
            X_t = pd.DataFrame(X_t, columns=X.columns, index=X.index)
        return model[-1], X_t
    return model, X

def compute_global_shap_importance(
    model: Any,
    X_sample: pd.DataFrame,
    feature_names: Optional[List[str]] = None
) -> pd.DataFrame:
    """Computes global feature importances using mean absolute SHAP values."""
    if feature_names is None and hasattr(X_sample, 'columns'):
        feature_names = list(X_sample.columns)
        
    try:
        model, X_sample = _unwrap(model, X_sample)
        # Check model type
        model_name = type(model).__name__.lower()
        if "randomforest" in model_name or "xgb" in model_name or "decisiontree" in model_name:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
        else:
            explainer = shap.Explainer(model, X_sample)
            shap_values = explainer(X_sample).values

        # Format 2D matrix
        if isinstance(shap_values, list):
            # Binary classification list: pick positive class (index 1)
            sv = np.abs(shap_values[1]).mean(axis=0)
        elif len(shap_values.shape) == 3:
            sv = np.abs(shap_values[:, :, 1]).mean(axis=0)
        else:
            sv = np.abs(shap_values).mean(axis=0)

        df_imp = pd.DataFrame({
            "feature": feature_names if feature_names else [f"feature_{i}" for i in range(len(sv))],
            "importance": np.round(sv, 4)
        }).sort_values(by="importance", ascending=False).reset_index(drop=True)

        return df_imp

    except Exception as e:
        print(f"SHAP global calculation fallback triggered: {e}")
        # Fallback to feature_importances_ or coef_ if present
        if hasattr(model, 'feature_importances_'):
            vals = model.feature_importances_
        elif hasattr(model, 'coef_'):
            vals = np.abs(model.coef_[0])
        else:
            vals = np.ones(X_sample.shape[1]) / X_sample.shape[1]

        df_imp = pd.DataFrame({
            "feature": feature_names if feature_names else [f"feature_{i}" for i in range(len(vals))],
            "importance": np.round(vals, 4)
        }).sort_values(by="importance", ascending=False).reset_index(drop=True)
        return df_imp

def compute_local_shap_explanation(
    model: Any,
    X_sample: pd.DataFrame,
    record_index: int,
    feature_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Computes SHAP values and base value for a specific record index."""
    if feature_names is None and hasattr(X_sample, 'columns'):
        feature_names = list(X_sample.columns)
        
    single_record = X_sample.iloc[[record_index]]
    
    try:
        model, X_model = _unwrap(model, X_sample)
        model_record = X_model.iloc[[record_index]]
        model_name = type(model).__name__.lower()
        if "randomforest" in model_name or "xgb" in model_name or "decisiontree" in model_name:
            explainer = shap.TreeExplainer(model)
            shap_vals = explainer.shap_values(model_record)
            base_val = explainer.expected_value
        else:
            explainer = shap.Explainer(model, X_model)
            exp_res = explainer(model_record)
            shap_vals = exp_res.values
            base_val = exp_res.base_values

        if isinstance(shap_vals, list):
            sv = shap_vals[1][0]
            bv = base_val[1] if isinstance(base_val, (list, np.ndarray)) else base_val
        elif len(shap_vals.shape) == 3:
            sv = shap_vals[0, :, 1]
            bv = base_val[0, 1] if len(base_val.shape) > 1 else base_val[0]
        else:
            sv = shap_vals[0]
            bv = base_val[0] if isinstance(base_val, (list, np.ndarray)) else base_val

        df_local = pd.DataFrame({
            "feature": feature_names,
            "feature_value": single_record.iloc[0].values,
            "shap_value": np.round(sv, 4)
        }).sort_values(by="shap_value", key=abs, ascending=False).reset_index(drop=True)

        return {
            "base_value": float(bv),
            "record_index": record_index,
            "local_explanation": df_local
        }

    except Exception as e:
        print(f"SHAP local explanation fallback triggered: {e}")
        df_local = pd.DataFrame({
            "feature": feature_names,
            "feature_value": single_record.iloc[0].values,
            "shap_value": 0.0
        })
        return {
            "base_value": 0.5,
            "record_index": record_index,
            "local_explanation": df_local
        }
