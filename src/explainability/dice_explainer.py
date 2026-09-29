import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
import dice_ml

def generate_counterfactual_explanation(
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    record: pd.Series,
    actionable_features: Optional[List[str]] = None,
    immutable_features: Optional[List[str]] = None,
    total_CFs: int = 2
) -> Dict[str, Any]:
    """Generates actionable counterfactual recourse suggestions for rejected applicants using DiCE or recourse search."""
    record_df = pd.DataFrame([record])
    
    # Feature list
    all_features = list(X_train.columns)
    
    if actionable_features is None:
        actionable_features = [f for f in all_features if not any(imm in f.lower() for imm in ['age', 'sex', 'race', 'gender', 'foreign'])]

    # Filter actionable features to existing encoded columns
    valid_actionable = [f for f in actionable_features if f in all_features]
    features_to_vary = valid_actionable if len(valid_actionable) > 0 else all_features

    try:
        # Construct DiCE Data & Model wrappers
        d = dice_ml.Data(
            dataframe=pd.concat([X_train, y_train.rename("target")], axis=1),
            outcome_name="target",
            continuous_features=[c for c in X_train.columns if X_train[c].dtype in [np.float64, np.int64]]
        )
        
        m = dice_ml.Model(model=model, backend="sklearn")
        exp = dice_ml.Dice(d, m, method="random")

        # Generate Counterfactuals
        cf = exp.generate_counterfactuals(
            record_df,
            total_CFs=total_CFs,
            desired_class=1,
            features_to_vary=features_to_vary
        )
        
        cf_df = cf.cf_examples_list[0].final_cfs_df
        if cf_df is not None and not cf_df.empty:
            cf_df = cf_df.drop(columns=["target"], errors="ignore")
            # Build changes table
            changes = []
            orig_vals = record_df.iloc[0]
            rec_vals = cf_df.iloc[0]
            
            for feat in all_features:
                o = orig_vals[feat]
                r = rec_vals[feat]
                if abs(float(o) - float(r)) > 1e-4:
                    changes.append({
                        "feature": feat,
                        "original_value": round(float(o), 2),
                        "suggested_value": round(float(r), 2),
                        "delta": round(float(r) - float(o), 2)
                    })

            return {
                "success": True,
                "counterfactual_df": cf_df,
                "actionable_changes": changes,
                "method": "DiCE-ML"
            }

    except Exception as e:
        print(f"DiCE optimization fallback triggered ({e}), running recourse gradient search...")

    # Fallback Actionable Recourse Search Engine
    changes = []
    orig_record = record.copy()
    rec_record = record.copy()
    
    # Identify top numeric actionable features
    num_actionable = [f for f in features_to_vary if np.issubdtype(X_train[f].dtype, np.number)]

    base_prob = float(model.predict_proba(record_df)[:, 1][0]) if hasattr(model, 'predict_proba') else 0.0

    for feat in num_actionable:
        std_val = float(X_train[feat].std())
        if std_val == 0:
            std_val = 1.0
            
        # Test positive and negative steps along std dev
        for step in [-1.0 * std_val, -0.5 * std_val, 0.5 * std_val, 1.0 * std_val]:
            test_rec = rec_record.copy()
            test_rec[feat] += step
            test_df = pd.DataFrame([test_rec])
            
            new_prob = float(model.predict_proba(test_df)[:, 1][0]) if hasattr(model, 'predict_proba') else 0.0
            if new_prob > base_prob:
                rec_record[feat] += step
                changes.append({
                    "feature": feat,
                    "original_value": round(float(orig_record[feat]), 2),
                    "suggested_value": round(float(rec_record[feat]), 2),
                    "delta": round(float(step), 2)
                })
                break

    return {
        "success": len(changes) > 0,
        "counterfactual_df": pd.DataFrame([rec_record]),
        "actionable_changes": changes,
        "method": "Actionable Recourse Search"
    }
