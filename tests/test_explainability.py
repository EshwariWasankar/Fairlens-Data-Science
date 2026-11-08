import pytest
import numpy as np
import pandas as pd
from src.data.loader import load_raw_dataset
from src.data.cleaner import clean_dataset
from src.data.preprocessor import preprocess_dataset
from src.models.trainer import train_model
from src.explainability.shap_explainer import compute_global_shap_importance, compute_local_shap_explanation
from src.explainability.dice_explainer import generate_counterfactual_explanation

def test_shap_explanations_validity():
    df_raw = load_raw_dataset("german_credit")
    df_clean = clean_dataset(df_raw, "german_credit")
    preproc = preprocess_dataset(df_clean, target_col="credit_risk", protected_col="sex")
    model, _ = train_model(preproc, model_type="random_forest")
    
    global_imp = compute_global_shap_importance(model, preproc.X_test)
    assert not global_imp.empty
    assert "feature" in global_imp.columns
    assert "importance" in global_imp.columns
    
    local_exp = compute_local_shap_explanation(model, preproc.X_test, record_index=0)
    assert "base_value" in local_exp
    assert "local_explanation" in local_exp

def test_dice_immutable_attributes_preserved():
    df_raw = load_raw_dataset("german_credit")
    df_clean = clean_dataset(df_raw, "german_credit")
    preproc = preprocess_dataset(df_clean, target_col="credit_risk", protected_col="sex")
    model, _ = train_model(preproc, model_type="random_forest")
    
    record = preproc.X_test.iloc[0]
    immutable_features = ["sex", "age", "age_group", "foreign_worker"]
    actionable_features = ["duration_in_month", "credit_amount", "savings_account"]
    
    res = generate_counterfactual_explanation(
        model, preproc.X_train, preproc.y_train, record,
        actionable_features=actionable_features,
        immutable_features=immutable_features
    )
    
    assert res["success"] is True
    changes = res.get("actionable_changes", [])
    for c in changes:
        feat = c["feature"]
        assert not any(imm in feat for imm in ["sex", "race", "age", "foreign_worker"]), \
            f"DiCE violated immutability constraint! Changed protected/immutable feature '{feat}'"
