import pytest
import numpy as np
import pandas as pd
from src.data.loader import load_raw_dataset
from src.data.cleaner import clean_dataset
from src.data.preprocessor import preprocess_dataset
from src.models.trainer import train_model
from src.fairness.mitigators import train_reweighed_model, train_exponentiated_gradient_model, train_threshold_optimizer_model

def test_mitigation_techniques_execution():
    df_raw = load_raw_dataset("german_credit")
    df_clean = clean_dataset(df_raw, "german_credit")
    preproc = preprocess_dataset(df_clean, target_col="credit_risk", protected_col="sex")
    
    # Pre-processing: Reweighing
    rew_model = train_reweighed_model(preproc, model_type="random_forest")
    assert rew_model is not None
    preds_rew = rew_model.predict(preproc.X_test)
    assert len(preds_rew) == len(preproc.X_test)
    
    # In-processing: Exponentiated Gradient
    eg_model = train_exponentiated_gradient_model(preproc, model_type="logistic_regression")
    assert eg_model is not None
    preds_eg = eg_model.predict(preproc.X_test_scaled)
    assert len(preds_eg) == len(preproc.X_test_scaled)

    # Post-processing: Threshold Optimizer
    base_model, _ = train_model(preproc, model_type="random_forest")
    to_model = train_threshold_optimizer_model(base_model, preproc, model_type="random_forest")
    assert to_model is not None
    preds_to = to_model.predict(preproc.X_test, sensitive_features=preproc.s_test)
    assert len(preds_to) == len(preproc.X_test)
