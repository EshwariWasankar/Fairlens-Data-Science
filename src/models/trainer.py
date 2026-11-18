import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier
from src.config import MODELS_DIR
from src.data.preprocessor import PreprocessedData

def train_model(
    data: PreprocessedData,
    model_type: str = "random_forest",
    dataset_key: str = "default",
    random_state: int = 42,
    save: bool = True
) -> Tuple[Any, float]:
    """Trains a classifier on preprocessed data and returns (model, mean_cv_f1_score).

    `save=False` for live/interactive runs: the file `models/<dataset>_<model>_baseline.joblib` is
    the benchmark model that the dashboard loads, so a run with a different protected attribute or
    uploaded data must never overwrite it.
    """
    
    if model_type == "logistic_regression":
        model = LogisticRegression(C=1.0, max_iter=1000, random_state=random_state)
        X_train_input = data.X_train_scaled
    elif model_type == "random_forest":
        model = RandomForestClassifier(n_estimators=100, max_depth=8, min_samples_split=5, random_state=random_state)
        X_train_input = data.X_train
    elif model_type == "xgboost":
        model = XGBClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.1, 
            random_state=random_state, eval_metric='logloss', use_label_encoder=False
        )
        X_train_input = data.X_train
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # 5-fold Stratified Cross-Validation on training set
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    cv_scores = cross_val_score(model, X_train_input, data.y_train, cv=skf, scoring='f1_macro')
    mean_cv_f1 = float(np.mean(cv_scores))

    # Fit final model on full training partition
    model.fit(X_train_input, data.y_train)

    if save:
        save_path = MODELS_DIR / f"{dataset_key}_{model_type}_baseline.joblib"
        joblib.dump(model, save_path)

    return model, mean_cv_f1

def model_matches_features(model: Any, feature_names: list) -> bool:
    """True if a (possibly stale) saved model was trained on exactly these feature columns."""
    names = getattr(model, "feature_names_in_", None)
    if names is not None:
        return list(names) == list(feature_names)
    n = getattr(model, "n_features_in_", None)
    return n is None or n == len(feature_names)

def raw_feature_model(model: Any, model_type: str, scaler: Any) -> Any:
    """Returns a model that accepts RAW (unscaled) feature frames.

    Logistic regression is fit on StandardScaler output, so feeding it raw X (as SHAP local
    explanations and DiCE recourse did) silently produces wrong predictions. Wrapping it with the
    already-fitted training scaler fixes that; tree models are returned unchanged.
    """
    if model_type != "logistic_regression":
        return model
    from sklearn.pipeline import Pipeline
    return Pipeline([("scaler", scaler), ("model", model)])

def get_model_input_data(model_type: str, data: PreprocessedData, split: str = "test") -> Tuple[Any, pd.Series, pd.Series]:
    """Retrieves appropriate feature matrix (scaled vs raw) based on model type and requested split."""
    if split == "train":
        X = data.X_train_scaled if model_type == "logistic_regression" else data.X_train
        y = data.y_train
        s = data.s_train
    elif split == "val":
        X = data.X_val_scaled if model_type == "logistic_regression" else data.X_val
        y = data.y_val
        s = data.s_val
    else: # test
        X = data.X_test_scaled if model_type == "logistic_regression" else data.X_test
        y = data.y_test
        s = data.s_test

    return X, y, s
