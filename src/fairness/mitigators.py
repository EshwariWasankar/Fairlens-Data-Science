import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from fairlearn.reductions import ExponentiatedGradient, DemographicParity, EqualizedOdds
from fairlearn.postprocessing import ThresholdOptimizer

from src.data.preprocessor import PreprocessedData

def compute_reweighing_weights(y: pd.Series, s: pd.Series) -> np.ndarray:
    """Computes pre-processing Reweighing sample weights to eliminate base-rate outcome disparity."""
    df = pd.DataFrame({'y': y, 's': s})
    n = len(df)
    
    # Calculate marginal probabilities
    p_s = df['s'].value_counts(normalize=True).to_dict()
    p_y = df['y'].value_counts(normalize=True).to_dict()
    
    # Calculate joint probabilities
    joint_counts = df.groupby(['s', 'y']).size().to_dict()
    
    weights = []
    for _, row in df.iterrows():
        s_val = row['s']
        y_val = row['y']
        
        p_joint = joint_counts.get((s_val, y_val), 1) / n
        expected_p = p_s[s_val] * p_y[y_val]
        
        w = expected_p / p_joint if p_joint > 0 else 1.0
        weights.append(w)
        
    return np.array(weights)

def train_reweighed_model(
    data: PreprocessedData,
    model_type: str = "random_forest",
    random_state: int = 42
) -> Any:
    """Applies Pre-processing Reweighing mitigation by fitting classifier with group-outcome balanced sample weights."""
    sample_weights = compute_reweighing_weights(data.y_train, data.s_train)

    if model_type == "logistic_regression":
        model = LogisticRegression(C=1.0, max_iter=1000, random_state=random_state)
        model.fit(data.X_train_scaled, data.y_train, sample_weight=sample_weights)
    elif model_type == "random_forest":
        model = RandomForestClassifier(n_estimators=100, max_depth=8, min_samples_split=5, random_state=random_state)
        model.fit(data.X_train, data.y_train, sample_weight=sample_weights)
    elif model_type == "xgboost":
        model = XGBClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.1, 
            random_state=random_state, eval_metric='logloss', use_label_encoder=False
        )
        model.fit(data.X_train, data.y_train, sample_weight=sample_weights)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    return model

def train_exponentiated_gradient_model(
    data: PreprocessedData,
    model_type: str = "logistic_regression",
    constraint: str = "demographic_parity",
    eps: float = 0.01,
    random_state: int = 42
) -> Any:
    """Applies In-processing Exponentiated Gradient mitigation with direct fairness constraints."""
    if model_type == "logistic_regression":
        base_estimator = LogisticRegression(C=1.0, max_iter=1000, random_state=random_state)
        X_tr = data.X_train_scaled
    elif model_type == "random_forest":
        base_estimator = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=random_state)
        X_tr = data.X_train
    elif model_type == "xgboost":
        base_estimator = XGBClassifier(
            n_estimators=50, max_depth=3, learning_rate=0.1, 
            random_state=random_state, eval_metric='logloss', use_label_encoder=False
        )
        X_tr = data.X_train
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    if constraint == "equalized_odds":
        fair_constraint = EqualizedOdds()
    else:
        fair_constraint = DemographicParity()

    mitigated_model = ExponentiatedGradient(
        estimator=base_estimator,
        constraints=fair_constraint,
        eps=eps,
        max_iter=50
    )
    
    mitigated_model.fit(X_tr, data.y_train, sensitive_features=data.s_train)
    return mitigated_model

def train_threshold_optimizer_model(
    unmitigated_model: Any,
    data: PreprocessedData,
    model_type: str = "random_forest",
    constraint: str = "demographic_parity"
) -> Any:
    """Applies Post-processing Threshold Optimizer mitigation to adjust group-wise decision thresholds."""
    if constraint == "equalized_odds":
        predict_method = "predict"
    else:
        predict_method = "auto"

    post_model = ThresholdOptimizer(
        estimator=unmitigated_model,
        constraints="demographic_parity" if constraint == "demographic_parity" else "equalized_odds",
        predict_method=predict_method,
        prefit=True
    )

    X_val = data.X_val_scaled if model_type == "logistic_regression" else data.X_val
    post_model.fit(X_val, data.y_val, sensitive_features=data.s_val)
    return post_model
