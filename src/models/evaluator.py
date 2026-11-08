import numpy as np
import pandas as pd
from typing import Dict, Any
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, balanced_accuracy_score, confusion_matrix

def evaluate_predictive_performance(model: Any, X: Any, y_true: pd.Series) -> Dict[str, Any]:
    """Computes predictive performance metrics, per-class breakdown, balanced accuracy, and confusion matrix for a trained classifier."""
    y_pred = model.predict(X)
    
    # Probabilities for ROC-AUC
    try:
        y_prob = model.predict_proba(X)[:, 1]
        roc_auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        roc_auc = 0.5

    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    
    # Class 1 (Favorable/Positive) metrics
    prec_1 = float(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
    rec_1 = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
    f1_1 = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))
    
    # Class 0 (Unfavorable/Negative) metrics
    prec_0 = float(precision_score(y_true, y_pred, pos_label=0, zero_division=0))
    rec_0 = float(recall_score(y_true, y_pred, pos_label=0, zero_division=0))
    f1_0 = float(f1_score(y_true, y_pred, pos_label=0, zero_division=0))

    macro_f1 = float(f1_score(y_true, y_pred, average='macro', zero_division=0))

    # Confusion matrix: [[TN, FP], [FN, TP]]
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    return {
        "accuracy": np.round(acc, 4),
        "balanced_accuracy": np.round(bal_acc, 4),
        "precision": np.round(prec_1, 4),
        "recall": np.round(rec_1, 4),
        "f1_score": np.round(f1_1, 4),
        "precision_class_0": np.round(prec_0, 4),
        "recall_class_0": np.round(rec_0, 4),
        "f1_class_0": np.round(f1_0, 4),
        "precision_class_1": np.round(prec_1, 4),
        "recall_class_1": np.round(rec_1, 4),
        "f1_class_1": np.round(f1_1, 4),
        "macro_f1": np.round(macro_f1, 4),
        "roc_auc": np.round(roc_auc, 4),
        "confusion_matrix": {"TN": tn, "FP": fp, "FN": fn, "TP": tp}
    }
