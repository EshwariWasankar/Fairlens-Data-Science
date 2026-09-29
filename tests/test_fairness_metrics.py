import pytest
import numpy as np
from src.fairness.metrics import compute_fairness_metrics

def test_fairness_metrics_perfectly_fair():
    # Synthetic perfectly fair predictions (identical selection rates and error rates across groups)
    y_true = np.array([1, 0, 1, 0, 1, 0, 1, 0])
    y_pred = np.array([1, 0, 1, 0, 1, 0, 1, 0])
    s = np.array(['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B'])
    
    m = compute_fairness_metrics(y_true, y_pred, s)
    assert m["demographic_parity_diff"] == pytest.approx(0.0, abs=1e-3)
    assert m["equalized_odds_diff"] == pytest.approx(0.0, abs=1e-3)
    assert m["equal_opportunity_diff"] == pytest.approx(0.0, abs=1e-3)
    assert m["predictive_parity_diff"] == pytest.approx(0.0, abs=1e-3)

def test_fairness_metrics_maximally_unfair():
    # Group A gets 100% positive predictions, Group B gets 0% positive predictions
    y_true = np.array([1, 1, 1, 1, 1, 1, 1, 1])
    y_pred = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    s = np.array(['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B'])
    
    m = compute_fairness_metrics(y_true, y_pred, s)
    assert m["demographic_parity_diff"] == pytest.approx(1.0, abs=1e-3)
    assert m["equal_opportunity_diff"] == pytest.approx(1.0, abs=1e-3)
