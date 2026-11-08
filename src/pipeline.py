import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from src.config import DATASET_CONFIGS
from src.data.loader import load_raw_dataset
from src.data.cleaner import clean_dataset
from src.data.preprocessor import preprocess_dataset, PreprocessedData
from src.analysis.eda import get_dataset_kpis, get_group_disparity_table, get_numeric_correlation_matrix
from src.analysis.proxy import compute_proxy_leakage
from src.models.trainer import train_model, get_model_input_data, raw_feature_model
from src.models.evaluator import evaluate_predictive_performance
from src.fairness.metrics import compute_fairness_metrics
from src.fairness.bootstrap import compute_bootstrap_confidence_intervals, compute_paired_bootstrap_difference
from src.fairness.mitigators import train_reweighed_model, train_exponentiated_gradient_model, train_threshold_optimizer_model
from src.explainability.shap_explainer import compute_global_shap_importance, compute_local_shap_explanation
from src.explainability.dice_explainer import generate_counterfactual_explanation
from src.explainability.insights import generate_plain_language_summary
from src.db.db_manager import log_dataset_metadata, log_experiment_run, log_proxy_analysis_results

def run_fairlens_pipeline(
    dataset_key: str,
    protected_attribute: Optional[str] = None,
    model_type: str = "random_forest",
    custom_df: Optional[pd.DataFrame] = None,
    custom_target_col: Optional[str] = None,
    custom_protected_col: Optional[str] = None
) -> Dict[str, Any]:
    """Runs the end-to-end FairLens audit, proxy, mitigation, and explainability pipeline."""
    
    # 1. Load Data
    if custom_df is not None and not custom_df.empty:
        dataset_name = "Custom Uploaded Dataset"
        df_raw = custom_df
        target_col = custom_target_col if custom_target_col else "target"
        protected_col = custom_protected_col if custom_protected_col else "sex"
        actionable_cols = [c for c in df_raw.columns if c not in [target_col, protected_col]]
        immutable_cols = [protected_col]
    else:
        cfg = DATASET_CONFIGS[dataset_key]
        dataset_name = cfg["name"]
        df_raw = load_raw_dataset(dataset_key)
        target_col = cfg["target_column"]
        protected_col = protected_attribute if protected_attribute else cfg["default_protected_attribute"]
        actionable_cols = cfg.get("actionable_features", [])
        immutable_cols = cfg.get("immutable_features", [])

    # Register Metadata in SQLite
    log_dataset_metadata(dataset_key, dataset_name, target_col, [protected_col], len(df_raw))

    # 2. Clean Data
    df_clean = clean_dataset(df_raw, dataset_key)
    
    # 3. Preprocess Data
    data: PreprocessedData = preprocess_dataset(
        df_clean, target_col=target_col, protected_col=protected_col
    )

    # 4. Exploratory Data Analysis
    eda_target = 'target' if 'target' in df_clean.columns else target_col
    fav_value = int(DATASET_CONFIGS.get(dataset_key, {}).get("favorable_target_value", 1))
    kpis = get_dataset_kpis(df_clean, target_col=eda_target, protected_col=protected_col, favorable_value=fav_value)
    disparity_table = get_group_disparity_table(df_clean, target_col=eda_target, protected_col=protected_col, favorable_value=fav_value)
    corr_matrix = get_numeric_correlation_matrix(df_clean)

    # 5. Proxy Feature Leakage Analysis
    proxy_df = compute_proxy_leakage(df_clean, protected_col=protected_col, target_col=eda_target)
    log_proxy_analysis_results(dataset_key, protected_col, proxy_df)

    # Extract privileged and unprivileged demographic categories from config if present
    priv_group = None
    unread_group = None
    if dataset_key in DATASET_CONFIGS:
        prot_cfg = DATASET_CONFIGS[dataset_key].get("protected_attributes", {}).get(protected_col, {})
        priv_group = prot_cfg.get("privileged")
        unread_group = prot_cfg.get("unprivileged")

    # 6. Train Baseline Model (System Under Audit)
    baseline_model, cv_f1 = train_model(data, model_type=model_type, dataset_key=dataset_key)
    X_test, y_test, s_test = get_model_input_data(model_type, data, split="test")

    # Predictive Performance
    baseline_pred_metrics = evaluate_predictive_performance(baseline_model, X_test, y_test)
    baseline_pred_metrics["cv_f1_score"] = cv_f1

    # Baseline Predictions
    y_pred_base = baseline_model.predict(X_test)

    # 7. Baseline Fairness Audit with Bootstrap 95% CIs
    baseline_fairness = compute_bootstrap_confidence_intervals(
        y_true=y_test.values,
        y_pred=y_pred_base,
        s=s_test.values,
        n_bootstraps=500,
        privileged_group=priv_group,
        unprivileged_group=unread_group
    )
    baseline_fairness["f1_score"] = baseline_pred_metrics["f1_score"]
    baseline_fairness["roc_auc"] = baseline_pred_metrics["roc_auc"]
    baseline_fairness["accuracy"] = baseline_pred_metrics["accuracy"]

    log_experiment_run(dataset_key, model_type, "baseline", baseline_pred_metrics, baseline_fairness)

    # 8. Mitigation Algorithms Execution
    mitigations = {}
    
    # a) Pre-processing: Reweighing
    try:
        rew_model = train_reweighed_model(data, model_type=model_type)
        y_pred_rew = rew_model.predict(X_test)
        rew_pred_metrics = evaluate_predictive_performance(rew_model, X_test, y_test)
        rew_fairness = compute_bootstrap_confidence_intervals(
            y_true=y_test.values, y_pred=y_pred_rew, s=s_test.values, n_bootstraps=500,
            privileged_group=priv_group, unprivileged_group=unread_group
        )
        rew_sig_all = compute_paired_bootstrap_difference(
            y_true=y_test.values, y_pred_base=y_pred_base, y_pred_mit=y_pred_rew, s=s_test.values,
            n_bootstraps=500, privileged_group=priv_group, unprivileged_group=unread_group
        )
        mitigations["Reweighing"] = {
            "model": rew_model,
            "metrics": rew_fairness,
            "pred_metrics": rew_pred_metrics,
            "significance": rew_sig_all["demographic_parity_diff"],
            "all_significance": rew_sig_all,
            "f1_score": rew_pred_metrics["f1_score"],
            "accuracy": rew_pred_metrics["accuracy"]
        }
        log_experiment_run(dataset_key, model_type, "Reweighing", rew_pred_metrics, rew_fairness, rew_sig_all["demographic_parity_diff"])
    except Exception as e:
        print(f"Reweighing mitigation error: {e}")

    # b) In-processing: Exponentiated Gradient
    try:
        # Use selected optimal epsilon from verified benchmark selection
        # Epsilon chosen on the validation set by master_eval.py (single source of truth)
        selected_eps = 0.01
        try:
            import json
            from src.config import DATA_DIR
            with open(DATA_DIR / "verified_benchmark_results.json", encoding="utf-8") as fh:
                selected_eps = float(json.load(fh).get("selected_epsilons", {}).get(dataset_key, 0.01))
        except Exception:
            pass

        eg_model = train_exponentiated_gradient_model(
            data, model_type=model_type, constraint="demographic_parity", eps=selected_eps, dataset_key=dataset_key
        )
        X_test_in = data.X_test_scaled if model_type == "logistic_regression" else data.X_test
        y_pred_eg = eg_model.predict(X_test_in)
        eg_pred_metrics = evaluate_predictive_performance(eg_model, X_test_in, y_test)
        eg_fairness = compute_bootstrap_confidence_intervals(
            y_true=y_test.values, y_pred=y_pred_eg, s=s_test.values, n_bootstraps=500,
            privileged_group=priv_group, unprivileged_group=unread_group
        )
        eg_sig_all = compute_paired_bootstrap_difference(
            y_true=y_test.values, y_pred_base=y_pred_base, y_pred_mit=y_pred_eg, s=s_test.values,
            n_bootstraps=500, privileged_group=priv_group, unprivileged_group=unread_group
        )
        mitigations["Exponentiated Gradient"] = {
            "model": eg_model,
            "metrics": eg_fairness,
            "pred_metrics": eg_pred_metrics,
            "significance": eg_sig_all["demographic_parity_diff"],
            "all_significance": eg_sig_all,
            "f1_score": eg_pred_metrics["f1_score"],
            "accuracy": eg_pred_metrics["accuracy"]
        }
        log_experiment_run(dataset_key, model_type, "Exponentiated Gradient", eg_pred_metrics, eg_fairness, eg_sig_all["demographic_parity_diff"])
    except Exception as e:
        print(f"Exponentiated Gradient mitigation error: {e}")

    # c) Post-processing: Threshold Optimizer
    try:
        to_model = train_threshold_optimizer_model(baseline_model, data, model_type=model_type)
        X_test_in = data.X_test_scaled if model_type == "logistic_regression" else data.X_test
        y_pred_to = to_model.predict(X_test_in, sensitive_features=s_test, random_state=42)
        
        class ThresholdWrapper:
            def __init__(self, tm, sf):
                self.tm = tm
                self.sf = sf
            def predict(self, X):
                return self.tm.predict(X, sensitive_features=self.sf, random_state=42)
            def predict_proba(self, X):
                return np.asarray(self.tm._pmf_predict(X, sensitive_features=self.sf), dtype=np.float64)

        to_wrapper = ThresholdWrapper(to_model, s_test)
        to_pred_metrics = evaluate_predictive_performance(to_wrapper, X_test_in, y_test)
        to_fairness = compute_bootstrap_confidence_intervals(
            y_true=y_test.values, y_pred=y_pred_to, s=s_test.values, n_bootstraps=500,
            privileged_group=priv_group, unprivileged_group=unread_group
        )
        to_sig_all = compute_paired_bootstrap_difference(
            y_true=y_test.values, y_pred_base=y_pred_base, y_pred_mit=y_pred_to, s=s_test.values,
            n_bootstraps=500, privileged_group=priv_group, unprivileged_group=unread_group
        )
        mitigations["Threshold Optimizer"] = {
            "model": to_model,
            "metrics": to_fairness,
            "pred_metrics": to_pred_metrics,
            "significance": to_sig_all["demographic_parity_diff"],
            "all_significance": to_sig_all,
            "f1_score": to_pred_metrics["f1_score"],
            "accuracy": to_pred_metrics["accuracy"]
        }
        log_experiment_run(dataset_key, model_type, "Threshold Optimizer", to_pred_metrics, to_fairness, to_sig_all["demographic_parity_diff"])
    except Exception as e:
        print(f"Threshold Optimizer mitigation error: {e}")

    # 9. Global Explainability (SHAP)
    shap_importance = compute_global_shap_importance(raw_feature_model(baseline_model, model_type, data.scaler), data.X_test, data.feature_names)

    # 10. Plain Language Insights
    primary_mitigation = mitigations.get("Reweighing", list(mitigations.values())[0] if mitigations else {})
    insights = generate_plain_language_summary(
        dataset_name=dataset_name,
        protected_attribute=protected_col,
        kpis=kpis,
        proxy_df=proxy_df,
        baseline_fairness=baseline_fairness,
        mitigation_res=primary_mitigation,
        technique_name="Reweighing"
    )

    return {
        "dataset_key": dataset_key,
        "dataset_name": dataset_name,
        "target_col": target_col,
        "protected_col": protected_col,
        "actionable_cols": actionable_cols,
        "immutable_cols": immutable_cols,
        "raw_df": df_raw,
        "clean_df": df_clean,
        "preprocessed_data": data,
        "kpis": kpis,
        "disparity_table": disparity_table,
        "corr_matrix": corr_matrix,
        "proxy_df": proxy_df,
        "baseline_model": baseline_model,
        "baseline_pred_metrics": baseline_pred_metrics,
        "baseline_fairness": baseline_fairness,
        "mitigations": mitigations,
        "shap_importance": shap_importance,
        "insights": insights
    }
