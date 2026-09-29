import sys
import os
import json
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Force UTF-8 output encoding for Windows terminal compatibility
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from src.config import DATASET_CONFIGS, DATA_DIR
from src.data.loader import load_raw_dataset
from src.data.cleaner import clean_dataset
from src.data.preprocessor import preprocess_dataset
from src.models.trainer import train_model, get_model_input_data
from src.models.evaluator import evaluate_predictive_performance
from src.fairness.metrics import compute_fairness_metrics
from src.fairness.bootstrap import compute_bootstrap_confidence_intervals, compute_paired_bootstrap_difference
from src.fairness.mitigators import train_reweighed_model, train_exponentiated_gradient_model, train_threshold_optimizer_model
from src.explainability.shap_explainer import compute_global_shap_importance

datasets = ["german_credit", "compas", "adult_income"]
models = ["logistic_regression", "random_forest", "xgboost"]
seeds = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
eps_values = [0.005, 0.01, 0.02, 0.05, 0.1]

print("="*80)
print("1. DEDUPLICATION AUDIT (RAW ROW COUNTS BEFORE AND AFTER EXACT ROW CLEANING)")
print("="*80)

dedup_info = {}
for d_key in datasets:
    raw_df = load_raw_dataset(d_key)
    raw_len = len(raw_df)
    clean_df = clean_dataset(raw_df, d_key)
    clean_len = len(clean_df)
    exact_duplicates_removed = raw_len - clean_len
    dedup_info[d_key] = {
        "raw_rows": raw_len,
        "clean_rows": clean_len,
        "exact_duplicates_removed": exact_duplicates_removed
    }
    print(f"[{d_key.upper()}] Raw Rows: {raw_len:,} | Post Exact-Row Clean Rows: {clean_len:,} | Duplicates Removed: {exact_duplicates_removed}")

print("\n" + "="*80)
print("2. BASELINE MODEL SELECTION (RULE: BEST MEAN VALIDATION F1 OVER 10 SEEDS)")
print("="*80)

val_f1_summary = {}
selected_baseline_models = {}

for d_key in datasets:
    cfg = DATASET_CONFIGS[d_key]
    raw_df = load_raw_dataset(d_key)
    clean_df = clean_dataset(raw_df, d_key)
    prot_col = cfg["default_protected_attribute"]
    target_col = cfg["target_column"]
    
    val_f1_summary[d_key] = {}
    
    for m_type in models:
        val_f1_scores = []
        for s in seeds:
            data = preprocess_dataset(clean_df, target_col=target_col, protected_col=prot_col, random_state=s)
            model, _ = train_model(data, model_type=m_type, dataset_key=d_key, random_state=s)
            X_val, y_val, _ = get_model_input_data(m_type, data, split="val")
            val_metrics = evaluate_predictive_performance(model, X_val, y_val)
            val_f1_scores.append(val_metrics["f1_score"])
            
        mean_val_f1 = float(np.mean(val_f1_scores))
        std_val_f1 = float(np.std(val_f1_scores))
        val_f1_summary[d_key][m_type] = {"mean": round(mean_val_f1, 4), "std": round(std_val_f1, 4)}
        print(f"  {d_key:15s} | {m_type:20s} | Mean Val F1 (10 seeds): {mean_val_f1:.4f} ± {std_val_f1:.4f}")

    # Select model with highest mean validation F1
    best_m = max(val_f1_summary[d_key].keys(), key=lambda k: val_f1_summary[d_key][k]["mean"])
    selected_baseline_models[d_key] = best_m
    print(f"--> SELECTION RULE RESULT [{d_key.upper()}]: '{best_m}' selected as baseline classifier (Best Val F1 = {val_f1_summary[d_key][best_m]['mean']:.4f})\n")

print("\n" + "="*80)
print("3. EXPONENTIATED GRADIENT EPSILON SWEEP (DemographicParity & EqualizedOdds)")
print("="*80)

eg_sweep_results = {}

for d_key in datasets:
    cfg = DATASET_CONFIGS[d_key]
    raw_df = load_raw_dataset(d_key)
    clean_df = clean_dataset(raw_df, d_key)
    prot_col = cfg["default_protected_attribute"]
    target_col = cfg["target_column"]
    prot_cfg = cfg.get("protected_attributes", {}).get(prot_col, {})
    priv_g = prot_cfg.get("privileged")
    unread_g = prot_cfg.get("unprivileged")
    
    m_type = selected_baseline_models[d_key]
    data = preprocess_dataset(clean_df, target_col=target_col, protected_col=prot_col, random_state=42)
    eg_sweep_results[d_key] = {}

    print(f"\n--- Exponentiated Gradient Epsilon Sweep for {d_key.upper()} (Baseline: {m_type}) ---")

    for constraint in ["demographic_parity", "equalized_odds"]:
        eg_sweep_results[d_key][constraint] = []
        for eps in eps_values:
            eg_model = train_exponentiated_gradient_model(data, model_type=m_type, constraint=constraint, random_state=42)
            if hasattr(eg_model, 'eps'):
                eg_model.eps = eps
                
            X_test_in = data.X_test_scaled if m_type == "logistic_regression" else data.X_test
            y_pred_eg = eg_model.predict(X_test_in)
            
            p_met = evaluate_predictive_performance(eg_model, X_test_in, data.y_test)
            f_met = compute_fairness_metrics(data.y_test.values, y_pred_eg, data.s_test.values, privileged_group=priv_g, unprivileged_group=unread_g)
            
            row_res = {
                "eps": eps,
                "accuracy": f"{p_met['accuracy']*100:.2f}%",
                "f1": f"{p_met['f1_score']:.4f}",
                "dp_diff": f"{f_met['demographic_parity_diff']:.4f}",
                "eo_diff": f"{f_met['equalized_odds_diff']:.4f}",
                "eopp_diff": f"{f_met['equal_opportunity_diff']:.4f}",
                "pp_diff": f"{f_met['predictive_parity_diff']:.4f}"
            }
            eg_sweep_results[d_key][constraint].append(row_res)
            print(f"Constraint: {constraint:18s} | eps={eps:<5} | Acc: {p_met['accuracy']*100:.2f}% | F1: {p_met['f1_score']:.4f} | DP: {f_met['demographic_parity_diff']:.4f} | EO: {f_met['equalized_odds_diff']:.4f} | EOpp: {f_met['equal_opportunity_diff']:.4f} | PP: {f_met['predictive_parity_diff']:.4f}")

print("\n" + "="*80)
print("4. FULL 10-SEED MITIGATION EVALUATION & PAIRED BOOTSTRAP CIs OF DIFFERENCE")
print("="*80)

verified_results = {
    "dedup_info": dedup_info,
    "val_f1_summary": val_f1_summary,
    "selected_baseline_models": selected_baseline_models,
    "eg_sweep_results": eg_sweep_results,
    "experiments": {}
}

for d_key in datasets:
    cfg = DATASET_CONFIGS[d_key]
    raw_df = load_raw_dataset(d_key)
    clean_df = clean_dataset(raw_df, d_key)
    prot_col = cfg["default_protected_attribute"]
    target_col = cfg["target_column"]
    prot_cfg = cfg.get("protected_attributes", {}).get(prot_col, {})
    priv_g = prot_cfg.get("privileged")
    unread_g = prot_cfg.get("unprivileged")
    
    m_type = selected_baseline_models[d_key]
    print(f"\n==========================================================")
    print(f"DATASET: {d_key.upper()} (Selected Baseline Model: {m_type})")
    print(f"==========================================================")

    # Storage across 10 seeds
    variant_results = {
        "Baseline": {"f1": [], "acc": [], "dp": [], "eo": [], "eopp": [], "pp": []},
        "Reweighing": {"f1": [], "acc": [], "dp": [], "eo": [], "eopp": [], "pp": []},
        "Exponentiated Gradient": {"f1": [], "acc": [], "dp": [], "eo": [], "eopp": [], "pp": []},
        "Threshold Optimizer": {"f1": [], "acc": [], "dp": [], "eo": [], "eopp": [], "pp": []}
    }

    # Store seed 42 predictions for paired bootstrap CIs
    data_s42 = preprocess_dataset(clean_df, target_col=target_col, protected_col=prot_col, random_state=42)
    base_model_s42, _ = train_model(data_s42, model_type=m_type, dataset_key=d_key, random_state=42)
    X_test_s42, y_test_s42, s_test_s42 = get_model_input_data(m_type, data_s42, split="test")
    y_pred_base_s42 = base_model_s42.predict(X_test_s42)

    rew_model_s42 = train_reweighed_model(data_s42, model_type=m_type, random_state=42)
    y_pred_rew_s42 = rew_model_s42.predict(X_test_s42)

    eg_model_s42 = train_exponentiated_gradient_model(data_s42, model_type=m_type, constraint="demographic_parity", random_state=42)
    X_test_in_s42 = data_s42.X_test_scaled if m_type == "logistic_regression" else data_s42.X_test
    y_pred_eg_s42 = eg_model_s42.predict(X_test_in_s42)

    to_model_s42 = train_threshold_optimizer_model(base_model_s42, data_s42, model_type=m_type)
    y_pred_to_s42 = to_model_s42.predict(X_test_in_s42, sensitive_features=s_test_s42)

    for s in seeds:
        data = preprocess_dataset(clean_df, target_col=target_col, protected_col=prot_col, random_state=s)
        base_m, _ = train_model(data, model_type=m_type, dataset_key=d_key, random_state=s)
        X_test, y_test, s_test = get_model_input_data(m_type, data, split="test")
        y_pred_b = base_m.predict(X_test)

        bp = evaluate_predictive_performance(base_m, X_test, y_test)
        bf = compute_fairness_metrics(y_test.values, y_pred_b, s_test.values, privileged_group=priv_g, unprivileged_group=unread_g)

        variant_results["Baseline"]["f1"].append(bp["f1_score"])
        variant_results["Baseline"]["acc"].append(bp["accuracy"])
        variant_results["Baseline"]["dp"].append(bf["demographic_parity_diff"])
        variant_results["Baseline"]["eo"].append(bf["equalized_odds_diff"])
        variant_results["Baseline"]["eopp"].append(bf["equal_opportunity_diff"])
        variant_results["Baseline"]["pp"].append(bf["predictive_parity_diff"])

        # Reweighing
        rew_m = train_reweighed_model(data, model_type=m_type, random_state=s)
        y_pred_r = rew_m.predict(X_test)
        rp = evaluate_predictive_performance(rew_m, X_test, y_test)
        rf = compute_fairness_metrics(y_test.values, y_pred_r, s_test.values, privileged_group=priv_g, unprivileged_group=unread_g)
        variant_results["Reweighing"]["f1"].append(rp["f1_score"])
        variant_results["Reweighing"]["acc"].append(rp["accuracy"])
        variant_results["Reweighing"]["dp"].append(rf["demographic_parity_diff"])
        variant_results["Reweighing"]["eo"].append(rf["equalized_odds_diff"])
        variant_results["Reweighing"]["eopp"].append(rf["equal_opportunity_diff"])
        variant_results["Reweighing"]["pp"].append(rf["predictive_parity_diff"])

        # Exponentiated Gradient
        eg_m = train_exponentiated_gradient_model(data, model_type=m_type, constraint="demographic_parity", random_state=s)
        X_test_in = data.X_test_scaled if m_type == "logistic_regression" else data.X_test
        y_pred_e = eg_m.predict(X_test_in)
        ep = evaluate_predictive_performance(eg_m, X_test_in, y_test)
        ef = compute_fairness_metrics(y_test.values, y_pred_e, s_test.values, privileged_group=priv_g, unprivileged_group=unread_g)
        variant_results["Exponentiated Gradient"]["f1"].append(ep["f1_score"])
        variant_results["Exponentiated Gradient"]["acc"].append(ep["accuracy"])
        variant_results["Exponentiated Gradient"]["dp"].append(ef["demographic_parity_diff"])
        variant_results["Exponentiated Gradient"]["eo"].append(ef["equalized_odds_diff"])
        variant_results["Exponentiated Gradient"]["eopp"].append(ef["equal_opportunity_diff"])
        variant_results["Exponentiated Gradient"]["pp"].append(ef["predictive_parity_diff"])

        # Threshold Optimizer
        to_m = train_threshold_optimizer_model(base_m, data, model_type=m_type)
        y_pred_t = to_m.predict(X_test_in, sensitive_features=s_test)
        class TW:
            def __init__(self, tm, sf): self.tm = tm; self.sf = sf
            def predict(self, X): return self.tm.predict(X, sensitive_features=self.sf)
            def predict_proba(self, X): p = self.predict(X); return np.column_stack([1-p, p])
        tp = evaluate_predictive_performance(TW(to_m, s_test), X_test_in, y_test)
        tf = compute_fairness_metrics(y_test.values, y_pred_t, s_test.values, privileged_group=priv_g, unprivileged_group=unread_g)
        variant_results["Threshold Optimizer"]["f1"].append(tp["f1_score"])
        variant_results["Threshold Optimizer"]["acc"].append(tp["accuracy"])
        variant_results["Threshold Optimizer"]["dp"].append(tf["demographic_parity_diff"])
        variant_results["Threshold Optimizer"]["eo"].append(tf["equalized_odds_diff"])
        variant_results["Threshold Optimizer"]["eopp"].append(tf["equal_opportunity_diff"])
        variant_results["Threshold Optimizer"]["pp"].append(tf["predictive_parity_diff"])

    # Compute paired bootstrap differences (Delta = baseline - mitigated) for Seed 42 partition
    paired_ci_rew = compute_paired_bootstrap_difference(y_test_s42.values, y_pred_base_s42, y_pred_rew_s42, s_test_s42.values, privileged_group=priv_g, unprivileged_group=unread_g)
    paired_ci_eg = compute_paired_bootstrap_difference(y_test_s42.values, y_pred_base_s42, y_pred_eg_s42, s_test_s42.values, privileged_group=priv_g, unprivileged_group=unread_g)
    paired_ci_to = compute_paired_bootstrap_difference(y_test_s42.values, y_pred_base_s42, y_pred_to_s42, s_test_s42.values, privileged_group=priv_g, unprivileged_group=unread_g)

    summary_rows = []
    for var in ["Baseline", "Reweighing", "Exponentiated Gradient", "Threshold Optimizer"]:
        res = variant_results[var]
        p_ci = None if var == "Baseline" else paired_ci_rew if var == "Reweighing" else paired_ci_eg if var == "Exponentiated Gradient" else paired_ci_to
        
        dp_delta = p_ci["demographic_parity_diff"]["mean_delta"] if p_ci else "N/A"
        dp_ci_str = f"[{p_ci['demographic_parity_diff']['ci_low']:.4f}, {p_ci['demographic_parity_diff']['ci_high']:.4f}]" if p_ci else "N/A"
        is_sig = str(p_ci["demographic_parity_diff"]["is_significant"]) if p_ci else "N/A"

        summary_rows.append({
            "Variant": var,
            "F1 (10-seed mean)": f"{np.mean(res['f1']):.4f} ± {np.std(res['f1']):.4f}",
            "Accuracy (10-seed mean)": f"{np.mean(res['acc'])*100:.2f}% ± {np.std(res['acc'])*100:.2f}%",
            "DP Diff": f"{np.mean(res['dp']):.4f} ± {np.std(res['dp']):.4f}",
            "EO Diff": f"{np.mean(res['eo']):.4f} ± {np.std(res['eo']):.4f}",
            "EOpp Diff": f"{np.mean(res['eopp']):.4f} ± {np.std(res['eopp']):.4f}",
            "PP Diff": f"{np.mean(res['pp']):.4f} ± {np.std(res['pp']):.4f}",
            "Paired Reduction Delta (DP)": str(dp_delta),
            "95% CI (Delta DP)": dp_ci_str,
            "Significant (CI > 0)": is_sig
        })

    df_summary = pd.DataFrame(summary_rows)
    print(df_summary.to_string(index=False))

    verified_results["experiments"][d_key] = {
        "summary_table": summary_rows,
        "raw_metrics_10_seeds": {
            var: {k: [round(v, 4) for v in vals] for k, vals in variant_results[var].items()}
            for var in variant_results
        },
        "paired_ci_seed42": {
            var: {
                m: {k: round(v, 4) if isinstance(v, (float, int)) else v for k, v in paired_ci[m].items() if k != 'delta_distribution'}
                for m in paired_ci
            }
            for var, paired_ci in [("Reweighing", paired_ci_rew), ("Exponentiated Gradient", paired_ci_eg), ("Threshold Optimizer", paired_ci_to)]
        }
    }

print("\n" + "="*80)
print("5. COMPAS GROUP COMPARISON DISAGGREGATION (Black vs Caucasian & All Groups n>=30)")
print("="*80)

compas_data = preprocess_dataset(clean_dataset(load_raw_dataset("compas"), "compas"), target_col="two_year_recid", protected_col="race", random_state=42)
compas_m_type = selected_baseline_models["compas"]
compas_model, _ = train_model(compas_data, model_type=compas_m_type, dataset_key="compas", random_state=42)
X_test_comp, y_test_comp, s_test_comp = get_model_input_data(compas_m_type, compas_data, split="test")
y_pred_comp = compas_model.predict(X_test_comp)

print("\nCOMPAS Test-Set Subgroup Sample Sizes (N_test = %d):" % len(s_test_comp))
print(s_test_comp.value_counts().to_string())

# (a) Black vs Caucasian specifically
ci_bvsw = compute_bootstrap_confidence_intervals(
    y_true=y_test_comp.values, y_pred=y_pred_comp, s=s_test_comp.values,
    n_bootstraps=1000, privileged_group="Caucasian", unprivileged_group="African-American", random_state=42
)
print("\n(a) Black vs Caucasian Demographic Parity Gap:")
print("    Point Estimate: %.4f | 95%% Bootstrap CI: [%.4f, %.4f]" % (
    ci_bvsw["demographic_parity_diff"]["val"],
    ci_bvsw["demographic_parity_diff"]["ci_low"],
    ci_bvsw["demographic_parity_diff"]["ci_high"]
))

# (b) All groups with n >= 30 in test set
ci_all30 = compute_bootstrap_confidence_intervals(
    y_true=y_test_comp.values, y_pred=y_pred_comp, s=s_test_comp.values,
    n_bootstraps=1000, privileged_group=None, unprivileged_group=None, random_state=42
)
print("\n(b) All-Groups (n >= 30) Max-Min Demographic Parity Gap:")
print("    Point Estimate: %.4f | 95%% Bootstrap CI: [%.4f, %.4f]" % (
    ci_all30["demographic_parity_diff"]["val"],
    ci_all30["demographic_parity_diff"]["ci_low"],
    ci_all30["demographic_parity_diff"]["ci_high"]
))

verified_results["compas_subgroup_disaggregation"] = {
    "test_set_counts": s_test_comp.value_counts().to_dict(),
    "black_vs_caucasian_ci": {k: v for k, v in ci_bvsw["demographic_parity_diff"].items() if k != 'bootstrap_distribution'},
    "all_groups_n30_ci": {k: v for k, v in ci_all30["demographic_parity_diff"].items() if k != 'bootstrap_distribution'}
}

print("\n" + "="*80)
print("6. UNEDITED TOP-5 GLOBAL SHAP FEATURE NAMES PER DATASET")
print("="*80)

shap_top5 = {}
for d_key in datasets:
    cfg = DATASET_CONFIGS[d_key]
    clean_df = clean_dataset(load_raw_dataset(d_key), d_key)
    prot_col = cfg["default_protected_attribute"]
    target_col = cfg["target_column"]
    
    data = preprocess_dataset(clean_df, target_col=target_col, protected_col=prot_col, random_state=42)
    m_type = selected_baseline_models[d_key]
    model, _ = train_model(data, model_type=m_type, dataset_key=d_key, random_state=42)
    
    shap_df = compute_global_shap_importance(model, data.X_test, data.feature_names)
    top5_features = shap_df.head(5)["feature"].tolist()
    shap_top5[d_key] = top5_features
    print(f"Top-5 SHAP Features for {d_key.upper()} ({m_type}):")
    for idx, feat in enumerate(top5_features, 1):
        print(f"  {idx}. {feat}")

verified_results["shap_top5"] = shap_top5

# Save single source of truth JSON file
json_out_path = DATA_DIR / "verified_benchmark_results.json"
with open(json_out_path, "w", encoding="utf-8") as f:
    json.dump(verified_results, f, indent=2)

print("\n" + "="*80)
print(f"SINGLE SOURCE OF TRUTH SAVED TO: {json_out_path}")
print("="*80)
