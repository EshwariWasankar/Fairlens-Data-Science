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

from src.config import DATASET_CONFIGS, DATA_DIR, MODELS_DIR
import joblib
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
    if "eg_models_in_ram" not in locals():
        eg_models_in_ram = {}
    eg_models_in_ram[d_key] = {}

    print(f"\n--- Exponentiated Gradient Epsilon Sweep for {d_key.upper()} (Baseline: {m_type}) ---")

    for constraint in ["demographic_parity", "equalized_odds"]:
        eg_sweep_results[d_key][constraint] = []
        eg_models_in_ram[d_key][constraint] = {}
        for eps in eps_values:
            # Keep trained models in RAM during the sweep - do not persist intermediate models to disk
            eg_model = train_exponentiated_gradient_model(
                data, model_type=m_type, constraint=constraint, eps=eps, random_state=42, dataset_key=d_key, save_model=False
            )
            eg_models_in_ram[d_key][constraint][eps] = eg_model
                
            X_test_in = data.X_test_scaled if m_type == "logistic_regression" else data.X_test
            y_pred_eg = eg_model.predict(X_test_in)
            
            p_met = evaluate_predictive_performance(eg_model, X_test_in, data.y_test)
            f_met = compute_fairness_metrics(data.y_test.values, y_pred_eg, data.s_test.values, privileged_group=priv_g, unprivileged_group=unread_g)
            
            # Validation-set metrics: epsilon is SELECTED on these (test set is only reported)
            X_val_in = data.X_val_scaled if m_type == "logistic_regression" else data.X_val
            y_pred_eg_val = eg_model.predict(X_val_in)
            val_acc = float((y_pred_eg_val == data.y_val.values).mean())
            val_f = compute_fairness_metrics(data.y_val.values, y_pred_eg_val, data.s_val.values, privileged_group=priv_g, unprivileged_group=unread_g)

            u_classes, u_counts = np.unique(y_pred_eg, return_counts=True)
            counts_dict = dict(zip([int(c) for c in u_classes], [int(c) for c in u_counts]))

            row_res = {
                "eps": eps,
                "accuracy": f"{p_met['accuracy']*100:.2f}%",
                "accuracy_raw": float(p_met['accuracy']),
                "f1": f"{p_met['f1_score']:.4f}",
                "dp_diff": f"{f_met['demographic_parity_diff']:.4f}",
                "dp_diff_raw": float(f_met['demographic_parity_diff']),
                "eo_diff": f"{f_met['equalized_odds_diff']:.4f}",
                "eopp_diff": f"{f_met['equal_opportunity_diff']:.4f}",
                "pp_diff": f"{f_met['predictive_parity_diff']:.4f}",
                "class_counts": counts_dict,
                "val_accuracy_raw": val_acc,
                "val_dp_diff_raw": float(val_f['demographic_parity_diff'])
            }
            eg_sweep_results[d_key][constraint].append(row_res)
            print(f"Constraint: {constraint:18s} | eps={eps:<5} | Acc: {p_met['accuracy']*100:.2f}% | F1: {p_met['f1_score']:.4f} | DP: {f_met['demographic_parity_diff']:.4f} | EO: {f_met['equalized_odds_diff']:.4f} | EOpp: {f_met['equal_opportunity_diff']:.4f} | PP: {f_met['predictive_parity_diff']:.4f} | Counts: {counts_dict}")

print("\n" + "="*80)
print("3B. EPSILON SELECTION ON VALIDATION SET (RULE: MAX FAIRNESS GAIN WITH VAL ACC LOSS <= 5.0 PP VS BASELINE)")
print("="*80)

selected_epsilons = {}
selection_tables = {}

for d_key in datasets:
    cfg = DATASET_CONFIGS[d_key]
    m_type = selected_baseline_models[d_key]
    clean_df = clean_dataset(load_raw_dataset(d_key), d_key)
    prot_col = cfg["default_protected_attribute"]
    target_col = cfg["target_column"]
    prot_cfg = cfg.get("protected_attributes", {}).get(prot_col, {})
    priv_g = prot_cfg.get("privileged")
    unread_g = prot_cfg.get("unprivileged")
    
    data = preprocess_dataset(clean_df, target_col=target_col, protected_col=prot_col, random_state=42)
    base_m, _ = train_model(data, model_type=m_type, dataset_key=d_key, random_state=42)
    X_test, y_test, s_test = get_model_input_data(m_type, data, split="test")
    base_p = evaluate_predictive_performance(base_m, X_test, y_test)
    base_f = compute_fairness_metrics(y_test.values, base_m.predict(X_test), s_test.values, privileged_group=priv_g, unprivileged_group=unread_g)
    
    base_acc_pct = base_p["accuracy"] * 100
    base_dp = base_f["demographic_parity_diff"]

    # Baseline on the VALIDATION split — the selection rule never looks at test data
    X_val_b, y_val_b, s_val_b = get_model_input_data(m_type, data, split="val")
    y_val_pred_b = base_m.predict(X_val_b)
    base_val_acc_pct = float((y_val_pred_b == y_val_b.values).mean()) * 100
    base_val_dp = compute_fairness_metrics(y_val_b.values, y_val_pred_b, s_val_b.values, privileged_group=priv_g, unprivileged_group=unread_g)["demographic_parity_diff"]
    
    dp_sweep = eg_sweep_results[d_key]["demographic_parity"]
    
    # Selection rule (validation set): max fairness gain (min val DP diff) subject to val acc loss <= 5.0 pp
    # Ties on DP are broken by higher validation accuracy, then smaller eps.
    eligible = [r for r in dp_sweep if (base_val_acc_pct - (r["val_accuracy_raw"] * 100)) <= 5.0]
    if eligible:
        best_r = min(eligible, key=lambda x: (x["val_dp_diff_raw"], -x["val_accuracy_raw"], x["eps"]))
        chosen_eps = best_r["eps"]
        selection_reason = f"Satisfies <= 5.0pp validation loss rule with maximum fairness gain (lowest val DP diff = {best_r['val_dp_diff_raw']:.4f})"
    else:
        # Fallback: minimum validation accuracy loss
        best_r = min(dp_sweep, key=lambda x: (base_val_acc_pct - (x["val_accuracy_raw"] * 100), x["eps"]))
        chosen_eps = best_r["eps"]
        selection_reason = f"No eps achieves <= 5.0pp validation accuracy loss; fallback to minimal val accuracy loss ({base_val_acc_pct - best_r['val_accuracy_raw']*100:.2f}pp loss)"
    
    selected_epsilons[d_key] = chosen_eps
    
    # Save ONLY the final selected winner model to disk as a .joblib file, discarding intermediate iterations
    winning_model = eg_models_in_ram[d_key]["demographic_parity"][chosen_eps]
    final_eg_path = MODELS_DIR / f"{d_key}_{m_type}_exponentiated_gradient.joblib"
    joblib.dump(winning_model, final_eg_path)
    # Also save the keyed version for seed 42
    joblib.dump(winning_model, MODELS_DIR / f"{d_key}_{m_type}_exponentiated_gradient_demographic_parity_eps{chosen_eps}_seed42.joblib")
    print(f"--> Saved final selected Exponentiated Gradient model to: {final_eg_path}")
    
    # Print table (eps, accuracy loss, DP diff, selected eps)
    table_rows = []
    for r in dp_sweep:
        loss_pp = base_val_acc_pct - (r["val_accuracy_raw"] * 100)
        is_selected = (r["eps"] == chosen_eps)
        table_rows.append({
            "Epsilon": r["eps"],
            "Val Accuracy": f"{r['val_accuracy_raw']*100:.2f}%",
            "Val Accuracy Loss (pp)": f"{loss_pp:+.2f}",
            "Val DP Diff": f"{r['val_dp_diff_raw']:.4f}",
            "Val Fairness Gain (DP reduction)": f"{base_val_dp - r['val_dp_diff_raw']:+.4f}",
            "Test Accuracy": r["accuracy"],
            "Test DP Diff": r["dp_diff"],
            "Acc Loss <= 5pp": loss_pp <= 5.0,
            "Selected": "YES (Chosen)" if is_selected else "No"
        })
    
    selection_tables[d_key] = table_rows
    print(f"\n--- Epsilon Selection Table for {d_key.upper()} (Baseline {m_type}: Val Acc={base_val_acc_pct:.2f}%, Val DP={base_val_dp:.4f} | Test Acc={base_acc_pct:.2f}%, Test DP={base_dp:.4f}) ---")
    print(pd.DataFrame(table_rows).to_string(index=False))
    print(f"--> SELECTED EPSILON for {d_key.upper()}: eps = {chosen_eps}")
    print(f"    Rationale: {selection_reason}")

print("\n" + "="*80)
print("4. FULL 10-SEED MITIGATION EVALUATION & PAIRED BOOTSTRAP CIs OF DIFFERENCE")
print("="*80)

verified_results = {
    "dedup_info": dedup_info,
    "val_f1_summary": val_f1_summary,
    "selected_baseline_models": selected_baseline_models,
    "eg_sweep_results": eg_sweep_results,
    "selected_epsilons": selected_epsilons,
    "epsilon_selection_tables": selection_tables,
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
    eg_eps = selected_epsilons[d_key]
    print(f"\n==========================================================")
    print(f"DATASET: {d_key.upper()} (Selected Baseline Model: {m_type} | Selected EG eps: {eg_eps})")
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

    # Use winning Exponentiated Gradient model already resident in RAM
    if d_key in eg_models_in_ram and "demographic_parity" in eg_models_in_ram[d_key] and eg_eps in eg_models_in_ram[d_key]["demographic_parity"]:
        eg_model_s42 = eg_models_in_ram[d_key]["demographic_parity"][eg_eps]
    else:
        eg_model_s42 = train_exponentiated_gradient_model(data_s42, model_type=m_type, constraint="demographic_parity", eps=eg_eps, random_state=42, dataset_key=d_key, save_model=False)

    X_test_in_s42 = data_s42.X_test_scaled if m_type == "logistic_regression" else data_s42.X_test
    y_pred_eg_s42 = eg_model_s42.predict(X_test_in_s42)

    # Train and save the final selected Threshold Optimizer winner to disk
    to_model_s42 = train_threshold_optimizer_model(
        base_model_s42, data_s42, model_type=m_type, save_model=True, dataset_key=d_key
    )
    y_pred_to_s42 = to_model_s42.predict(X_test_in_s42, sensitive_features=s_test_s42, random_state=42)

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

        # Exponentiated Gradient (keep intermediate seeds in RAM, do not save to disk)
        eg_m = train_exponentiated_gradient_model(data, model_type=m_type, constraint="demographic_parity", eps=eg_eps, random_state=s, dataset_key=d_key, save_model=False)
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

        # Threshold Optimizer (keep intermediate seeds in RAM, do not save to disk)
        to_m = train_threshold_optimizer_model(base_m, data, model_type=m_type, save_model=False, dataset_key=d_key)
        y_pred_t = to_m.predict(X_test_in, sensitive_features=s_test, random_state=42)
        class TW:
            def __init__(self, tm, sf): self.tm = tm; self.sf = sf
            def predict(self, X): return self.tm.predict(X, sensitive_features=self.sf, random_state=42)
            def predict_proba(self, X): return np.asarray(self.tm._pmf_predict(X, sensitive_features=self.sf), dtype=np.float64)
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

    # Precompute 500-resample single-model bootstrap CIs on seed 42 partition for instant dashboard rendering
    base_ci_500 = compute_bootstrap_confidence_intervals(y_test_s42.values, y_pred_base_s42, s_test_s42.values, n_bootstraps=500, privileged_group=priv_g, unprivileged_group=unread_g, random_state=42)
    rew_ci_500 = compute_bootstrap_confidence_intervals(y_test_s42.values, y_pred_rew_s42, s_test_s42.values, n_bootstraps=500, privileged_group=priv_g, unprivileged_group=unread_g, random_state=42)
    eg_ci_500 = compute_bootstrap_confidence_intervals(y_test_s42.values, y_pred_eg_s42, s_test_s42.values, n_bootstraps=500, privileged_group=priv_g, unprivileged_group=unread_g, random_state=42)
    to_ci_500 = compute_bootstrap_confidence_intervals(y_test_s42.values, y_pred_to_s42, s_test_s42.values, n_bootstraps=500, privileged_group=priv_g, unprivileged_group=unread_g, random_state=42)

    base_pred_s42 = evaluate_predictive_performance(base_model_s42, X_test_s42, y_test_s42)
    rew_pred_s42 = evaluate_predictive_performance(rew_model_s42, X_test_s42, y_test_s42)
    eg_pred_s42 = evaluate_predictive_performance(eg_model_s42, X_test_in_s42, y_test_s42)
    to_pred_s42 = evaluate_predictive_performance(TW(to_model_s42, s_test_s42), X_test_in_s42, y_test_s42)

    def _strip_ci(ci_obj):
        return {
            m: {k: round(v, 4) if isinstance(v, (float, int)) else v for k, v in data.items() if k != 'bootstrap_distribution'}
            for m, data in ci_obj.items()
        }

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
        "baseline_fairness": _strip_ci(base_ci_500),
        "baseline_pred_metrics": {k: round(v, 4) if isinstance(v, float) else v for k, v in base_pred_s42.items()},
        "mitigations": {
            "Reweighing": {
                "metrics": _strip_ci(rew_ci_500),
                "pred_metrics": {k: round(v, 4) if isinstance(v, float) else v for k, v in rew_pred_s42.items()},
                "significance": {k: round(v, 4) if isinstance(v, (float, int)) else v for k, v in paired_ci_rew["demographic_parity_diff"].items() if k != 'delta_distribution'}
            },
            "Exponentiated Gradient": {
                "metrics": _strip_ci(eg_ci_500),
                "pred_metrics": {k: round(v, 4) if isinstance(v, float) else v for k, v in eg_pred_s42.items()},
                "significance": {k: round(v, 4) if isinstance(v, (float, int)) else v for k, v in paired_ci_eg["demographic_parity_diff"].items() if k != 'delta_distribution'}
            },
            "Threshold Optimizer": {
                "metrics": _strip_ci(to_ci_500),
                "pred_metrics": {k: round(v, 4) if isinstance(v, float) else v for k, v in to_pred_s42.items()},
                "significance": {k: round(v, 4) if isinstance(v, (float, int)) else v for k, v in paired_ci_to["demographic_parity_diff"].items() if k != 'delta_distribution'}
            }
        },
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
def _json_serializable(obj):
    if isinstance(obj, dict):
        return {k: _json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_json_serializable(v) for v in obj]
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, (np.floating, float)):
        return float(obj)
    elif isinstance(obj, (np.integer, int)):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return [_json_serializable(x) for x in obj.tolist()]
    return obj

json_out_path = DATA_DIR / "verified_benchmark_results.json"
with open(json_out_path, "w", encoding="utf-8") as f:
    json.dump(_json_serializable(verified_results), f, indent=2)

print("\n" + "="*80)
print(f"SINGLE SOURCE OF TRUTH SAVED TO: {json_out_path}")
print("="*80)
