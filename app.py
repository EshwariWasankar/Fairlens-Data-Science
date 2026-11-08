import os
import json
import joblib
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime
from typing import Optional, Any, Tuple

from src.config import DATASET_CONFIGS, REPORTS_DIR, DATA_DIR, MODELS_DIR, get_outcome_labels
from src.pipeline import run_fairlens_pipeline
from src.data.loader import validate_custom_csv, load_raw_dataset
from src.data.cleaner import clean_dataset
from src.data.preprocessor import preprocess_dataset, PreprocessedData
from src.analysis.eda import get_dataset_kpis, get_group_disparity_table, get_numeric_correlation_matrix
from src.analysis.proxy import compute_proxy_leakage
from src.models.trainer import train_model, get_model_input_data, raw_feature_model
from src.fairness.intersectional import compute_intersectional_metrics
from src.fairness.calibration import compute_group_calibration
from src.fairness.frontier import pareto_frontier, sweep_points
from src.reporting.model_card import generate_model_card
from src.benchmark_store import uses_precomputed_benchmark
from src.explainability.shap_explainer import compute_global_shap_importance, compute_local_shap_explanation
from src.explainability.dice_explainer import generate_counterfactual_explanation
from src.explainability.insights import generate_plain_language_summary
from src.reporting.pdf_generator import generate_pdf_audit_report, generate_html_audit_report
from src.db.db_manager import get_experiment_history

# Page Configuration
st.set_page_config(
    page_title="FairLens | Algorithmic Bias Auditing & Mitigation Platform",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load Custom CSS Design System
def load_css():
    css_path = Path("style.css")
    if css_path.exists():
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# Consistent Institutional Color Palette
PALETTE = ["#2563eb", "#0d9488", "#d97706", "#7c3aed", "#dc2626", "#4f46e5"]
MODEL_LABELS = {"random_forest": "Random Forest", "logistic_regression": "Logistic Regression", "xgboost": "XGBoost"}

# Sidebar Configuration Controls
st.sidebar.markdown('<div class="sidebar-section-title">DATASET SELECTION</div>', unsafe_allow_html=True)

dataset_options = {
    "german_credit": "German Credit Data (Statlog)",
    "compas": "COMPAS Recidivism (ProPublica)",
    "adult_income": "Adult Census Income",
    "custom": "Upload Custom CSV"
}

selected_dataset_key = st.sidebar.selectbox(
    "Select Benchmark Dataset",
    options=list(dataset_options.keys()),
    format_func=lambda x: dataset_options[x]
)

custom_df = None
custom_target = None
custom_protected = None

if selected_dataset_key == "custom":
    st.sidebar.markdown('<div class="sidebar-section-title">CUSTOM CSV INGESTION</div>', unsafe_allow_html=True)
    uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])
    
    if uploaded_file is not None:
        try:
            custom_df = pd.read_csv(uploaded_file)
            st.sidebar.caption(f"Loaded: {len(custom_df):,} rows")
            
            cols = list(custom_df.columns)
            custom_target = st.sidebar.selectbox("Select Target Column", options=cols)
            custom_protected = st.sidebar.selectbox("Select Protected Attribute Column", options=[c for c in cols if c != custom_target])
            
            valid, msg = validate_custom_csv(custom_df, custom_target, custom_protected)
            if not valid:
                st.sidebar.error(msg)
                st.stop()
        except Exception as e:
            st.sidebar.error(f"Error reading CSV: {e}")
            st.stop()
    else:
        st.info("Upload a custom dataset CSV in the sidebar to initiate live bias audit.")
        st.stop()

if selected_dataset_key != "custom":
    cfg = DATASET_CONFIGS[selected_dataset_key]
    protected_attrs = list(cfg["protected_attributes"].keys())
    default_prot = cfg["default_protected_attribute"]
    
    selected_protected = st.sidebar.selectbox(
        "Protected Demographic Attribute",
        options=protected_attrs,
        index=protected_attrs.index(default_prot) if default_prot in protected_attrs else 0
    )
else:
    selected_protected = custom_protected

st.sidebar.markdown('<div class="sidebar-section-title">CLASSIFIER MODEL</div>', unsafe_allow_html=True)
selected_model_type = st.sidebar.selectbox(
    "Classifier Architecture (System Under Audit)",
    options=["random_forest", "logistic_regression", "xgboost"],
    format_func=lambda x: "Random Forest Classifier" if x == "random_forest" else "Logistic Regression Baseline" if x == "logistic_regression" else "XGBoost Classifier"
)

st.sidebar.markdown('<div class="sidebar-section-title">AUDIT METHODOLOGY</div>', unsafe_allow_html=True)
st.sidebar.markdown("""
- **Partitioning:** 60% Train / 20% Val / 20% Test (Stratified)
- **Bootstrap Validation:** 500 Resamples (95% CI)
- **Significance Test:** Paired Bootstrap 95% CI (Difference vs Baseline)
""")

# ==============================================================================
# STREAMLIT CACHING & PRE-COMPUTED RESOURCE LOADERS
# ==============================================================================

# 1. Data Loading (@st.cache_data): load CSVs and clean dataframes into memory once
@st.cache_data(show_spinner=False)
def load_cached_raw_dataset(dataset_key: str) -> pd.DataFrame:
    """Loads CSV dataset from disk into memory once."""
    return load_raw_dataset(dataset_key)

@st.cache_data(show_spinner=False)
def load_cached_clean_dataset(dataset_key: str) -> pd.DataFrame:
    """Cleans dataset once and stores cleaned DataFrame in cache."""
    raw_df = load_cached_raw_dataset(dataset_key)
    return clean_dataset(raw_df, dataset_key)

@st.cache_data(show_spinner=False)
def get_cached_preprocessed_data(dataset_key: str, target_col: str, protected_col: str) -> PreprocessedData:
    """Preprocesses dataset into train/val/test splits once."""
    clean_df = load_cached_clean_dataset(dataset_key)
    return preprocess_dataset(clean_df, target_col=target_col, protected_col=protected_col, random_state=42)

@st.cache_data(show_spinner=False)
def get_cached_eda(dataset_key: str, protected_col: str):
    """Calculates EDA KPIs, group disparity breakdown, correlation matrix, and proxy leakage once."""
    clean_df = load_cached_clean_dataset(dataset_key)
    eda_target = 'target' if 'target' in clean_df.columns else DATASET_CONFIGS[dataset_key]["target_column"]
    fav_value = int(DATASET_CONFIGS[dataset_key].get("favorable_target_value", 1))
    kpis = get_dataset_kpis(clean_df, target_col=eda_target, protected_col=protected_col, favorable_value=fav_value)
    disparity_table = get_group_disparity_table(clean_df, target_col=eda_target, protected_col=protected_col, favorable_value=fav_value)
    corr_matrix = get_numeric_correlation_matrix(clean_df)
    proxy_df = compute_proxy_leakage(clean_df, protected_col=protected_col, target_col=eda_target)
    return kpis, disparity_table, corr_matrix, proxy_df

# 2. Metric Fetching (@st.cache_data): strictly read pre-computed statistical benchmarks
@st.cache_data(show_spinner=False)
def load_precomputed_benchmarks() -> dict:
    """Reads pre-computed verified benchmark results containing 500-resample bootstrap CIs and metrics."""
    json_path = DATA_DIR / "verified_benchmark_results.json"
    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

# 3. Model Loading (@st.cache_resource): load .joblib AI models once so they do not rebuild on tab switches
@st.cache_resource(show_spinner=False)
def load_cached_joblib_file(model_path_str: str) -> Any:
    """Loads pre-trained .joblib model binary from disk using st.cache_resource."""
    return joblib.load(model_path_str)

@st.cache_resource(show_spinner=False)
def get_cached_baseline_model(dataset_key: str, model_type: str) -> Any:
    """Loads pre-trained baseline classifier from disk or fits once if absent."""
    candidate_names = [
        f"{dataset_key}_{model_type}_baseline.joblib",
        f"default_{model_type}_baseline.joblib"
    ]
    for cname in candidate_names:
        p = MODELS_DIR / cname
        if p.exists():
            return load_cached_joblib_file(str(p))
    cfg = DATASET_CONFIGS[dataset_key]
    data = get_cached_preprocessed_data(dataset_key, cfg["target_column"], cfg["default_protected_attribute"])
    model, _ = train_model(data, model_type=model_type, dataset_key=dataset_key, random_state=42)
    return model

@st.cache_resource(show_spinner=False)
def get_cached_mitigated_model(dataset_key: str, model_type: str, technique: str, eps: Optional[float] = None) -> Optional[Any]:
    """Loads pre-trained mitigated model from disk via st.cache_resource."""
    if technique == "Exponentiated Gradient":
        candidates = [
            MODELS_DIR / f"{dataset_key}_{model_type}_exponentiated_gradient.joblib"
        ]
        if eps is not None:
            candidates.append(MODELS_DIR / f"{dataset_key}_{model_type}_exponentiated_gradient_demographic_parity_eps{eps}_seed42.joblib")
        for p in candidates:
            if p.exists():
                return load_cached_joblib_file(str(p))
    elif technique == "Threshold Optimizer":
        p = MODELS_DIR / f"{dataset_key}_{model_type}_threshold_optimizer.joblib"
        if p.exists():
            return load_cached_joblib_file(str(p))
    return None

@st.cache_data(show_spinner=False)
def get_cached_global_shap(dataset_key: str, model_type: str) -> pd.DataFrame:
    """Computes global SHAP feature importance once and caches result in memory."""
    base_model = get_cached_baseline_model(dataset_key, model_type)
    cfg = DATASET_CONFIGS[dataset_key]
    data = get_cached_preprocessed_data(dataset_key, cfg["target_column"], cfg["default_protected_attribute"])
    X_in = data.X_test_scaled if model_type == "logistic_regression" else data.X_test
    return compute_global_shap_importance(base_model, X_in, data.feature_names)


# Execute or retrieve pipeline results
outcome_labels = get_outcome_labels(selected_dataset_key)
is_benchmark_match = False
benchmarks = load_precomputed_benchmarks()
if selected_dataset_key != "custom":
    cfg = DATASET_CONFIGS[selected_dataset_key]
    dataset_name = cfg["name"]
    target_col = cfg["target_column"]
    protected_col = selected_protected
    actionable_cols = cfg.get("actionable_features", [])
    immutable_cols = cfg.get("immutable_features", [])

    # Fast cached EDA and Preprocessing
    kpis, disparity_table, corr_matrix, proxy_df = get_cached_eda(selected_dataset_key, protected_col)
    preproc_data = get_cached_preprocessed_data(selected_dataset_key, target_col, protected_col)
    baseline_model = get_cached_baseline_model(selected_dataset_key, selected_model_type)

    benchmarks = load_precomputed_benchmarks()
    exp_bench = benchmarks.get("experiments", {}).get(selected_dataset_key, {})

    # Check if pre-computed metrics exist for this benchmark
    is_benchmark_match = uses_precomputed_benchmark(
        benchmarks, selected_dataset_key, protected_col, selected_model_type, cfg["default_protected_attribute"]
    )

    if is_benchmark_match:
        # Strictly read the heavy statistical metrics from verified benchmark JSON
        baseline_fairness = exp_bench["baseline_fairness"]
        baseline_pred_metrics = exp_bench["baseline_pred_metrics"]
        mitigations = exp_bench["mitigations"]

        # Attach cached model objects to mitigations for recourse/explainability
        for mit_name in mitigations:
            mit_m = get_cached_mitigated_model(
                selected_dataset_key, selected_model_type, mit_name, 
                eps=benchmarks.get("selected_epsilons", {}).get(selected_dataset_key)
            )
            mitigations[mit_name]["model"] = mit_m

        # Compute selection rate group breakdown for Tab 3 chart from cached test predictions
        X_test, y_test, s_test = preproc_data.X_test, preproc_data.y_test, preproc_data.s_test
        X_in = preproc_data.X_test_scaled if selected_model_type == "logistic_regression" else X_test
        y_pred_base = baseline_model.predict(X_in)
        temp_gr = pd.DataFrame({"group": s_test.astype(str), "pred": y_pred_base}).groupby("group")["pred"].mean().reset_index()
        temp_gr.columns = ["group", "selection_rate"]
        baseline_fairness["demographic_parity_diff"]["group_rates"] = temp_gr.to_dict(orient="records")

        # Global SHAP importance
        shap_importance = get_cached_global_shap(selected_dataset_key, selected_model_type)

        # Plain language summary
        primary_mit_res = mitigations.get("Reweighing", list(mitigations.values())[0])
        insights = generate_plain_language_summary(
            dataset_name=dataset_name,
            protected_attribute=protected_col,
            kpis=kpis,
            proxy_df=proxy_df,
            baseline_fairness=baseline_fairness,
            mitigation_res=primary_mit_res,
            technique_name="Reweighing"
        )

        pipeline_results = {
            "dataset_key": selected_dataset_key,
            "dataset_name": dataset_name,
            "target_col": target_col,
            "protected_col": protected_col,
            "actionable_cols": actionable_cols,
            "immutable_cols": immutable_cols,
            "kpis": kpis,
            "disparity_table": disparity_table,
            "corr_matrix": corr_matrix,
            "proxy_df": proxy_df,
            "baseline_model": baseline_model,
            "baseline_pred_metrics": baseline_pred_metrics,
            "baseline_fairness": baseline_fairness,
            "mitigations": mitigations,
            "shap_importance": shap_importance,
            "insights": insights,
            "preprocessed_data": preproc_data
        }
    else:
        with st.spinner("Executing FairLens pipeline..."):
            pipeline_results = run_fairlens_pipeline(selected_dataset_key, protected_attribute=selected_protected, model_type=selected_model_type)
            kpis = pipeline_results["kpis"]
            proxy_df = pipeline_results["proxy_df"]
            baseline_pred_metrics = pipeline_results["baseline_pred_metrics"]
            baseline_fairness = pipeline_results["baseline_fairness"]
            mitigations = pipeline_results["mitigations"]
            shap_importance = pipeline_results["shap_importance"]
            insights = pipeline_results["insights"]
            preproc_data = pipeline_results["preprocessed_data"]
else:
    with st.spinner("Auditing uploaded custom dataset..."):
        pipeline_results = run_fairlens_pipeline(
            "custom",
            custom_df=custom_df,
            custom_target_col=custom_target,
            custom_protected_col=custom_protected,
            model_type=selected_model_type
        )
        kpis = pipeline_results["kpis"]
        proxy_df = pipeline_results["proxy_df"]
        baseline_pred_metrics = pipeline_results["baseline_pred_metrics"]
        baseline_fairness = pipeline_results["baseline_fairness"]
        mitigations = pipeline_results["mitigations"]
        shap_importance = pipeline_results["shap_importance"]
        insights = pipeline_results["insights"]
        preproc_data = pipeline_results["preprocessed_data"]

# Header Wordmark & Metadata Subtitle
model_label = "Random Forest Classifier" if selected_model_type == "random_forest" else "Logistic Regression Baseline" if selected_model_type == "logistic_regression" else "XGBoost Classifier"
timestamp_str = datetime.now().strftime('%Y-%m-%d %H:%M')

st.markdown(f"""
<div class="main-header">
    <div class="main-title">FAIRLENS | ALGORITHMIC BIAS AUDITING & MITIGATION PLATFORM</div>
    <div class="main-subtitle">
        Dataset: <strong>{pipeline_results['dataset_name']}</strong> &nbsp;|&nbsp; 
        Protected Attribute: <strong>{pipeline_results['protected_col']}</strong> &nbsp;|&nbsp; 
        Target: <strong>{pipeline_results.get('target_col', 'target')}</strong> &nbsp;|&nbsp; 
        Model: <strong>{model_label}</strong> &nbsp;|&nbsp; 
        Audit Timestamp: <strong>{timestamp_str}</strong>
    </div>
</div>
""", unsafe_allow_html=True)

# 6 Main Clean Text Navigation Tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Overview & EDA",
    "Proxy Leakage Analysis",
    "Fairness Audit",
    "Mitigation Comparison",
    "Explainability & Recourse",
    "Audit Report & Export"
])

# ==========================================
# TAB 1: OVERVIEW & EDA
# ==========================================
with tab1:
    st.markdown("### Dataset Summary & Base Outcome Disparity")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Total {outcome_labels['subject']}s / Records</div>
            <div class="kpi-val">{kpis['total_records']:,}</div>
            <div class="kpi-subtext">Deduplicated Sample Size</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Base Favorable Rate</div>
            <div class="kpi-val">{kpis['overall_positive_rate']*100:.2f}%</div>
            <div class="kpi-subtext">Historical {outcome_labels['favorable_rate']}</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        disp_val = kpis['base_rate_disparity']*100
        badge = '<span class="status-badge-red">Significant Disparity</span>' if disp_val > 10 else '<span class="status-badge-amber">Moderate Disparity</span>' if disp_val > 5 else '<span class="status-badge-green">Low Disparity</span>'
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Demographic Base Gap</div>
            <div class="kpi-val">{disp_val:.2f}%</div>
            <div class="kpi-subtext">{badge}</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Protected Subgroups</div>
            <div class="kpi-val">{kpis['protected_groups_count']}</div>
            <div class="kpi-subtext">Evaluated Subcategories</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="insight-card">
        <div class="insight-title">EXPLORATORY DISPARITY AUDIT FINDING</div>
        {insights['base_statement']}
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns([1, 1])
    
    with c1:
        st.markdown(f"#### Favorable Outcome ({outcome_labels['favorable_rate']}) by Demographic Group")
        disp_df = pipeline_results["disparity_table"]
        if not disp_df.empty:
            fig_bar = px.bar(
                disp_df,
                x='Protected_Group',
                y='Favorable_Rate',
                text_auto='.2f',
                color='Protected_Group',
                color_discrete_sequence=PALETTE,
                title=f"Historical {outcome_labels['favorable_rate']} (%)"
            )
            fig_bar.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0b0f19",
                plot_bgcolor="#111827",
                height=360,
                showlegend=False,
                yaxis_title=f"{outcome_labels['favorable_rate']} (%)"
            )
            st.plotly_chart(fig_bar, use_container_width=True)

    with c2:
        st.markdown("#### Subgroup Applicant Volume & Representation")
        if not disp_df.empty:
            fig_vol = px.bar(
                disp_df,
                x='Protected_Group',
                y='Total_Applicants',
                text_auto=',d',
                color='Protected_Group',
                color_discrete_sequence=PALETTE,
                title="Total Applicant Count by Group"
            )
            fig_vol.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0b0f19",
                plot_bgcolor="#111827",
                height=360,
                showlegend=False,
                yaxis_title="Applicant Count"
            )
            st.plotly_chart(fig_vol, use_container_width=True)

    st.markdown("#### Feature Pairwise Correlation Matrix")
    corr_df = pipeline_results["corr_matrix"]
    if not corr_df.empty:
        fig_heat = px.imshow(
            corr_df,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="Blues",
            title="Pearson Correlation Matrix (Numerical Features)"
        )
        fig_heat.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0b0f19",
            plot_bgcolor="#111827",
            height=400
        )
        st.plotly_chart(fig_heat, use_container_width=True)

# ==========================================
# TAB 2: PROXY LEAKAGE ANALYSIS
# ==========================================
with tab2:
    st.markdown("### Indirect Feature Proxy Leakage Ranking")
    
    st.markdown(f"""
    <div class="insight-card">
        <div class="insight-title">PROXY LEAKAGE AUDIT FINDING</div>
        {insights['proxy_statement']}
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns([3, 2])
    
    with c1:
        st.markdown("#### Feature Mutual Information Leakage Ranking")
        if not proxy_df.empty:
            fig_proxy = px.bar(
                proxy_df.head(10),
                x='mutual_information',
                y='feature',
                orientation='h',
                color='risk_level',
                color_discrete_map={
                    "High Proxy Risk": "#dc2626",
                    "Moderate Proxy Risk": "#d97706",
                    "Low Proxy Risk": "#059669"
                },
                title=f"Mutual Information with Protected Attribute: '{pipeline_results['protected_col']}'"
            )
            fig_proxy.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0b0f19",
                plot_bgcolor="#111827",
                height=400,
                xaxis_title="Mutual Information Score",
                yaxis=dict(autorange="reversed")
            )
            st.plotly_chart(fig_proxy, use_container_width=True)

    with c2:
        st.markdown("#### Proxy Leakage Scores Table")
        if not proxy_df.empty:
            st.dataframe(
                proxy_df[['feature', 'mutual_information', 'correlation', 'risk_level']],
                column_config={
                    "feature": "Feature Name",
                    "mutual_information": st.column_config.NumberColumn("Mutual Information", format="%.4f"),
                    "correlation": st.column_config.NumberColumn("Correlation", format="%.4f"),
                    "risk_level": "Proxy Risk Level"
                },
                hide_index=True,
                use_container_width=True,
                height=400
            )

# ==========================================
# TAB 3: FAIRNESS AUDIT
# ==========================================
with tab3:
    st.markdown(f"### Baseline Model Fairness Audit — {model_label}")
    
    st.markdown(f"""
    <div class="insight-card">
        <div class="insight-title">STATISTICAL BIAS AUDIT FINDING</div>
        {insights['audit_statement']}
    </div>
    """, unsafe_allow_html=True)

    # 4 Metric Cards with 95% Bootstrap CIs
    m1, m2, m3, m4 = st.columns(4)
    
    dp_info = baseline_fairness["demographic_parity_diff"]
    eo_info = baseline_fairness["equalized_odds_diff"]
    eopp_info = baseline_fairness["equal_opportunity_diff"]
    pp_info = baseline_fairness["predictive_parity_diff"]

    with m1:
        badge = '<span class="status-badge-red">Significant Disparity</span>' if dp_info.get('is_significant', dp_info['ci_low'] > 0.02) else '<span class="status-badge-green">Not Significant</span>'
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Demographic Parity Diff</div>
            <div class="kpi-val">{dp_info['val']:.4f}</div>
            <div class="kpi-subtext">95% CI: [{dp_info['ci_low']:.4f}, {dp_info['ci_high']:.4f}]</div>
            <div style="margin-top: 6px;">{badge}</div>
        </div>
        """, unsafe_allow_html=True)

    with m2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Equalized Odds Diff</div>
            <div class="kpi-val">{eo_info['val']:.4f}</div>
            <div class="kpi-subtext">95% CI: [{eo_info['ci_low']:.4f}, {eo_info['ci_high']:.4f}]</div>
            <div style="margin-top: 6px;"><span class="status-badge-amber">Equal Error Rate</span></div>
        </div>
        """, unsafe_allow_html=True)

    with m3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Equal Opportunity Diff</div>
            <div class="kpi-val">{eopp_info['val']:.4f}</div>
            <div class="kpi-subtext">95% CI: [{eopp_info['ci_low']:.4f}, {eopp_info['ci_high']:.4f}]</div>
            <div style="margin-top: 6px;"><span class="status-badge-amber">TPR Disparity</span></div>
        </div>
        """, unsafe_allow_html=True)

    with m4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Predictive Parity Diff</div>
            <div class="kpi-val">{pp_info['val']:.4f}</div>
            <div class="kpi-subtext">95% CI: [{pp_info['ci_low']:.4f}, {pp_info['ci_high']:.4f}]</div>
            <div style="margin-top: 6px;"><span class="status-badge-amber">Precision Gap</span></div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns([1, 1])

    with c1:
        st.markdown("#### Baseline Model Predictive Performance Breakdown")
        perf_df = pd.DataFrame([{
            "F1 (Class 1)": f"{baseline_pred_metrics.get('f1_class_1', baseline_pred_metrics.get('f1_score', 0.0)):.4f}",
            "Precision (Class 1)": f"{baseline_pred_metrics.get('precision_class_1', baseline_pred_metrics.get('precision', 0.0)):.4f}",
            "Recall (Class 1)": f"{baseline_pred_metrics.get('recall_class_1', baseline_pred_metrics.get('recall', 0.0)):.4f}",
            "F1 (Class 0)": f"{baseline_pred_metrics.get('f1_class_0', 0.0):.4f}",
            "Precision (Class 0)": f"{baseline_pred_metrics.get('precision_class_0', 0.0):.4f}",
            "Recall (Class 0)": f"{baseline_pred_metrics.get('recall_class_0', 0.0):.4f}",
            "Balanced Accuracy": f"{baseline_pred_metrics.get('balanced_accuracy', 0.0):.4f}",
            "Overall Accuracy": f"{baseline_pred_metrics.get('accuracy', 0.0):.4f}",
            "ROC-AUC": f"{baseline_pred_metrics.get('roc_auc', 0.0):.4f}"
        }]).T.reset_index()
        perf_df.columns = ["Metric Name", "Value"]
        st.dataframe(perf_df, hide_index=True, use_container_width=True)

    with c2:
        st.markdown("#### Model Selection Rate Breakdown by Subgroup")
        gr_df = pd.DataFrame(baseline_fairness["demographic_parity_diff"].get("group_rates", []))
        if not gr_df.empty:
            fig_gr = px.bar(
                gr_df,
                x='group',
                y='selection_rate',
                text_auto='.3f',
                color='group',
                color_discrete_sequence=PALETTE,
                title=f"Selection Rate = share predicted '{outcome_labels['by_value'][1]}' (label 1)"
            )
            fig_gr.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0b0f19",
                plot_bgcolor="#111827",
                height=360,
                showlegend=False,
                yaxis_title="Selection Rate"
            )
            st.plotly_chart(fig_gr, use_container_width=True)

    # ---- Shared baseline test-set predictions for the extra audits ----
    _X_base_in = preproc_data.X_test_scaled if selected_model_type == "logistic_regression" else preproc_data.X_test
    _base_model = pipeline_results["baseline_model"]
    _y_pred_base_t = np.asarray(_base_model.predict(_X_base_in))
    _clean_for_attrs = pipeline_results.get("clean_df")
    if _clean_for_attrs is None and selected_dataset_key != "custom":
        _clean_for_attrs = load_cached_clean_dataset(selected_dataset_key)

    st.markdown("---")
    st.markdown("### Intersectional Fairness Audit")
    st.caption("Single-attribute audits can hide disparities that only appear for combinations of attributes. "
               "Subgroups with fewer than 30 test-set members are listed but excluded from the gap.")
    _cfg_attrs = list(DATASET_CONFIGS.get(selected_dataset_key, {}).get("protected_attributes", {}).keys())
    _other_attrs = [a for a in _cfg_attrs if a != protected_col and _clean_for_attrs is not None and a in _clean_for_attrs.columns]
    if _other_attrs:
        _default_second = {"compas": "sex", "adult_income": "age_group", "german_credit": "age_group"}.get(selected_dataset_key)
        second_attr = st.selectbox(
            f"Intersect '{protected_col}' with",
            options=_other_attrs,
            index=_other_attrs.index(_default_second) if _default_second in _other_attrs else 0,
        )
        inter = compute_intersectional_metrics(
            preproc_data.y_test.values, _y_pred_base_t,
            preproc_data.s_test, _clean_for_attrs.loc[preproc_data.s_test.index, second_attr],
        )
        inter["label"] = f"{protected_col} × {second_attr}"
        st.session_state["fl_intersectional"] = inter
        ic1, ic2, ic3 = st.columns(3)
        ic1.metric("Intersectional DP gap (max−min)", f"{inter['gaps']['demographic_parity_diff']:.4f}")
        ic2.metric(f"'{protected_col}' alone", f"{inter['single_attribute_dp']['attr_a']:.4f}")
        ic3.metric(f"'{second_attr}' alone", f"{inter['single_attribute_dp']['attr_b']:.4f}")
        _it = inter["table"]
        fig_int = px.bar(
            _it[_it["included"]], x="attr_a", y="selection_rate", color="attr_b", barmode="group",
            text_auto=".2f", color_discrete_sequence=PALETTE,
            hover_data={"count": True, "tpr": ":.3f", "fpr": ":.3f"},
            title=f"Selection rate by {protected_col} × {second_attr} (label 1 = {outcome_labels['by_value'][1]})",
        )
        fig_int.update_layout(template="plotly_dark", paper_bgcolor="#0b0f19", plot_bgcolor="#111827", height=380,
                              xaxis_title=protected_col, yaxis_title="Selection Rate", legend_title=second_attr)
        st.plotly_chart(fig_int, use_container_width=True)
        _show = _it.rename(columns={"attr_a": protected_col, "attr_b": second_attr}).drop(columns=["group"])
        st.dataframe(_show.round(4), hide_index=True, use_container_width=True)
    else:
        st.info("No second protected attribute is configured for this dataset.")

    st.markdown("---")
    st.markdown("### Calibration by Group")
    st.caption("Does a predicted risk mean the same thing for every group? A well-calibrated model's points sit on the diagonal. "
               "When base rates differ, calibration and equal error rates cannot both hold — the core of the COMPAS debate.")
    try:
        _y_prob_t = _base_model.predict_proba(_X_base_in)[:, 1]
        calib = compute_group_calibration(preproc_data.y_test.values, _y_prob_t, preproc_data.s_test.values)
        st.session_state["fl_calibration"] = calib
        cc1, cc2 = st.columns([3, 2])
        with cc1:
            fig_cal = go.Figure()
            fig_cal.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect calibration",
                                         line=dict(dash="dash", color="#64748b")))
            for i, (g, cdf) in enumerate(calib["curves"].groupby("group", sort=False)):
                fig_cal.add_trace(go.Scatter(x=cdf["mean_predicted"], y=cdf["observed_rate"], mode="lines+markers",
                                             name=str(g), line=dict(color=PALETTE[i % len(PALETTE)]),
                                             customdata=cdf["count"], hovertemplate="pred %{x:.2f} · observed %{y:.2f} · n=%{customdata}"))
            fig_cal.update_layout(template="plotly_dark", paper_bgcolor="#0b0f19", plot_bgcolor="#111827", height=400,
                                  title=f"Reliability curve: P('{outcome_labels['by_value'][1]}')",
                                  xaxis_title="Mean predicted probability", yaxis_title="Observed rate",
                                  xaxis=dict(range=[0, 1]), yaxis=dict(range=[0, 1]))
            st.plotly_chart(fig_cal, use_container_width=True)
        with cc2:
            st.dataframe(calib["summary"].rename(columns={"brier_score": "Brier", "ece": "ECE"}).round(4),
                         hide_index=True, use_container_width=True)
            st.caption(f"Overall Brier {calib['overall']['brier_score']:.4f} · ECE {calib['overall']['ece']:.4f}. "
                       "Lower is better. Groups with n < 30 omitted.")
    except Exception as e:
        st.info(f"Calibration needs predicted probabilities; unavailable for this model ({e}).")

# ==========================================
# TAB 4: MITIGATION COMPARISON
# ==========================================
with tab4:
    st.markdown("### Multi-Stage Mitigation Strategy Evaluation")
    
    st.markdown(f"""
    <div class="insight-card">
        <div class="insight-title">MITIGATION & ACCURACY TRADEOFF FINDING</div>
        {insights['mitigation_statement']}
    </div>
    """, unsafe_allow_html=True)

    comp_rows = []
    comp_rows.append({
        "Stage": "Baseline",
        "Technique": "Unmitigated Baseline",
        "F1-Score": baseline_pred_metrics['f1_score'],
        "Balanced Acc": baseline_pred_metrics['balanced_accuracy'],
        "ROC-AUC": baseline_pred_metrics['roc_auc'],
        "Demographic Parity Diff": baseline_fairness['demographic_parity_diff']['val'],
        "Equalized Odds Diff": baseline_fairness['equalized_odds_diff']['val'],
        "Significance (p < 0.05)": "Reference Baseline"
    })

    for name, mres in mitigations.items():
        metrics = mres["metrics"]
        pmetrics = mres["pred_metrics"]
        sig = mres["significance"]
        stage = "Pre-processing" if name == "Reweighing" else "In-processing" if name == "Exponentiated Gradient" else "Post-processing"
        
        comp_rows.append({
            "Stage": stage,
            "Technique": name,
            "F1-Score": pmetrics['f1_score'],
            "Balanced Acc": pmetrics['balanced_accuracy'],
            "ROC-AUC": pmetrics['roc_auc'],
            "Demographic Parity Diff": metrics['demographic_parity_diff']['val'],
            "Equalized Odds Diff": metrics['equalized_odds_diff']['val'],
            "Significance (95% CI > 0)": f"{'Significant' if sig['is_significant'] else 'Not Significant'} (Δ={sig.get('mean_delta', 0.0):.4f}, [{sig.get('ci_low', 0.0):.4f}, {sig.get('ci_high', 0.0):.4f}])"
        })

    comp_df = pd.DataFrame(comp_rows)
    
    st.markdown("#### Comparative Mitigation Results Table")
    st.dataframe(comp_df, hide_index=True, use_container_width=True)

    # Epsilon Sweep & Selection Audit Section
    if selected_dataset_key in ["german_credit", "compas", "adult_income"]:
        import json
        from src.config import DATA_DIR
        json_path = DATA_DIR / "verified_benchmark_results.json"
        if json_path.exists():
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    bench_data = json.load(f)
                eps_tables = bench_data.get("epsilon_selection_tables", {})
                if selected_dataset_key in eps_tables:
                    with st.expander("Epsilon Optimization & Selection Audit (Validation Set Rule: Max Fairness Gain with <=5pp Accuracy Loss)"):
                        st.markdown("""
                        **Optimization Rule:** Epsilon ($\\epsilon$) controls the fairness constraint tolerance in Exponentiated Gradient. The optimal $\\epsilon$ is selected to maximize fairness gain (minimize Demographic Parity difference) while keeping model accuracy loss $\\le 5.0$ percentage points vs the unmitigated baseline. If all candidates exceed 5pp loss (as on COMPAS), the deterministic fallback selects the $\\epsilon$ that minimizes accuracy loss.
                        """)
                        eps_df = pd.DataFrame(eps_tables[selected_dataset_key])
                        st.dataframe(eps_df, hide_index=True, use_container_width=True)
            except Exception as e:
                pass

    c1, c2 = st.columns([1, 1])

    with c1:
        st.markdown("#### Demographic Parity Disparity Comparison")
        fig_comp_bar = px.bar(
            comp_df,
            x='Technique',
            y='Demographic Parity Diff',
            color='Technique',
            color_discrete_sequence=PALETTE,
            text_auto='.4f',
            title="Absolute Demographic Parity Gap"
        )
        fig_comp_bar.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0b0f19",
            plot_bgcolor="#111827",
            height=380,
            showlegend=False
        )
        st.plotly_chart(fig_comp_bar, use_container_width=True)

    with c2:
        st.markdown("#### Accuracy vs Fairness Pareto Trade-off")
        fig_scatter = px.scatter(
            comp_df,
            x='Demographic Parity Diff',
            y='F1-Score',
            text='Technique',
            color='Stage',
            color_discrete_sequence=PALETTE,
            size=[18]*len(comp_df),
            title="Pareto Scatter: F1 Performance vs Demographic Parity Gap"
        )
        fig_scatter.update_traces(textposition='top center')
        fig_scatter.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0b0f19",
            plot_bgcolor="#111827",
            height=380
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown("---")
    st.markdown("### Fairness–Accuracy Frontier (Exponentiated Gradient ε-sweep)")
    _sw = sweep_points(benchmarks, selected_dataset_key) if is_benchmark_match else pd.DataFrame()
    if _sw.empty:
        st.caption("The ε-sweep is precomputed for the benchmark configuration only "
                   f"({MODEL_LABELS.get(benchmarks.get('selected_baseline_models', {}).get(selected_dataset_key, ''), 'n/a')} with the default protected attribute).")
    else:
        fig_fr = go.Figure()
        _styles = {"demographic_parity": ("DP-constrained EG", "#3b82f6"), "equalized_odds": ("EO-constrained EG", "#a855f7")}
        for cons, cdf in _sw.groupby("constraint"):
            name, color = _styles.get(cons, (cons, "#94a3b8"))
            fig_fr.add_trace(go.Scatter(x=cdf["dp_diff"], y=cdf["accuracy"], mode="markers", name=name,
                                        customdata=cdf["eps"], marker=dict(size=10, color=color),
                                        hovertemplate=name + "<br>ε=%{customdata}<br>DP %{x:.4f} · acc %{y:.4f}<extra></extra>"))
        _others = [("Baseline", baseline_fairness["demographic_parity_diff"]["val"], baseline_pred_metrics["accuracy"])]
        _others += [(n, m["metrics"]["demographic_parity_diff"]["val"], m["pred_metrics"]["accuracy"])
                    for n, m in mitigations.items() if n != "Exponentiated Gradient"]
        for i, (n, dp, acc) in enumerate(_others):
            fig_fr.add_trace(go.Scatter(x=[dp], y=[acc], mode="markers", name=n,
                                        marker=dict(size=13, symbol="diamond", color=["#f59e0b", "#10b981", "#ef4444"][i % 3]),
                                        hovertemplate=n + "<br>DP %{x:.4f} · acc %{y:.4f}<extra></extra>"))
        _all = pd.concat([_sw[["dp_diff", "accuracy"]], pd.DataFrame([(dp, acc) for _, dp, acc in _others], columns=["dp_diff", "accuracy"])])
        _front = pareto_frontier(_all)
        fig_fr.add_trace(go.Scatter(x=_front["dp_diff"], y=_front["accuracy"], mode="lines", name="Pareto frontier",
                                    line=dict(color="#e2e8f0", dash="dot", shape="hv")))
        _sel = _sw[_sw["selected"]]
        if not _sel.empty:
            fig_fr.add_trace(go.Scatter(x=_sel["dp_diff"], y=_sel["accuracy"], mode="markers", name="Selected ε (validation rule)",
                                        customdata=_sel["eps"], hovertemplate="Selected ε=%{customdata}<extra></extra>",
                                        marker=dict(size=20, symbol="circle-open", line=dict(width=3, color="#facc15"))))
        _xmax = float(max(_all["dp_diff"].max(), 0.01))
        fig_fr.update_xaxes(range=[-0.05 * _xmax, 1.08 * _xmax])
        fig_fr.update_layout(template="plotly_dark", paper_bgcolor="#0b0f19", plot_bgcolor="#111827", height=460,
                             xaxis_title="Demographic Parity Diff (test, lower = fairer)", yaxis_title="Accuracy (test)",
                             title="Every ε tested, plus the other mitigations (seed-42 test split)")
        st.plotly_chart(fig_fr, use_container_width=True)
        st.caption("Points on the dotted line are Pareto-optimal: nothing else is both fairer and more accurate. "
                   "ε is chosen on the validation split, so the selected point need not be on the test-set frontier.")

# ==========================================
# TAB 5: EXPLAINABILITY & RECOURSE
# ==========================================
with tab5:
    st.markdown("### Individual Decision Auditing & Actionable Recourse")
    
    c1, c2 = st.columns([1, 1])
    
    with c1:
        st.markdown("#### Global Feature Importance (SHAP)")
        if not shap_importance.empty:
            fig_shap = px.bar(
                shap_importance.head(10),
                x='importance',
                y='feature',
                orientation='h',
                title="Top 10 Global Features Driving Model Predictions",
                color_discrete_sequence=['#2563eb']
            )
            fig_shap.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0b0f19",
                plot_bgcolor="#111827",
                height=380,
                yaxis=dict(autorange="reversed")
            )
            st.plotly_chart(fig_shap, use_container_width=True)

    with c2:
        st.markdown("#### Select Applicant for Individual Audit")
        X_test = preproc_data.X_test
        y_test = preproc_data.y_test
        explain_model = raw_feature_model(pipeline_results["baseline_model"], selected_model_type, preproc_data.scaler)
        y_pred = explain_model.predict(X_test)
        
        rejected_indices = np.where(y_pred == outcome_labels["unfavorable_value"])[0]
        if len(rejected_indices) == 0:
            rejected_indices = list(range(len(X_test)))

        selected_rec_idx = st.selectbox(
            f"Select {outcome_labels['subject']} Row Index (Test Partition, predicted unfavorable)",
            options=rejected_indices,
            index=0
        )

        record_series = X_test.iloc[selected_rec_idx]
        _dec = int(y_pred[selected_rec_idx])
        st.caption(f"Selected {outcome_labels['subject']} Index: {selected_rec_idx} | Model Baseline Decision: {outcome_labels['by_value'].get(_dec, _dec)} (label {_dec})")

    st.markdown("---")
    
    rec_c1, rec_c2 = st.columns([1, 1])

    with rec_c1:
        st.markdown("#### Local SHAP Feature Contribution Waterfall")
        local_shap = compute_local_shap_explanation(
            explain_model,
            X_test,
            selected_rec_idx,
            preproc_data.feature_names
        )
        
        local_df = local_shap["local_explanation"].head(8)
        fig_local = px.bar(
            local_df,
            x='shap_value',
            y='feature',
            orientation='h',
            color='shap_value',
            color_continuous_scale="RdBu",
            title=f"SHAP contributions to P(label 1) for {outcome_labels['subject']} #{selected_rec_idx}"
        )
        fig_local.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0b0f19",
            plot_bgcolor="#111827",
            height=380,
            yaxis=dict(autorange="reversed")
        )
        st.plotly_chart(fig_local, use_container_width=True)

    with rec_c2:
        st.markdown(f"#### Actionable Counterfactual Recourse (DiCE Search → {outcome_labels['favorable']})")
        with st.spinner("Searching actionable recourse path..."):
            cf_res = generate_counterfactual_explanation(
                explain_model,
                preproc_data.X_train,
                preproc_data.y_train,
                record_series,
                actionable_features=pipeline_results.get("actionable_cols"),
                immutable_features=pipeline_results.get("immutable_cols"),
                desired_class=outcome_labels["favorable_value"]
            )
            
            changes = cf_res.get("actionable_changes", [])
            if len(changes) > 0:
                st.dataframe(pd.DataFrame(changes), hide_index=True, use_container_width=True)
            else:
                st.caption("No simple single-step recourse found within constraints.")

# ==========================================
# TAB 6: AUDIT REPORT & EXPORT
# ==========================================
with tab6:
    st.markdown("### Compliance & Academic Audit Report Export")
    
    primary_mitigation_name = list(mitigations.keys())[0] if mitigations else "Reweighing"
    primary_mitigation_data = mitigations.get(primary_mitigation_name, {})

    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### Executive Summary Preview")
        st.markdown(f"""
        <div class="insight-card">
            {insights['executive_summary']}
        </div>
        """, unsafe_allow_html=True)
        
        pdf_path = generate_pdf_audit_report(
            dataset_name=pipeline_results["dataset_name"],
            protected_attribute=pipeline_results["protected_col"],
            kpis=kpis,
            proxy_df=proxy_df,
            baseline_metrics=baseline_fairness,
            mitigation_metrics=primary_mitigation_data,
            insights=insights,
            technique_name=primary_mitigation_name
        )

        if pdf_path.exists():
            with open(pdf_path, "rb") as f:
                st.download_button(
                    label="Download PDF Audit Report",
                    data=f,
                    file_name=pdf_path.name,
                    mime="application/pdf"
                )

        st.markdown("#### Model Card & Datasheet")
        _eps_sel = benchmarks.get("selected_epsilons", {}).get(selected_dataset_key)
        model_card_md = generate_model_card(
            dataset_key=selected_dataset_key,
            dataset_name=pipeline_results["dataset_name"],
            model_label=model_label,
            protected_col=pipeline_results["protected_col"],
            kpis=kpis,
            baseline_pred_metrics=baseline_pred_metrics,
            baseline_fairness=baseline_fairness,
            mitigations=mitigations,
            split_sizes={"train": len(preproc_data.y_train), "val": len(preproc_data.y_val), "test": len(preproc_data.y_test)},
            feature_count=len(preproc_data.feature_names),
            selected_eps=_eps_sel,
            dedup_info=benchmarks.get("dedup_info", {}).get(selected_dataset_key),
            metrics_source="precomputed benchmark JSON (seed-42 split)" if is_benchmark_match else "live run (seed-42 split)",
            intersectional=st.session_state.get("fl_intersectional"),
            calibration=st.session_state.get("fl_calibration"),
        )
        st.download_button(
            label="Download Model Card + Datasheet (Markdown)",
            data=model_card_md.encode("utf-8"),
            file_name=f"FairLens_Model_Card_{selected_dataset_key}_{selected_model_type}.md",
            mime="text/markdown",
        )
        with st.expander("Preview model card"):
            st.markdown(model_card_md)

    with col2:
        st.markdown("#### SQLite Logged Experiment History")
        exp_db_df = get_experiment_history(selected_dataset_key)
        if not exp_db_df.empty:
            st.dataframe(exp_db_df, hide_index=True, use_container_width=True, height=280)
            
            csv_data = exp_db_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Experiment History CSV",
                data=csv_data,
                file_name=f"FairLens_Experiments_{selected_dataset_key}.csv",
                mime="text/csv"
            )
