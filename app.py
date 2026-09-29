import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime

from src.config import DATASET_CONFIGS, REPORTS_DIR
from src.pipeline import run_fairlens_pipeline
from src.data.loader import validate_custom_csv
from src.explainability.shap_explainer import compute_local_shap_explanation
from src.explainability.dice_explainer import generate_counterfactual_explanation
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
- **Significance Test:** Wilcoxon Signed-Rank Test (p < 0.05)
""")

# Run Main Pipeline Execution wrapper
@st.cache_data(show_spinner=False)
def cached_run_pipeline(dataset_key: str, protected_attr: str, model_type: str):
    return run_fairlens_pipeline(dataset_key, protected_attribute=protected_attr, model_type=model_type)

with st.spinner("Executing FairLens pipeline (cleaning, proxy analysis, model training & statistical audit)..."):
    if selected_dataset_key == "custom":
        pipeline_results = run_fairlens_pipeline(
            "custom",
            custom_df=custom_df,
            custom_target_col=custom_target,
            custom_protected_col=custom_protected,
            model_type=selected_model_type
        )
    else:
        pipeline_results = cached_run_pipeline(selected_dataset_key, selected_protected, selected_model_type)

# Unpack Key Objects from Pipeline
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
        Target: <strong>target</strong> &nbsp;|&nbsp; 
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
            <div class="kpi-label">Total Applicants / Records</div>
            <div class="kpi-val">{kpis['total_records']:,}</div>
            <div class="kpi-subtext">Deduplicated Sample Size</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Base Favorable Rate</div>
            <div class="kpi-val">{kpis['overall_positive_rate']*100:.2f}%</div>
            <div class="kpi-subtext">Historical Base Approval</div>
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
        st.markdown("#### Favorable Outcome (Approval) Rate by Demographic Group")
        disp_df = pipeline_results["disparity_table"]
        if not disp_df.empty:
            fig_bar = px.bar(
                disp_df,
                x='Protected_Group',
                y='Favorable_Rate',
                text_auto='.2f',
                color='Protected_Group',
                color_discrete_sequence=PALETTE,
                title="Historical Approval Rate (%)"
            )
            fig_bar.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0b0f19",
                plot_bgcolor="#111827",
                height=360,
                showlegend=False,
                yaxis_title="Approval Rate (%)"
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
        badge = '<span class="status-badge-red">Significant Disparity</span>' if dp_info['ci_low'] > 0.02 else '<span class="status-badge-green">Acceptable Bound</span>'
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
                title="Predicted Positive Rate (Selection Rate) by Group"
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
            "Significance (p < 0.05)": f"{'Significant' if sig['is_significant'] else 'Not Significant'} (p={sig['p_value']:.4f})"
        })

    comp_df = pd.DataFrame(comp_rows)
    
    st.markdown("#### Comparative Mitigation Results Table")
    st.dataframe(comp_df, hide_index=True, use_container_width=True)

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
            title="Demographic Parity Difference Reduction"
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
        y_pred = pipeline_results["baseline_model"].predict(X_test)
        
        rejected_indices = np.where(y_pred == 0)[0]
        if len(rejected_indices) == 0:
            rejected_indices = list(range(len(X_test)))

        selected_rec_idx = st.selectbox(
            "Select Applicant Row Index (Test Partition)",
            options=rejected_indices,
            index=0
        )

        record_series = X_test.iloc[selected_rec_idx]
        st.caption(f"Selected Applicant Index: {selected_rec_idx} | Model Baseline Decision: REJECTED (0)")

    st.markdown("---")
    
    rec_c1, rec_c2 = st.columns([1, 1])

    with rec_c1:
        st.markdown("#### Local SHAP Feature Contribution Waterfall")
        local_shap = compute_local_shap_explanation(
            pipeline_results["baseline_model"],
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
            title=f"Feature SHAP Contributions for Applicant #{selected_rec_idx}"
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
        st.markdown("#### Actionable Counterfactual Recourse (DiCE Search)")
        with st.spinner("Searching actionable recourse path..."):
            cf_res = generate_counterfactual_explanation(
                pipeline_results["baseline_model"],
                preproc_data.X_train,
                preproc_data.y_train,
                record_series,
                actionable_features=pipeline_results.get("actionable_cols"),
                immutable_features=pipeline_results.get("immutable_cols")
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
