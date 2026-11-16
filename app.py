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
    page_title="FairLens",
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

# ---------------------------------------------------------------- design tokens
import html as _html
import plotly.io as pio

BRAND = "#0f766e"      # single accent colour (teal)
MUTED = "#94a3b8"
PALETTE = [BRAND, "#64748b", "#d97706", "#6366f1", "#be123c", "#0891b2"]
RISK_COLORS = {"High Proxy Risk": "#be123c", "Moderate Proxy Risk": "#d97706", "Low Proxy Risk": MUTED}
DIVERGING = [[0.0, "#b45309"], [0.5, "#f8fafc"], [1.0, BRAND]]
METHOD_COLORS = {"Baseline": "#94a3b8", "Reweighing": "#d97706",
                 "Exponentiated Gradient": BRAND, "Threshold Optimizer": "#6366f1"}
MODEL_LABELS = {"random_forest": "Random Forest", "logistic_regression": "Logistic Regression", "xgboost": "XGBoost"}

pio.templates["fairlens"] = go.layout.Template(layout=go.Layout(
    font=dict(family="IBM Plex Sans, -apple-system, Segoe UI, sans-serif", size=12, color="#334155"),
    paper_bgcolor="white", plot_bgcolor="white", colorway=PALETTE,
    margin=dict(l=8, r=8, t=16, b=8),
    xaxis=dict(gridcolor="#eef1f5", linecolor="#cbd5e1", zeroline=False, ticks="outside", tickcolor="#cbd5e1", title_font_size=12),
    yaxis=dict(gridcolor="#eef1f5", linecolor="#cbd5e1", zeroline=False, title_font_size=12),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font_size=12, title_font_size=12),
    hoverlabel=dict(bgcolor="white", bordercolor="#e2e8f0", font=dict(size=12, color="#0f172a")),
    bargap=0.35,
), data=dict(bar=[go.Bar(textposition="outside", textangle=0, cliponaxis=False, marker_line_width=0)]))
pio.templates.default = "plotly_white+fairlens"


def section(title: str, subtitle: Optional[str] = None, level: int = 3) -> None:
    tag = "h3" if level == 3 else "h4"
    sub = f'<p class="fl-sub">{_html.escape(subtitle)}</p>' if subtitle else ""
    st.markdown(f'<{tag} class="fl-{tag}">{_html.escape(title)}</{tag}>{sub}', unsafe_allow_html=True)


def finding(text: str) -> None:
    st.markdown(f'<div class="fl-note"><div class="fl-note-label">What this shows</div>{text}</div>',
                unsafe_allow_html=True)


def cards(items: list) -> None:
    """Row of metric cards: label, value, optional note and status=(kind, text) with kind in good/warn/bad/neutral."""
    cells = []
    for it in items:
        status = it.get("status")
        status_html = (f'<div class="fl-status fl-{status[0]}"><span class="fl-dot"></span>{_html.escape(status[1])}</div>'
                       if status else "")
        note_html = f'<div class="fl-card-note">{_html.escape(it["note"])}</div>' if it.get("note") else ""
        size = " fl-card-value-sm" if it.get("small") else ""
        cells.append(f'<div class="fl-card"><div class="fl-card-label">{_html.escape(it["label"])}</div>'
                     f'<div class="fl-card-value{size}">{_html.escape(str(it["value"]))}</div>{note_html}{status_html}</div>')
    st.markdown(f'<div class="fl-cards" style="grid-template-columns: repeat({len(items)}, minmax(0, 1fr));">'
                + "".join(cells) + "</div>", unsafe_allow_html=True)


def divider() -> None:
    st.markdown('<hr class="fl-hr">', unsafe_allow_html=True)


def show(fig) -> None:
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

# Sidebar Configuration Controls
st.sidebar.markdown('<div class="fl-side-h fl-side-first">Audit setup</div>', unsafe_allow_html=True)

dataset_options = {
    "german_credit": "German Credit",
    "compas": "COMPAS recidivism",
    "adult_income": "Adult Income",
    "custom": "Upload a CSV…"
}

selected_dataset_key = st.sidebar.selectbox(
    "Dataset",
    options=list(dataset_options.keys()),
    format_func=lambda x: dataset_options[x]
)

custom_df = None
custom_target = None
custom_protected = None

if selected_dataset_key == "custom":
    uploaded_file = st.sidebar.file_uploader("CSV file", type=["csv"])
    
    if uploaded_file is not None:
        try:
            custom_df = pd.read_csv(uploaded_file)
            st.sidebar.caption(f"{len(custom_df):,} rows loaded")
            
            cols = list(custom_df.columns)
            custom_target = st.sidebar.selectbox("Outcome column", options=cols)
            custom_protected = st.sidebar.selectbox("Protected attribute", options=[c for c in cols if c != custom_target])
            
            valid, msg = validate_custom_csv(custom_df, custom_target, custom_protected)
            if not valid:
                st.sidebar.error(msg)
                st.stop()
        except Exception as e:
            st.sidebar.error(f"Could not read this CSV: {e}")
            st.stop()
    else:
        st.info("Upload a CSV in the sidebar to start an audit.")
        st.stop()

if selected_dataset_key != "custom":
    cfg = DATASET_CONFIGS[selected_dataset_key]
    protected_attrs = list(cfg["protected_attributes"].keys())
    default_prot = cfg["default_protected_attribute"]
    
    selected_protected = st.sidebar.selectbox(
        "Protected attribute",
        options=protected_attrs,
        index=protected_attrs.index(default_prot) if default_prot in protected_attrs else 0
    )
else:
    selected_protected = custom_protected

selected_model_type = st.sidebar.selectbox(
    "Model",
    options=["random_forest", "logistic_regression", "xgboost"],
    format_func=lambda x: MODEL_LABELS[x]
)

st.sidebar.markdown(
    '<div class="fl-side-method"><div class="fl-side-h">Method</div>'
    '<p>Stratified 60 / 20 / 20 train, validation and test split.</p>'
    '<p>95% bootstrap intervals from 500 resamples; mitigation effects tested with a paired bootstrap.</p>'
    '<p>Exponentiated Gradient tolerance chosen on the validation split.</p></div>',
    unsafe_allow_html=True)

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

def _model_matches_features(model: Any, feature_names: tuple) -> bool:
    """True when a saved model was fitted on exactly the feature set the dashboard feeds it."""
    names = getattr(model, "feature_names_in_", None)
    if names is None and hasattr(model, "get_booster"):
        names = model.get_booster().feature_names
    if names is not None:
        return list(names) == list(feature_names)
    n_in = getattr(model, "n_features_in_", None)
    return n_in is None or n_in == len(feature_names)


@st.cache_resource(show_spinner=False)
def get_cached_baseline_model(dataset_key: str, model_type: str, feature_names: tuple = ()) -> Any:
    """Loads the saved baseline classifier, or refits it (without touching disk) if the file is
    missing, unreadable (e.g. written by a different xgboost version) or fitted on other features."""
    for cname in (f"{dataset_key}_{model_type}_baseline.joblib", f"default_{model_type}_baseline.joblib"):
        p = MODELS_DIR / cname
        if p.exists():
            try:
                model = load_cached_joblib_file(str(p))
            except Exception:
                continue
            if not feature_names or _model_matches_features(model, feature_names):
                return model
    cfg = DATASET_CONFIGS[dataset_key]
    data = get_cached_preprocessed_data(dataset_key, cfg["target_column"], cfg["default_protected_attribute"])
    model, _ = train_model(data, model_type=model_type, dataset_key=dataset_key, random_state=42, save=False)
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
                try:
                    return load_cached_joblib_file(str(p))
                except Exception:
                    continue
    elif technique == "Threshold Optimizer":
        p = MODELS_DIR / f"{dataset_key}_{model_type}_threshold_optimizer.joblib"
        if p.exists():
            try:
                return load_cached_joblib_file(str(p))
            except Exception:
                return None
    return None

@st.cache_data(show_spinner=False)
def get_cached_global_shap(dataset_key: str, model_type: str) -> pd.DataFrame:
    """Computes global SHAP feature importance once and caches result in memory."""
    cfg = DATASET_CONFIGS[dataset_key]
    data = get_cached_preprocessed_data(dataset_key, cfg["target_column"], cfg["default_protected_attribute"])
    base_model = get_cached_baseline_model(dataset_key, model_type, tuple(data.X_test.columns))
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

    benchmarks = load_precomputed_benchmarks()
    exp_bench = benchmarks.get("experiments", {}).get(selected_dataset_key, {})

    # Check if pre-computed metrics exist for this benchmark
    is_benchmark_match = uses_precomputed_benchmark(
        benchmarks, selected_dataset_key, protected_col, selected_model_type, cfg["default_protected_attribute"]
    )

    if is_benchmark_match:
        baseline_model = get_cached_baseline_model(selected_dataset_key, selected_model_type, tuple(preproc_data.X_test.columns))
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

# ==============================================================================
# PRESENTATION
# ==============================================================================
model_label = MODEL_LABELS.get(selected_model_type, selected_model_type)
subject = outcome_labels["subject"].lower()
fav_rate_name = outcome_labels["favorable_rate"]
label1_name = outcome_labels["by_value"][1]
prot = pipeline_results["protected_col"]

st.markdown(f"""
<div class="fl-top">
  <div class="fl-brand"><span class="fl-mark"></span>FairLens</div>
  <div class="fl-context">
    <span>{pipeline_results['dataset_name']}</span><span class="fl-sep">/</span>
    <span>{prot}</span><span class="fl-sep">/</span>
    <span>{model_label}</span>
  </div>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["Overview", "Proxies", "Fairness", "Mitigation", "Explanations", "Report"]
)

# ------------------------------------------------------------------ Overview
with tab1:
    section("Dataset overview", f"Outcomes in the cleaned data, before any model is trained.")

    disp_val = kpis['base_rate_disparity'] * 100
    gap_status = ("bad", "Large gap") if disp_val > 10 else ("warn", "Moderate gap") if disp_val > 5 else ("good", "Small gap")
    cards([
        dict(label="Records", value=f"{kpis['total_records']:,}", note="after removing duplicates"),
        dict(label=fav_rate_name, value=f"{kpis['overall_positive_rate']*100:.1f}%", note=f"favorable outcome: {outcome_labels['favorable'].lower()}"),
        dict(label=f"Gap across {prot} groups", value=f"{disp_val:.1f} pts", status=gap_status),
        dict(label=f"Groups in {prot}", value=f"{kpis['protected_groups_count']}", note="distinct values"),
    ])
    finding(insights['base_statement'])

    disp_df = pipeline_results["disparity_table"]
    c1, c2 = st.columns(2, gap="large")
    with c1:
        section(f"{fav_rate_name} by {prot}", level=4)
        if not disp_df.empty:
            fig = px.bar(disp_df, x='Protected_Group', y='Favorable_Rate', text_auto='.1f',
                         color='Protected_Group', color_discrete_sequence=PALETTE)
            fig.update_layout(height=320, showlegend=False, xaxis_title=None, yaxis_title="%")
            show(fig)
    with c2:
        section(f"Group sizes", level=4)
        if not disp_df.empty:
            fig = px.bar(disp_df, x='Protected_Group', y='Total_Applicants', text_auto=',d',
                         color='Protected_Group', color_discrete_sequence=PALETTE)
            fig.update_layout(height=320, showlegend=False, xaxis_title=None, yaxis_title=f"{outcome_labels['subject']}s")
            show(fig)

    section("Correlation between numeric features", level=4)
    corr_df = pipeline_results["corr_matrix"]
    if not corr_df.empty:
        fig = px.imshow(corr_df, text_auto=".2f", aspect="auto", zmin=-1, zmax=1, color_continuous_scale=DIVERGING)
        fig.update_layout(height=420, coloraxis_colorbar=dict(thickness=10, len=0.8, title=None))
        show(fig)

# ------------------------------------------------------------------ Proxies
with tab2:
    section("Proxy features",
            f"The model never sees '{prot}', but other features can still encode it. "
            f"Higher mutual information means a feature reveals more about {prot}.")
    finding(insights['proxy_statement'])

    c1, c2 = st.columns([3, 2], gap="large")
    with c1:
        if not proxy_df.empty:
            fig = px.bar(proxy_df.head(10), x='mutual_information', y='feature', orientation='h',
                         color='risk_level', color_discrete_map=RISK_COLORS)
            fig.update_layout(height=400, xaxis_title=f"Mutual information with {prot}", yaxis_title=None,
                              yaxis=dict(autorange="reversed"), legend_title_text=None)
            show(fig)
    with c2:
        if not proxy_df.empty:
            st.dataframe(
                proxy_df[['feature', 'mutual_information', 'correlation', 'risk_level']],
                column_config={
                    "feature": "Feature",
                    "mutual_information": st.column_config.NumberColumn("Mutual info.", format="%.4f"),
                    "correlation": st.column_config.NumberColumn("Correlation", format="%.3f"),
                    "risk_level": "Risk",
                },
                hide_index=True, width="stretch", height=400,
            )

# ------------------------------------------------------------------ Fairness
with tab3:
    section(f"Baseline fairness · {model_label}",
            "Gaps between groups on the test set, with 95% bootstrap intervals (500 resamples).")
    finding(insights['audit_statement'])

    dp_info = baseline_fairness["demographic_parity_diff"]
    eo_info = baseline_fairness["equalized_odds_diff"]
    eopp_info = baseline_fairness["equal_opportunity_diff"]
    pp_info = baseline_fairness["predictive_parity_diff"]

    def _sig(info):
        if "is_significant" not in info:
            return None
        return ("bad", "Significant") if bool(info["is_significant"]) else ("neutral", "Not significant")

    def _ci(info):
        if "signed_ci_low" in info:
            return f"signed 95% CI {info['signed_ci_low']:.3f} to {info['signed_ci_high']:.3f}"
        return f"95% CI {info['ci_low']:.3f} to {info['ci_high']:.3f}"

    cards([
        dict(label="Demographic parity gap", value=f"{dp_info['val']:.3f}", note=_ci(dp_info), status=_sig(dp_info)),
        dict(label="Equalized odds gap", value=f"{eo_info['val']:.3f}", note=_ci(eo_info)),
        dict(label="Equal opportunity gap", value=f"{eopp_info['val']:.3f}", note=_ci(eopp_info), status=_sig(eopp_info)),
        dict(label="Predictive parity gap", value=f"{pp_info['val']:.3f}", note=_ci(pp_info), status=_sig(pp_info)),
    ])
    st.caption("Gaps are absolute differences between groups. Significance uses the signed difference, "
               "because an interval on an absolute value almost never contains zero.")

    c1, c2 = st.columns(2, gap="large")
    with c1:
        section("Model performance", level=4)
        perf_df = pd.DataFrame([
            ("Accuracy", baseline_pred_metrics.get('accuracy', 0.0)),
            ("Balanced accuracy", baseline_pred_metrics.get('balanced_accuracy', 0.0)),
            ("ROC-AUC", baseline_pred_metrics.get('roc_auc', 0.0)),
            (f"F1 · {outcome_labels['by_value'][1]}", baseline_pred_metrics.get('f1_class_1', baseline_pred_metrics.get('f1_score', 0.0))),
            (f"Precision · {outcome_labels['by_value'][1]}", baseline_pred_metrics.get('precision_class_1', baseline_pred_metrics.get('precision', 0.0))),
            (f"Recall · {outcome_labels['by_value'][1]}", baseline_pred_metrics.get('recall_class_1', baseline_pred_metrics.get('recall', 0.0))),
            (f"F1 · {outcome_labels['by_value'][0]}", baseline_pred_metrics.get('f1_class_0', 0.0)),
            (f"Precision · {outcome_labels['by_value'][0]}", baseline_pred_metrics.get('precision_class_0', 0.0)),
            (f"Recall · {outcome_labels['by_value'][0]}", baseline_pred_metrics.get('recall_class_0', 0.0)),
        ], columns=["Metric", "Value"])
        st.dataframe(perf_df, hide_index=True, width="stretch",
                     column_config={"Value": st.column_config.NumberColumn(format="%.3f")})
    with c2:
        section("Selection rate by group", level=4)
        st.caption(f"Share of each group the model predicts as '{label1_name}'.")
        gr_df = pd.DataFrame(dp_info.get("group_rates", []))
        if not gr_df.empty:
            fig = px.bar(gr_df, x='group', y='selection_rate', text_auto='.3f',
                         color='group', color_discrete_sequence=PALETTE)
            fig.update_layout(height=300, showlegend=False, xaxis_title=None, yaxis_title="Selection rate")
            show(fig)

    # Shared baseline test-set predictions for the extra audits
    _X_base_in = preproc_data.X_test_scaled if selected_model_type == "logistic_regression" else preproc_data.X_test
    _base_model = pipeline_results["baseline_model"]
    _y_pred_base_t = np.asarray(_base_model.predict(_X_base_in))
    _clean_for_attrs = pipeline_results.get("clean_df")
    if _clean_for_attrs is None and selected_dataset_key != "custom":
        _clean_for_attrs = load_cached_clean_dataset(selected_dataset_key)

    divider()
    section("Intersectional audit",
            "Some gaps only show up for combinations of attributes. Groups with fewer than 30 people in the test set are listed but left out of the gap.")
    _cfg_attrs = list(DATASET_CONFIGS.get(selected_dataset_key, {}).get("protected_attributes", {}).keys())
    _other_attrs = [a for a in _cfg_attrs if a != protected_col and _clean_for_attrs is not None and a in _clean_for_attrs.columns]
    if _other_attrs:
        _default_second = {"compas": "sex", "adult_income": "age_group", "german_credit": "age_group"}.get(selected_dataset_key)
        sc1, _ = st.columns([1, 2])
        with sc1:
            second_attr = st.selectbox(
                f"Combine {protected_col} with",
                options=_other_attrs,
                index=_other_attrs.index(_default_second) if _default_second in _other_attrs else 0,
            )
        inter = compute_intersectional_metrics(
            preproc_data.y_test.values, _y_pred_base_t,
            preproc_data.s_test, _clean_for_attrs.loc[preproc_data.s_test.index, second_attr],
        )
        inter["label"] = f"{protected_col} × {second_attr}"
        st.session_state["fl_intersectional"] = inter
        cards([
            dict(label=f"{protected_col} × {second_attr}", value=f"{inter['gaps']['demographic_parity_diff']:.3f}", note="largest selection-rate gap"),
            dict(label=f"{protected_col} alone", value=f"{inter['single_attribute_dp']['attr_a']:.3f}", note="same rows"),
            dict(label=f"{second_attr} alone", value=f"{inter['single_attribute_dp']['attr_b']:.3f}", note="same rows"),
        ])
        _it = inter["table"]
        c1, c2 = st.columns([3, 2], gap="large")
        with c1:
            fig = px.bar(_it[_it["included"]], x="attr_a", y="selection_rate", color="attr_b", barmode="group",
                         text_auto=".2f", color_discrete_sequence=PALETTE,
                         hover_data={"count": True, "tpr": ":.3f", "fpr": ":.3f"})
            fig.update_layout(height=340, xaxis_title=None, yaxis_title="Selection rate",
                              legend=dict(title=dict(text=f"{second_attr}  ", side="left")))
            show(fig)
        with c2:
            _show = _it.rename(columns={"attr_a": protected_col, "attr_b": second_attr, "count": "n",
                                        "selection_rate": "Selection", "tpr": "TPR", "fpr": "FPR",
                                        "precision": "Precision", "included": "In gap"}).drop(columns=["group"])
            st.dataframe(_show.round(3), hide_index=True, width="stretch")
    else:
        st.caption("No second protected attribute is configured for this dataset.")

    divider()
    section("Calibration by group",
            f"Does a predicted probability of '{label1_name}' mean the same thing for every group? "
            "Well-calibrated groups sit on the diagonal. When base rates differ, calibration and equal error rates cannot both hold.")
    try:
        _y_prob_t = _base_model.predict_proba(_X_base_in)[:, 1]
        calib = compute_group_calibration(preproc_data.y_test.values, _y_prob_t, preproc_data.s_test.values)
        st.session_state["fl_calibration"] = calib
        cc1, cc2 = st.columns([3, 2], gap="large")
        with cc1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfectly calibrated",
                                     line=dict(dash="dot", color=MUTED, width=1)))
            for i, (g, cdf) in enumerate(calib["curves"].groupby("group", sort=False)):
                fig.add_trace(go.Scatter(x=cdf["mean_predicted"], y=cdf["observed_rate"], mode="lines+markers",
                                         name=str(g), line=dict(color=PALETTE[i % len(PALETTE)], width=2),
                                         marker=dict(size=6), customdata=cdf["count"],
                                         hovertemplate="predicted %{x:.2f} · observed %{y:.2f} · n=%{customdata}<extra>" + str(g) + "</extra>"))
            fig.update_layout(height=380, xaxis_title="Mean predicted probability", yaxis_title="Observed rate",
                              xaxis=dict(range=[0, 1]), yaxis=dict(range=[0, 1]))
            show(fig)
        with cc2:
            st.dataframe(calib["summary"].rename(columns={"group": "Group", "count": "n", "brier_score": "Brier", "ece": "ECE",
                                                          "mean_predicted": "Mean predicted", "observed_rate": "Observed"}).round(3),
                         hide_index=True, width="stretch")
            st.caption(f"Overall Brier {calib['overall']['brier_score']:.3f}, ECE {calib['overall']['ece']:.3f}. "
                       "Lower is better. Groups with fewer than 30 people are omitted.")
    except Exception as e:
        st.caption(f"Calibration needs predicted probabilities, which this model does not provide ({e}).")

# ------------------------------------------------------------------ Mitigation
with tab4:
    section("Mitigation",
            "Three ways to reduce the demographic parity gap: reweighting the training data, constraining training, "
            "and adjusting decision thresholds per group.")
    finding(insights['mitigation_statement'])

    comp_rows = [{
        "Method": "Baseline",
        "F1": baseline_pred_metrics['f1_score'],
        "Balanced acc.": baseline_pred_metrics['balanced_accuracy'],
        "ROC-AUC": baseline_pred_metrics['roc_auc'],
        "Parity gap": baseline_fairness['demographic_parity_diff']['val'],
        "Odds gap": baseline_fairness['equalized_odds_diff']['val'],
        "Reduction (95% CI)": "—",
    }]
    for name, mres in mitigations.items():
        sig = mres["significance"]
        stage = "Pre-processing" if name == "Reweighing" else "In-processing" if name == "Exponentiated Gradient" else "Post-processing"
        mark = "  ✓" if sig['is_significant'] else ""
        comp_rows.append({
            "Method": name,
            "Stage": stage,
            "F1": mres["pred_metrics"]['f1_score'],
            "Balanced acc.": mres["pred_metrics"]['balanced_accuracy'],
            "ROC-AUC": mres["pred_metrics"]['roc_auc'],
            "Parity gap": mres["metrics"]['demographic_parity_diff']['val'],
            "Odds gap": mres["metrics"]['equalized_odds_diff']['val'],
            "Reduction (95% CI)": f"{sig.get('mean_delta', 0.0):+.3f}  ({sig.get('ci_low', 0.0):.3f} to {sig.get('ci_high', 0.0):.3f}){mark}",
        })
    comp_rows[0]["Stage"] = "—"
    comp_df = pd.DataFrame(comp_rows)[["Method", "Stage", "F1", "Balanced acc.", "ROC-AUC", "Parity gap", "Odds gap", "Reduction (95% CI)"]]
    st.dataframe(comp_df, hide_index=True, width="stretch",
                 column_config={c: st.column_config.NumberColumn(format="%.3f")
                                for c in ["F1", "Balanced acc.", "ROC-AUC", "Parity gap", "Odds gap"]})
    st.caption("Reduction = baseline parity gap minus mitigated gap, from a paired bootstrap on the same resampled rows. "
               "✓ means the interval is entirely above zero.")

    eps_tables = benchmarks.get("epsilon_selection_tables", {})
    if is_benchmark_match and selected_dataset_key in eps_tables:
        with st.expander("How ε was chosen for Exponentiated Gradient"):
            st.markdown(
                "ε is the tolerance on the fairness constraint. On the **validation** split we pick the ε with the "
                "smallest parity gap among those that lose at most 5 accuracy points against the baseline. If none "
                "qualifies, we take the one that loses the least accuracy. The test split is only used for reporting."
            )
            st.dataframe(pd.DataFrame(eps_tables[selected_dataset_key]), hide_index=True, width="stretch")

    c1, c2 = st.columns(2, gap="large")
    with c1:
        section("Parity gap by method", level=4)
        fig = px.bar(comp_df, x='Method', y='Parity gap', text_auto='.3f',
                     color='Method', color_discrete_map=METHOD_COLORS)
        fig.update_layout(height=340, showlegend=False, xaxis_title=None, yaxis_title="Demographic parity gap")
        show(fig)
    with c2:
        section("F1 against parity gap", level=4)
        fig = px.scatter(comp_df, x='Parity gap', y='F1', color='Method',
                         color_discrete_map=METHOD_COLORS)
        fig.update_traces(marker=dict(size=13, line=dict(width=1, color="white")))
        fig.update_layout(height=340, legend_title_text=None, xaxis_title="Demographic parity gap (lower is fairer)")
        show(fig)

    divider()
    section("Fairness–accuracy frontier",
            "Every ε tried for Exponentiated Gradient, next to the other methods. Points on the dotted line are "
            "Pareto-optimal: nothing else is both fairer and more accurate.")
    _sw = sweep_points(benchmarks, selected_dataset_key) if is_benchmark_match else pd.DataFrame()
    if _sw.empty:
        st.caption("The ε sweep is precomputed for the benchmark setup only "
                   f"({MODEL_LABELS.get(benchmarks.get('selected_baseline_models', {}).get(selected_dataset_key, ''), 'n/a')} "
                   "with the default protected attribute). Switch to that setup to see it.")
    else:
        fig = go.Figure()
        _styles = {"demographic_parity": ("EG · parity constraint", BRAND), "equalized_odds": ("EG · odds constraint", "#5eb8ad")}
        for cons, cdf in _sw.groupby("constraint"):
            name, color = _styles.get(cons, (cons, MUTED))
            fig.add_trace(go.Scatter(x=cdf["dp_diff"], y=cdf["accuracy"], mode="markers", name=name,
                                     customdata=cdf["eps"], marker=dict(size=9, color=color, opacity=0.85),
                                     hovertemplate=name + "<br>ε=%{customdata}<br>gap %{x:.4f} · accuracy %{y:.4f}<extra></extra>"))
        _others = [("Baseline", baseline_fairness["demographic_parity_diff"]["val"], baseline_pred_metrics["accuracy"])]
        _others += [(n, m["metrics"]["demographic_parity_diff"]["val"], m["pred_metrics"]["accuracy"])
                    for n, m in mitigations.items() if n != "Exponentiated Gradient"]
        _ocolors = {**METHOD_COLORS, "Baseline": "#475569"}
        for n, dp, acc in _others:
            fig.add_trace(go.Scatter(x=[dp], y=[acc], mode="markers", name=n,
                                     marker=dict(size=12, symbol="diamond", color=_ocolors.get(n, MUTED), line=dict(width=1, color="white")),
                                     hovertemplate=n + "<br>gap %{x:.4f} · accuracy %{y:.4f}<extra></extra>"))
        _all = pd.concat([_sw[["dp_diff", "accuracy"]], pd.DataFrame([(dp, acc) for _, dp, acc in _others], columns=["dp_diff", "accuracy"])])
        _front = pareto_frontier(_all)
        fig.add_trace(go.Scatter(x=_front["dp_diff"], y=_front["accuracy"], mode="lines", name="Pareto frontier",
                                 line=dict(color="#94a3b8", dash="dot", shape="hv", width=1.5), hoverinfo="skip"))
        _sel = _sw[_sw["selected"]]
        if not _sel.empty:
            fig.add_trace(go.Scatter(x=_sel["dp_diff"], y=_sel["accuracy"], mode="markers", name="Chosen ε",
                                     customdata=_sel["eps"], hovertemplate="chosen ε=%{customdata}<extra></extra>",
                                     marker=dict(size=20, symbol="circle-open", line=dict(width=2, color="#0f172a"))))
        _xmax = float(max(_all["dp_diff"].max(), 0.01))
        fig.update_xaxes(range=[-0.05 * _xmax, 1.08 * _xmax])
        fig.update_layout(height=440, xaxis_title="Demographic parity gap (test, lower is fairer)", yaxis_title="Accuracy (test)")
        show(fig)
        st.caption("ε is chosen on the validation split, so the chosen point need not sit on the test-set frontier.")

# ------------------------------------------------------------------ Explanations
with tab5:
    section("Explanations", f"Which features drive the model, and what would change an individual {subject}'s outcome.")

    X_test = preproc_data.X_test
    explain_model = raw_feature_model(pipeline_results["baseline_model"], selected_model_type, preproc_data.scaler)
    y_pred = explain_model.predict(X_test)

    c1, c2 = st.columns([3, 2], gap="large")
    with c1:
        section("Most influential features", level=4)
        st.caption("Mean absolute SHAP value across the test set.")
        if not shap_importance.empty:
            fig = px.bar(shap_importance.head(10), x='importance', y='feature', orientation='h',
                         color_discrete_sequence=[BRAND])
            fig.update_layout(height=360, xaxis_title="Mean |SHAP|", yaxis_title=None, yaxis=dict(autorange="reversed"))
            show(fig)
    with c2:
        section(f"Choose {'an' if subject[:1] in 'aeiou' else 'a'} {subject}", level=4)
        st.caption(f"Only {subject}s the model predicts as '{outcome_labels['unfavorable']}' are listed.")
        rejected_indices = np.where(y_pred == outcome_labels["unfavorable_value"])[0]
        if len(rejected_indices) == 0:
            rejected_indices = list(range(len(X_test)))
        selected_rec_idx = st.selectbox(f"{outcome_labels['subject']} (test-set row)", options=rejected_indices, index=0)
        record_series = X_test.iloc[selected_rec_idx]
        _dec = int(y_pred[selected_rec_idx])
        cards([dict(label="Model decision", value=outcome_labels['by_value'].get(_dec, str(_dec)),
                    note=f"row {selected_rec_idx} · label {_dec}", small=True)])

    divider()
    rc1, rc2 = st.columns(2, gap="large")
    with rc1:
        section("Why this prediction", level=4)
        st.caption(f"Contribution of each feature to the probability of '{label1_name}'.")
        local_shap = compute_local_shap_explanation(explain_model, X_test, selected_rec_idx, preproc_data.feature_names)
        local_df = local_shap["local_explanation"].head(8).copy()
        local_df["direction"] = np.where(local_df["shap_value"] >= 0, f"towards '{label1_name}'", "away")
        fig = px.bar(local_df, x='shap_value', y='feature', orientation='h', color="direction",
                     color_discrete_map={f"towards '{label1_name}'": BRAND, "away": "#d97706"})
        fig.update_layout(height=340, xaxis_title="SHAP value", yaxis_title=None,
                          yaxis=dict(autorange="reversed"), legend_title_text=None)
        show(fig)
    with rc2:
        section(f"What would change the outcome", level=4)
        st.caption(f"Smallest changes to actionable features that lead to '{outcome_labels['favorable']}' (DiCE). "
                   "Protected and immutable attributes are held fixed.")
        with st.spinner("Searching for counterfactuals…"):
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
            _cf = pd.DataFrame(changes).rename(columns={"feature": "Feature", "original_value": "Now",
                                                        "suggested_value": "Change to", "delta": "Difference"})
            st.dataframe(_cf, hide_index=True, width="stretch")
        else:
            st.caption("No simple change within the allowed features flips this decision.")

# ------------------------------------------------------------------ Report
with tab6:
    section("Report", "Download the audit for this dataset and model, or review past runs.")

    primary_mitigation_name = list(mitigations.keys())[0] if mitigations else "Reweighing"
    primary_mitigation_data = mitigations.get(primary_mitigation_name, {})

    col1, col2 = st.columns([3, 2], gap="large")
    with col1:
        section("Summary", level=4)
        st.markdown("<div class='fl-prose'>" + "".join(
            f"<p>{insights[k]}</p>" for k in ["base_statement", "proxy_statement", "audit_statement", "mitigation_statement"]
        ) + "</div>", unsafe_allow_html=True)

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

        b1, b2 = st.columns(2)
        with b1:
            if pdf_path.exists():
                with open(pdf_path, "rb") as f:
                    st.download_button("Download PDF report", data=f, file_name=pdf_path.name,
                                       mime="application/pdf", type="primary", width="stretch")
        with b2:
            st.download_button("Download model card", data=model_card_md.encode("utf-8"),
                               file_name=f"FairLens_Model_Card_{selected_dataset_key}_{selected_model_type}.md",
                               mime="text/markdown", width="stretch")
        with st.expander("Preview model card and datasheet"):
            st.markdown(model_card_md)

    with col2:
        section("Run history", level=4)
        st.caption("Every audit is logged to a local SQLite database.")
        exp_db_df = get_experiment_history(selected_dataset_key)
        if not exp_db_df.empty:
            _hist = exp_db_df.rename(columns={"timestamp": "When", "model_type": "Model", "mitigation_technique": "Method",
                                              "f1_score": "F1", "demographic_parity_diff": "Parity gap",
                                              "equalized_odds_diff": "Odds gap", "p_value": "p"})
            _hist["Model"] = _hist["Model"].map(lambda m: MODEL_LABELS.get(m, m))
            _hist["Method"] = _hist["Method"].replace({"baseline": "Baseline"})
            _hist["When"] = pd.to_datetime(_hist["When"], errors="coerce").dt.strftime("%d %b %H:%M")
            st.dataframe(_hist[["When", "Model", "Method", "F1", "Parity gap", "Odds gap", "p"]], hide_index=True,
                         width="stretch", height=300,
                         column_config={c: st.column_config.NumberColumn(format="%.3f") for c in ["F1", "Parity gap", "Odds gap", "p"]})
            st.download_button("Download history (CSV)", data=exp_db_df.to_csv(index=False).encode('utf-8'),
                               file_name=f"FairLens_Experiments_{selected_dataset_key}.csv", mime="text/csv")
        else:
            st.caption("No runs logged yet.")
