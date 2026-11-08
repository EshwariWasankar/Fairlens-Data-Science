# ⚖️ FairLens: Generalized Algorithmic Bias Auditing, Explainability & Mitigation Framework

**FairLens** is a dataset-agnostic Data Science & Machine Learning pipeline designed to systematically audit tabular classification models for demographic bias, quantify indirect proxy leakage via Mutual Information, compare multi-stage fairness mitigations with statistical significance validation, and provide dual global/local explainability (SHAP + DiCE Counterfactuals).

---

## 🎯 Key Features & Methodological Rigor

1. **Dataset-Agnostic Core Architecture**: Bundles 3 independent benchmark datasets (**COMPAS Recidivism**, **German Credit Data**, **Adult Census Income**) and supports **Custom CSV Upload**.
2. **Indirect Proxy Feature Leakage Quantification**: Computes Mutual Information (MI) and Point-Biserial Correlations between non-protected features and demographic attributes to reveal hidden proxy signals (e.g., zip code or housing status acting as proxies for race/sex).
3. **Statistical Validation (No Fluff)**:
   - **500-Resample Bootstrap 95% Confidence Intervals** around all 4 fairness metrics (*Demographic Parity*, *Equalized Odds*, *Equal Opportunity*, *Predictive Parity*).
   - **Paired Hypothesis Significance Testing**: 95% Paired Bootstrap Confidence Intervals of the difference ($\Delta = \text{Baseline} - \text{Mitigated}$) across 1,000 resamples, certifying statistical significance when the interval strictly excludes zero.
4. **Multi-Stage Mitigation Comparison**:
   - **Pre-Processing:** Reweighing (Sample Weighting)
   - **In-Processing:** Exponentiated Gradient (Fairness Constrained Optimization with $\epsilon$ selection)
   - **Post-Processing:** Threshold Optimizer (Group-wise Decision Threshold Tuning)
5. **Dual Explainability Paradigm**:
   - **SHAP (Shapley Additive exPlanations):** Global feature reliance bar charts & local waterfall force plots for individual decision auditing.
   - **DiCE Actionable Counterfactual Recourse:** Generates actionable recourse paths for rejected applicants while strictly preserving immutable attributes (e.g. sex, race, age).
6. **Automated Compliance & Report Exporting**: One-click generation of PDF & HTML audit reports aligned with RBI Fair Practices, NITI Aayog Responsible AI guidelines, and India's DPDP Act (2023).

---

## 📊 Verified Benchmark Results (Single Source of Truth)

Metrics evaluated over 10 random train/test split seeds ($42 \dots 51$) with 95% paired bootstrap confidence intervals of the difference ($\Delta = \text{Baseline} - \text{Mitigated}$):

| Dataset | Selected Model | Variant | F1 (10-seed mean) | Accuracy (10-seed mean) | DP Diff | EO Diff | EOpp Diff | PP Diff | Paired $\Delta$ DP Reduction | 95% CI ($\Delta$ DP) | Significant (CI > 0) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **German Credit** | Random Forest | Baseline | 0.8287 ± 0.0150 | 72.75% ± 2.24% | 0.0324 ± 0.0214 | 0.0739 ± 0.0301 | 0.0275 ± 0.0204 | 0.0802 ± 0.0168 | Reference | Reference | Reference |
| German Credit | Random Forest | Reweighing | 0.8310 ± 0.0098 | 72.95% ± 1.52% | 0.0435 ± 0.0295 | 0.0873 ± 0.0252 | 0.0425 ± 0.0302 | 0.0695 ± 0.0209 | +0.0003 | [-0.0801, 0.0916] | False |
| German Credit | Random Forest | Exponentiated Grad ($\epsilon=0.005$) | 0.8293 ± 0.0081 | 72.10% ± 1.11% | 0.0332 ± 0.0110 | 0.0783 ± 0.0554 | 0.0285 ± 0.0157 | 0.0875 ± 0.0182 | +0.0061 | [-0.0649, 0.0840] | False |
| German Credit | Random Forest | Threshold Opt | 0.8282 ± 0.0192 | 73.95% ± 3.22% | 0.0478 ± 0.0407 | 0.1208 ± 0.0735 | 0.0315 ± 0.0263 | 0.0914 ± 0.0183 | -0.0411 | [-0.1426, 0.0948] | False |
| **COMPAS** | Random Forest | Baseline | 0.6344 ± 0.0107 | 68.71% ± 0.79% | 0.2473 ± 0.0204 | 0.2416 ± 0.0321 | 0.2416 ± 0.0321 | 0.0495 ± 0.0258 | Reference | Reference | Reference |
| COMPAS | Random Forest | Reweighing | 0.6293 ± 0.0088 | 68.35% ± 0.54% | 0.2408 ± 0.0202 | 0.2319 ± 0.0364 | 0.2319 ± 0.0364 | 0.0560 ± 0.0357 | +0.0020 | [-0.0142, 0.0161] | False |
| COMPAS | Random Forest | Exponentiated Grad ($\epsilon=0.010$) | 0.3996 ± 0.0166 | 60.27% ± 0.64% | 0.0592 ± 0.0252 | 0.0789 ± 0.0487 | 0.0756 ± 0.0523 | 0.1655 ± 0.0650 | +0.2076 | [0.1476, 0.2591] | **True** |
| COMPAS | Random Forest | Threshold Opt | 0.6056 ± 0.0198 | 66.40% ± 1.14% | 0.0271 ± 0.0124 | 0.0596 ± 0.0239 | 0.0445 ± 0.0274 | 0.1513 ± 0.0261 | +0.2158 | [0.1318, 0.2680] | **True** |
| **Adult Income** | XGBoost | Baseline | 0.6971 ± 0.0059 | 86.85% ± 0.22% | 0.1729 ± 0.0056 | 0.0944 ± 0.0245 | 0.0925 ± 0.0269 | 0.0273 ± 0.0255 | Reference | Reference | Reference |
| Adult Income | XGBoost | Reweighing | 0.6810 ± 0.0058 | 86.52% ± 0.26% | 0.0933 ± 0.0059 | 0.1372 ± 0.0256 | 0.1372 ± 0.0256 | 0.1741 ± 0.0146 | +0.0766 | [0.0622, 0.0918] | **True** |
| Adult Income | XGBoost | Exponentiated Grad ($\epsilon=0.100$) | 0.5938 ± 0.0116 | 84.31% ± 0.36% | 0.0202 ± 0.0069 | 0.2584 ± 0.0278 | 0.2584 ± 0.0278 | 0.3009 ± 0.0250 | +0.1422 | [0.1208, 0.1613] | **True** |
| Adult Income | XGBoost | Threshold Opt | 0.6397 ± 0.0109 | 85.02% ± 0.31% | 0.0089 ± 0.0061 | 0.3104 ± 0.0231 | 0.3104 ± 0.0231 | 0.3511 ± 0.0188 | +0.1440 | [0.1062, 0.1701] | **True** |

*All results stored in `data/verified_benchmark_results.json` as the single source of truth.*

---

## 📂 Project Directory Structure

```
ds wala project/
├── dataset_config.yaml            # Config mapping datasets, targets, protected attributes, and actionable features
├── app.py                         # Interactive Streamlit Dashboard (6 Main Tabs)
├── style.css                      # Glassmorphism dark-theme styling
├── data/
│   ├── raw/                       # Auto-downloaded benchmark CSVs (COMPAS, German Credit, Adult Income)
│   ├── processed/                 # Cleaned datasets
│   └── fairlens.db                # SQLite Experiment & Proxy Tracking Database
├── models/                        # Serialized joblib trained model binaries
├── reports/                       # Generated HTML and PDF Audit Reports
├── src/
│   ├── config.py                  # Configuration loader
│   ├── pipeline.py                # Master pipeline orchestrator
│   ├── data/
│   │   ├── downloader.py          # Auto-downloader with SSL bypass & mirror fallbacks
│   │   ├── loader.py              # Ingestion & custom CSV schema validator
│   │   ├── cleaner.py             # Cleaning, deduplication, missing value imputation, IQR outlier flags
│   │   └── preprocessor.py        # Joint stratified 60/20/20 split & leakage-free scaling
│   ├── analysis/
│   │   ├── eda.py                 # KPI calculations & subgroup outcome rate breakdown
│   │   └── proxy.py               # Mutual Information & correlation proxy leakage ranking
│   ├── models/
│   │   ├── trainer.py             # Model training (Logistic Regression, Random Forest, XGBoost)
│   │   └── evaluator.py           # Predictive performance metrics (F1, Precision, Recall, ROC-AUC)
│   ├── fairness/
│   │   ├── metrics.py             # 4 Fairness Metrics (Demographic Parity, Equalized Odds, etc.)
│   │   ├── bootstrap.py           # Bootstrap 95% CIs & paired difference bootstrap test
│   │   └── mitigators.py          # Pre-processing, In-processing, and Post-processing mitigators
│   ├── explainability/
│   │   ├── shap_explainer.py      # SHAP global & local waterfall explanations
│   │   ├── dice_explainer.py      # DiCE counterfactual recourse generator
│   │   └── insights.py            # Natural language audit narrative generator
│   ├── db/
│   │   └── db_manager.py          # SQLite database logging interface
│   └── reporting/
│       └── pdf_generator.py       # HTML and PDF audit report compiler
├── requirements.txt               # Dependencies
└── README.md                      # Documentation
```

---

## 🚀 Quickstart & Setup Guide

### 1. Installation
Clone the repository and install the dependencies:
```bash
pip install -r requirements.txt
```

### 2. Download Datasets
Run the automated downloader script to populate `data/raw/`:
```bash
python -m src.data.downloader
```

### 3. Launch Interactive Streamlit Dashboard
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser.

---

## 🔬 Demonstration / Viva Walkthrough Guide

1. **Tab 1 — Overview & EDA:** Select **German Credit Data**. Observe applicant count (1,000) and base rate disparity between male and female applicants (72% vs 61% approval).
2. **Tab 2 — Proxy Leakage:** Inspect the Mutual Information ranking. Observe how features like `housing` or `employment_duration` leak protected sex/age identity.
3. **Tab 3 — Baseline Fairness Audit:** Switch to **COMPAS**. The baseline Random Forest has a Demographic Parity difference of 0.239 whose signed-gap 95% CI [0.183, 0.292] excludes zero (significant). For contrast, German Credit's gap (0.015) is *not* significant.
4. **Tab 4 — Mitigation Comparison:** On COMPAS or Adult Income, Exponentiated Gradient and Threshold Optimizer cut the Demographic Parity gap with paired-bootstrap significance (CI of the reduction > 0); on Adult Income, Reweighing is significant too. Observe the accuracy cost on the Pareto scatter plot.
5. **Tab 5 — Explainability & Recourse:** Select a rejected female applicant. View her local SHAP waterfall explanation (identifying `credit_amount` or `duration_in_month` as rejection drivers) and view her DiCE Actionable Recourse table (e.g. reducing duration by 6 months).
6. **Tab 6 — Audit Report:** Click **Download PDF Audit Report** to export the publication-ready compliance report.
