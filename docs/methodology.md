# 🔬 FairLens: Empirical Data Science & Statistical Methodology

This document details the quantitative methods, statistical validation protocols, and data integrity safeguards implemented in **FairLens**.

---

## 1. Data Processing & Leakage Prevention Protocol

### A. Data Leakage Bug Discovery & Resolution
During early development, an empirical check flagged an impossible $F_1 = 1.000$ performance score across all models. A root-cause audit revealed two data leakage pathways:
1. **Target Alias Duplication**: `clean_dataset()` attached a standardized `'target'` column to `cleaned_df` in addition to original target names (`credit_risk`, `two_year_recid`, `income`). When preprocessing dropped dataset-specific target names, the `'target'` alias remained inside $\mathbf{X}$, feeding ground-truth labels directly into models.
2. **COMPAS Algorithm Outputs & Future Recidivism Labels**: Raw COMPAS files contained post-hoc algorithm scores (`decile_score`, `decile_score.1`, `score_text`, `v_decile_score`, `v_score_text`) and future arrest/recidivism events (`is_recid`, `violent_recid`, `is_violent_recid`, `r_case_number`).

**Resolution**:
- Target alias columns were removed entirely at the cleaning stage so that datasets contain **exactly one target column** named `'target'`.
- All COMPAS algorithm outputs, future recidivism labels, and census sampling weights (`fnlwgt`) are strictly excluded from feature matrix construction.
- Automated pytest regression tests (`test_no_leakage.py`) permanently guard against target re-entry.

### B. Joint Stratified Splitting & Partition Separation
To preserve demographic and outcome proportions without row duplication:
- Datasets are split into **60% Train / 20% Validation / 20% Test** partitions.
- Splitting is stratified jointly on the target variable $Y$ and protected attribute $S$ ($\text{strata} = Y \times S$).
- Feature matrix row deduplication is performed prior to splitting, guaranteeing **0 row index overlap** across partitions (`test_no_train_test_overlap.py`).
- Continuous feature scalers (`StandardScaler`) are `.fit()` **strictly on the training partition $X_{\text{train}}$** and `.transform()`-applied to $X_{\text{val}}$ and $X_{\text{test}}$.

---

## 2. Proxy Feature Leakage Quantification

Removing protected attributes ($S$) from training data ("Fairness Through Unawareness") is insufficient because non-protected features ($X$) encode demographic identity. FairLens quantifies proxy leakage via **Mutual Information (MI)**:

$$I(X_j; S) = \sum_{x \in X_j} \sum_{s \in S} P(x, s) \log \left( \frac{P(x, s)}{P(x)P(s)} \right)$$

### Risk Classification:
- **High Proxy Risk**: $MI \ge 0.10$ (e.g., COMPAS `priors_count` $MI=0.184$, Adult `occupation` $MI=0.114$).
- **Moderate Proxy Risk**: $0.03 \le MI < 0.10$ (e.g., German Credit `housing` $MI=0.052$).
- **Low Proxy Risk**: $MI < 0.03$.

---

## 3. Fairness Metrics & Subgroup Analysis

FairLens computes four fairness metrics across demographic groups:

1. **Demographic Parity Difference ($DP_{\text{diff}}$)**:
   $$DP_{\text{diff}} = |P(\hat{Y}=1 | S=\text{privileged}) - P(\hat{Y}=1 | S=\text{unprivileged})|$$
2. **Equalized Odds Difference ($EO_{\text{diff}}$)**:
   $$EO_{\text{diff}} = \max\left(|TPR_{\text{priv}} - TPR_{\text{unread}}|, |FPR_{\text{priv}} - FPR_{\text{unread}}|\right)$$
3. **Equal Opportunity Difference ($EOpp_{\text{diff}}$)**:
   $$EOpp_{\text{diff}} = |TPR_{\text{priv}} - TPR_{\text{unread}}|$$
4. **Predictive Parity Difference ($PP_{\text{diff}}$)**:
   $$PP_{\text{diff}} = |Precision_{\text{priv}} - Precision_{\text{unread}}|$$

### Subgroup Group Size Safeguard:
Small subgroups ($n < 10$, e.g., Native American $n=2$ in COMPAS) can artificially inflate max-min metric spreads. FairLens compares primary privileged vs unprivileged categories and filters $n < 10$ subgroups when computing overall spreads.

---

## 4. Statistical Validation: 500-Resample Bootstrap & Paired Hypothesis Testing

- **95% Confidence Intervals**: 500 bootstrap resamples with replacement generated on test set predictions to estimate $2.5^{\text{th}}$ and $97.5^{\text{th}}$ percentile interval bounds $[CI_{\text{low}}, CI_{\text{high}}]$.
- **Paired Hypothesis Significance Testing**: **Wilcoxon Signed-Rank Test** executed on paired bootstrap distributions comparing baseline vs mitigated models on identical test set partitions ($p < 0.05$ flags statistically significant bias reduction).

---

## 5. Multi-Stage Mitigation Suite

1. **Pre-Processing (Reweighing)**: Re-balances training instances using sample weights:
   $$W(s, y) = \frac{P(S=s) \cdot P(Y=y)}{P(S=s, Y=y)}$$
2. **In-Processing (Exponentiated Gradient)**: Solves a constrained optimization problem minimizing prediction loss subject to demographic parity bounds ($\epsilon = 0.01$).
3. **Post-Processing (Threshold Optimizer)**: Adjusts group-specific decision thresholds to equalize outcome rates.

---

## 6. Dual Explainability Paradigm

- **SHAP (Shapley Additive exPlanations)**: Computes Shapley values using `TreeExplainer` or `LinearExplainer` for global feature ranking and local decision waterfalls.
- **DiCE Counterfactual Recourse**: Solves for actionable feature modifications to flip negative decisions for rejected applicants while holding immutable attributes (sex, race, age, foreign-worker status) fixed.
