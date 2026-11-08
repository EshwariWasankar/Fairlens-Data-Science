# FairLens Pipeline Auditing & Quality Assurance Documentation

This document provides a comprehensive, non-technical history of every data science, algorithmic, and software issue identified and resolved across the FairLens project lifecycle. It is written in plain language so that non-technical leaders, auditors, reviewers, and incoming developers can understand how each bug occurred, how it was caught, how it was remedied, and how the fixes affected the project's reported benchmark metrics.

---

## Table of Contents
1. [Executive Summary](#1-executive-summary)
2. [Catalog of Project Issues & Corrections](#2-catalog-of-project-issues--corrections)
   - [Problem 1: Target Variable Leakage (Duplicate Label Column)](#problem-1-target-variable-leakage-duplicate-label-column)
   - [Problem 2: COMPAS Algorithmic-Output Leakage](#problem-2-compas-algorithmic-output-leakage)
   - [Problem 3: Cleaning Deduplication Bug (Dropped Valid Observations)](#problem-3-cleaning-deduplication-bug-dropped-valid-observations)
   - [Problem 4: Wilcoxon Statistical Error on Bootstrap Replicates](#problem-4-wilcoxon-statistical-error-on-bootstrap-replicates)
   - [Problem 5: Adult Income Exponentiated Gradient Apparent Degeneracy (Bug 1)](#problem-5-adult-income-exponentiated-gradient-apparent-degeneracy-bug-1)
   - [Problem 6: German Credit Epsilon Sweep Caching and Parameter Forwarding Bug (Bug 2)](#problem-6-german-credit-epsilon-sweep-caching-and-parameter-forwarding-bug-bug-2)
   - [Problem 7: Inconsistent Epsilon Selection & Fallback Rule Definition (Item 3)](#problem-7-inconsistent-epsilon-selection--fallback-rule-definition-item-3)
   - [Problem 8: Modern Pandas 3.0 String Categorical Incompatibility](#problem-8-modern-pandas-30-string-categorical-incompatibility)
   - [Problem 9: Stale Claims and Outdated Numbers in Existing Documentation](#problem-9-stale-claims-and-outdated-numbers-in-existing-documentation)
   - [Problem 10: German Credit EDA Zero-Metrics & Blank Charts Bug](#problem-10-german-credit-eda-zero-metrics--blank-charts-bug)
   - [Problem 11: `personal_status_sex` Direct Proxy Leakage](#problem-11-personal_status_sex-direct-proxy-leakage)
   - [Problem 12: Model Serialization Bloat During Hyperparameter Sweeps](#problem-12-model-serialization-bloat-during-hyperparameter-sweeps)
   - [Problem 13: Streamlit Dashboard UI Lag & Live Bootstrap Bottlenecks](#problem-13-streamlit-dashboard-ui-lag--live-bootstrap-bottlenecks)
   - [Problem 14: Dynamic Proxy Column Dropping for Parent Features](#problem-14-dynamic-proxy-column-dropping-for-parent-features)
   - [Problem 15: Infinite RecursionError in Mitigator Deserialization](#problem-15-infinite-recursionerror-in-mitigator-deserialization)
   - [Problems 16–22: 2026-11-08 Audit Fixes](#problems-1622-2026-11-08-audit-fixes)
3. [Before and After Benchmark Comparison](#3-before-and-after-benchmark-comparison)
4. [Methodological Glossary for Non-Technical Readers](#4-methodological-glossary-for-non-technical-readers)

---

## 1. Executive Summary

FairLens is an automated software framework that audits tabular machine learning models for demographic disparities (fairness) and applies mitigation strategies to ensure fair automated decision-making. 

To ensure FairLens can be trusted in production environments and academic publications, every component of its data preparation, model training, fairness evaluation, and statistical validation must be methodologically sound. During rigorous development and verification, our team identified and corrected critical issues ranging from data leakage to algorithmic cache misses. 

Documenting these issues transparently is essential for scientific integrity: resolving methodological defects increases trust in FairLens's final conclusions.

---

## 2. Catalog of Project Issues & Corrections

### Problem 1: Target Variable Leakage (Duplicate Label Column)

#### What Was Wrong
During initial testing, several classification models achieved impossibly high accuracy scores (up to 100% accuracy and F1 scores of 1.000). While this looked impressive at first glance, real-world credit approval and recidivism prediction models never achieve perfect accuracy.

#### How It Was Found
During automated verification, engineering checks flagged that the model's accuracy remained flawless even when noise was added. A deep inspection of the feature matrix $X$ fed into the algorithms revealed that the target answer (whether a person was approved or arrested) was accidentally present inside the dataset twice under two different column names.

#### Root Cause
During data transformation and feature encoding, an alias column representing the target outcome was retained in the predictor table $X$ instead of being strictly isolated into the outcome label $y$. The machine learning algorithm simply "looked up" the answer from this duplicate column during training and testing.

#### How It Was Fixed
The feature preprocessing pipeline (`src/data/preprocessor.py`) was overhauled to explicitly drop all target column names and aliases before generating features. An automated unit test (`tests/test_no_leakage.py`) was created to ensure no column having a 1.0 correlation or matching the target label could ever enter the training set.

#### Result
The artificial 100% accuracy vanished. Baseline models settled into realistic, statistically credible ranges (e.g., German Credit F1 ~0.83, COMPAS F1 ~0.63, Adult Income F1 ~0.70).

---

### Problem 2: COMPAS Algorithmic-Output Leakage

#### What Was Wrong
In the COMPAS dataset (used to study recidivism risk predictions), models initially appeared far more accurate and less biased than standard criminal justice literature indicated.

#### How It Was Found
A line-by-line audit of the input features was conducted. We discovered that columns such as `decile_score`, `score_text`, `v_decile_score`, and `v_score_text` were being fed into our models as ordinary input features.

#### Root Cause
The raw ProPublica COMPAS dataset contains both raw demographic/criminal records and the output scores assigned by Northpointe's commercial risk assessment software. These score columns are the *downstream output* of the scoring system being audited; they are not independent background facts about a defendant. By training on them, our models were predicting recidivism based on another algorithm's pre-computed conclusions rather than authentic defendant history.

#### How It Was Fixed
All algorithmic score columns and assessment deciles were placed on a mandatory exclusion list in `src/data/cleaner.py` and `src/data/preprocessor.py`. Models were restricted to genuine pre-trial historical features: age, prior conviction count, charge severity, and juvenile citation records.

#### Result
COMPAS baseline accuracy returned to genuine predictive numbers (Random Forest F1 ~0.63, Accuracy ~67.6%), revealing the true, documented racial disparity between African-American and Caucasian defendants (Demographic Parity difference of ~0.238).

---

### Problem 3: Cleaning Deduplication Bug (Dropped Valid Observations)

#### What Was Wrong
When checking the cleaned dataset sizes against standard academic literature, our benchmark datasets were missing hundreds of legitimate rows.

#### How It Was Found
We compared the raw dataset record counts with the cleaned dataset row counts. For example, German Credit, which universally contains 1,000 borrowers, was missing dozens of valid borrower profiles.

#### Root Cause
The cleaning script had an aggressive deduplication command that removed rows if they shared identical values across a subset of demographic features. In demographic datasets, two distinct individuals frequently share the same age, marital status, and credit history category. Dropping these rows was treating distinct individuals as duplicate entries.

#### How It Was Fixed
In `src/data/cleaner.py`, the deduplication logic was corrected to perform strict, full-row exact duplicate checking (`raw_df.drop_duplicates()`), preserving all unique individuals even if their demographic profiles were identical.

#### Result
Cleaned datasets now match standard benchmarks:
- **German Credit:** 1,000 rows retained (0 duplicates dropped).
- **COMPAS:** 6,172 rows retained (only 5 exact duplicate rows dropped).
- **Adult Income:** 32,537 rows retained (only 24 exact duplicate rows dropped).

---

### Problem 4: Wilcoxon Statistical Error on Bootstrap Replicates

#### What Was Wrong
When determining whether a fairness improvement (e.g., from Reweighing or Exponentiated Gradient) was statistically significant, the system produced erratic or artificially confident $p$-values.

#### How It Was Found
A mathematical review of `src/fairness/bootstrap.py` revealed that the code was passing 1,000 bootstrap resamples directly into a classical hypothesis test known as the Wilcoxon signed-rank test (`scipy.stats.wilcoxon`).

#### Root Cause
The Wilcoxon signed-rank test assumes that every observation in the sample is an independent, real-world data point. However, bootstrap replicates are created by randomly resampling the *same* test partition over and over again with replacement. Feeding 1,000 bootstrap draws into a test that assumes 1,000 independent patients or borrowers artificially inflates the effective sample size, manufacturing false statistical significance where none exists.

#### How It Was Fixed
The invalid Wilcoxon test was removed. In its place, we implemented a mathematically rigorous **Paired Bootstrap Confidence Interval** method. For each bootstrap iteration, we compute the exact difference between the baseline prediction and the mitigated prediction on the same set of individuals:
$$\Delta = \text{Metric}_{\text{baseline}} - \text{Metric}_{\text{mitigated}}$$
Across 1,000 bootstrap iterations, we construct an empirical 95% confidence interval $[CI_{\text{lower}}, CI_{\text{upper}}]$. A fairness gain is certified as statistically significant if and only if the 95% interval strictly excludes zero.

#### Result
The fairness reporting is now mathematically solid, defensible for peer review, and avoids spurious claims of statistical significance.

---

### Problem 5: Adult Income Exponentiated Gradient Apparent Degeneracy (Bug 1)

#### What Was Wrong
Early notes in `PROJECT_STATUS.md` and `README.md` stated that Exponentiated Gradient on the Adult Income dataset was "degenerate," claiming it produced identical scores across all epsilon values (Acc 24.09%, F1 0.3883, DP 0.0000, std 0.0000). Because approximately 24% of the Adult dataset belongs to the positive class (> $50K income), it was suspected that the model collapsed into predicting "positive" for everyone to trivially achieve zero fairness difference.

#### How It Was Found
We performed an empirical investigation by capturing the raw class distribution counts:
`np.unique(y_pred, return_counts=True)`
and checking the code in `master_eval.py` and `src/fairness/mitigators.py`.

#### Root Cause
1. **Parameter Passing Failure:** In `master_eval.py`, `train_exponentiated_gradient_model` was called without the `eps` parameter, causing it to fall back to the function's default `eps=0.01`. The script then ran `eg_model.eps = eps` *after* model fitting had already completed. Fairlearn does not retroactively refit or alter a trained model when its attributes are modified post-hoc.
2. **Stale/False Status Claim:** When actually trained with the baseline XGBoost classifier and the Demographic Parity constraint, the model does **not** collapse. At `eps=0.005`, the model predicts **5,529 negative records** and **979 positive records** (a 15.0% positive prediction rate), achieving an accuracy of **84.22%** and an F1 score of **0.5939**.
3. **Solver Warnings:** Deprecated arguments (`use_label_encoder=False`) in XGBoost generated warnings that masked internal solver diagnostics.

#### How It Was Fixed
- Modified `train_exponentiated_gradient_model` to accept and pass `eps` directly into Fairlearn's `ExponentiatedGradient(estimator=base_estimator, constraints=fair_constraint, eps=eps)`.
- Removed deprecated XGBoost parameters.
- Updated `master_eval.py` to forward `eps=eps` at fit time and log prediction class counts for verification.

#### Result
The apparent degeneracy was disproven and fixed. Adult Income Exponentiated Gradient is fully functional, producing robust predictions that reduce Demographic Parity difference by **93%** (from 0.1742 down to 0.0124) while preserving 84.22% accuracy.

---

### Problem 6: German Credit Epsilon Sweep Caching and Parameter Forwarding Bug (Bug 2)

#### What Was Wrong
In the German Credit epsilon sweep, testing five different tolerance values ($0.005, 0.01, 0.02, 0.05, 0.1$) produced identical results for $0.005, 0.02, 0.05$, and $0.1$. Only $0.01$ showed different numbers.

#### How It Was Found
Section 2 of `PROJECT_STATUS.md` flagged this anomaly. In a normal optimization sweep, relaxing epsilon should gradually trade off accuracy for fairness. Producing identical numbers across four different thresholds was a red flag.

#### Root Cause
Two flaws interacted to cause this behavior:
1. **Missing Parameter Call:** As in Problem 5, `train_exponentiated_gradient_model` was called without `eps=eps`. The default parameter in the function signature was `0.01`.
2. **Deficient Cache Keying:** The model persistence logic on disk saved models without including `eps` in the filename. Even when different values were intended, the code reloaded the cached `0.01` model from disk.

#### How It Was Fixed
1. In `src/fairness/mitigators.py`, disk caching was updated to use a comprehensive cache key:
   `f"{dataset_key}_{model_type}_exponentiated_gradient_{constraint}_eps{eps}_seed{random_state}.joblib"`
2. Stale, improperly keyed `.joblib` files in `models/` were identified and purged.
3. In `master_eval.py`, `eps=eps` was explicitly forwarded to `train_exponentiated_gradient_model`.

#### Result
Correction (2026-11-08): the bullet list previously shown here did not match `data/verified_benchmark_results.json`. With the cache bug fixed, the DemographicParity sweep on German Credit returns the **same** model for eps 0.005–0.05 (test DP 0.0208) and differs only at eps 0.1 (test DP 0.0082). This is genuine: German Credit's baseline disparity is already small, so the constraint is not binding. See the sweep table in Section 3A.

---

### Problem 7: Inconsistent Epsilon Selection & Fallback Rule Definition (Item 3)

#### What Was Wrong
The project's rule for picking the best epsilon was defined as: *"Select the epsilon that maximizes fairness gain while keeping accuracy loss within 5.0 percentage points of the unmitigated baseline."* However, this rule was not implemented consistently in code. Crucially, on the COMPAS dataset, every single epsilon tested had an accuracy loss greater than 5.0 percentage points (losses ranged between 6.72 and 8.01 percentage points), leaving the selection undefined without an automated fallback rule.

#### How It Was Found
During automated pipeline execution, attempting to filter by `loss <= 5.0` returned an empty list for COMPAS, exposing the absence of a deterministic fallback policy.

#### Root Cause
The codebase lacked a formal, centralized decision function that handled the edge case where no hyperparameter satisfies the 5-point constraint.

#### How It Was Fixed
We implemented a robust selection procedure in `master_eval.py`:
1. Calculate the baseline accuracy and baseline demographic parity difference.
2. Filter candidate models where $\text{Accuracy}_{\text{baseline}} - \text{Accuracy}_{\text{mitigated}} \le 5.0$ percentage points.
3. If valid candidates exist, select the candidate that achieves the lowest Demographic Parity difference (highest fairness gain).
4. If **no** candidate meets the 5 percentage point threshold (as on COMPAS), apply the formalized fallback rule: **select the epsilon that minimizes accuracy loss**.

#### Result
Epsilon selection is reproducible and deterministic. Since 2026-11-08 it is evaluated **on the validation split only** (Problem 20):
- **German Credit:** Selected $\epsilon = 0.005$ (validation accuracy loss within 5.0pp): val acc loss = +1.50pp, val DP = 0.0065; on test: acc = 70.50%, DP = 0.0208.
- **COMPAS:** Selected $\epsilon = 0.010$ under the fallback rule (no candidate within 5pp; minimal validation accuracy loss): val acc loss = +7.21pp, val DP = 0.0736; on test: acc = 60.40%, DP = 0.0281.
- **Adult Income:** Selected $\epsilon = 0.100$ (validation accuracy loss within 5.0pp): val acc loss = +2.72pp, val DP = 0.0184; on test: acc = 84.31%, DP = 0.0106.

---

### Problem 8: Modern Pandas 3.0 String Categorical Incompatibility

#### What Was Wrong
During local testing on modern Python environments, running `pytest` failed with:
`ValueError: invalid literal for int() with base 10: 'Male'` in `src/analysis/proxy.py`.

#### How It Was Found
Execution of the unit test suite (`tests/test_pipeline_integration.py`).

#### Root Cause
Under modern versions of Pandas (Pandas 3.0+), string columns are represented as `StringDtype()` rather than legacy Python `object` pointers. In `src/analysis/proxy.py` line 31, the check:
`if str(df[col].dtype) == "object":`
failed to identify string columns, causing the code to attempt converting raw text strings like `'Male'` directly into integers with `astype(int)`.

#### How It Was Fixed
Line 31 of `src/analysis/proxy.py` was updated to support both legacy and modern Pandas string types:
`if str(df[col].dtype) in ("object", "string") or isinstance(df[col].dtype, pd.StringDtype) or pd.api.types.is_string_dtype(df[col]):`

#### Result
All 16 automated tests across the test suite pass with 100% success.

---

### Problem 9: Stale Claims and Outdated Numbers in Existing Documentation

#### What Was Wrong
`README.md` and `PROJECT_STATUS.md` contained provisional numbers from early test runs that had known bugs (such as the unpassed epsilon in Exponentiated Gradient). For instance, `PROJECT_STATUS.md` still displayed the provisional Adult Income row as `Acc 24.09%`, and `README.md` listed preliminary figures that did not match the re-verified 10-seed benchmarks.

#### How It Was Found
Systematic cross-checking of all repository markdown files against the raw benchmark output file (`data/verified_benchmark_results.json`).

#### Resolution
To maintain complete clarity, this document (`DOCUMENTATION.md`) establishes the ground truth for all benchmark figures. The project instructions specify that `README.md`, final reports, and dashboard text should be updated in a subsequent pass directly from these re-verified results.

---

### Problem 10: German Credit EDA Zero-Metrics & Blank Charts Bug

#### What Was Wrong
In the Streamlit dashboard under the Overview & EDA tab for German Credit, the KPI cards for *Base Favorable Rate*, *Demographic Base Gap*, and *Protected Subgroups* displayed `0.00%` or `0`, and the outcome distribution bar charts rendered completely blank.

#### How It Was Found
Visual and programmatic inspection of `app.py`, `src/pipeline.py`, and `src/analysis/eda.py`.

#### Root Cause
During data cleaning (`src/data/cleaner.py`), the original target column name (`credit_risk`) was standardized and renamed to `target`, dropping the original name. When `src/pipeline.py` called `get_dataset_kpis` and `get_group_disparity_table` with `target_col="credit_risk"`, the EDA module strictly checked `if target_col not in df.columns: return 0.0`. Because `"credit_risk"` was no longer a column in `df_clean`, both functions immediately returned empty/zero structures.

#### How It Was Fixed
1. In `src/analysis/eda.py`, implemented `_resolve_columns` to dynamically resolve target and protected attributes, checking the provided parameter, falling back to `'target'`, and inspecting standard domain aliases (`credit_risk`, `two_year_recid`, `income`).
2. Implemented `_extract_binary_target` to safely handle UCI Statlog `1=Good / 2=Bad` encodings and string representations.
3. In `src/pipeline.py`, passed `eda_target = 'target' if 'target' in df_clean.columns else target_col`.
4. In `app.py`, updated header subtitle to dynamically reflect `pipeline_results.get('target_col', 'target')`.

#### Result
The EDA tab for German Credit now properly displays:
- Total Applicants: **1,000**
- Base Favorable Rate: **70.00%** (700 good credit records)
- Demographic Base Gap: **7.48%** (Male approval: 72.32%, Female approval: 64.84%)
- Protected Subgroups: **2** (female: 310, male: 690)
- Both outcome rate and volume bar charts render with full interactivity.

---

### Problem 11: `personal_status_sex` Direct Proxy Leakage

#### What Was Wrong
In the German Credit proxy analysis ranking, the feature `personal_status_sex` exhibited a $-1.0000$ Pearson correlation and high Mutual Information ($0.6196$) with the protected attribute `sex`. Because this feature directly concatenates marital status and gender, it leaked the protected attribute directly into the proxy feature set.

#### How It Was Found
Inspecting the top rows of `proxy_df` in `src/analysis/proxy.py` and the generated compliance report.

#### Root Cause
While `src/data/preprocessor.py` excluded `personal_status_sex` from the model training matrix, `clean_dataset` in `src/data/cleaner.py` retained it in `df_clean`. When `compute_proxy_leakage` analyzed `df_clean`, `personal_status_sex` was treated as an ordinary candidate feature rather than being excluded.

#### How It Was Fixed
1. In `src/data/cleaner.py`, ensured `'sex'` is extracted if not already present, then explicitly dropped `'personal_status_sex'` from `cleaned_df`.
2. In `src/analysis/proxy.py`, added `'personal_status_sex'` to the mandatory `drop_cols` list.

#### Result
`personal_status_sex` is completely eliminated from the proxy analysis. The top proxy features for German Credit now legitimately reflect genuine indirect demographic channels (e.g., `age`, `credit_history`, `employment_duration`, `purpose`).

---

### Problem 12: Model Serialization Bloat During Hyperparameter Sweeps

#### What Was Wrong
The hyperparameter sweep and multi-seed evaluation routines saved a `.joblib` binary file to `models/` for every single epsilon value and random seed iteration. This resulted in 98 serialized model files consuming excessive disk space and cluttering the artifact directory with unselected candidates.

#### How It Was Found
Directory inspection revealed dozens of intermediate artifacts formatted as `..._eps{eps}_seed{seed}.joblib`.

#### Root Cause
In `src/fairness/mitigators.py`, `train_exponentiated_gradient_model` unconditionally executed `joblib.dump(mitigated_model, cache_path)` during every call. As a result, every intermediate sweep evaluation across 5 epsilons, 2 constraints, 3 datasets, and 10 seeds persisted to disk.

#### How It Was Fixed
1. Refactored `train_exponentiated_gradient_model` and `train_threshold_optimizer_model` to accept `save_model: bool = False` (default `False`). Intermediate candidates are retained strictly in RAM.
2. In `master_eval.py`, candidate models are held in an in-memory dictionary during the sweep. The 5 percentage point accuracy loss rule evaluates the models in RAM to select the winner.
3. Only the final selected winning model is serialized to disk as a `.joblib` file (e.g. `{dataset}_{model}_exponentiated_gradient.joblib` and `{dataset}_{model}_threshold_optimizer.joblib`).
4. Intermediate seed and unselected epsilon binaries were purged from `models/`, reducing the model directory from 98 files to 16 clean, production-ready artifacts.

---

### Problem 13: Streamlit Dashboard UI Lag & Live Bootstrap Bottlenecks

#### What Was Wrong
The Streamlit dashboard (`app.py`) froze during dataset selection and tab switching, leading to 30-40+ second delays and UI unresponsiveness.

#### How It Was Found
Profiling Streamlit execution during tab switching and dataset selection.

#### Root Cause
1. **Live Bootstrap Recalculation:** The dashboard was invoking `run_fairlens_pipeline`, which ran 3,500 live bootstrap resamples (500 resamples for baseline + 3 mitigations + paired differences) on the frontend.
2. **Missing Granular Caching:** Raw CSV loading and data preprocessing were not decoupled from model training and statistical auditing.
3. **Model Re-Instantiation:** Serialized `.joblib` models were rebuilt or re-read from disk on tab switches rather than held in persistent application memory.

#### How It Was Fixed
1. **Data Caching:** Applied `@st.cache_data` to `load_cached_raw_dataset`, `load_cached_clean_dataset`, `get_cached_preprocessed_data`, and `get_cached_eda`.
2. **Model Caching:** Applied `@st.cache_resource` to `load_cached_joblib_file`, `get_cached_baseline_model`, and `get_cached_mitigated_model`, ensuring `.joblib` models are loaded into memory once and reused across tab switches.
3. **Pre-Computed Metric Fetching:** Augmented `data/verified_benchmark_results.json` with pre-computed 500-resample bootstrap confidence intervals (`baseline_fairness`), predictive metrics, and mitigation evaluations. For benchmark datasets, `app.py` strictly reads these metrics directly from the JSON file, eliminating live frontend bootstrap execution.

#### Result
Frontend metric retrieval latency dropped from ~35,000 ms to ~5.4 ms. Switching tabs and datasets is instantaneous.

---

### Problem 14: Dynamic Proxy Column Dropping for Parent Features

#### What Was Wrong
In datasets where a protected attribute was derived from a continuous or composite parent feature (e.g., `age_group` derived by binning continuous `age`, or `sex` derived from German Credit's `personal_status_sex`), toggling the protected attribute left the parent feature inside the feature matrix $X$ and the proxy analyzer pool. As a result, the proxy mutual information ranking was dominated by the trivial parent feature (e.g., $0.45+$ mutual information and $0.70+$ correlation), and downstream models retained direct access to the unbinned demographic information.

#### How It Was Found
Auditing proxy leakage tables and unit tests (`tests/test_no_leakage.py`) when toggling protected attributes between `sex` and `age_group`.

#### Root Cause
Static column drop lists did not dynamically react to which protected attribute was actively selected. When `age_group` was chosen as the protected attribute, continuous `age` was treated as an ordinary candidate feature rather than being excluded as the direct parent column.

#### How It Was Fixed
1. In `src/data/preprocessor.py`, right before feature matrix $X$ is finalized and encoded, dynamic drop rules were added:
   - If `protected_col == 'age_group'`, continuous `'age'` is appended to `drop_cols`.
   - If `protected_col == 'sex'`, `'personal_status_sex'` is appended to `drop_cols`.
2. In `src/analysis/proxy.py`, identical dynamic rules were integrated into `compute_proxy_leakage` to ensure candidate proxy evaluations ignore direct parent columns.
3. In `tests/test_no_leakage.py`, added explicit test assertions verifying `'age'` is absent from features when `age_group` is selected.

#### Result
Downstream models and proxy risk tables cannot access direct parent columns. Top proxy features now reflect genuine indirect demographic channels (e.g., `housing`, `employment_duration`, `credit_amount`), and leakage tests pass.

---

### Problem 15: Infinite RecursionError in Mitigator Deserialization

#### What Was Wrong
When loading cached mitigation models from disk using `joblib.load()` (such as `ThresholdOptimizer` or custom wrapped estimators), Python crashed with an unhandled `RecursionError: maximum recursion depth exceeded`.

#### How It Was Found
Attempting to unpickle serialized models stored in `models/` during pipeline execution and dashboard startup.

#### Root Cause
`_Float64PredictProbaWrapper` defined a naive `__getattr__(self, name)` that forwarded all unrecognized attribute lookups to `getattr(self.estimator, name)`. During object deserialization, Python's unpickler (`joblib`/`pickle`) inspects internal magic methods (such as `__setstate__`, `__getstate__`, or `__dict__`) *before* the instance's `estimator` attribute is restored into `self.__dict__`. Because `self.estimator` was not yet set, referring to `self.estimator` inside `__getattr__` triggered another lookup for `'estimator'`, calling `__getattr__('estimator')` indefinitely until Python's recursion limit was reached.

#### How It Was Fixed
1. Updated `__getattr__` in `src/fairness/mitigators.py` to explicitly block magic methods and guard against premature access before `estimator` is loaded:
   ```python
   def __getattr__(self, name: str) -> Any:
       # Prevent infinite recursion during joblib/pickle loading
       if name.startswith('__') and name.endswith('__'):
           raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
       
       # Prevent recursion if estimator isn't loaded yet
       if name == "estimator" or "estimator" not in self.__dict__:
           raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
           
       return getattr(self.estimator, name)
   ```
2. Implemented an explicit `__sklearn_tags__(self)` method to cleanly delegate scikit-learn 1.6+ estimator tag inspections to `self.estimator.__sklearn_tags__()` without triggering dunder recursion or `AttributeError`.

#### Result
All serialized models (`threshold_optimizer.joblib`, `exponentiated_gradient.joblib`) across all datasets deserialize cleanly with zero `RecursionError`.

---

### Problems 16–22: 2026-11-08 Audit Fixes

The following issues were found in a full re-audit (tests, benchmark JSON, and the dashboard) and fixed before regenerating every number in this document.

#### Problem 16: Fairness values reported as the bootstrap mean (inflated)
- What was wrong: `compute_bootstrap_confidence_intervals` reported `val` as the mean of the bootstrap distribution. For absolute-gap metrics that mean is biased upward, so small gaps looked much larger: German Credit's baseline DP was shown as 0.0433 while the actual test-set value is 0.0147.
- How it was discovered: The Fairness Audit tab showed group selection rates of 0.855 vs 0.870 (gap ≈ 0.015) right next to a DP card reading 0.0433.
- Fix: `val` is now the full-test-set point estimate; the bootstrap mean is kept separately as `bootstrap_mean`.
- Result: Reported values match the group rates. Regression test: `test_bootstrap_value_is_full_sample_point_estimate`.

#### Problem 17: Non-reproducible Exponentiated Gradient / Threshold Optimizer predictions
- What was wrong: Fairlearn's `ExponentiatedGradient.predict` and `ThresholdOptimizer.predict` are randomized (`random_state=None`). The same fitted COMPAS model gave 59.3%–61.1% accuracy across 5 calls, and accuracy/F1 were computed on a different random draw than the fairness metrics. Epsilon choices that differed by <1pp were therefore partly noise.
- Fix: `SeededExponentiatedGradient` (in `src/fairness/mitigators.py`) fixes the prediction seed; Threshold Optimizer is always called with `random_state=42`.
- Result: Repeated runs give identical numbers. Regression tests: `test_exponentiated_gradient_predictions_are_reproducible`, `test_threshold_optimizer_seeded_predictions_are_reproducible`.

#### Problem 18: Exponentiated Gradient ROC-AUC always 0.5
- What was wrong: `ExponentiatedGradient` has no `predict_proba`, so the evaluator silently fell back to ROC-AUC = 0.5; Threshold Optimizer's AUC was computed from hard labels.
- Fix: Both now expose the mixture/threshold positive-class probability (`_pmf_predict`).
- Result: Real AUCs are reported (e.g. Adult Income EG: 0.7259).

#### Problem 19: Significance tested on the absolute gap
- What was wrong: "CI excludes zero" was evaluated on the CI of |gap|, which essentially never contains 0, so it could not detect "no disparity". The text also used an ad-hoc `ci_low > 0.02` threshold.
- Fix: Each simple gap metric (DP, EOpp, PP) is also bootstrapped as a *signed* gap between two fixed reference groups; `is_significant` = signed 95% CI excludes 0.
- Result: German Credit's baseline disparity (DP 0.0147, signed CI [-0.0855, 0.1215]) is correctly reported as **not** significant; COMPAS (signed CI [0.1833, 0.2916]) and Adult Income (signed CI [0.1321, 0.1815]) are significant.

#### Problem 20: Epsilon selected on the test set
- What was wrong: `master_eval.py` chose epsilon by comparing test-set accuracy and DP — test-set leakage into a hyperparameter choice.
- Fix: The sweep now also records validation-set metrics, and the selection rule (max fairness gain with ≤ 5pp accuracy loss, minimal-loss fallback) runs only on the validation split. Test metrics are reported but never used for selection.
- Result: Selected epsilons are now German Credit 0.005, COMPAS 0.01, Adult Income 0.1 (previously 0.005 / 0.05 / 0.005).

#### Problem 21: Dashboard showed the benchmark model's metrics for any selected model
- What was wrong: `app.py` read precomputed metrics whenever the protected attribute was the default, ignoring the selected classifier — e.g. Adult Income + Logistic Regression displayed the XGBoost DP of 0.1564.
- Fix: `src/benchmark_store.uses_precomputed_benchmark` also requires the selected model to be the benchmarked one; any other selection is computed live.
- Result: Regression test `test_dashboard_uses_benchmark_only_for_benchmarked_model`.

#### Problem 22: Test suite overwrote committed artifacts
- What was wrong: `train_model()` always writes `models/*.joblib` and the pipeline logs to `data/fairlens.db`, so every `pytest` run silently replaced committed models.
- Fix: `tests/conftest.py` redirects `MODELS_DIR` and `DB_PATH` to a temp directory; generated artifacts (`models/*.joblib`, `data/fairlens.db`, `reports/*.html|pdf`) are now git-ignored.

---

## 3. Before and After Benchmark Comparison

The table below contrasts the flawed or provisional results previously reported against the verified numbers produced after resolving all bugs.

### A. Exponentiated Gradient Epsilon Sweeps (Seed 42)

DemographicParity constraint, seed-42 split. Selection uses **only** the validation columns; test columns are reported for transparency. All predictions are seeded (Problem 17).

| Dataset | Epsilon ($\epsilon$) | Val Acc | Val DP | Test Acc | Test DP | Test Class Counts (0 / 1) | Selected |
|---|---|---|---|---|---|---|---|
| **German Credit** | $\epsilon = 0.005$ | 72.00% | 0.0065 | 70.50% | 0.0208 | 19 / 181 | **YES** |
| German Credit | $\epsilon = 0.010$ | 72.00% | 0.0065 | 70.50% | 0.0208 | 19 / 181 |  |
| German Credit | $\epsilon = 0.020$ | 72.00% | 0.0065 | 70.50% | 0.0208 | 19 / 181 |  |
| German Credit | $\epsilon = 0.050$ | 72.00% | 0.0065 | 70.50% | 0.0208 | 19 / 181 |  |
| German Credit | $\epsilon = 0.100$ | 70.50% | 0.0549 | 69.50% | 0.0082 | 15 / 185 |  |
| **COMPAS** | $\epsilon = 0.005$ | 57.62% | 0.0392 | 59.43% | 0.0491 | 912 / 323 |  |
| COMPAS | $\epsilon = 0.010$ | 59.08% | 0.0736 | 60.40% | 0.0281 | 994 / 241 | **YES** |
| COMPAS | $\epsilon = 0.020$ | 58.02% | 0.0530 | 60.08% | 0.0374 | 968 / 267 |  |
| COMPAS | $\epsilon = 0.050$ | 57.70% | 0.0526 | 60.49% | 0.0262 | 911 / 324 |  |
| COMPAS | $\epsilon = 0.100$ | 58.10% | 0.0381 | 60.89% | 0.0401 | 1014 / 221 |  |
| **Adult Income** | $\epsilon = 0.005$ | 84.36% | 0.0235 | 84.36% | 0.0145 | 5544 / 964 |  |
| Adult Income | $\epsilon = 0.010$ | 84.45% | 0.0207 | 84.27% | 0.0111 | 5550 / 958 |  |
| Adult Income | $\epsilon = 0.020$ | 84.46% | 0.0232 | 84.28% | 0.0118 | 5553 / 955 |  |
| Adult Income | $\epsilon = 0.050$ | 84.39% | 0.0214 | 84.36% | 0.0118 | 5550 / 958 |  |
| Adult Income | $\epsilon = 0.100$ | 84.37% | 0.0184 | 84.31% | 0.0106 | 5543 / 965 | **YES** |

---

### B. Final 10-Seed Mitigation Evaluation (Mean $\pm$ Std across 10 Seeds)

#### German Credit (Selected Baseline: Random Forest | Selected EG $\epsilon = 0.005$)
| Technique | F1 Score | Accuracy | Demographic Parity Diff | Equalized Odds Diff | Paired Delta (DP) | Significant (CI > 0) |
|---|---|---|---|---|---|---|
| **Unmitigated Baseline** | 0.8287 ± 0.0150 | 72.75% ± 2.24% | 0.0324 ± 0.0214 | 0.0739 ± 0.0301 | N/A | N/A |
| **Reweighing** | 0.8310 ± 0.0098 | 72.95% ± 1.52% | 0.0435 ± 0.0295 | 0.0873 ± 0.0252 | +0.0003 [-0.0801, 0.0916] | False |
| **Exponentiated Gradient** | 0.8293 ± 0.0081 | 72.10% ± 1.11% | 0.0332 ± 0.0110 | 0.0783 ± 0.0554 | +0.0061 [-0.0649, 0.0840] | False |
| **Threshold Optimizer** | 0.8282 ± 0.0192 | 73.95% ± 3.22% | 0.0478 ± 0.0407 | 0.1208 ± 0.0735 | -0.0411 [-0.1426, 0.0948] | False |

#### COMPAS (Selected Baseline: Random Forest | Selected EG $\epsilon = 0.010$)
| Technique | F1 Score | Accuracy | Demographic Parity Diff | Equalized Odds Diff | Paired Delta (DP) | Significant (CI > 0) |
|---|---|---|---|---|---|---|
| **Unmitigated Baseline** | 0.6344 ± 0.0107 | 68.71% ± 0.79% | 0.2473 ± 0.0204 | 0.2416 ± 0.0321 | N/A | N/A |
| **Reweighing** | 0.6293 ± 0.0088 | 68.35% ± 0.54% | 0.2408 ± 0.0202 | 0.2319 ± 0.0364 | +0.0020 [-0.0142, 0.0161] | False |
| **Exponentiated Gradient** | 0.3996 ± 0.0166 | 60.27% ± 0.64% | 0.0592 ± 0.0252 | 0.0789 ± 0.0487 | +0.2076 [0.1476, 0.2591] | **True** |
| **Threshold Optimizer** | 0.6056 ± 0.0198 | 66.40% ± 1.14% | 0.0271 ± 0.0124 | 0.0596 ± 0.0239 | +0.2158 [0.1318, 0.2680] | **True** |

#### Adult Income (Selected Baseline: XGBoost | Selected EG $\epsilon = 0.100$)
| Technique | F1 Score | Accuracy | Demographic Parity Diff | Equalized Odds Diff | Paired Delta (DP) | Significant (CI > 0) |
|---|---|---|---|---|---|---|
| **Unmitigated Baseline** | 0.6971 ± 0.0059 | 86.85% ± 0.22% | 0.1729 ± 0.0056 | 0.0944 ± 0.0245 | N/A | N/A |
| **Reweighing** | 0.6810 ± 0.0058 | 86.52% ± 0.26% | 0.0933 ± 0.0059 | 0.1372 ± 0.0256 | +0.0766 [0.0622, 0.0918] | **True** |
| **Exponentiated Gradient** | 0.5938 ± 0.0116 | 84.31% ± 0.36% | 0.0202 ± 0.0069 | 0.2584 ± 0.0278 | +0.1422 [0.1208, 0.1613] | **True** |
| **Threshold Optimizer** | 0.6397 ± 0.0109 | 85.02% ± 0.31% | 0.0089 ± 0.0061 | 0.3104 ± 0.0231 | +0.1440 [0.1062, 0.1701] | **True** |

---

## 4. Methodological Glossary for Non-Technical Readers

- **Target Leakage:** A flaw where the information the model is trying to predict is accidentally included in the training data, creating a false impression of predictive mastery.
- **Demographic Parity Difference:** The difference in positive outcome rates (e.g., credit approval rate) between the privileged group and the unprivileged group. A value of $0.00$ represents perfect statistical equality across groups.
- **Equalized Odds Difference:** A fairness metric that checks whether an algorithm has equal accuracy (both true positive rate and false positive rate) across different demographic groups.
- **Epsilon ($\epsilon$):** A mathematical tolerance dial used in the Exponentiated Gradient algorithm. A smaller epsilon strictly forces the algorithm to prioritize fairness, whereas a larger epsilon allows slight fairness deviations to preserve higher predictive accuracy.
- **Paired Bootstrap:** A statistical technique that tests whether an algorithm's improvement is genuine or merely lucky noise by recalculating performance thousands of times on re-sampled subsets of the exact same test subjects.
- **In-Processing Mitigation (Exponentiated Gradient):** A technique that changes how the model learns during training by framing fairness as a mathematical game between a predictor and an auditor.
- **Post-Processing Mitigation (Threshold Optimizer):** A technique that takes a pre-trained model and adjusts the approval/denial cutoffs independently for each demographic group to achieve fairness.
- **Pre-Processing Mitigation (Reweighing):** A technique that adjusts the weights of individual training examples before training so that underrepresented or disadvantaged groups have a balanced voice.
