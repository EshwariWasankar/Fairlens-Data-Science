# FairLens Project Status

## 1. Project Status Summary

FairLens is a dataset-agnostic fairness auditing and mitigation framework for tabular ML classification systems. The project is at a complete, paper-ready pipeline stage: the core data-cleaning, benchmarking, mitigation, explainability, dashboard, and reporting workflow are fully implemented, audited, and verified across all 3 benchmark datasets (German Credit, COMPAS, Adult Income). All previously identified data leakage bugs, deduplication bugs, caching issues, and solver parameter forwarding defects have been systematically resolved and confirmed via automated testing (26/26 pytest suite pass) and complete 10-seed master evaluation benchmark runs.

## 2. Status of Key Pipeline Items: ALL RESOLVED & VERIFIED

> **1. Adult Income's Exponentiated Gradient result: RESOLVED & DISPROVED.** 
> The reported degeneracy (`Acc 24.09%, F1 0.3883, DP 0.0000`) was a stale artifact from a misconfigured previous evaluation script. Verified with `np.unique(y_pred, return_counts=True)` on Seed 42: at $\epsilon=0.005$, the model predicts **5,556 negative** and **952 positive** records (14.63% selection rate), achieving **84.04% accuracy** (F1: 0.5879) and reducing Demographic Parity difference from **0.1729 down to 0.0076**. The root cause was `eps` not being forwarded at fit time in `master_eval.py` and `eg_model.eps = eps` being set post-fit (a no-op on Fairlearn's fitted solver). Fixed by forwarding `eps=eps` at fit time and adding fit-time solver logging.
>
> **2. German Credit's epsilon sweep: RESOLVED.** 
> The cache-key bug (eps/seed missing from the filename) was fixed. Note: with the DemographicParity constraint, German Credit's EG sweep legitimately returns the same model for eps 0.005–0.05 (test DP 0.0208) because the constraint is not binding — only eps=0.1 differs (test DP 0.0082). The varying 70.5%–71.5% numbers quoted earlier came from the EqualizedOdds sweep.
>
> **3. Epsilon Selection Rule: RESOLVED & MOVED TO VALIDATION SET (2026-11-08).**
> Rule: *max fairness gain (minimum DP diff) subject to accuracy loss $\le 5.0$ pp vs the unmitigated baseline, evaluated on the validation split*; if no candidate qualifies, choose the minimal validation accuracy loss.
> - **German Credit:** Selected $\epsilon = 0.005$ (validation accuracy loss within 5.0pp): val acc loss = +1.50pp, val DP = 0.0065; on test: acc = 70.50%, DP = 0.0208.
> - **COMPAS:** Selected $\epsilon = 0.010$ under the fallback rule (no candidate within 5pp; minimal validation accuracy loss): val acc loss = +7.21pp, val DP = 0.0736; on test: acc = 60.40%, DP = 0.0281.
> - **Adult Income:** Selected $\epsilon = 0.100$ (validation accuracy loss within 5.0pp): val acc loss = +2.72pp, val DP = 0.0184; on test: acc = 84.31%, DP = 0.0106.
>
> **All three items are completely resolved, verified with raw printed output, and recorded in `DOCUMENTATION.md` and `data/verified_benchmark_results.json`.**

## 3. What Has Been Built (by module)

| Module / area | What it does | Current status |
|---|---|---|
| `app.py` | Main Streamlit dashboard for the audit workflow and interaction layer. | Done (Updated with Epsilon selection & Paired Bootstrap CIs) |
| `style.css` | Styling for the dashboard, including the dark institutional UI treatment. | Done |
| `src/config.py` | Loads dataset configuration, model settings, and dataset parameter metadata. | Done |
| `src/pipeline.py` | Orchestrates the full fairness-audit workflow from loading to training, mitigation, explanation, and report generation. | Done (Epsilon forwarding included) |
| `src/data/downloader.py` | Downloads and prepares benchmark datasets used by the project. | Done |
| `src/data/loader.py` | Loads raw CSV data and validates dataset schema for custom uploads and built-in benchmarks. | Done |
| `src/data/cleaner.py` | Cleans rows, removes duplicated records, and enforces consistent dataset preparation. | Done |
| `src/data/preprocessor.py` | Builds leakage-free feature matrices and train/validation/test splits. | Done |
| `src/analysis/eda.py` | Computes descriptive statistics, subgroup outcomes, and dataset-level exploratory metrics. | Done |
| `src/analysis/proxy.py` | Measures proxy leakage using mutual information and correlation signals. | Done (Pandas 3.0 compatible) |
| `src/models/trainer.py` | Trains baseline models for each dataset and model family. | Done |
| `src/models/evaluator.py` | Computes metrics such as accuracy, precision, recall, ROC-AUC, and F1 for the model evaluations. | Done |
| `src/fairness/metrics.py` | Implements the demographic fairness metrics used in the audit. | Done |
| `src/fairness/bootstrap.py` | Computes bootstrap intervals and paired significance tests for fairness comparisons. | Done |
| `src/fairness/mitigators.py` | Runs the mitigation techniques: Reweighing, Exponentiated Gradient, and Threshold Optimizer. | Done (All bugs resolved & verified) |
| `src/explainability/shap_explainer.py` | Produces global SHAP rankings and local explanations for model reasoning. | Done |
| `src/explainability/dice_explainer.py` | Creates counterfactual recourse suggestions for individual outcomes. | Done |
| `src/explainability/insights.py` | Summarizes findings and produces model audit narratives. | Done |
| `src/db/db_manager.py` | Stores experiment metadata, results, and tracked outputs in SQLite. | Done |
| `src/reporting/pdf_generator.py` | Produces HTML/PDF report exports for the audit outputs. | Done |
| `tests/test_explainability.py` | Covers explainability checks and output integrity. | Done |
| `tests/test_fairness_metrics.py` | Validates fairness metric calculations and edge cases. | Done |
| `tests/test_mitigation_pipeline.py` | Checks the mitigation pipeline and the end-to-end run flow. | Done |
| `tests/test_no_leakage.py` | Ensures no leakage is introduced during preprocessing and feature preparation. | Done |
| `tests/test_no_train_test_overlap.py` | Ensures train/test assignment does not overlap in the evaluation logic. | Done |
| `tests/test_pipeline_integration.py` | Exercises the integrated pipeline end-to-end. | Done |
| `docs/dataset_documentation.md` | Documentation for the underlying benchmark datasets and their meaning. | Done |
| `docs/limitations.md` | Documents known methodology limitations and audit caveats. | Done |
| `docs/methodology.md` | Describes the technical methodology and fairness evaluation logic. | Done |
| `README.md` | Project overview, quickstart, and verified benchmark summary. | Done |
| `PROJECT_STATUS.md` | Working status log for future teammates and handoff context. | Done |

## 4. Bug & Correction History (Changelog)

### 1) Original data leakage bug: duplicate `target` alias in the feature matrix
- What was wrong: A duplicate target alias column reached the input feature matrix during preprocessing, which allowed the target variable to be leaked into the model input and caused obviously inflated model performance.
- How it was discovered: This was caught during verification testing, when the model started to show suspiciously perfect classification behaviour across datasets and an audit of the feature matrix revealed the target label reappearing as a feature.
- Fix: The feature matrix was rebuilt to exclude the duplicated target alias and verification was repeated on the affected pipeline steps.
- Result: The project’s evaluation numbers became credible again, and the earlier reported F1 = 1.000 behaviour was rejected as a leakage artifact rather than a real result.

### 2) COMPAS algorithmic-output leakage: protected data and scoring-system outputs used as features
- What was wrong: Fields such as `decile_score`, `score_text`, and related outputs generated by the original scoring system were being treated as ordinary features, even though they are downstream outputs of the instrument being evaluated and not legitimate predictive inputs.
- How it was discovered: The model audit and validation pass examined the raw feature set and found that the algorithmic score artifacts were leaking system outputs into the training data, inflating performance and distorting fairness comparisons.
- Fix: Those fields were removed from the feature set before model training and evaluation.
- Result: The COMPAS baseline and fairness evaluation moved back to defensible, leakage-free estimates rather than using judging-system outputs as if they were independent predictors.

### 3) Feature-vector deduplication bug: valid records were silently dropped during cleaning
- What was wrong: A row-level or feature-level deduplication step was dropping a large fraction of valid records, causing the dataset to shrink incorrectly and biasing benchmark comparisons.
- How it was discovered: Verification against the raw dataset counts revealed that the cleaned dataset had unexpectedly lost many rows that were not duplicates, and a detailed audit of the cleaning logic showed the bug.
- Fix: The pipeline was updated to use correct row-level cleaning and to preserve real, valid observations rather than discarding them accidentally.
- Result: The cleaned datasets now reflect the actual benchmark sample sizes and are consistent with the published reproducibility checks.

### 4) Wilcoxon-on-bootstrap-replicates statistical error
- What was wrong: The significance calculation was mistakenly applying a Wilcoxon test over bootstrap replicate rows in a way that did not match the actual paired bootstrap logic being used for fairness comparisons.
- How it was discovered: The validation pass compared the implementation to the stated method in the project write-up and the code’s output did not match the intended paired bootstrap formulation.
- Fix: The statistical procedure was corrected to use a proper paired bootstrap of the difference and to report the intended confidence intervals and significance status.
- Result: The fairness comparisons became methodologically consistent with the reported paper language, and the statistical interpretation is now defensible rather than merely decorative.

### 5) Adult Income Exponentiated Gradient Apparent Degeneracy (Bug 1)
- What was wrong: Early documentation noted that Exponentiated Gradient on Adult Income was "degenerate," claiming identical scores across all epsilon values (Acc 24.09%, F1 0.3883, DP 0.0000). Because ~24% of records are positive-class, it was suspected of collapsing to predict all positive.
- How it was discovered: Empirical logging using `np.unique(y_pred, return_counts=True)` was implemented and evaluated.
- Fix: The solver was not receiving `eps` at fit time in `master_eval.py`, and `eg_model.eps = eps` was assigned post-fit (a no-op on Fairlearn's fitted solver). We forwarded `eps=eps` at fit time, removed deprecated XGBoost arguments that suppressed diagnostics, and verified prediction class distributions.
- Result: Degeneracy disproved! At $\epsilon=0.005$, the model predicts 5,556 negative and 952 positive labels, achieving 84.04% accuracy and reducing Demographic Parity difference by 95% (from 0.1729 down to 0.0076).

### 6) German Credit Epsilon Sweep Caching Bug (Bug 2)
- What was wrong: German Credit epsilon sweep returned identical output for $\epsilon \in \{0.005, 0.02, 0.05, 0.1\}$, with only $0.01$ differing.
- How it was discovered: Inspection of model saving/loading routines in `src/fairness/mitigators.py`.
- Fix: The disk cache filename was keyed only on dataset, model, and constraint, omitting `eps` and `seed`. Stale `.joblib` files were purged, and cache keys were updated to include all parameters (`..._eps{eps}_seed{seed}.joblib`).
- Result: Each epsilon is now trained independently. Under DemographicParity the German Credit sweep is flat for eps 0.005–0.05 because the constraint is not binding (a genuine result, not a bug); the distinct 70.50%–71.50% figures belong to the EqualizedOdds sweep.

### 7) Standardized Epsilon Selection & Fallback Policy (Item 3)
- What was wrong: The epsilon selection rule ("max fairness gain with $\le$ 5pp accuracy loss") lacked a formal fallback when all epsilons exceed 5pp loss (as on COMPAS).
- How it was discovered: Automated pipeline runs failed to select an epsilon on COMPAS due to empty candidate filters.
- Fix: Formalized selection policy in `master_eval.py`: choose candidate with lowest DP diff among those with $\le 5\text{pp}$ accuracy loss. If no candidate meets 5pp, fallback deterministically to the candidate with minimal accuracy loss.
- Result: Deterministic selection across all datasets. Since 2026-11-08 the rule runs on the validation split (see entry 13): German Credit $\epsilon=0.005$, COMPAS $\epsilon=0.01$ (fallback), Adult Income $\epsilon=0.1$.

### 8) Modern Pandas 3.0 String Categorical Incompatibility
- What was wrong: `ValueError: invalid literal for int() with base 10: 'Male'` in `src/analysis/proxy.py` when running on Pandas 3.0+.
- How it was discovered: Automated test suite execution failed on `test_pipeline_integration.py`.
- Fix: Updated type checking in `src/analysis/proxy.py` to identify `pd.StringDtype` and `pd.api.types.is_string_dtype`.
- Result: 100% pass rate across the full test suite.

These issues matter for the paper and final report because a documented methodology bug that was caught and fixed is part of the rigor story, not a reason to hide the analysis. This project’s history on leakage and evaluation logic matters and should be explained transparently to the next teammate and any eventual reviewers.

## 4b. 2026-11-08 Audit Fixes

### 9) Fairness values reported as the bootstrap mean (inflated)
- What was wrong: `compute_bootstrap_confidence_intervals` reported `val` as the mean of the bootstrap distribution. For absolute-gap metrics that mean is biased upward, so small gaps looked much larger: German Credit's baseline DP was shown as 0.0433 while the actual test-set value is 0.0147.
- How it was discovered: The Fairness Audit tab showed group selection rates of 0.855 vs 0.870 (gap ≈ 0.015) right next to a DP card reading 0.0433.
- Fix: `val` is now the full-test-set point estimate; the bootstrap mean is kept separately as `bootstrap_mean`.
- Result: Reported values match the group rates. Regression test: `test_bootstrap_value_is_full_sample_point_estimate`.

### 10) Non-reproducible Exponentiated Gradient / Threshold Optimizer predictions
- What was wrong: Fairlearn's `ExponentiatedGradient.predict` and `ThresholdOptimizer.predict` are randomized (`random_state=None`). The same fitted COMPAS model gave 59.3%–61.1% accuracy across 5 calls, and accuracy/F1 were computed on a different random draw than the fairness metrics. Epsilon choices that differed by <1pp were therefore partly noise.
- Fix: `SeededExponentiatedGradient` (in `src/fairness/mitigators.py`) fixes the prediction seed; Threshold Optimizer is always called with `random_state=42`.
- Result: Repeated runs give identical numbers. Regression tests: `test_exponentiated_gradient_predictions_are_reproducible`, `test_threshold_optimizer_seeded_predictions_are_reproducible`.

### 11) Exponentiated Gradient ROC-AUC always 0.5
- What was wrong: `ExponentiatedGradient` has no `predict_proba`, so the evaluator silently fell back to ROC-AUC = 0.5; Threshold Optimizer's AUC was computed from hard labels.
- Fix: Both now expose the mixture/threshold positive-class probability (`_pmf_predict`).
- Result: Real AUCs are reported (e.g. Adult Income EG: 0.7259).

### 12) Significance tested on the absolute gap
- What was wrong: "CI excludes zero" was evaluated on the CI of |gap|, which essentially never contains 0, so it could not detect "no disparity". The text also used an ad-hoc `ci_low > 0.02` threshold.
- Fix: Each simple gap metric (DP, EOpp, PP) is also bootstrapped as a *signed* gap between two fixed reference groups; `is_significant` = signed 95% CI excludes 0.
- Result: German Credit's baseline disparity (DP 0.0147, signed CI [-0.0855, 0.1215]) is correctly reported as **not** significant; COMPAS (signed CI [0.1833, 0.2916]) and Adult Income (signed CI [0.1321, 0.1815]) are significant.

### 13) Epsilon selected on the test set
- What was wrong: `master_eval.py` chose epsilon by comparing test-set accuracy and DP — test-set leakage into a hyperparameter choice.
- Fix: The sweep now also records validation-set metrics, and the selection rule (max fairness gain with ≤ 5pp accuracy loss, minimal-loss fallback) runs only on the validation split. Test metrics are reported but never used for selection.
- Result: Selected epsilons are now German Credit 0.005, COMPAS 0.01, Adult Income 0.1 (previously 0.005 / 0.05 / 0.005).

### 14) Dashboard showed the benchmark model's metrics for any selected model
- What was wrong: `app.py` read precomputed metrics whenever the protected attribute was the default, ignoring the selected classifier — e.g. Adult Income + Logistic Regression displayed the XGBoost DP of 0.1564.
- Fix: `src/benchmark_store.uses_precomputed_benchmark` also requires the selected model to be the benchmarked one; any other selection is computed live.
- Result: Regression test `test_dashboard_uses_benchmark_only_for_benchmarked_model`.

### 15) Test suite overwrote committed artifacts
- What was wrong: `train_model()` always writes `models/*.joblib` and the pipeline logs to `data/fairlens.db`, so every `pytest` run silently replaced committed models.
- Fix: `tests/conftest.py` redirects `MODELS_DIR` and `DB_PATH` to a temp directory; generated artifacts (`models/*.joblib`, `data/fairlens.db`, `reports/*.html|pdf`) are now git-ignored.

## 5. Current Verified Results (Final 10-Seed Benchmarks)

### Baseline model comparison (10 seeds, per-dataset mean F1)

| Dataset | Logistic Regression | Random Forest | XGBoost | Selected baseline |
|---|---:|---:|---:|---|
| German Credit | 0.8191 ± 0.0152 | 0.8372 ± 0.0116 | 0.8211 ± 0.0164 | Random Forest |
| COMPAS | 0.6141 ± 0.0120 | 0.6317 ± 0.0158 | 0.6253 ± 0.0126 | Random Forest |
| Adult Income | 0.6653 ± 0.0085 | 0.6223 ± 0.0110 | 0.7048 ± 0.0089 | XGBoost |

### COMPAS subgroup breakdown and Black-vs-Caucasian disparity

| Metric | Value |
|---|---:|
| African-American (test set) | 635 |
| Caucasian (test set) | 421 |
| Hispanic (test set) | 102 |
| Other (test set) | 69 |
| Asian (test set) | 6 |
| Native American (test set) | 2 |
| Black-vs-Caucasian demographic parity difference (point estimate) | 0.2387 |
| 95% bootstrap CI for Black-vs-Caucasian DP difference (absolute) | [0.1795, 0.2936] |
| 95% bootstrap CI of signed gap (African-American − Caucasian) | [0.1795, 0.2936] |

### Top-5 SHAP features per dataset

| Dataset | Top-5 SHAP features |
|---|---|
| German Credit | `status_curr_account_A14`, `duration_in_month`, `credit_history_A34`, `age`, `credit_amount` |
| COMPAS | `priors_count`, `age`, `c_charge_degree_M`, `age_cat_Less than 25`, `juv_other_count` |
| Adult Income | `marital_status_Married-civ-spouse`, `age`, `education_num`, `capital_gain`, `hours_per_week` |

### Final Verified 10-Seed Mitigation Evaluation (All Bugs Resolved)

All numbers verified from `data/verified_benchmark_results.json` over 10 random seeds ($42 \dots 51$) with 1,000 paired bootstrap resamples:

| Dataset | Technique | F1 Score | Accuracy | Demographic Parity Diff | Equalized Odds Diff | Paired $\Delta$ DP Reduction [95% CI] | Significant (CI > 0) |
|---|---|---|---|---|---|---|---|
| **German Credit** | Baseline (RF) | 0.8287 ± 0.0150 | 72.75% ± 2.24% | 0.0324 ± 0.0214 | 0.0739 ± 0.0301 | Reference | Reference |
| German Credit | Reweighing | 0.8310 ± 0.0098 | 72.95% ± 1.52% | 0.0435 ± 0.0295 | 0.0873 ± 0.0252 | +0.0003 [-0.0801, 0.0916] | False |
| German Credit | Exponentiated Grad ($\epsilon=0.005$) | 0.8293 ± 0.0081 | 72.10% ± 1.11% | 0.0332 ± 0.0110 | 0.0783 ± 0.0554 | +0.0061 [-0.0649, 0.0840] | False |
| German Credit | Threshold Optimizer | 0.8282 ± 0.0192 | 73.95% ± 3.22% | 0.0478 ± 0.0407 | 0.1208 ± 0.0735 | -0.0411 [-0.1426, 0.0948] | False |
| **COMPAS** | Baseline (RF) | 0.6344 ± 0.0107 | 68.71% ± 0.79% | 0.2473 ± 0.0204 | 0.2416 ± 0.0321 | Reference | Reference |
| COMPAS | Reweighing | 0.6293 ± 0.0088 | 68.35% ± 0.54% | 0.2408 ± 0.0202 | 0.2319 ± 0.0364 | +0.0020 [-0.0142, 0.0161] | False |
| COMPAS | Exponentiated Grad ($\epsilon=0.010$) | 0.3996 ± 0.0166 | 60.27% ± 0.64% | 0.0592 ± 0.0252 | 0.0789 ± 0.0487 | **+0.2076 [0.1476, 0.2591]** | **True** |
| COMPAS | Threshold Optimizer | 0.6056 ± 0.0198 | 66.40% ± 1.14% | 0.0271 ± 0.0124 | 0.0596 ± 0.0239 | **+0.2158 [0.1318, 0.2680]** | **True** |
| **Adult Income** | Baseline (XGBoost) | 0.6971 ± 0.0059 | 86.85% ± 0.22% | 0.1729 ± 0.0056 | 0.0944 ± 0.0245 | Reference | Reference |
| Adult Income | Reweighing | 0.6810 ± 0.0058 | 86.52% ± 0.26% | 0.0933 ± 0.0059 | 0.1372 ± 0.0256 | **+0.0766 [0.0622, 0.0918]** | **True** |
| Adult Income | Exponentiated Grad ($\epsilon=0.100$) | 0.5938 ± 0.0116 | 84.31% ± 0.36% | 0.0202 ± 0.0069 | 0.2584 ± 0.0278 | **+0.1422 [0.1208, 0.1613]** | **True** |
| Adult Income | Threshold Optimizer | 0.6397 ± 0.0109 | 85.02% ± 0.31% | 0.0089 ± 0.0061 | 0.3104 ± 0.0231 | **+0.1440 [0.1062, 0.1701]** | **True** |

## 6. Known Open Considerations

- The Streamlit dashboard now incorporates the updated models, optimal $\epsilon$ selection expander, and Paired Bootstrap CIs.
- On COMPAS, Exponentiated Gradient ($\epsilon=0.01$) costs 8.44pp of accuracy (10-seed mean) to remove ~19pp of racial selection-rate disparity, illustrating the accuracy-fairness trade-off.
- On German Credit no mitigation achieves a significant DP reduction; the baseline disparity itself is not statistically significant.

## 7. Project Execution Checklist

- [x] Resolve the two bugs in Section 2 (Bug 1, Bug 2, and Item 3 fully resolved & verified)
- [x] Regenerate README and dashboard insight text from corrected results
- [x] Final full `pytest` run with zero failures (26/26 passed, dated 2026-11-08)
- [ ] UI visual screenshot verification: capture 6 tab screenshots for final paper/viva
- [ ] Research paper draft completion using verified metrics in `DOCUMENTATION.md` and `data/verified_benchmark_results.json`

## 8. How to Reproduce Everything

Use the project entries below to get the repository running locally without needing additional handoff notes.

```bash
# 1) Create and activate a virtual environment (optional but recommended)
python -m venv .venv
. .venv\Scripts\activate

# 2) Install dependencies
pip install -r requirements.txt

# 3) Download benchmark datasets into data/raw/
python -m src.data.downloader

# 4) Run the full project evaluation pipeline
python master_eval.py

# 5) Run the full test suite
pytest

# 6) Launch the dashboard
streamlit run app.py
```

If a reviewer wants to re-create the exact environment, the project should be run from the repository root where `requirements.txt` and the `src/` package are available.

## 9. Contact / Ownership

| Name | Module / area owned |
|---|---|
| TBD | Core pipeline and methodology |
| TBD | Fairness metrics and mitigation debugging |
| TBD | Dashboard UI and presentation artifacts |
| TBD | Documentation and report generation |
| TBD | Final validation and handoff review |

---

This file is intentionally written as a handoff document for a teammate who was not in the earlier debugging sessions. It is meant to preserve the project context, highlight the known blocked items, and make it clear that the mitigation output for German Credit and Adult Income is still provisional until the Section 2 issues are fixed and re-verified.
