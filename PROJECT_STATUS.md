# FairLens Project Status

## 1. Project Status Summary

FairLens is a dataset-agnostic fairness auditing and mitigation framework for tabular ML classification systems. The project is at a near-complete pipeline stage: the core data-cleaning, benchmarking, mitigation, explainability, and reporting workflow are already implemented and largely verified, but the project is not yet paper-ready because two of the mitigation results are still blocked by unresolved bugs. The most important thing for any teammate to keep in mind before changing anything is that this project has already had at least one serious data leakage bug and one caching bug; any modification to the preprocessing or mitigation logic should be treated as a high-risk change and must be re-verified with the full test suite and the relevant benchmark outputs before being trusted.

## 2. CRITICAL — Assigned Next Task: Fix These Two Bugs Before Any Further Mitigation or Paper Work

> **1. Adult Income's Exponentiated Gradient result is degenerate.** Across all 5 epsilon values tested (0.005 to 0.1), the result is identical to 4 decimal places (Acc 24.09%, F1 0.3883, DP 0.0000), and across all 10 seeds in the full evaluation the standard deviation is exactly 0.0000. Since ~24% of Adult Income records are positive-class, this pattern is consistent with the model collapsing to "always predict positive" — a useless constant classifier that trivially achieves demographic parity. Whoever picks this up should print the actual predicted class distribution (count of each predicted class) for this specific model/dataset combination to confirm. If it is a constant classifier: check whether `ExponentiatedGradient` is receiving properly scaled/typed data for this dataset specifically, check solver convergence warnings (they are likely being suppressed), and check whether the `eps` parameter is actually reaching the underlying solver call (add a print statement confirming the exact `eps` value used at fit time). This result must not be used in any report or the paper until it's either fixed or confirmed genuine with evidence.
>
>
> **2. German Credit's epsilon sweep returns identical output for eps = 0.005, 0.02, 0.05, and 0.1.** Only eps = 0.01 differs. Whoever picks this up should check whether the training/caching function for this mitigation is keyed only on `dataset` (or `dataset + technique`) rather than `dataset + technique + eps` — if so, later eps values in the loop are returning a cached model from the first run instead of retraining. Fix the cache key and re-run the full sweep for German Credit specifically to confirm genuinely different results per eps.
>
>
> **3. Related to both of the above: confirm, in code and in output, which epsilon value was actually selected for each dataset's final mitigation evaluation, and why** (the stated rule was "max fairness gain with ≤5pt accuracy loss" — this needs to be confirmed as applied consistently per dataset, with the chosen eps printed for COMPAS and Adult Income specifically, not just German Credit).
>
>
> **These three items must be resolved and re-verified with raw output before any of the current mitigation numbers for Adult Income or German Credit are used in the paper or final report.**

**Whoever picks this up: resolve all three items above and re-verify with raw output before using the affected numbers anywhere in the paper or reports.**

## 3. What Has Been Built (by module)

| Module / area | What it does | Current status |
|---|---|---|
| `app.py` | Main Streamlit dashboard for the audit workflow and interaction layer. | Done |
| `style.css` | Styling for the dashboard, including the dark institutional UI treatment. | Done |
| `src/config.py` | Loads dataset configuration, model settings, and dataset parameter metadata. | Done |
| `src/pipeline.py` | Orchestrates the full fairness-audit workflow from loading to training, mitigation, explanation, and report generation. | Done |
| `src/data/downloader.py` | Downloads and prepares benchmark datasets used by the project. | Done |
| `src/data/loader.py` | Loads raw CSV data and validates dataset schema for custom uploads and built-in benchmarks. | Done |
| `src/data/cleaner.py` | Cleans rows, removes duplicated records, and enforces consistent dataset preparation. | Done |
| `src/data/preprocessor.py` | Builds leakage-free feature matrices and train/validation/test splits. | Done |
| `src/analysis/eda.py` | Computes descriptive statistics, subgroup outcomes, and dataset-level exploratory metrics. | Done |
| `src/analysis/proxy.py` | Measures proxy leakage using mutual information and correlation signals. | Done |
| `src/models/trainer.py` | Trains baseline models for each dataset and model family. | Done |
| `src/models/evaluator.py` | Computes metrics such as accuracy, precision, recall, ROC-AUC, and F1 for the model evaluations. | Done |
| `src/fairness/metrics.py` | Implements the demographic fairness metrics used in the audit. | Done |
| `src/fairness/bootstrap.py` | Computes bootstrap intervals and paired significance tests for fairness comparisons. | Done |
| `src/fairness/mitigators.py` | Runs the mitigation techniques: Reweighing, Exponentiated Gradient, and Threshold Optimizer. | Blocked by the bugs in Section 2 |
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

These issues matter for the paper and final report because a documented methodology bug that was caught and fixed is part of the rigor story, not a reason to hide the analysis. This project’s history on leakage and evaluation logic matters and should be explained transparently to the next teammate and any eventual reviewers.

## 5. Current Verified Results (Final Numbers, Excluding the Two Blocked Items)

### Baseline model comparison (10 seeds, per-dataset mean F1)

| Dataset | Logistic Regression | Random Forest | XGBoost | Selected baseline |
|---|---:|---:|---:|---|
| German Credit | 0.8191 ± 0.0152 | 0.8369 ± 0.0115 | 0.8213 ± 0.0163 | Random Forest |
| COMPAS | 0.6141 ± 0.0120 | 0.6316 ± 0.0161 | 0.6253 ± 0.0126 | Random Forest |
| Adult Income | 0.6653 ± 0.0085 | 0.6225 ± 0.0111 | 0.7048 ± 0.0089 | XGBoost |

### COMPAS subgroup breakdown and Black-vs-Caucasian disparity

| Metric | Value |
|---|---:|
| African-American (test set) | 635 |
| Caucasian (test set) | 421 |
| Hispanic (test set) | 102 |
| Other (test set) | 69 |
| Asian (test set) | 6 |
| Native American (test set) | 2 |
| Black-vs-Caucasian demographic parity difference (point estimate) | 0.2383 |
| 95% bootstrap CI for Black-vs-Caucasian DP difference | [0.1808, 0.2948] |

### Top-5 SHAP features per dataset

| Dataset | Top-5 SHAP features |
|---|---|
| German Credit | `status_curr_account_A14`, `duration_in_month`, `credit_history_A34`, `age`, `credit_amount` |
| COMPAS | `priors_count`, `age`, `c_charge_degree_M`, `age_cat_Less than 25`, `juv_other_count` |
| Adult Income | `marital_status_Married-civ-spouse`, `age`, `education_num`, `capital_gain`, `hours_per_week` |

### Mitigation tables currently in project results

The mitigation tables from the verified benchmark outputs are still useful for the pipeline workflow, but the following rows are marked as provisional and blocked pending the Section 2 fixes because they rely on the unresolved Exponentiated Gradient and epsilon-caching issues.

| Dataset | Mitigation | F1 | Accuracy | Demographic Parity Diff | Status |
|---|---|---:|---:|---:|---|
| German Credit | Reweighing | 0.8282 ± 0.0107 | 72.65% ± 1.61% | 0.0333 ± 0.0207 | provisional — blocked pending Section 2 fixes |
| German Credit | Exponentiated Gradient | 0.8278 ± 0.0061 | 71.80% ± 0.95% | 0.0206 ± 0.0214 | provisional — blocked pending Section 2 fixes |
| German Credit | Threshold Optimizer | 0.8276 ± 0.0212 | 73.85% ± 3.57% | 0.0560 ± 0.0394 | provisional — blocked pending Section 2 fixes |
| Adult Income | Reweighing | 0.6810 ± 0.0058 | 86.52% ± 0.26% | 0.0933 ± 0.0059 | provisional — blocked pending Section 2 fixes |
| Adult Income | Exponentiated Gradient | 0.3883 ± 0.0000 | 24.09% ± 0.00% | 0.0000 ± 0.0000 | provisional — blocked pending Section 2 fixes |
| Adult Income | Threshold Optimizer | 0.6391 ± 0.0108 | 84.98% ± 0.32% | 0.0092 ± 0.0047 | provisional — blocked pending Section 2 fixes |

## 6. Known Open Issues / Things to Double-Check (Beyond Section 2)

- The redesigned dashboard UI has not yet been visually confirmed with screenshots. The code may compile, but a human still needs to open the app and click through all tabs to confirm the institutional design language (no emoji, consistent font and colors, bar charts rather than donuts, correct label placement, and a readable correlation heatmap).
- The final PDF/HTML report and dashboard narrative text should be regenerated only after the blocked mitigation bugs are fixed and the numbers are re-verified.
- All results should be treated as versioned artifacts and tracked from the same run that produced the final output, rather than mixing old and new numbers across different evaluations.
- Any future change to the preprocessing or mitigation implementation should be treated as a potentially high-impact change because the project already caught two major methodology bugs in the verification phase.

## 7. Remaining Tasks for Paper-Ready / Complete Project Execution

- [ ] Resolve the two bugs in Section 2
- [ ] **UI still needs improvement.** The dashboard redesign has not been visually confirmed with screenshots — someone needs to run the app, click through all 6 tabs, and confirm the institutional design system (no emoji, consistent fonts/colors, bar charts not donuts, correctly labeled correlation heatmap) actually renders as intended, not just that the code compiles. Save screenshots of every tab into a `docs/screenshots/` folder for the paper and viva.
- [ ] Regenerate the PDF/HTML report, README, and dashboard insight text from the corrected results once Section 2's bugs are fixed.
- [ ] Full manual click-through of the deployed/local app confirming no runtime errors, for all dataset × model × mitigation combinations.
- [ ] Decide and document: Streamlit Community Cloud deployment, or local-only for the viva.
- [ ] Write the research paper draft using the verified results (Methodology, Results, Discussion sections can draw directly from this file and `docs/methodology.md`/`docs/limitations.md`).
- [ ] Prepare final presentation slides with the corrected numbers.
- [ ] Final full `pytest` run with zero failures, pasted into this file as proof, dated.

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
