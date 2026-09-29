# ⚠️ FairLens: Technical Scope & Methodological Limitations

## 1. Scope & Design Boundaries

1. **Tabular Supervised Classification Scope**: FairLens is designed for static, batch-processed tabular datasets with binary classification targets. Real-time streaming transaction scoring, deep learning, unstructured text/vision data, and multi-class target problems are outside the scope.
2. **Impossibility Theorem Constraints**: Per Kleinberg et al. (2016) and Chouldechova (2017), no classification algorithm can simultaneously optimize Demographic Parity, Equalized Odds, and Predictive Parity unless base outcome rates across groups are identical. FairLens computes all four metrics to surface these mathematical trade-offs explicitly.
3. **Data Leakage Safeguards**: Data leakage must be audited proactively during feature engineering. FairLens includes automated regression test guards (`pytest`) to prevent target alias re-entry.

4. **Post-Processing Inference Requirement**: Post-processing mitigation (Threshold Optimizer) requires access to the protected demographic attribute at inference time to select group-specific decision thresholds. In certain legal/regulatory frameworks (such as US Equal Credit Opportunity Act or specific anti-discrimination doctrines prohibiting disparate treatment), using protected attributes explicitly at the point of decision is restricted, even when done to promote fair outcomes. Pre-processing (Reweighing) and in-processing methods avoid this inference constraint by producing group-agnostic model artifacts.

---

## 2. Regulatory Alignment

FairLens implementations align with regulatory transparency expectations:
- **RBI Fair Practices Code & Digital Lending Guidelines (India)**: Requires non-discriminatory, transparent credit scoring decisions.
- **NITI Aayog Responsible AI Framework**: Calls for statistical bias auditing, proxy detection, and local explainability in algorithmic scoring systems.
- **Digital Personal Data Protection Act (DPDP 2023)**: Mandates accountable automated processing of personal and demographic data.
