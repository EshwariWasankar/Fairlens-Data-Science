# 📊 FairLens: Dataset Documentation

FairLens benchmark evaluations are performed on three static, independent tabular datasets representing high-stakes decision domains.

---

## 1. COMPAS Recidivism Dataset (ProPublica)

- **Domain**: Criminal Justice / Pretrial Risk Assessment.
- **Source**: ProPublica 2016 COMPAS Investigation Repository.
- **Records**: 7,214 defendants.
- **Target Column**: `two_year_recid` (0 = No Recidivism / Favorable, 1 = Recidivism / Unfavorable).
- **Protected Attributes**: `race` (Privileged: Caucasian, Unprivileged: African-American), `sex` (Privileged: Male, Unprivileged: Female).
- **Predictive Features**: `age`, `juv_fel_count`, `juv_misd_count`, `juv_other_count`, `priors_count`, `c_charge_degree`, `days_b_screening_arrest`.
- **Actionable Features**: `priors_count`, `days_b_screening_arrest`.
- **Immutable Features**: `age`, `race`, `sex`, `c_charge_degree`, `age_cat`.
- **Excluded Leakage Columns**: `decile_score`, `decile_score.1`, `score_text`, `v_decile_score`, `v_score_text`, `is_recid`, `violent_recid`, `is_violent_recid`, `r_case_number`, dates, metadata.

---

## 2. German Credit Data (Statlog)

- **Domain**: Financial Lending / Credit Risk Scoring.
- **Source**: UCI Machine Learning Repository (Statlog German Credit Dataset #144).
- **Records**: 1,000 credit applicants.
- **Target Column**: `credit_risk` (1 = Good Credit / Favorable, 0 = Bad Credit / Unfavorable).
- **Protected Attributes**: `sex` (Privileged: male, Unprivileged: female), `age_group` (Privileged: older >25, Unprivileged: young <=25), `foreign_worker` (Privileged: no, Unprivileged: yes).
- **Predictive Features**: 20 financial & demographic attributes (`duration_in_month`, `credit_amount`, `savings_account`, `housing`, `employment_duration`, etc.).
- **Actionable Features**: `duration_in_month`, `credit_amount`, `savings_account`, `employment_duration`, `installment_rate`.
- **Immutable Features**: `sex`, `age`, `age_group`, `foreign_worker`, `personal_status_sex`.

---

## 3. Adult Census Income Dataset

- **Domain**: Employment & Economic Opportunity.
- **Source**: UCI Machine Learning Repository (Adult Dataset #2).
- **Records**: 48,842 individuals (32,561 train / 16,281 test raw split).
- **Target Column**: `income` (1 = >50K / Favorable, 0 = <=50K / Unfavorable).
- **Protected Attributes**: `sex` (Privileged: Male, Unprivileged: Female), `race` (Privileged: White, Unprivileged: Black), `age_group` (Privileged: prime_age 25-60, Unprivileged: young <25).
- **Predictive Features**: `age`, `workclass`, `education`, `education_num`, `marital_status`, `occupation`, `relationship`, `race`, `capital_gain`, `capital_loss`, `hours_per_week`, `native_country`.
- **Actionable Features**: `education_num`, `hours_per_week`, `capital_gain`, `capital_loss`, `workclass`, `occupation`.
- **Immutable Features**: `sex`, `race`, `age`, `age_group`, `native_country`.
- **Excluded Feature**: `fnlwgt` (census sampling weight, removed to prevent noise/leakage).
