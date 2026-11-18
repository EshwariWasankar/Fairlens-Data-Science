"""Generate data/demo/loan_applications_demo.csv: a synthetic loan-approval dataset for demonstrating
FairLens's CSV upload. Approvals depend on credit score, income, loan size, existing loans and job
tenure, plus a deliberate penalty for female applicants, so the audit has a real bias to find.
Women are concentrated in sectors that past lending disfavoured and earn slightly less on average,
so employer sector and income act as proxies for gender even though gender is never a model input.

Run:  python data/demo/make_demo_loans.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(2026)
n = 4000

gender = rng.choice(["Male", "Female"], n, p=[0.58, 0.42])
female = gender == "Female"
age = rng.integers(21, 61, n)
marital = np.where(rng.random(n) < np.clip((age - 20) / 25, 0.15, 0.85), "Married", "Single")
education = rng.choice(["12th Pass", "Diploma", "Graduate", "Postgraduate"], n, p=[0.15, 0.2, 0.45, 0.2])
edu_bonus = pd.Series(education).map({"12th Pass": 0.0, "Diploma": 0.15, "Graduate": 0.35, "Postgraduate": 0.55}).to_numpy()
employment = rng.choice(["Salaried", "Self-employed", "Business"], n, p=[0.6, 0.25, 0.15])
city_tier = rng.choice(["Tier 1", "Tier 2", "Tier 3"], n, p=[0.4, 0.35, 0.25])
SECTORS = ["IT", "Manufacturing", "Transport", "Finance", "Retail", "Education", "Healthcare"]
p_male = [0.24, 0.20, 0.14, 0.14, 0.12, 0.08, 0.08]
p_female = [0.14, 0.05, 0.02, 0.12, 0.17, 0.26, 0.24]
sector = np.where(female, rng.choice(SECTORS, n, p=p_female), rng.choice(SECTORS, n, p=p_male))
# historical lending favoured some sectors; these sectors are mostly male, so sector carries the gender bias
sector_bias = pd.Series(sector).map({"IT": 0.35, "Manufacturing": 0.3, "Transport": 0.25, "Finance": 0.2,
                                     "Retail": -0.1, "Education": -0.45, "Healthcare": -0.45}).to_numpy()

income = np.exp(rng.normal(1.75 + edu_bonus - 0.25 * female + 0.01 * (age - 30), 0.45))   # lakh INR per year
income = np.round(np.clip(income, 1.8, 60), 2)
loan_amount = np.round(np.clip(income * rng.uniform(0.8, 4.5, n), 1, 150), 2)            # lakh INR
loan_term = rng.choice([12, 24, 36, 60, 84], n, p=[0.1, 0.2, 0.3, 0.25, 0.15])
credit_score = np.clip(np.round(rng.normal(705, 65, n)), 350, 900).astype(int)
existing_loans = rng.choice([0, 1, 2, 3, 4], n, p=[0.35, 0.3, 0.2, 0.1, 0.05])
years_in_job = np.clip(np.round(rng.gamma(2.2, 3.2, n), 1), 0, 35)

logit = (0.022 * (credit_score - 700)
         + 0.9 * np.log(income / 6)
         - 0.55 * np.log(loan_amount / income)
         - 0.35 * existing_loans
         + 0.05 * np.minimum(years_in_job, 15)
         + 0.25 * (employment == "Salaried")
         + sector_bias
         + 0.6
         - 0.6 * female                       # the historical bias the audit should detect
         + rng.normal(0, 0.6, n))
approved = rng.random(n) < 1 / (1 + np.exp(-logit))

df = pd.DataFrame({
    "loan_status": np.where(approved, "Approved", "Rejected"),
    "gender": gender,
    "age": age,
    "marital_status": marital,
    "education": education,
    "employment_type": employment,
    "employer_sector": sector,
    "city_tier": city_tier,
    "annual_income_lakh": income,
    "loan_amount_lakh": loan_amount,
    "loan_term_months": loan_term,
    "credit_score": credit_score,
    "existing_loans": existing_loans,
    "years_in_job": years_in_job,
})
df.loc[rng.choice(n, 60, replace=False), "years_in_job"] = np.nan   # a few gaps, as in real data

out = Path(__file__).parent / "loan_applications_demo.csv"
df.to_csv(out, index=False)
print(f"wrote {out} ({len(df)} rows)")
print(df.groupby("gender")["loan_status"].apply(lambda s: (s == "Approved").mean()).round(3))
