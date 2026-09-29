import os
import ssl
import pandas as pd
import numpy as np
import urllib.request
from pathlib import Path
from src.config import RAW_DATA_DIR, DATASET_CONFIGS

# Create unverified SSL context to prevent CERTIFICATE_VERIFY_FAILED on Windows
ssl_context = ssl._create_unverified_context()

def download_compas() -> Path:
    target_path = RAW_DATA_DIR / "compas-scores-two-years.csv"
    if target_path.exists() and target_path.stat().st_size > 1000:
        print("COMPAS dataset already exists.")
        return target_path

    url = DATASET_CONFIGS["compas"]["url"]
    print(f"Fetching COMPAS dataset from {url}...")
    req = urllib.request.urlopen(url, context=ssl_context)
    df = pd.read_csv(req)
    df.to_csv(target_path, index=False)
    print(f"Saved COMPAS dataset to {target_path}")
    return target_path

def download_german_credit() -> Path:
    target_path = RAW_DATA_DIR / "german_credit.csv"
    if target_path.exists() and target_path.stat().st_size > 1000:
        print("German Credit dataset already exists.")
        return target_path

    print("Fetching German Credit dataset...")
    # Try ucimlrepo or fallback mirrors
    df = None
    try:
        from ucimlrepo import fetch_ucirepo
        statlog = fetch_ucirepo(id=144)
        X = statlog.data.features
        y = statlog.data.targets
        df = pd.concat([X, y], axis=1)
        if 'Attribute21' in df.columns:
            df.rename(columns={'Attribute21': 'credit_risk'}, inplace=True)
    except Exception as e:
        print(f"ucimlrepo failed ({e}), trying GitHub mirror...")

    if df is None or df.empty:
        try:
            # German Credit mirrored CSV with standard names
            mirror_url = "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/germancredit.csv"
            req = urllib.request.urlopen(mirror_url, context=ssl_context)
            df_raw = pd.read_csv(req)
            # This mirror has columns like Default (1=Good, 2=Bad or 0=Bad), let's format it
            if 'Default' in df_raw.columns:
                df_raw['credit_risk'] = df_raw['Default'].apply(lambda x: 1 if x == 1 else 0)
                df_raw.drop(columns=['Default'], inplace=True)
            elif 'default' in df_raw.columns:
                df_raw['credit_risk'] = df_raw['default'].apply(lambda x: 1 if x == 1 else 0)
                df_raw.drop(columns=['default'], inplace=True)
            df = df_raw
        except Exception as e2:
            print(f"Mirror fetch failed ({e2}), fetching from UCI with unverified SSL...")
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
            col_names = [
                'status_curr_account', 'duration_in_month', 'credit_history', 'purpose',
                'credit_amount', 'savings_account', 'employment_duration', 'installment_rate',
                'personal_status_sex', 'other_debtors', 'present_residence', 'property',
                'age', 'other_installment_plans', 'housing', 'existing_credits',
                'job', 'num_dependents', 'telephone', 'foreign_worker', 'credit_risk'
            ]
            req = urllib.request.urlopen(url, context=ssl_context)
            df = pd.read_csv(req, sep=r'\s+', names=col_names)

    # Standardize German credit target and sex column
    if 'credit_risk' in df.columns:
        # Check if values are 1 and 2 or 0 and 1
        u_vals = df['credit_risk'].unique()
        if 2 in u_vals:
            df['credit_risk'] = df['credit_risk'].apply(lambda x: 1 if x == 1 else 0)

    # Decode personal_status_sex if present
    if 'personal_status_sex' in df.columns and 'sex' not in df.columns:
        def parse_sex(val):
            val_str = str(val).lower()
            if val_str in ['a92', 'a95'] or 'female' in val_str:
                return 'female'
            return 'male'
        df['sex'] = df['personal_status_sex'].apply(parse_sex)

    # If age column exists, ensure age_group is present
    if 'age' in df.columns and 'age_group' not in df.columns:
        df['age_group'] = df['age'].apply(lambda a: "older (>25)" if a > 25 else "young (<=25)")

    df.to_csv(target_path, index=False)
    print(f"Saved German Credit dataset to {target_path}")
    return target_path

def download_adult_income() -> Path:
    target_path = RAW_DATA_DIR / "adult_income.csv"
    if target_path.exists() and target_path.stat().st_size > 1000:
        print("Adult Income dataset already exists.")
        return target_path

    print("Fetching Adult Census Income dataset...")
    df = None
    try:
        from ucimlrepo import fetch_ucirepo
        adult = fetch_ucirepo(id=2)
        X = adult.data.features
        y = adult.data.targets
        df = pd.concat([X, y], axis=1)
    except Exception as e:
        print(f"ucimlrepo failed ({e}), using raw GitHub mirror / direct UCI URL...")

    if df is None or df.empty:
        try:
            mirror_url = "https://raw.githubusercontent.com/tirthajack/Adult-Census-Income-Prediction/master/adult.csv"
            req = urllib.request.urlopen(mirror_url, context=ssl_context)
            df = pd.read_csv(req)
        except Exception as e2:
            print(f"GitHub mirror failed ({e2}), downloading UCI data...")
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
            col_names = [
                'age', 'workclass', 'fnlwgt', 'education', 'education_num',
                'marital_status', 'occupation', 'relationship', 'race', 'sex',
                'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income'
            ]
            req = urllib.request.urlopen(url, context=ssl_context)
            df = pd.read_csv(req, names=col_names, skipinitialspace=True)

    # Standardize column names
    df.columns = [c.replace('-', '_').strip() for c in df.columns]

    df.to_csv(target_path, index=False)
    print(f"Saved Adult Income dataset to {target_path}")
    return target_path

def download_all_datasets():
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    compas_p = download_compas()
    german_p = download_german_credit()
    adult_p = download_adult_income()
    return {
        "compas": compas_p,
        "german_credit": german_p,
        "adult_income": adult_p
    }

if __name__ == "__main__":
    download_all_datasets()

