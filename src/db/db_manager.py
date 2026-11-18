import sqlite3
import json
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List, Optional
from src.config import DB_PATH

def init_db():
    """Initializes the SQLite database schema if tables do not exist."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Datasets Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS datasets (
        dataset_id TEXT PRIMARY KEY,
        name TEXT,
        target_column TEXT,
        protected_attributes TEXT,
        num_records INTEGER,
        date_added DATETIME
    );
    """)

    # Experiments Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS experiments (
        experiment_id INTEGER PRIMARY KEY AUTOINCREMENT,
        dataset_id TEXT,
        model_type TEXT,
        mitigation_technique TEXT,
        f1_score REAL,
        roc_auc REAL,
        demographic_parity_diff REAL,
        dp_ci_low REAL,
        dp_ci_high REAL,
        equalized_odds_diff REAL,
        equal_opportunity_diff REAL,
        predictive_parity_diff REAL,
        p_value REAL,
        is_significant INTEGER,
        timestamp DATETIME,
        FOREIGN KEY (dataset_id) REFERENCES datasets (dataset_id)
    );
    """)

    # Proxy Analysis Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS proxy_analysis (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        dataset_id TEXT,
        feature_name TEXT,
        protected_attribute TEXT,
        mutual_information REAL,
        correlation REAL,
        risk_level TEXT,
        FOREIGN KEY (dataset_id) REFERENCES datasets (dataset_id)
    );
    """)

    conn.commit()
    conn.close()

def log_dataset_metadata(dataset_id: str, name: str, target_col: str, protected_attrs: list, num_records: int):
    """Registers dataset metadata in SQLite DB."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
    INSERT OR REPLACE INTO datasets (dataset_id, name, target_column, protected_attributes, num_records, date_added)
    VALUES (?, ?, ?, ?, ?, ?)
    """, (dataset_id, name, target_col, json.dumps(protected_attrs), num_records, datetime.now().isoformat(sep=' ', timespec='seconds')))
    
    conn.commit()
    conn.close()

def log_experiment_run(
    dataset_id: str,
    model_type: str,
    mitigation_technique: Optional[str],
    metrics: Dict[str, Any],
    fairness_res: Dict[str, Any],
    significance_res: Optional[Dict[str, Any]] = None
):
    """Logs an auditing or mitigation experiment run to SQLite DB."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    dp_data = fairness_res.get("demographic_parity_diff", {})
    eo_data = fairness_res.get("equalized_odds_diff", {})
    eopp_data = fairness_res.get("equal_opportunity_diff", {})
    pp_data = fairness_res.get("predictive_parity_diff", {})

    p_val = significance_res.get("p_value", 1.0) if significance_res else 1.0
    is_sig = int(significance_res.get("is_significant", False)) if significance_res else 0

    cursor.execute("""
    INSERT INTO experiments (
        dataset_id, model_type, mitigation_technique, f1_score, roc_auc,
        demographic_parity_diff, dp_ci_low, dp_ci_high,
        equalized_odds_diff, equal_opportunity_diff, predictive_parity_diff,
        p_value, is_significant, timestamp
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        dataset_id, model_type, mitigation_technique if mitigation_technique else "baseline",
        metrics.get("f1_score", 0.0), metrics.get("roc_auc", 0.0),
        dp_data.get("val", 0.0), dp_data.get("ci_low", 0.0), dp_data.get("ci_high", 0.0),
        eo_data.get("val", 0.0), eopp_data.get("val", 0.0), pp_data.get("val", 0.0),
        p_val, is_sig, datetime.now().isoformat(sep=' ', timespec='seconds')
    ))

    conn.commit()
    conn.close()

def log_proxy_analysis_results(dataset_id: str, protected_attr: str, proxy_df: pd.DataFrame):
    """Stores proxy leakage scores in SQLite DB."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Clear old entries for this dataset + attribute combination
    cursor.execute("DELETE FROM proxy_analysis WHERE dataset_id = ? AND protected_attribute = ?", (dataset_id, protected_attr))

    for _, row in proxy_df.iterrows():
        cursor.execute("""
        INSERT INTO proxy_analysis (dataset_id, feature_name, protected_attribute, mutual_information, correlation, risk_level)
        VALUES (?, ?, ?, ?, ?, ?)
        """, (
            dataset_id, str(row['feature']), protected_attr,
            float(row['mutual_information']), float(row['correlation']), str(row['risk_level'])
        ))

    conn.commit()
    conn.close()

def get_experiment_history(dataset_id: Optional[str] = None) -> pd.DataFrame:
    """Queries logged experiments table from SQLite DB."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    query = "SELECT * FROM experiments"
    if dataset_id:
        query += f" WHERE dataset_id = '{dataset_id}'"
    query += " ORDER BY timestamp DESC"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df
