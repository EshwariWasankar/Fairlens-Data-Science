import os
import yaml
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
DB_PATH = DATA_DIR / "fairlens.db"
CONFIG_FILE = BASE_DIR / "dataset_config.yaml"

# Ensure directories exist
for path in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, REPORTS_DIR]:
    path.mkdir(parents=True, exist_ok=True)

def load_dataset_configs():
    if not CONFIG_FILE.exists():
        raise FileNotFoundError(f"Config file not found at {CONFIG_FILE}")
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

DATASET_CONFIGS = load_dataset_configs()


_DEFAULT_OUTCOME_LABELS = {
    "subject": "Individual",
    "favorable": "Favorable (1)",
    "unfavorable": "Unfavorable (0)",
    "favorable_rate": "Favorable Outcome Rate",
}


def get_outcome_labels(dataset_key: str) -> dict:
    """Dataset-specific wording for model decisions plus which label value is favorable.

    Keeps the UI from hard-coding credit language ("REJECTED") for datasets such as COMPAS,
    where label 1 means recidivism (unfavorable).
    """
    cfg = DATASET_CONFIGS.get(dataset_key, {})
    labels = {**_DEFAULT_OUTCOME_LABELS, **cfg.get("outcome_labels", {})}
    fav = int(cfg.get("favorable_target_value", 1))
    labels["favorable_value"] = fav
    labels["unfavorable_value"] = 1 - fav
    labels["by_value"] = {fav: labels["favorable"], 1 - fav: labels["unfavorable"]}
    return labels
