"""Compute every number used in the paper (seed-42 split, benchmark models) -> paper/paper_stats.json."""
import json, sys, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.config import DATASET_CONFIGS, get_outcome_labels
from src.data.loader import load_raw_dataset
from src.data.cleaner import clean_dataset
from src.data.preprocessor import preprocess_dataset
from src.analysis.eda import get_dataset_kpis
from src.analysis.proxy import compute_proxy_leakage
from src.models.trainer import train_model, get_model_input_data
from src.fairness.intersectional import compute_intersectional_metrics
from src.fairness.calibration import compute_group_calibration
import src.models.trainer as T, tempfile
T.MODELS_DIR = Path(tempfile.mkdtemp())          # never touch committed artifacts

J = json.load(open(ROOT / "data/verified_benchmark_results.json"))
SECOND = {"german_credit": "age_group", "compas": "sex", "adult_income": "age_group"}
out = {}
for dk, cfg in DATASET_CONFIGS.items():
    prot = cfg["default_protected_attribute"]; lab = get_outcome_labels(dk)
    clean = clean_dataset(load_raw_dataset(dk), dk)
    data = preprocess_dataset(clean, target_col=cfg["target_column"], protected_col=prot, random_state=42)
    m_type = J["selected_baseline_models"][dk]
    model, _ = train_model(data, model_type=m_type, dataset_key=dk, random_state=42)
    X, y, s = get_model_input_data(m_type, data, "test")
    yp = model.predict(X); pr = model.predict_proba(X)[:, 1]
    kp = get_dataset_kpis(clean, "target", prot, favorable_value=lab["favorable_value"])
    proxy = compute_proxy_leakage(clean, protected_col=prot, target_col="target").head(3)
    inter = compute_intersectional_metrics(y.values, yp, s, clean.loc[s.index, SECOND[dk]])
    cal = compute_group_calibration(y.values, pr, s.values)
    it = inter["table"]
    out[dk] = {
        "n_clean": len(clean), "n_features": len(data.feature_names),
        "splits": [len(data.y_train), len(data.y_val), len(data.y_test)],
        "protected": prot, "second_attr": SECOND[dk], "model": m_type,
        "favorable_rate": kp["overall_positive_rate"], "base_gap": kp["base_rate_disparity"],
        "group_fav_rates": kp["group_rates"],
        "label1_rate": float(clean["target"].mean()),
        "proxy_top3": proxy[["feature", "mutual_information"]].values.tolist(),
        "intersectional": {"gap": inter["gaps"]["demographic_parity_diff"], "eo_gap": inter["gaps"]["equalized_odds_diff"],
                           "single": inter["single_attribute_dp"], "hi": inter["highest_selection_group"], "lo": inter["lowest_selection_group"],
                           "n_incl": inter["n_included"], "n_excl": inter["n_excluded"],
                           "table": it[["group", "count", "selection_rate", "tpr", "fpr", "included"]].round(4).values.tolist()},
        "calibration": {"overall": cal["overall"], "gaps": cal["gaps"],
                        "summary": cal["summary"].round(4).to_dict(orient="records"),
                        "curves": cal["curves"].round(4).to_dict(orient="records")},
    }
    print(dk, out[dk]["n_features"], out[dk]["splits"], round(out[dk]["intersectional"]["gap"], 4),
          {k: round(v, 4) for k, v in out[dk]["intersectional"]["single"].items()},
          out[dk]["intersectional"]["hi"], "|", out[dk]["intersectional"]["lo"])
    print("   calib", [(r["group"], r["count"], r["brier_score"], r["ece"]) for r in out[dk]["calibration"]["summary"]])
    print("   fav", round(kp["overall_positive_rate"], 4), {g: round(v, 4) for g, v in kp["group_rates"].items()}, "proxy", out[dk]["proxy_top3"])
json.dump(out, open(ROOT / "paper/paper_stats.json", "w"), indent=1, default=float)
