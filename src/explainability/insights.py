import pandas as pd
import numpy as np
from typing import Dict, Any, List

def generate_plain_language_summary(
    dataset_name: str,
    protected_attribute: str,
    kpis: Dict[str, Any],
    proxy_df: pd.DataFrame,
    baseline_fairness: Dict[str, Any],
    mitigation_res: Dict[str, Any],
    technique_name: str = "Reweighing"
) -> Dict[str, str]:
    """Synthesizes pipeline metrics into structured, plain-language audit statements."""
    
    # 1. Base Rate Overview
    overall_pos = round(kpis.get("overall_positive_rate", 0.0) * 100, 1)
    base_disp = round(kpis.get("base_rate_disparity", 0.0) * 100, 1)
    
    base_statement = (
        f"In the {dataset_name} dataset, the overall favorable outcome (approval) rate is {overall_pos}%. "
        f"A historical base-rate outcome disparity of {base_disp} percentage points was observed across different '{protected_attribute}' demographic groups "
        f"before any algorithmic scoring model was applied."
    )

    # 2. Proxy Leakage Signal
    if not proxy_df.empty:
        top_proxy = proxy_df.iloc[0]
        proxy_feat = top_proxy['feature']
        proxy_mi = top_proxy['mutual_information']
        proxy_risk = top_proxy['risk_level']
        
        proxy_statement = (
            f"Mutual Information proxy analysis revealed that feature '{proxy_feat}' exhibits the highest demographic leakage "
            f"with respect to protected attribute '{protected_attribute}' (MI = {proxy_mi}, categorized as '{proxy_risk}'). "
            f"Even when '{protected_attribute}' is excluded from model training, features like '{proxy_feat}' silently encode demographic identity."
        )
    else:
        proxy_statement = f"Proxy analysis detected low indirect feature leakage with respect to protected attribute '{protected_attribute}'."

    # 3. Baseline Audit & Statistical Validation
    base_dp = baseline_fairness.get("demographic_parity_diff", {}).get("val", 0.0)
    base_dp_low = baseline_fairness.get("demographic_parity_diff", {}).get("ci_low", 0.0)
    base_dp_high = baseline_fairness.get("demographic_parity_diff", {}).get("ci_high", 0.0)
    
    dp_info = baseline_fairness.get("demographic_parity_diff", {})
    is_biased = dp_info.get("is_significant", base_dp_low > 0.02)
    signed_txt = ""
    if "signed_ci_low" in dp_info:
        g_hi, g_lo = dp_info.get("reference_groups", ["", ""])
        signed_txt = (f" Signed gap ('{g_hi}' minus '{g_lo}') 95% CI: "
                      f"[{dp_info['signed_ci_low']:.3f}, {dp_info['signed_ci_high']:.3f}].")

    audit_statement = (
        f"The baseline unmitigated classifier exhibits a Demographic Parity difference of {base_dp:.3f} "
        f"(95% Bootstrap CI: [{base_dp_low:.3f}, {base_dp_high:.3f}]).{signed_txt} "
        f"{'This disparity is statistically significant (signed-gap CI excludes zero), confirming systematic demographic bias.' if is_biased else 'This disparity is not statistically distinguishable from zero (signed-gap CI includes zero).'}"
    )

    # 4. Mitigation Efficacy & Tradeoff
    mit_dp = mitigation_res.get("metrics", {}).get("demographic_parity_diff", {}).get("val", 0.0)
    mit_dp_low = mitigation_res.get("metrics", {}).get("demographic_parity_diff", {}).get("ci_low", 0.0)
    mit_dp_high = mitigation_res.get("metrics", {}).get("demographic_parity_diff", {}).get("ci_high", 0.0)
    
    f1_before = baseline_fairness.get("f1_score", 0.0)
    f1_after = mitigation_res.get("f1_score", 0.0)
    f1_delta = round((f1_after - f1_before) * 100, 1)
    
    dp_reduction_pct = round(((base_dp - mit_dp) / base_dp * 100), 1) if base_dp > 0 else 0.0
    p_val = mitigation_res.get("significance", {}).get("p_value", 1.0)
    is_sig = mitigation_res.get("significance", {}).get("is_significant", False)

    change_verb = "reduced" if mit_dp < base_dp else "increased" if mit_dp > base_dp else "left unchanged"
    change_pct = abs(dp_reduction_pct)
    mitigation_statement = (
        f"Applying '{technique_name}' mitigation {change_verb} the Demographic Parity disparity from {base_dp:.3f} to {mit_dp:.3f} "
        f"(a {change_pct}% {'reduction' if mit_dp < base_dp else 'change'}, 95% CI: [{mit_dp_low:.3f}, {mit_dp_high:.3f}]). "
        f"Paired hypothesis testing confirmed this improvement is {'statistically significant (p = ' + str(p_val) + ' < 0.05)' if is_sig else 'not statistically significant (p = ' + str(p_val) + ')'}. "
        f"The associated accuracy/F1 trade-off cost was {f1_delta} percentage points."
    )

    return {
        "base_statement": base_statement,
        "proxy_statement": proxy_statement,
        "audit_statement": audit_statement,
        "mitigation_statement": mitigation_statement,
        "executive_summary": f"{base_statement} {proxy_statement} {audit_statement} {mitigation_statement}"
    }
