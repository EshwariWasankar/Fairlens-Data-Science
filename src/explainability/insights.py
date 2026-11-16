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
    
    # 1. Outcomes in the data
    overall_pos = kpis.get("overall_positive_rate", 0.0) * 100
    base_disp = kpis.get("base_rate_disparity", 0.0) * 100
    base_statement = (
        f"{overall_pos:.1f}% of records in {dataset_name} have the favorable outcome. "
        f"Between '{protected_attribute}' groups this rate differs by up to {base_disp:.1f} percentage points, "
        f"before any model is involved."
    )

    # 2. Proxies
    if not proxy_df.empty:
        top = proxy_df.iloc[0]
        proxy_statement = (
            f"'{top['feature']}' carries the most information about '{protected_attribute}' "
            f"(mutual information {float(top['mutual_information']):.3f}, {str(top['risk_level']).lower()}). "
            f"Leaving '{protected_attribute}' out of the features does not remove this signal."
        )
    else:
        proxy_statement = f"No feature carries much information about '{protected_attribute}'."

    # 3. Baseline gap and its significance
    dp = baseline_fairness.get("demographic_parity_diff", {})
    base_dp = float(dp.get("val", 0.0))
    is_biased = bool(dp.get("is_significant", dp.get("ci_low", 0.0) > 0.02))
    audit_statement = (
        f"The model's demographic parity gap is {base_dp:.3f} "
        f"(95% CI {float(dp.get('ci_low', 0.0)):.3f} to {float(dp.get('ci_high', 0.0)):.3f})."
    )
    if "signed_ci_low" in dp:
        g_hi, g_lo = dp.get("reference_groups", ["", ""])
        audit_statement += (
            f" Measured as '{g_hi}' minus '{g_lo}', the 95% CI is "
            f"{float(dp['signed_ci_low']):.3f} to {float(dp['signed_ci_high']):.3f},"
        )
        audit_statement += " so the gap is statistically significant." if is_biased else " so the gap is not statistically significant."
    else:
        audit_statement += " The gap is statistically significant." if is_biased else " The gap is not statistically significant."

    # 4. Mitigation effect
    mit_dp = float(mitigation_res.get("metrics", {}).get("demographic_parity_diff", {}).get("val", 0.0))
    if mit_dp < base_dp:
        pct = (base_dp - mit_dp) / base_dp * 100 if base_dp > 0 else 0.0
        change = f"lowers the gap from {base_dp:.3f} to {mit_dp:.3f} ({pct:.0f}% smaller)"
    elif mit_dp > base_dp:
        change = f"raises the gap from {base_dp:.3f} to {mit_dp:.3f}"
    else:
        change = f"leaves the gap unchanged at {base_dp:.3f}"
    sig = mitigation_res.get("significance", {})
    p_val = float(sig.get("p_value", 1.0))
    sig_text = (f"The paired bootstrap shows a statistically significant reduction (p = {p_val:.3f})."
                if sig.get("is_significant", False)
                else f"The paired bootstrap finds no statistically significant reduction (p = {p_val:.3f}).")
    f1_delta = (float(mitigation_res.get("f1_score", 0.0)) - float(baseline_fairness.get("f1_score", 0.0))) * 100
    mitigation_statement = f"{technique_name} {change}. {sig_text} F1 changes by {f1_delta:+.1f} points."

    return {
        "base_statement": base_statement,
        "proxy_statement": proxy_statement,
        "audit_statement": audit_statement,
        "mitigation_statement": mitigation_statement,
        "executive_summary": f"{base_statement} {proxy_statement} {audit_statement} {mitigation_statement}"
    }
