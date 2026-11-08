import os
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
from src.config import REPORTS_DIR

def generate_html_audit_report(
    dataset_name: str,
    protected_attribute: str,
    kpis: Dict[str, Any],
    proxy_df: pd.DataFrame,
    baseline_metrics: Dict[str, Any],
    mitigation_metrics: Dict[str, Any],
    insights: Dict[str, str],
    technique_name: str = "Reweighing"
) -> str:
    """Generates an HTML compliance and algorithmic fairness audit report."""
    
    proxy_rows = ""
    if not proxy_df.empty:
        for _, r in proxy_df.head(5).iterrows():
            risk_color = "#ef4444" if "High" in str(r['risk_level']) else "#f59e0b" if "Moderate" in str(r['risk_level']) else "#10b981"
            proxy_rows += f"""
            <tr>
                <td style="padding: 10px; border-bottom: 1px solid #334155; font-weight: 600;">{r['feature']}</td>
                <td style="padding: 10px; border-bottom: 1px solid #334155;">{r['mutual_information']:.4f}</td>
                <td style="padding: 10px; border-bottom: 1px solid #334155;">{r['correlation']:.4f}</td>
                <td style="padding: 10px; border-bottom: 1px solid #334155; color: {risk_color}; font-weight: bold;">{r['risk_level']}</td>
            </tr>
            """

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>FairLens Algorithmic Fairness Audit Report - {dataset_name}</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background-color: #0f172a;
                color: #f8fafc;
                margin: 0;
                padding: 40px;
                line-height: 1.6;
            }}
            .container {{
                max-width: 900px;
                margin: 0 auto;
                background: #1e293b;
                padding: 40px;
                border-radius: 12px;
                box-shadow: 0 10px 25px rgba(0,0,0,0.5);
                border: 1px solid #334155;
            }}
            .header {{
                border-bottom: 2px solid #3b82f6;
                padding-bottom: 20px;
                margin-bottom: 30px;
            }}
            h1 {{ color: #60a5fa; margin: 0 0 10px 0; font-size: 28px; }}
            .subtitle {{ color: #94a3b8; font-size: 14px; margin: 0; }}
            .section {{
                margin-bottom: 35px;
            }}
            h2 {{
                color: #38bdf8;
                font-size: 20px;
                border-left: 4px solid #3b82f6;
                padding-left: 10px;
                margin-top: 0;
            }}
            .card-grid {{
                display: grid;
                grid-template-columns: repeat(3, 1fr);
                gap: 15px;
                margin-bottom: 20px;
            }}
            .card {{
                background: #0f172a;
                padding: 15px;
                border-radius: 8px;
                border: 1px solid #334155;
                text-align: center;
            }}
            .card-val {{ font-size: 22px; font-weight: bold; color: #60a5fa; }}
            .card-lbl {{ font-size: 12px; color: #94a3b8; margin-top: 5px; }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin-top: 15px;
                background: #0f172a;
                border-radius: 8px;
                overflow: hidden;
            }}
            th {{
                background: #334155;
                color: #f8fafc;
                text-align: left;
                padding: 12px;
                font-size: 13px;
            }}
            .insight-box {{
                background: rgba(59, 130, 246, 0.1);
                border-left: 4px solid #3b82f6;
                padding: 15px;
                border-radius: 4px;
                margin-top: 15px;
                color: #e2e8f0;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>FairLens Algorithmic Fairness Audit Report</h1>
                <p class="subtitle">Dataset under audit: <strong>{dataset_name}</strong> | Protected Attribute: <strong>{protected_attribute}</strong> | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>

            <div class="section">
                <h2>1. Executive Audit Summary</h2>
                <div class="insight-box">
                    {insights.get('executive_summary', 'No summary generated.')}
                </div>
            </div>

            <div class="section">
                <h2>2. Dataset KPIs & Disparity Signal</h2>
                <div class="card-grid">
                    <div class="card">
                        <div class="card-val">{kpis.get('total_records', 0):,}</div>
                        <div class="card-lbl">Total Records</div>
                    </div>
                    <div class="card">
                        <div class="card-val">{kpis.get('overall_positive_rate', 0.0)*100:.1f}%</div>
                        <div class="card-lbl">Base Approval Rate</div>
                    </div>
                    <div class="card">
                        <div class="card-val">{kpis.get('base_rate_disparity', 0.0)*100:.1f}%</div>
                        <div class="card-lbl">Demographic Outcome Gap</div>
                    </div>
                </div>
            </div>

            <div class="section">
                <h2>3. Top Indirect Proxy Features</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Feature Name</th>
                            <th>Mutual Information</th>
                            <th>Correlation</th>
                            <th>Proxy Risk Level</th>
                        </tr>
                    </thead>
                    <tbody>
                        {proxy_rows}
                    </tbody>
                </table>
            </div>

            <div class="section">
                <h2>4. Baseline vs Mitigated Model Fairness Comparison</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Metric Name</th>
                            <th>Baseline Value (95% CI)</th>
                            <th>Mitigated Value ({technique_name})</th>
                            <th>Statistical Significance</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td style="padding: 10px; border-bottom: 1px solid #334155;">Demographic Parity Difference</td>
                            <td style="padding: 10px; border-bottom: 1px solid #334155;">{baseline_metrics.get('demographic_parity_diff', {}).get('val', 0.0):.3f} [{baseline_metrics.get('demographic_parity_diff', {}).get('ci_low', 0.0):.3f}, {baseline_metrics.get('demographic_parity_diff', {}).get('ci_high', 0.0):.3f}]</td>
                            <td style="padding: 10px; border-bottom: 1px solid #334155;">{mitigation_metrics.get('metrics', {}).get('demographic_parity_diff', {}).get('val', 0.0):.3f}</td>
                            <td style="padding: 10px; border-bottom: 1px solid #334155; color: #10b981; font-weight: bold;">{'Significant (p < 0.05)' if mitigation_metrics.get('significance', {}).get('is_significant') else 'Not Significant'}</td>
                        </tr>
                        <tr>
                            <td style="padding: 10px; border-bottom: 1px solid #334155;">Equalized Odds Difference</td>
                            <td style="padding: 10px; border-bottom: 1px solid #334155;">{baseline_metrics.get('equalized_odds_diff', {}).get('val', 0.0):.3f}</td>
                            <td style="padding: 10px; border-bottom: 1px solid #334155;">{mitigation_metrics.get('metrics', {}).get('equalized_odds_diff', {}).get('val', 0.0):.3f}</td>
                            <td style="padding: 10px; border-bottom: 1px solid #334155;">Evaluated</td>
                        </tr>
                    </tbody>
                </table>
            </div>
        </div>
    </body>
    </html>
    """
    return html_content

def generate_pdf_audit_report(
    dataset_name: str,
    protected_attribute: str,
    kpis: Dict[str, Any],
    proxy_df: pd.DataFrame,
    baseline_metrics: Dict[str, Any],
    mitigation_metrics: Dict[str, Any],
    insights: Dict[str, str],
    technique_name: str = "Reweighing"
) -> Path:
    """Generates PDF audit report using ReportLab or saves HTML report artifact."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    file_name = f"FairLens_Audit_Report_{dataset_name.replace(' ', '_')}.pdf"
    file_path = REPORTS_DIR / file_name
    html_path = REPORTS_DIR / f"FairLens_Audit_Report_{dataset_name.replace(' ', '_')}.html"

    html_code = generate_html_audit_report(
        dataset_name, protected_attribute, kpis, proxy_df, baseline_metrics, mitigation_metrics, insights, technique_name
    )

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_code)

    # Use ReportLab for PDF generation
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib import colors

        doc = SimpleDocTemplate(str(file_path), pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
        styles = getSampleStyleSheet()
        
        title_style = ParagraphStyle('TitleStyle', parent=styles['Heading1'], fontSize=20, textColor=colors.HexColor("#1e3a8a"), spaceAfter=10)
        heading_style = ParagraphStyle('HeadingStyle', parent=styles['Heading2'], fontSize=14, textColor=colors.HexColor("#2563eb"), spaceBefore=15, spaceAfter=8)
        body_style = ParagraphStyle('BodyStyle', parent=styles['Normal'], fontSize=10, textColor=colors.HexColor("#334155"), leading=14)
        
        story = []
        story.append(Paragraph(f"FairLens Algorithmic Fairness Audit Report", title_style))
        story.append(Paragraph(f"Dataset: <b>{dataset_name}</b> | Protected Attribute: <b>{protected_attribute}</b> | Date: {datetime.now().strftime('%Y-%m-%d')}", body_style))
        story.append(Spacer(1, 15))

        story.append(Paragraph("1. Executive Summary", heading_style))
        story.append(Paragraph(insights.get('executive_summary', ''), body_style))
        story.append(Spacer(1, 15))

        story.append(Paragraph("2. Top Indirect Proxy Features", heading_style))
        p_table_data = [["Feature Name", "Mutual Info", "Correlation", "Proxy Risk"]]
        if not proxy_df.empty:
            for _, r in proxy_df.head(5).iterrows():
                p_table_data.append([str(r['feature']), f"{r['mutual_information']:.4f}", f"{r['correlation']:.4f}", str(r['risk_level'])])
        
        t = Table(p_table_data, colWidths=[180, 100, 100, 120])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#f1f5f9")),
            ('TEXTCOLOR', (0,0), (-1,0), colors.HexColor("#0f172a")),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0,0), (-1,0), 8),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
        ]))
        story.append(t)
        story.append(Spacer(1, 15))

        story.append(Paragraph("3. Baseline vs Mitigated Fairness Comparison", heading_style))
        base_dp = baseline_metrics.get('demographic_parity_diff', {}).get('val', 0.0)
        mit_dp = mitigation_metrics.get('metrics', {}).get('demographic_parity_diff', {}).get('val', 0.0)
        sig = "Yes (p < 0.05)" if mitigation_metrics.get('significance', {}).get('is_significant') else "No"

        m_table_data = [
            ["Metric", "Baseline (95% CI)", f"Mitigated ({technique_name})", "Statistically Significant"],
            ["Demographic Parity Diff", f"{base_dp:.3f}", f"{mit_dp:.3f}", sig]
        ]
        t2 = Table(m_table_data, colWidths=[160, 130, 110, 100])
        t2.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#f1f5f9")),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
        ]))
        story.append(t2)

        doc.build(story)
        print(f"ReportLab PDF generated at {file_path}")
        return file_path

    except Exception as e:
        print(f"PDF generation fallback to HTML: {e}")
        return html_path
