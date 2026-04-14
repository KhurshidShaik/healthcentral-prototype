import pandas as pd
import numpy as np
np.random.seed(42)

campaigns = [
    ("HC-DTC-001", "Diabetes Awareness — DTC", "DTC", "Diabetes", "Programmatic Display"),
    ("HC-HCP-002", "Oncology CME — HCP", "HCP", "Cancer", "Endemic Display"),
    ("HC-DTC-003", "Mental Health Stories — DTC", "DTC", "Mental Health", "Social + Native"),
    ("HC-HCP-004", "Autoimmune Treatment — HCP", "HCP", "Autoimmune", "Email + Endemic"),
    ("HC-DTC-005", "HIV Prevention — DTC", "DTC", "HIV/AIDS", "Programmatic + Social"),
    ("HC-DTC-006", "Skin Cancer Awareness — DTC", "DTC", "Skin Cancer", "Social + Display"),
    ("HC-HCP-007", "Cardiology Insights — HCP", "HCP", "Heart Disease", "Endemic + Email"),
    ("HC-DTC-008", "COPD Living Well — DTC", "DTC", "COPD", "Programmatic Display"),
    ("HC-HCP-009", "Rheumatology Update — HCP", "HCP", "Arthritis", "Endemic Display"),
    ("HC-DTC-010", "Migraine Relief — DTC", "DTC", "Migraine", "Social + Native"),
    ("HC-DTC-011", "Asthma Management — DTC", "DTC", "Asthma", "Programmatic + Social"),
    ("HC-HCP-012", "Dermatology CME — HCP", "HCP", "Skin Conditions", "Endemic + Email"),
]

weeks = pd.date_range("2026-01-06", periods=12, freq="W-MON")
rows = []
for cid, name, ctype, condition, channel in campaigns:
    base_imp = np.random.randint(180000, 420000)
    base_ctr = np.random.uniform(1.1, 2.2)
    base_cvr = np.random.uniform(1.8, 4.2)
    base_cpm = np.random.uniform(10, 24)
    for i, w in enumerate(weeks):
        growth = 1 + i * np.random.uniform(0.01, 0.04)
        imp = int(base_imp * growth * np.random.uniform(0.85, 1.15))
        ctr = round(base_ctr * np.random.uniform(0.92, 1.08), 2)
        clicks = int(imp * ctr / 100)
        cvr = round(base_cvr * np.random.uniform(0.9, 1.1), 2)
        conversions = int(clicks * cvr / 100)
        cpm = round(base_cpm * np.random.uniform(0.95, 1.05), 2)
        spend = round(imp / 1000 * cpm, 2)
        cpc = round(spend / max(clicks, 1), 2)
        rows.append({
            "campaign_id": cid, "campaign_name": name, "campaign_type": ctype,
            "condition_category": condition, "channel": channel,
            "week_start": w.strftime("%Y-%m-%d"),
            "impressions": imp, "clicks": clicks, "ctr_pct": ctr,
            "conversions": conversions, "cvr_pct": cvr,
            "spend_usd": spend, "cpm_usd": cpm, "cpc_usd": cpc,
            "source": "Sigma"
        })

df = pd.DataFrame(rows)
df.to_csv("data/sigma_campaign_metrics.csv", index=False)
print(f"Sigma export: {len(df)} rows, {df['campaign_id'].nunique()} campaigns, {len(weeks)} weeks")
