import pandas as pd
import numpy as np
np.random.seed(44)

campaign_ids = [f"HC-DTC-{i:03d}" for i in [1,3,5,6,8,10,11]] + [f"HC-HCP-{i:03d}" for i in [2,4,7,9,12]]
weeks = pd.date_range("2026-01-06", periods=12, freq="W-MON")

rows = []
for cid in campaign_ids:
    ctype = "HCP" if "HCP" in cid else "DTC"
    base_viewability = np.random.uniform(55, 85)
    base_completion = np.random.uniform(40, 75)
    base_freq = np.random.uniform(1.5, 4.5)
    for i, w in enumerate(weeks):
        fatigue = 1 + i * 0.015
        rows.append({
            "campaign_id": cid,
            "week_start": w.strftime("%Y-%m-%d"),
            "viewability_pct": round(base_viewability * np.random.uniform(0.93, 1.07), 1),
            "video_completion_rate_pct": round(base_completion * np.random.uniform(0.9, 1.1), 1) if np.random.random() > 0.4 else None,
            "avg_frequency": round(base_freq * fatigue * np.random.uniform(0.9, 1.1), 1),
            "unique_reach": int(np.random.randint(80000, 350000) / fatigue),
            "fraud_rate_pct": round(np.random.uniform(0.5, 5.2), 2),
            "brand_safety_pct": round(np.random.uniform(92, 99.5), 1),
            "above_fold_pct": round(np.random.uniform(45, 82), 1),
            "source": "DCM"
        })

df = pd.DataFrame(rows)
df.to_csv("data/dcm_delivery.csv", index=False)
print(f"DCM export: {len(df)} rows, {df['campaign_id'].nunique()} campaigns")
