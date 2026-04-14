import pandas as pd
import numpy as np
np.random.seed(43)

campaign_ids = [f"HC-DTC-{i:03d}" for i in [1,3,5,6,8,10,11]] + [f"HC-HCP-{i:03d}" for i in [2,4,7,9,12]]
weeks = pd.date_range("2026-01-06", periods=12, freq="W-MON")

engagement_profiles = {
    "DTC": {"avg_time": (90, 200), "pages": (1.8, 3.8), "bounce": (30, 55), "scroll": (40, 80), "return_rate": (8, 25)},
    "HCP": {"avg_time": (150, 320), "pages": (2.5, 5.0), "bounce": (20, 40), "scroll": (55, 90), "return_rate": (15, 40)},
}

rows = []
for cid in campaign_ids:
    ctype = "HCP" if "HCP" in cid else "DTC"
    profile = engagement_profiles[ctype]
    base_time = np.random.uniform(*profile["avg_time"])
    base_pages = np.random.uniform(*profile["pages"])
    base_bounce = np.random.uniform(*profile["bounce"])
    base_scroll = np.random.uniform(*profile["scroll"])
    base_return = np.random.uniform(*profile["return_rate"])
    for i, w in enumerate(weeks):
        improvement = 1 + i * 0.008
        rows.append({
            "campaign_id": cid,
            "week_start": w.strftime("%Y-%m-%d"),
            "avg_time_on_site_sec": round(base_time * improvement * np.random.uniform(0.9, 1.1)),
            "pages_per_session": round(base_pages * improvement * np.random.uniform(0.92, 1.08), 1),
            "bounce_rate_pct": round(base_bounce / improvement * np.random.uniform(0.93, 1.07), 1),
            "avg_scroll_depth_pct": round(base_scroll * improvement * np.random.uniform(0.92, 1.08), 1),
            "return_visit_rate_pct": round(base_return * improvement * np.random.uniform(0.88, 1.12), 1),
            "mobile_pct": round(np.random.uniform(52, 78), 1),
            "new_visitor_pct": round(np.random.uniform(40, 72), 1),
            "source": "Adobe Analytics"
        })

df = pd.DataFrame(rows)
df.to_csv("data/adobe_engagement.csv", index=False)
print(f"Adobe export: {len(df)} rows, {df['campaign_id'].nunique()} campaigns")
