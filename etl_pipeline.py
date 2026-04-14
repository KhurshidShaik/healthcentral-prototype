"""
ETL Pipeline — HealthCentral Campaign Performance
Integrates data from 3 sources: Sigma (core metrics), Adobe Analytics (engagement), DCM (delivery)
Author: Khurshid Shaik | BI Analyst Prototype
"""
import pandas as pd
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

def extract():
    sigma = pd.read_csv(os.path.join(DATA_DIR, "sigma_campaign_metrics.csv"))
    adobe = pd.read_csv(os.path.join(DATA_DIR, "adobe_engagement.csv"))
    dcm = pd.read_csv(os.path.join(DATA_DIR, "dcm_delivery.csv"))
    return sigma, adobe, dcm

def validate_source(df, name, required_cols):
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name}: missing columns {missing}")
    nulls = df[required_cols].isnull().sum()
    null_report = nulls[nulls > 0]
    return {
        "source": name,
        "rows": len(df),
        "campaigns": df["campaign_id"].nunique(),
        "null_fields": null_report.to_dict() if len(null_report) > 0 else "None",
        "status": "PASS"
    }

def transform(sigma, adobe, dcm):
    sigma["week_start"] = pd.to_datetime(sigma["week_start"])
    adobe["week_start"] = pd.to_datetime(adobe["week_start"])
    dcm["week_start"] = pd.to_datetime(dcm["week_start"])

    # Join on campaign_id + week_start
    merged = sigma.merge(
        adobe.drop(columns=["source"]),
        on=["campaign_id", "week_start"],
        how="left",
        suffixes=("", "_adobe")
    )
    merged = merged.merge(
        dcm.drop(columns=["source"]),
        on=["campaign_id", "week_start"],
        how="left",
        suffixes=("", "_dcm")
    )

    # Standardized metric calculations
    merged["roas"] = (merged["conversions"] * 35 / merged["spend_usd"]).round(2)  # assume $35 avg conversion value
    merged["cost_per_conversion"] = (merged["spend_usd"] / merged["conversions"].replace(0, 1)).round(2)
    merged["engagement_score"] = (
        (merged["avg_time_on_site_sec"].fillna(0) / 320 * 25) +
        (merged["avg_scroll_depth_pct"].fillna(0) / 100 * 25) +
        ((100 - merged["bounce_rate_pct"].fillna(50)) / 100 * 25) +
        (merged["return_visit_rate_pct"].fillna(0) / 40 * 25)
    ).clip(0, 100).round(1)

    # Quality score (viewability + brand safety + fraud)
    merged["quality_score"] = (
        (merged["viewability_pct"].fillna(60) / 100 * 40) +
        (merged["brand_safety_pct"].fillna(95) / 100 * 40) +
        ((100 - merged["fraud_rate_pct"].fillna(3)) / 100 * 20)
    ).clip(0, 100).round(1)

    merged["data_sources"] = "Sigma + Adobe + DCM"

    return merged

def load(df):
    output_path = os.path.join(DATA_DIR, "unified_campaign_performance.csv")
    df.to_csv(output_path, index=False)
    return output_path

def run_pipeline():
    print("=" * 60)
    print("HealthCentral Campaign ETL Pipeline")
    print("=" * 60)

    # Extract
    print("\n[1/4] EXTRACT — Reading source data...")
    sigma, adobe, dcm = extract()

    # Validate
    print("[2/4] VALIDATE — Checking data quality...")
    validations = [
        validate_source(sigma, "Sigma", ["campaign_id", "impressions", "clicks", "spend_usd"]),
        validate_source(adobe, "Adobe Analytics", ["campaign_id", "avg_time_on_site_sec", "bounce_rate_pct"]),
        validate_source(dcm, "DCM", ["campaign_id", "viewability_pct", "brand_safety_pct"]),
    ]
    for v in validations:
        print(f"   {v['source']}: {v['rows']} rows, {v['campaigns']} campaigns — {v['status']}")
        if v["null_fields"] != "None":
            print(f"     Nulls: {v['null_fields']}")

    # Transform
    print("[3/4] TRANSFORM — Joining sources, standardizing metrics...")
    unified = transform(sigma, adobe, dcm)
    print(f"   Unified dataset: {len(unified)} rows, {unified['campaign_id'].nunique()} campaigns")
    print(f"   Derived metrics: roas, cost_per_conversion, engagement_score, quality_score")
    print(f"   Columns: {len(unified.columns)}")

    # Load
    print("[4/4] LOAD — Writing unified dataset...")
    path = load(unified)
    print(f"   Output: {path}")

    print("\n" + "=" * 60)
    print("Pipeline complete.")
    print("=" * 60)
    return unified

if __name__ == "__main__":
    run_pipeline()
