"""
HealthCentral Campaign Intelligence Hub — FastAPI backend.
Serves the React UI and all data/agent API endpoints.
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from agent import run_agent
from etl_pipeline import run_pipeline

load_dotenv()

app = FastAPI(title="HealthCentral Campaign Intelligence API")

# ─── Load data once at startup ────────────────────────────────────────────────
print("Loading unified campaign dataset...")
DF: pd.DataFrame = run_pipeline()
DF["week_start"] = pd.to_datetime(DF["week_start"])
print(f"Ready — {len(DF)} rows, {DF['campaign_id'].nunique()} campaigns.")

BASE_DIR = Path(__file__).parent


# ─── Static UI ────────────────────────────────────────────────────────────────

@app.get("/")
def serve_ui():
    return FileResponse(BASE_DIR / "index.html", media_type="text/html")


# ─── Filters ──────────────────────────────────────────────────────────────────

@app.get("/api/filters")
def get_filters():
    campaigns = (
        DF[["campaign_id", "campaign_name", "campaign_type", "condition_category", "channel"]]
        .drop_duplicates()
        .sort_values("campaign_name")
        .to_dict(orient="records")
    )
    return {
        "campaign_types": ["All", "DTC", "HCP"],
        "conditions": ["All"] + sorted(DF["condition_category"].unique().tolist()),
        "channels": ["All"] + sorted(DF["channel"].unique().tolist()),
        "campaigns": campaigns,
        "date_range": {
            "min": DF["week_start"].min().strftime("%Y-%m-%d"),
            "max": DF["week_start"].max().strftime("%Y-%m-%d"),
        },
    }


# ─── Data endpoint ────────────────────────────────────────────────────────────

@app.get("/api/data")
def get_data(
    campaign_type: str = "All",
    condition: str = "All",
    channel: str = "All",
):
    filtered = DF.copy()
    if campaign_type != "All":
        filtered = filtered[filtered["campaign_type"] == campaign_type]
    if condition != "All":
        filtered = filtered[filtered["condition_category"] == condition]
    if channel != "All":
        filtered = filtered[filtered["channel"] == channel]

    if filtered.empty:
        raise HTTPException(status_code=404, detail="No data for the selected filters.")

    # ── KPIs ──
    total_impressions = int(filtered["impressions"].sum())
    total_clicks = int(filtered["clicks"].sum())
    total_conversions = int(filtered["conversions"].sum())
    total_spend = float(filtered["spend_usd"].sum())
    avg_ctr = float(total_clicks / total_impressions * 100) if total_impressions > 0 else 0
    avg_roas = float(filtered["roas"].mean())
    avg_engagement = float(filtered["engagement_score"].mean())
    avg_quality = float(filtered["quality_score"].mean())
    avg_viewability = float(filtered["viewability_pct"].mean())

    # ── Weekly trend ──
    weekly = (
        filtered.groupby("week_start")
        .agg(
            impressions=("impressions", "sum"),
            clicks=("clicks", "sum"),
            conversions=("conversions", "sum"),
            spend_usd=("spend_usd", "sum"),
            engagement_score=("engagement_score", "mean"),
            roas=("roas", "mean"),
            viewability_pct=("viewability_pct", "mean"),
        )
        .reset_index()
    )
    weekly["week_start"] = weekly["week_start"].dt.strftime("%Y-%m-%d")
    weekly["ctr"] = (weekly["clicks"] / weekly["impressions"] * 100).round(2)
    weekly = weekly.round(2)

    # ── Campaign aggregates ──
    camp_agg = (
        filtered.groupby(
            ["campaign_id", "campaign_name", "campaign_type", "condition_category", "channel"]
        )
        .agg(
            impressions=("impressions", "sum"),
            clicks=("clicks", "sum"),
            conversions=("conversions", "sum"),
            spend_usd=("spend_usd", "sum"),
            roas=("roas", "mean"),
            ctr_pct=("ctr_pct", "mean"),
            engagement_score=("engagement_score", "mean"),
            quality_score=("quality_score", "mean"),
            viewability_pct=("viewability_pct", "mean"),
            bounce_rate_pct=("bounce_rate_pct", "mean"),
            avg_frequency=("avg_frequency", "mean"),
            fraud_rate_pct=("fraud_rate_pct", "mean"),
            brand_safety_pct=("brand_safety_pct", "mean"),
        )
        .reset_index()
        .round(2)
    )
    camp_agg["ctr"] = (camp_agg["clicks"] / camp_agg["impressions"] * 100).round(2)

    # ── Channel mix ──
    channel_data = (
        filtered.groupby("channel")
        .agg(
            spend_usd=("spend_usd", "sum"),
            impressions=("impressions", "sum"),
            conversions=("conversions", "sum"),
            roas=("roas", "mean"),
        )
        .reset_index()
        .round(2)
    )

    return _clean_json({
        "kpis": {
            "impressions": total_impressions,
            "clicks": total_clicks,
            "conversions": total_conversions,
            "spend": round(total_spend, 2),
            "ctr": round(avg_ctr, 2),
            "roas": round(avg_roas, 2),
            "engagement": round(avg_engagement, 1),
            "quality": round(avg_quality, 1),
            "viewability": round(avg_viewability, 1),
            "n_campaigns": int(filtered["campaign_id"].nunique()),
        },
        "weekly_trend": weekly.to_dict(orient="records"),
        "campaigns": camp_agg.to_dict(orient="records"),
        "channel_mix": channel_data.to_dict(orient="records"),
    })


# ─── Chat / Agent endpoint ────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    filters: dict = {}
    history: list = []


@app.post("/api/chat")
def chat(req: ChatRequest):
    # Build live stats snapshot to inject into context
    filters = req.filters or {}
    from tools import _apply_filters
    filtered = _apply_filters(DF, filters)

    if filtered.empty:
        live_stats = {}
    else:
        total_imp = int(filtered["impressions"].sum())
        total_clk = int(filtered["clicks"].sum())
        live_stats = {
            "n_campaigns": int(filtered["campaign_id"].nunique()),
            "impressions": total_imp,
            "clicks": total_clk,
            "ctr": round(float(total_clk / total_imp * 100), 2) if total_imp > 0 else 0,
            "conversions": int(filtered["conversions"].sum()),
            "spend": round(float(filtered["spend_usd"].sum()), 2),
            "roas": round(float(filtered["roas"].mean()), 2),
            "engagement": round(float(filtered["engagement_score"].mean()), 1),
            "quality": round(float(filtered["quality_score"].mean()), 1),
        }

    try:
        result = run_agent(
            user_message=req.message,
            current_filters=filters,
            history=req.history,
            df=DF,
            live_stats=live_stats,
        )
        return _clean_json(result)
    except Exception as e:
        err = str(e)
        if "insufficient_quota" in err or "429" in err:
            msg = "OpenAI quota exceeded — please add credits at platform.openai.com/settings/billing and restart the server."
        elif "401" in err or "invalid_api_key" in err:
            msg = "Invalid OpenAI API key. Check your .env file."
        elif "APIConnectionError" in type(e).__name__:
            msg = "Could not reach OpenAI API — check your internet connection."
        else:
            msg = f"Agent error: {err}"
        return {"message": msg, "actions": [], "error": True}


# ─── Helper ───────────────────────────────────────────────────────────────────

def _clean_json(obj):
    """Replace NaN/Inf with None so JSON serialization doesn't choke."""
    if isinstance(obj, dict):
        return {k: _clean_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean_json(i) for i in obj]
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    return obj


# ─── Dev entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
