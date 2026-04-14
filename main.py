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
from fastapi.responses import StreamingResponse
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


# ─── Source CSV downloads ─────────────────────────────────────────────────────

SOURCE_FILES = {
    "sigma": ("sigma_campaign_metrics.csv", "Sigma_Campaign_Metrics.csv"),
    "adobe": ("adobe_engagement.csv", "Adobe_Engagement.csv"),
    "dcm":   ("dcm_delivery.csv", "DCM_Delivery.csv"),
}

@app.get("/api/source/{source}")
def download_source(source: str):
    if source not in SOURCE_FILES:
        raise HTTPException(status_code=404, detail="Unknown source")
    filename, download_name = SOURCE_FILES[source]
    path = BASE_DIR / "data" / filename
    return FileResponse(
        path,
        media_type="text/csv",
        filename=download_name,
        headers={"Content-Disposition": f"attachment; filename={download_name}"},
    )


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

    # ── Week-over-week deltas (last week vs prior week) ──
    weeks_sorted = sorted(filtered["week_start"].unique())
    wow = {}
    if len(weeks_sorted) >= 2:
        cur_w  = filtered[filtered["week_start"] == weeks_sorted[-1]]
        prev_w = filtered[filtered["week_start"] == weeks_sorted[-2]]
        def _pct(cur, prv):
            if prv == 0: return None
            return round((cur - prv) / abs(prv) * 100, 1)
        cur_imp  = int(cur_w["impressions"].sum())
        prev_imp = int(prev_w["impressions"].sum())
        cur_conv = int(cur_w["conversions"].sum())
        prev_conv= int(prev_w["conversions"].sum())
        cur_ctr  = float(cur_w["clicks"].sum() / cur_w["impressions"].sum() * 100) if cur_w["impressions"].sum() > 0 else 0
        prev_ctr = float(prev_w["clicks"].sum() / prev_w["impressions"].sum() * 100) if prev_w["impressions"].sum() > 0 else 0
        cur_roas = float(cur_w["roas"].mean())
        prev_roas= float(prev_w["roas"].mean())
        cur_eng  = float(cur_w["engagement_score"].mean())
        prev_eng = float(prev_w["engagement_score"].mean())
        wow = {
            "impressions": _pct(cur_imp, prev_imp),
            "ctr":         _pct(cur_ctr, prev_ctr),
            "conversions": _pct(cur_conv, prev_conv),
            "roas":        _pct(cur_roas, prev_roas),
            "engagement":  _pct(cur_eng, prev_eng),
        }

    # ── Health alerts ──
    alerts = []
    camp_agg_alert = (
        filtered.groupby(["campaign_id", "campaign_name"])
        .agg(viewability=("viewability_pct","mean"), fraud=("fraud_rate_pct","mean"),
             frequency=("avg_frequency","mean"), roas=("roas","mean"), bounce=("bounce_rate_pct","mean"))
        .reset_index()
    )
    low_view = camp_agg_alert[camp_agg_alert["viewability"] < 70]
    if not low_view.empty:
        alerts.append({"type":"warning","msg":f"{len(low_view)} campaign{'s' if len(low_view)>1 else ''} below 70% viewability benchmark","tab":"quality"})
    high_fraud = camp_agg_alert[camp_agg_alert["fraud"] > 3]
    if not high_fraud.empty:
        alerts.append({"type":"error","msg":f"{len(high_fraud)} campaign{'s' if len(high_fraud)>1 else ''} with fraud rate >3%","tab":"quality"})
    high_freq = camp_agg_alert[camp_agg_alert["frequency"] > 5]
    if not high_freq.empty:
        alerts.append({"type":"warning","msg":f"{len(high_freq)} campaign{'s' if len(high_freq)>1 else ''} with frequency >5x — fatigue risk","tab":"campaigns"})
    low_roas = camp_agg_alert[camp_agg_alert["roas"] < 1.0]
    if not low_roas.empty:
        alerts.append({"type":"error","msg":f"{len(low_roas)} campaign{'s' if len(low_roas)>1 else ''} with ROAS <1x — spending more than generating","tab":"campaigns"})

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
        "wow": wow,
        "alerts": alerts,
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
