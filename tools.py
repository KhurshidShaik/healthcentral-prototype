"""
Agent tool implementations — pure data functions, no Streamlit / FastAPI dependencies.
Each tool takes a pandas DataFrame + parameters and returns a structured dict.
"""
import pandas as pd
import numpy as np


# ─── Tool: filter_dashboard ────────────────────────────────────────────────────

def filter_dashboard(df: pd.DataFrame, campaign_type: str = "All",
                     condition: str = "All", channel: str = "All") -> dict:
    """
    Apply filters and return the new filter state + a data summary for the agent
    to narrate. The frontend applies the returned filters and re-fetches data.
    """
    new_filters = {
        "campaign_type": campaign_type,
        "condition": condition,
        "channel": channel,
    }

    filtered = _apply_filters(df, new_filters)
    n_campaigns = filtered["campaign_id"].nunique()
    total_spend = filtered["spend_usd"].sum()
    avg_roas = filtered["roas"].mean()
    avg_engagement = filtered["engagement_score"].mean()
    top = (filtered.groupby("campaign_name")["roas"].mean()
                   .sort_values(ascending=False)
                   .index[0] if n_campaigns > 0 else "N/A")

    return {
        "new_filters": new_filters,
        "result_summary": {
            "campaigns_visible": int(n_campaigns),
            "total_spend_usd": round(float(total_spend), 2),
            "avg_roas": round(float(avg_roas), 2),
            "avg_engagement_score": round(float(avg_engagement), 1),
            "top_performer": top,
        }
    }


# ─── Tool: explain_metric ──────────────────────────────────────────────────────

METRIC_EXPLANATIONS = {
    "roas": {
        "name": "ROAS (Return on Ad Spend)",
        "source": "Derived (Sigma + assumed $35 conversion value)",
        "formula": "ROAS = (Conversions × $35 avg value) / Spend",
        "benchmark": "Good: 2x+, Excellent: 3x+, Underperforming: <1.5x",
        "interpretation": "How many dollars of value generated per dollar spent. A 3x ROAS means $3 of conversion value for every $1 in media spend. Healthcare benchmarks are lower than e-commerce because conversion events (appointments, Rx) have longer lag times.",
        "watch_for": "Low ROAS with high engagement score suggests a landing page or attribution gap — value is being created but not captured.",
    },
    "ctr": {
        "name": "CTR (Click-Through Rate)",
        "source": "Sigma Computing",
        "formula": "CTR = Clicks / Impressions × 100",
        "benchmark": "Display: 0.35%, Social: 0.9% (industry avg). Our campaigns: 1.5–2.5% — well above benchmark.",
        "interpretation": "Measures ad creative effectiveness and audience relevance. Higher CTR signals strong message-audience fit.",
        "watch_for": "CTR can be misleading if click fraud is high — always cross-reference with DCM fraud rate.",
    },
    "engagement_score": {
        "name": "Engagement Score",
        "source": "Adobe Analytics (derived composite)",
        "formula": "(time_on_site/320 × 25) + (scroll_depth/100 × 25) + ((100 − bounce_rate)/100 × 25) + (return_visit_rate/40 × 25)",
        "benchmark": ">70 strong, 50–70 moderate, <50 needs attention",
        "interpretation": "Composite 0–100 score measuring post-click quality. Equally weights 4 Adobe signals: session depth (time), content consumption (scroll), initial relevance (inverse bounce), and loyalty (return visits). A score of 75 means users are genuinely engaging with HealthCentral content after clicking the ad.",
        "watch_for": "High CTR + low engagement = ad promise doesn't match landing content. Low time on site + low scroll = content isn't compelling enough.",
    },
    "quality_score": {
        "name": "Quality Score",
        "source": "DCM Ad Server (derived composite)",
        "formula": "(viewability/100 × 40) + (brand_safety/100 × 40) + ((100 − fraud_rate)/100 × 20)",
        "benchmark": ">85 excellent, 70–85 good, <70 investigate",
        "interpretation": "Composite 0–100 score measuring ad delivery quality from DCM. Viewability (40%) — was the ad actually seen? Brand safety (40%) — was it adjacent to appropriate content? Fraud rate (20%) — were clicks from real humans?",
        "watch_for": "Low quality score with high spend is wasted budget. Viewability below 70% means a significant portion of your impressions are never seen.",
    },
    "viewability": {
        "name": "Viewability Rate",
        "source": "DCM Ad Server",
        "formula": "% of impressions where ≥50% of the ad was in-view for ≥1 second (IAB standard)",
        "benchmark": "IAB benchmark: 70%. Good: >70%, Excellent: >80%",
        "interpretation": "Measures whether your ads were actually seen, not just technically served. A non-viewable impression is effectively wasted spend. Healthcare endemic placements (WebMD, Healthline) tend to have higher viewability than open exchange.",
        "watch_for": "Viewability below 60% on programmatic display suggests poor placement selection — review DCM placement reports and apply viewability-optimized PMPs.",
    },
    "cpm": {
        "name": "CPM (Cost Per Mille)",
        "source": "Sigma Computing",
        "formula": "CPM = (Spend / Impressions) × 1000",
        "benchmark": "Healthcare display: $18–$35. HCP endemic: $50–$120 (HCPs are premium audiences).",
        "interpretation": "Cost to reach 1,000 people. Higher CPM means a more targeted/valuable audience. HCP campaigns should have significantly higher CPMs than DTC — if they're similar, something is wrong with targeting.",
        "watch_for": "Unusually low CPM on HCP campaigns suggests the audience may not actually be physicians — verify endemic publisher targeting.",
    },
    "bounce_rate": {
        "name": "Bounce Rate",
        "source": "Adobe Analytics",
        "formula": "% of sessions where the user left after viewing only one page",
        "benchmark": "Excellent: <35%, Acceptable: 35–50%, Investigate: >50%",
        "interpretation": "High bounce rate means ad-driven traffic is landing and immediately leaving. Could indicate creative-to-landing mismatch, slow page load, or wrong audience reaching the page.",
        "watch_for": "Compare DTC vs HCP bounce rates separately — HCP campaigns often have lower bounce because physicians are highly motivated visitors.",
    },
    "frequency": {
        "name": "Ad Frequency",
        "source": "DCM Ad Server",
        "formula": "Avg number of times a unique user saw the ad per week",
        "benchmark": "Optimal: 3–4x/week. Fatigue risk: >5x. Underexposure: <2x",
        "interpretation": "How often your target audience sees the ad. Too low = insufficient brand impression; too high = creative fatigue and potential negative brand sentiment. Healthcare audiences respond well to message sequencing (awareness → education → action) when frequency is managed.",
        "watch_for": "High frequency combined with declining CTR week-over-week is a clear fatigue signal — rotate creative or apply frequency caps.",
    },
    "fraud_rate": {
        "name": "Invalid Traffic / Fraud Rate",
        "source": "DCM Ad Server",
        "formula": "% of clicks flagged as invalid/non-human by DCM's IVT detection",
        "benchmark": "Excellent: <1%, Acceptable: <3%, Escalate: >5%",
        "interpretation": "Invalid traffic wastes budget and distorts performance metrics — campaigns with high fraud look better on CTR than they really are. Healthcare advertisers are prime targets for ad fraud due to high CPMs.",
        "watch_for": "Any campaign with >4% fraud rate needs immediate investigation — review which publishers/placements are driving it via DCM placement reports.",
    },
    "brand_safety": {
        "name": "Brand Safety Rate",
        "source": "DCM Ad Server",
        "formula": "% of impressions served adjacent to brand-safe content",
        "benchmark": ">95% excellent, 90–95% acceptable, <90% needs review",
        "interpretation": "Critical for healthcare advertisers. A pharmaceutical or health brand appearing next to misinformation, graphic content, or inappropriate topics creates significant reputational and regulatory risk. Endemic publishers (WebMD, Healthline) have near-100% brand safety.",
        "watch_for": "Brand safety issues are most common on open exchange programmatic. If a campaign drops below 93%, audit placement exclusion lists.",
    },
    "cpc": {
        "name": "CPC (Cost Per Click)",
        "source": "Sigma Computing",
        "formula": "CPC = Spend / Clicks",
        "benchmark": "Healthcare display: $1–$3. Social: $0.50–$2. HCP endemic: $5–$20.",
        "interpretation": "How much you pay per visitor driven to the site. Should be evaluated alongside landing page conversion rate — a high CPC is acceptable if the visitor-to-conversion rate is strong.",
        "watch_for": "High CPC + high bounce rate = expensive visitors who immediately leave. This is the worst efficiency scenario.",
    },
    "cvr": {
        "name": "Conversion Rate (CVR)",
        "source": "Sigma Computing",
        "formula": "CVR = Conversions / Clicks × 100",
        "benchmark": "Healthcare: 1.5–3% is typical. >3% strong. <1% investigate.",
        "interpretation": "% of ad clicks that result in a meaningful action (symptom checker completion, appointment request, content download, etc.). Reflects the quality of both the landing experience and the audience targeting.",
        "watch_for": "CVR below 1% usually points to a landing page problem, not an ad problem — the audience is clicking but the experience isn't compelling enough to convert.",
    },
}

def explain_metric(metric_name: str) -> dict:
    """Return structured explanation for any known metric."""
    key = metric_name.lower().replace(" ", "_").replace("-", "_")
    # Fuzzy match
    for k in METRIC_EXPLANATIONS:
        if k in key or key in k:
            return {"found": True, "metric": METRIC_EXPLANATIONS[k]}

    return {
        "found": False,
        "available_metrics": list(METRIC_EXPLANATIONS.keys()),
        "message": f"No detailed explanation found for '{metric_name}'. Available: {', '.join(METRIC_EXPLANATIONS.keys())}",
    }


# ─── Tool: analyze_campaign ────────────────────────────────────────────────────

def analyze_campaign(df: pd.DataFrame, campaign_id: str) -> dict:
    """Deep-dive analysis of a single campaign across all 3 data sources."""
    camp = df[df["campaign_id"] == campaign_id]
    if camp.empty:
        # Try by name (partial match)
        camp = df[df["campaign_name"].str.contains(campaign_id, case=False, na=False)]
    if camp.empty:
        return {"error": f"Campaign '{campaign_id}' not found. Available: {df['campaign_id'].unique().tolist()}"}

    name = camp["campaign_name"].iloc[0]
    ctype = camp["campaign_type"].iloc[0]
    condition = camp["condition_category"].iloc[0]
    channel = camp["channel"].iloc[0]

    # Sigma aggregates
    total_impressions = int(camp["impressions"].sum())
    total_clicks = int(camp["clicks"].sum())
    total_conversions = int(camp["conversions"].sum())
    total_spend = float(camp["spend_usd"].sum())
    avg_ctr = float(camp["ctr_pct"].mean())
    avg_cpm = float(camp["cpm_usd"].mean())
    avg_cpc = float(camp["cpc_usd"].mean())
    avg_roas = float(camp["roas"].mean())
    cost_per_conv = float(camp["cost_per_conversion"].mean())

    # Adobe aggregates
    avg_time = float(camp["avg_time_on_site_sec"].mean())
    avg_pages = float(camp["pages_per_session"].mean())
    avg_bounce = float(camp["bounce_rate_pct"].mean())
    avg_scroll = float(camp["avg_scroll_depth_pct"].mean())
    avg_return = float(camp["return_visit_rate_pct"].mean())
    avg_mobile = float(camp["mobile_pct"].mean())
    avg_engagement = float(camp["engagement_score"].mean())

    # DCM aggregates
    avg_viewability = float(camp["viewability_pct"].mean())
    avg_frequency = float(camp["avg_frequency"].mean())
    avg_fraud = float(camp["fraud_rate_pct"].mean())
    avg_brand_safety = float(camp["brand_safety_pct"].mean())
    avg_quality = float(camp["quality_score"].mean())

    # Portfolio benchmarks for comparison
    portfolio_roas = float(df["roas"].mean())
    portfolio_engagement = float(df["engagement_score"].mean())
    portfolio_quality = float(df["quality_score"].mean())

    # Trend: is performance improving or declining?
    weekly_roas = camp.groupby("week_start")["roas"].mean()
    trend = "improving" if weekly_roas.iloc[-1] > weekly_roas.iloc[0] else "declining"
    roas_change = float(weekly_roas.iloc[-1] - weekly_roas.iloc[0])

    # Diagnose issues
    issues = []
    strengths = []

    if avg_roas >= portfolio_roas * 1.2:
        strengths.append(f"ROAS {avg_roas:.1f}x — {((avg_roas/portfolio_roas)-1)*100:.0f}% above portfolio avg")
    elif avg_roas < portfolio_roas * 0.8:
        issues.append(f"ROAS {avg_roas:.1f}x — {((portfolio_roas/avg_roas)-1)*100:.0f}% below portfolio avg ({portfolio_roas:.1f}x)")

    if avg_viewability < 70:
        issues.append(f"Viewability {avg_viewability:.0f}% — below IAB 70% benchmark [DCM]")
    elif avg_viewability >= 80:
        strengths.append(f"Viewability {avg_viewability:.0f}% — above 80% [DCM]")

    if avg_bounce > 50:
        issues.append(f"Bounce rate {avg_bounce:.0f}% — above 50% threshold [Adobe]")
    elif avg_bounce < 35:
        strengths.append(f"Bounce rate {avg_bounce:.0f}% — excellent [Adobe]")

    if avg_fraud > 3:
        issues.append(f"Fraud rate {avg_fraud:.1f}% — above 3% acceptable threshold [DCM]")

    if avg_frequency > 5:
        issues.append(f"Frequency {avg_frequency:.1f}x/week — fatigue risk (>5x) [DCM]")

    if avg_engagement >= portfolio_engagement * 1.15:
        strengths.append(f"Engagement score {avg_engagement:.0f}/100 — strong [Adobe]")
    elif avg_engagement < portfolio_engagement * 0.85:
        issues.append(f"Engagement score {avg_engagement:.0f}/100 — below portfolio avg ({portfolio_engagement:.0f}) [Adobe]")

    return {
        "campaign": {
            "id": campaign_id,
            "name": name,
            "type": ctype,
            "condition": condition,
            "channel": channel,
        },
        "sigma_metrics": {
            "impressions": total_impressions,
            "clicks": total_clicks,
            "conversions": total_conversions,
            "spend_usd": round(total_spend, 2),
            "ctr_pct": round(avg_ctr, 2),
            "cpm_usd": round(avg_cpm, 2),
            "cpc_usd": round(avg_cpc, 2),
            "roas": round(avg_roas, 2),
            "cost_per_conversion": round(cost_per_conv, 2),
        },
        "adobe_metrics": {
            "avg_time_on_site_sec": round(avg_time, 1),
            "pages_per_session": round(avg_pages, 1),
            "bounce_rate_pct": round(avg_bounce, 1),
            "avg_scroll_depth_pct": round(avg_scroll, 1),
            "return_visit_rate_pct": round(avg_return, 1),
            "mobile_pct": round(avg_mobile, 1),
            "engagement_score": round(avg_engagement, 1),
        },
        "dcm_metrics": {
            "viewability_pct": round(avg_viewability, 1),
            "avg_frequency": round(avg_frequency, 1),
            "fraud_rate_pct": round(avg_fraud, 1),
            "brand_safety_pct": round(avg_brand_safety, 1),
            "quality_score": round(avg_quality, 1),
        },
        "benchmarks": {
            "portfolio_avg_roas": round(portfolio_roas, 2),
            "portfolio_avg_engagement": round(portfolio_engagement, 1),
            "portfolio_avg_quality": round(portfolio_quality, 1),
        },
        "diagnosis": {
            "trend": trend,
            "roas_change_over_period": round(roas_change, 2),
            "strengths": strengths,
            "issues": issues,
        }
    }


# ─── Tool: compare_campaigns ───────────────────────────────────────────────────

def compare_campaigns(df: pd.DataFrame, campaign_ids: list) -> dict:
    """Side-by-side comparison of 2–4 campaigns across all metrics."""
    results = []
    for cid in campaign_ids:
        analysis = analyze_campaign(df, cid)
        if "error" not in analysis:
            results.append(analysis)

    if len(results) < 2:
        return {"error": "Need at least 2 valid campaign IDs to compare."}

    # Build comparison matrix
    comparison = []
    metrics = [
        ("ROAS", "sigma_metrics", "roas"),
        ("CTR %", "sigma_metrics", "ctr_pct"),
        ("Spend ($)", "sigma_metrics", "spend_usd"),
        ("Conversions", "sigma_metrics", "conversions"),
        ("Cost/Conv ($)", "sigma_metrics", "cost_per_conversion"),
        ("Engagement Score", "adobe_metrics", "engagement_score"),
        ("Bounce Rate %", "adobe_metrics", "bounce_rate_pct"),
        ("Time on Site (s)", "adobe_metrics", "avg_time_on_site_sec"),
        ("Viewability %", "dcm_metrics", "viewability_pct"),
        ("Fraud Rate %", "dcm_metrics", "fraud_rate_pct"),
        ("Quality Score", "dcm_metrics", "quality_score"),
    ]

    for label, section, field in metrics:
        row = {"metric": label}
        for r in results:
            row[r["campaign"]["name"]] = r[section][field]
        comparison.append(row)

    winner_by = {
        "roas": max(results, key=lambda r: r["sigma_metrics"]["roas"])["campaign"]["name"],
        "engagement": max(results, key=lambda r: r["adobe_metrics"]["engagement_score"])["campaign"]["name"],
        "quality": max(results, key=lambda r: r["dcm_metrics"]["quality_score"])["campaign"]["name"],
        "efficiency": min(results, key=lambda r: r["sigma_metrics"]["cost_per_conversion"])["campaign"]["name"],
    }

    return {
        "campaigns_compared": [r["campaign"] for r in results],
        "comparison_matrix": comparison,
        "winner_by": winner_by,
    }


# ─── Tool: get_portfolio_summary ───────────────────────────────────────────────

def get_portfolio_summary(df: pd.DataFrame, filters: dict = None) -> dict:
    """High-level portfolio health snapshot with filtered or full data."""
    filtered = _apply_filters(df, filters or {})

    n_campaigns = filtered["campaign_id"].nunique()
    total_impressions = int(filtered["impressions"].sum())
    total_clicks = int(filtered["clicks"].sum())
    total_conversions = int(filtered["conversions"].sum())
    total_spend = float(filtered["spend_usd"].sum())
    avg_ctr = float(total_clicks / total_impressions * 100) if total_impressions > 0 else 0
    avg_roas = float(filtered["roas"].mean())
    avg_engagement = float(filtered["engagement_score"].mean())
    avg_quality = float(filtered["quality_score"].mean())
    avg_viewability = float(filtered["viewability_pct"].mean())
    avg_fraud = float(filtered["fraud_rate_pct"].mean())

    # Top and bottom performers
    camp_agg = filtered.groupby(["campaign_id", "campaign_name", "campaign_type"])["roas"].mean()
    top_3 = camp_agg.nlargest(3).reset_index().to_dict(orient="records")
    bottom_3 = camp_agg.nsmallest(3).reset_index().to_dict(orient="records")

    # DTC vs HCP split
    dtc = filtered[filtered["campaign_type"] == "DTC"]
    hcp = filtered[filtered["campaign_type"] == "HCP"]

    # Alerts
    alerts = []
    if avg_viewability < 70:
        alerts.append(f"Portfolio avg viewability {avg_viewability:.0f}% is below IAB 70% benchmark")
    if avg_fraud > 3:
        alerts.append(f"Portfolio avg fraud rate {avg_fraud:.1f}% exceeds 3% threshold")
    high_freq = filtered.groupby("campaign_id")["avg_frequency"].mean()
    high_freq_camps = high_freq[high_freq > 5].index.tolist()
    if high_freq_camps:
        alerts.append(f"{len(high_freq_camps)} campaign(s) have frequency >5x — fatigue risk")

    return {
        "overview": {
            "campaigns": n_campaigns,
            "impressions": total_impressions,
            "clicks": total_clicks,
            "conversions": total_conversions,
            "total_spend_usd": round(total_spend, 2),
            "avg_ctr_pct": round(avg_ctr, 2),
            "avg_roas": round(avg_roas, 2),
            "avg_engagement_score": round(avg_engagement, 1),
            "avg_quality_score": round(avg_quality, 1),
            "avg_viewability_pct": round(avg_viewability, 1),
            "avg_fraud_rate_pct": round(avg_fraud, 1),
        },
        "by_type": {
            "DTC": {
                "campaigns": int(dtc["campaign_id"].nunique()),
                "spend_usd": round(float(dtc["spend_usd"].sum()), 2),
                "avg_roas": round(float(dtc["roas"].mean()), 2),
                "avg_engagement": round(float(dtc["engagement_score"].mean()), 1),
            },
            "HCP": {
                "campaigns": int(hcp["campaign_id"].nunique()),
                "spend_usd": round(float(hcp["spend_usd"].sum()), 2),
                "avg_roas": round(float(hcp["roas"].mean()), 2),
                "avg_engagement": round(float(hcp["engagement_score"].mean()), 1),
            },
        },
        "top_performers": top_3,
        "bottom_performers": bottom_3,
        "alerts": alerts,
    }


# ─── Tool: get_channel_analysis ────────────────────────────────────────────────

def get_channel_analysis(df: pd.DataFrame, filters: dict = None) -> dict:
    """Performance breakdown by channel."""
    filtered = _apply_filters(df, filters or {})

    channel_agg = filtered.groupby("channel").agg(
        campaigns=("campaign_id", "nunique"),
        impressions=("impressions", "sum"),
        clicks=("clicks", "sum"),
        conversions=("conversions", "sum"),
        spend_usd=("spend_usd", "sum"),
        avg_roas=("roas", "mean"),
        avg_ctr=("ctr_pct", "mean"),
        avg_viewability=("viewability_pct", "mean"),
        avg_engagement=("engagement_score", "mean"),
        avg_fraud=("fraud_rate_pct", "mean"),
    ).reset_index().round(2)

    channel_agg["cpc"] = (channel_agg["spend_usd"] / channel_agg["clicks"].replace(0, 1)).round(2)
    channel_agg["cost_per_conv"] = (channel_agg["spend_usd"] / channel_agg["conversions"].replace(0, 1)).round(2)

    records = channel_agg.to_dict(orient="records")

    best_roas_channel = channel_agg.loc[channel_agg["avg_roas"].idxmax(), "channel"]
    most_spend_channel = channel_agg.loc[channel_agg["spend_usd"].idxmax(), "channel"]
    best_viewability = channel_agg.loc[channel_agg["avg_viewability"].idxmax(), "channel"]

    return {
        "channels": records,
        "insights": {
            "best_roas_channel": best_roas_channel,
            "most_spend_channel": most_spend_channel,
            "best_viewability_channel": best_viewability,
        }
    }


# ─── Helper ────────────────────────────────────────────────────────────────────

def _apply_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    filtered = df.copy()
    if filters.get("campaign_type", "All") not in ("All", None, ""):
        filtered = filtered[filtered["campaign_type"] == filters["campaign_type"]]
    if filters.get("condition", "All") not in ("All", None, ""):
        filtered = filtered[filtered["condition_category"] == filters["condition"]]
    if filters.get("channel", "All") not in ("All", None, ""):
        filtered = filtered[filtered["channel"] == filters["channel"]]
    return filtered
