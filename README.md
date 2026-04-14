# HealthCentral — Campaign Performance Intelligence Hub

A multi-source BI prototype demonstrating how campaign data from **Sigma Computing**, **Adobe Analytics**, and **DCM ad servers** can be unified into a single reporting layer with a **conversational AI agent** for insight generation, dashboard control, and deep explainability.

Built by **Khurshid Shaik** as a proof-of-concept for the BI Analyst role at HealthCentral Corporation.

---

## Architecture

```
┌─────────────────┐  ┌──────────────────┐  ┌─────────────────┐
│ Sigma Computing  │  │ Adobe Analytics  │  │   DCM / DFA     │
│ (campaign KPIs)  │  │  (engagement)    │  │  (ad delivery)  │
└───────┬─────────┘  └────────┬─────────┘  └───────┬─────────┘
        │ CSV export          │ CSV export          │ CSV export
        └─────────────┬───────┴─────────────────────┘
                      ▼
        ┌─────────────────────────────┐
        │   Python ETL (pandas)       │
        │   Extract → Validate →      │
        │   Transform → Join → Load   │
        └─────────────┬───────────────┘
                      ▼
        ┌─────────────────────────────┐
        │  Unified Campaign Dataset   │
        │  (standardized metrics)     │
        └─────────────┬───────────────┘
                      ▼
        ┌─────────────────────────────┐
        │  FastAPI Backend            │
        │  /api/data  /api/chat       │
        │  /api/filters               │
        └─────────────┬───────────────┘
                      ▼
        ┌─────────────────────────────┐    ┌──────────────────────────┐
        │  React Dashboard (dark UI)  │◄───│  OpenAI GPT-4o Agent     │
        │  Overview · Campaigns       │    │  Function-calling tools   │
        │  Channels · Quality         │    │  filter · analyze ·      │
        │  + AI Chat Panel            │    │  explain · compare        │
        └─────────────────────────────┘    └──────────────────────────┘
```

---

## Data Sources (Simulated)

| Source | File | Metrics | Rows |
|--------|------|---------|------|
| Sigma Computing | `data/sigma_campaign_metrics.csv` | Impressions, clicks, CTR, conversions, spend, CPM, CPC | 144 |
| Adobe Analytics | `data/adobe_engagement.csv` | Time on site, pages/session, bounce rate, scroll depth, return rate | 144 |
| DCM Ad Server | `data/dcm_delivery.csv` | Viewability, video completion, frequency, reach, fraud rate, brand safety | 144 |

12 campaigns (7 DTC, 5 HCP) × 12 weeks (Jan–Mar 2026)

---

## Derived Metrics (ETL)

| Metric | Formula | Source |
|--------|---------|--------|
| **ROAS** | Conversions × $35 avg value / Spend | Sigma |
| **Engagement Score** (0–100) | Time on site (25%) + Scroll depth (25%) + Inverse bounce (25%) + Return visits (25%) | Adobe |
| **Quality Score** (0–100) | Viewability (40%) + Brand safety (40%) + Inverse fraud rate (20%) | DCM |
| **Cost per Conversion** | Spend / Conversions | Sigma |

---

## AI Agent Capabilities

The conversational agent (GPT-4o with function calling) understands natural language and can:

| What you say | What happens |
|---|---|
| *"Show me only HCP campaigns"* | Filters dashboard to HCP, charts re-render |
| *"Why is Cardiology Insights underperforming?"* | Pulls data from all 3 sources, diagnoses root cause |
| *"Explain how engagement score is calculated"* | Returns formula, benchmarks, what to watch for |
| *"Compare DTC vs HCP performance"* | Side-by-side metric comparison across all sources |
| *"Which channel has the best ROAS?"* | Runs channel analysis, surfaces the winner |
| *"Give me a portfolio overview"* | Summary KPIs, top/bottom performers, active alerts |

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/KhurshidShaik/healthcentral-prototype.git
cd healthcentral-prototype

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your OpenAI API key
echo "OPENAI_API_KEY=sk-your-key-here" > .env

# 4. Generate source data (if CSVs not present)
python generate_sigma.py
python generate_adobe.py
python generate_dcm.py

# 5. Launch
python main.py
```

Open **http://localhost:8000** — dashboard loads with AI chat panel.

---

## Project Structure

```
healthcentral-prototype/
├── main.py              # FastAPI server — serves UI + API endpoints
├── agent.py             # OpenAI GPT-4o function-calling agent
├── tools.py             # Agent tool implementations (data analysis functions)
├── prompts.py           # System prompt with full schema, benchmarks, metric definitions
├── etl_pipeline.py      # Extract → Validate → Transform → Join → Load
├── generate_sigma.py    # Simulated Sigma Computing data generator
├── generate_adobe.py    # Simulated Adobe Analytics data generator
├── generate_dcm.py      # Simulated DCM ad server data generator
├── index.html           # React dashboard (single file, CDN-loaded, no build step)
├── requirements.txt
└── data/
    ├── sigma_campaign_metrics.csv
    ├── adobe_engagement.csv
    ├── dcm_delivery.csv
    └── unified_campaign_performance.csv
```

---

## JD Alignment

| JD Requirement | How This Prototype Demonstrates It |
|---|---|
| "Own campaign performance data products" | FastAPI + React dashboard is a fully deployable campaign data product |
| "Translate multi-source data into insights" | 3 sources unified via ETL pipeline into one dataset |
| "Evolve from descriptive to prescriptive" | AI agent moves from summary → diagnostic → prescriptive recommendation |
| "Integrate Sigma, Adobe Analytics, ad servers" | All three tools simulated with realistic, schema-accurate data |
| "Define and standardize metrics" | Engagement Score, Quality Score, ROAS — documented with formulas |
| "AI-assisted reporting workflows" | Conversational agent answers questions, filters dashboard, explains every metric |
| "HCP and DTC campaigns" | 12 campaigns across both types with condition-level filtering |
| "Scalable, reliable data products" | Modular ETL + FastAPI + stateless React = production-deployable pattern |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python 3.10+, FastAPI, Uvicorn |
| AI Agent | OpenAI GPT-4o (function calling) |
| Data processing | pandas, numpy |
| Frontend | React 18 (CDN), Recharts, Tailwind CSS |
| Data sources | Simulated CSV exports (Sigma, Adobe, DCM schemas) |
