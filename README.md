# HealthCentral — Campaign Performance Intelligence Hub

A multi-source BI prototype demonstrating how campaign data from **Sigma Computing**, **Adobe Analytics**, and **DCM ad servers** can be unified into a single reporting layer with AI-assisted insight generation.

Built by **Khurshid Shaik** as a proof-of-concept for the BI Analyst role at HealthCentral Corporation.

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
        │  Streamlit Dashboard        │
        │  + AI Insight Layer         │
        └─────────────────────────────┘
```

## Data Sources (Simulated)

| Source | File | Metrics | Rows |
|--------|------|---------|------|
| Sigma Computing | `data/sigma_campaign_metrics.csv` | Impressions, clicks, CTR, conversions, spend, CPM, CPC | 144 |
| Adobe Analytics | `data/adobe_engagement.csv` | Time on site, pages/session, bounce rate, scroll depth, return rate | 144 |
| DCM Ad Server | `data/dcm_delivery.csv` | Viewability, video completion, frequency, reach, fraud rate, brand safety | 144 |

## Derived Metrics (ETL)

- **ROAS** — Return on ad spend (conversions × $35 avg value / spend)
- **Engagement Score** — Composite 0-100 from time on site, scroll depth, bounce rate, return visits
- **Quality Score** — Composite 0-100 from viewability, brand safety, fraud rate
- **Cost per Conversion** — Spend / conversions

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Generate source data
python generate_sigma.py
python generate_adobe.py
python generate_dcm.py

# Run ETL pipeline
python etl_pipeline.py

# Launch dashboard
streamlit run app.py
```

## Deploy to Streamlit Cloud (Free)

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set main file as `app.py`
5. Deploy — you'll get a shareable link

## JD Alignment

| JD Requirement | How This Prototype Demonstrates It |
|---------------|-----------------------------------|
| "Own campaign performance data products" | The entire dashboard is a campaign data product |
| "Translate multi-source data into insights" | 3 sources unified via ETL pipeline |
| "Evolve from descriptive to prescriptive" | AI insight button: summary → diagnostic → prescriptive |
| "Integrate Sigma, Adobe Analytics, ad servers" | Exact tools simulated with realistic data |
| "Define and standardize metrics" | Engagement score, quality score, ROAS — documented |
| "AI-assisted reporting workflows" | AI generates diagnostic + prescriptive recommendations |
| "HCP and DTC campaigns" | Both types with filtering and comparison |
| "Scalable, reliable data products" | Modular ETL pipeline + Streamlit = deployable |

## Tech Stack (All Free)

- **Python 3.10+** — ETL pipeline
- **pandas** — Data processing
- **Streamlit** — Dashboard UI
- **Plotly** — Interactive charts
- **Streamlit Cloud** — Free hosting
