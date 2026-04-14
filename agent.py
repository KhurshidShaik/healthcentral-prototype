"""
OpenAI function-calling agent for HealthCentral Campaign Intelligence.
One clean loop: message → tool calls → tool results → final response + actions.
"""
import json
import os
import re
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

from prompts import SYSTEM_PROMPT, build_context_message
from tools import (
    filter_dashboard,
    explain_metric,
    analyze_campaign,
    compare_campaigns,
    get_portfolio_summary,
    get_channel_analysis,
)

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ─── Tool JSON Schemas ─────────────────────────────────────────────────────────

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "filter_dashboard",
            "description": (
                "Update the dashboard filters to narrow the visible data. "
                "Call this when the user wants to see a specific campaign type, condition category, "
                "or channel. Always returns the new filter state so the frontend can re-render."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "campaign_type": {
                        "type": "string",
                        "enum": ["All", "DTC", "HCP"],
                        "description": "Filter by DTC (consumer) or HCP (healthcare professional) campaigns.",
                    },
                    "condition": {
                        "type": "string",
                        "enum": [
                            "All", "Diabetes", "Mental Health", "HIV/AIDS", "Skin Cancer",
                            "COPD", "Migraine", "Asthma", "Cancer", "Autoimmune",
                            "Heart Disease", "Arthritis", "Skin Conditions",
                        ],
                        "description": "Filter by health condition / therapeutic area.",
                    },
                    "channel": {
                        "type": "string",
                        "enum": [
                            "All", "Programmatic Display", "Social + Native",
                            "Programmatic + Social", "Social + Display",
                            "Endemic Display", "Email + Endemic", "Endemic + Email",
                        ],
                        "description": "Filter by advertising channel.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "explain_metric",
            "description": (
                "Explain any campaign metric in detail: formula, data source, benchmarks, "
                "and what to watch for. Call when the user asks 'what is X', 'how is X calculated', "
                "or 'explain X'. Available metrics: roas, ctr, engagement_score, quality_score, "
                "viewability, cpm, bounce_rate, frequency, fraud_rate, brand_safety, cpc, cvr."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "metric_name": {
                        "type": "string",
                        "description": "The metric to explain (e.g. 'roas', 'engagement_score', 'viewability').",
                    },
                },
                "required": ["metric_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_campaign",
            "description": (
                "Deep-dive analysis of a single campaign across all 3 data sources (Sigma, Adobe, DCM). "
                "Returns metrics, strengths, issues, and trend. Use when user asks about a specific campaign "
                "or asks why something is underperforming."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "campaign_id": {
                        "type": "string",
                        "description": (
                            "Campaign ID (e.g. 'HC-DTC-001') or partial name match "
                            "(e.g. 'Diabetes', 'Cardiology', 'Mental Health')."
                        ),
                    },
                },
                "required": ["campaign_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_campaigns",
            "description": (
                "Side-by-side comparison of 2–4 campaigns across all metrics. "
                "Use when user says 'compare X and Y', 'DTC vs HCP', or 'which is performing better'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "campaign_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of 2–4 campaign IDs or partial name strings.",
                        "minItems": 2,
                        "maxItems": 4,
                    },
                },
                "required": ["campaign_ids"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_portfolio_summary",
            "description": (
                "Overall portfolio health check with KPIs, top/bottom performers, DTC vs HCP split, "
                "and any active alerts. Use for 'how are we doing overall', 'portfolio summary', "
                "or 'give me an overview'."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_channel_analysis",
            "description": (
                "Performance breakdown by advertising channel (Programmatic Display, Social, Endemic, etc.). "
                "Use when user asks about channel performance, channel mix, or 'which channel is best'."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
]


# ─── Tool Router ───────────────────────────────────────────────────────────────

def _route_tool(name: str, args: dict, df: pd.DataFrame, current_filters: dict) -> tuple[dict, list]:
    """
    Execute the named tool and return (result_dict, actions_list).
    actions_list contains frontend directives (e.g. filter_update).
    """
    actions = []

    if name == "filter_dashboard":
        merged = {**current_filters, **args}
        result = filter_dashboard(df, **merged)
        actions.append({"type": "filter_update", "filters": result["new_filters"]})
        return result, actions

    if name == "explain_metric":
        return explain_metric(**args), actions

    if name == "analyze_campaign":
        return analyze_campaign(df, **args), actions

    if name == "compare_campaigns":
        return compare_campaigns(df, **args), actions

    if name == "get_portfolio_summary":
        return get_portfolio_summary(df, current_filters), actions

    if name == "get_channel_analysis":
        return get_channel_analysis(df, current_filters), actions

    return {"error": f"Unknown tool: {name}"}, actions


# ─── Main Agent Entry Point ────────────────────────────────────────────────────

def run_agent(user_message: str, current_filters: dict, history: list,
              df: pd.DataFrame, live_stats: dict) -> dict:
    """
    Full agent turn:
      1. Build message array (system + context + history + user msg)
      2. Call OpenAI with tools
      3. Handle tool calls (may loop for multi-tool turns)
      4. Return {message, actions}
    """
    # Build system + context
    context = build_context_message(current_filters, live_stats)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT + "\n\n" + context},
    ]

    # Append conversation history (cap at last 20 turns for context efficiency)
    for turn in history[-20:]:
        messages.append(turn)

    # Add current user message
    messages.append({"role": "user", "content": user_message})

    accumulated_actions = []
    max_tool_rounds = 3  # guard against infinite loops

    for _ in range(max_tool_rounds):
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=TOOL_DEFINITIONS,
            tool_choice="auto",
            temperature=0.2,
            max_tokens=450,
        )

        msg = response.choices[0].message

        # No tool call → final response
        if not msg.tool_calls:
            content, suggestions = _parse_suggestions(msg.content or "")
            return {
                "message": content,
                "actions": accumulated_actions,
                "suggestions": suggestions,
            }

        # Append assistant message with tool calls
        messages.append({"role": "assistant", "content": msg.content, "tool_calls": [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            }
            for tc in msg.tool_calls
        ]})

        # Execute each tool call and append results
        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments)
            result, actions = _route_tool(tc.function.name, args, df, current_filters)
            accumulated_actions.extend(actions)

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(result, default=str),
            })

    # Fallback if we hit max rounds
    return {
        "message": "I've processed your request. The dashboard has been updated.",
        "actions": accumulated_actions,
        "suggestions": [],
    }


# ─── Suggestion Parser ─────────────────────────────────────────────────────────

def _parse_suggestions(content: str) -> tuple[str, list]:
    """
    Extract SUGGEST:[...] from the end of the model response.
    Returns (clean_message, suggestions_list).
    """
    match = re.search(r'SUGGEST:\s*(\[.*?\])\s*$', content, re.DOTALL)
    if not match:
        return content.strip(), []
    try:
        suggestions = json.loads(match.group(1))
        clean = content[:match.start()].strip()
        return clean, suggestions[:3]
    except (json.JSONDecodeError, ValueError):
        return content.strip(), []
