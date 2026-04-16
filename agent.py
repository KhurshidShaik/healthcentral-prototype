"""
OpenAI function-calling agent for HealthCentral Campaign Intelligence.
One clean loop: message -> tool calls -> tool results -> final response + actions.

Langfuse v3 observability — trace structure per chat turn:

  chat-request (span, root)                    ← created in main.py
  ├── data-preparation (span)                  ← filter DataFrame, compute live KPIs
  └── agent-run (agent)                        ← AI reasoning loop (enables Agent Graph tab)
        ├── openai-call · round 1 (generation) ← LLM call #1: model decides what to do
        │     output: text OR { tool_calls: [...] }
        ├── tool:<name> · round 1 (tool)       ← tool executes, result fed back to model
        └── openai-call · round 2 (generation) ← LLM call #2: model synthesises tool results

Observation types and why they matter:
  - "span"       → generic step (any operation worth timing/logging)
  - "generation" → LLM call (Langfuse uses this to compute token cost + show model analytics)
  - "tool"       → function/tool execution (shown as distinct nodes in Agent Graph)
  - "agent"      → top-level AI reasoning unit (unlocks the Agent Graph view in Langfuse UI)
"""
import json
import os
import re
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Bridge env var naming: .env uses LANGFUSE_BASE_URL, v3 SDK expects LANGFUSE_HOST
if os.getenv("LANGFUSE_BASE_URL") and not os.getenv("LANGFUSE_HOST"):
    os.environ["LANGFUSE_HOST"] = os.getenv("LANGFUSE_BASE_URL")

# get_client()         → returns a process-level singleton Langfuse client
#                        (safe to import in multiple files — same instance is reused)
# propagate_attributes → context manager that stamps session_id, user_id, tags, and
#                        metadata onto every observation created inside its block
from langfuse import get_client, propagate_attributes

from prompts import SYSTEM_PROMPT, build_context_message
from tools import (
    filter_dashboard,
    explain_metric,
    analyze_campaign,
    compare_campaigns,
    get_portfolio_summary,
    get_channel_analysis,
)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

APP_VERSION = "1.0.0"
ENVIRONMENT = os.getenv("LANGFUSE_ENVIRONMENT", "development")

# ─── Langfuse Client ─────────────────────────────────────────────────────────
# Credentials are read automatically from env vars:
#   LANGFUSE_SECRET_KEY, LANGFUSE_PUBLIC_KEY, LANGFUSE_HOST
# auth_check() makes a quick ping to verify keys are valid before any user traffic.
langfuse = get_client()

if langfuse.auth_check():
    print("Langfuse auth OK — traces will be sent to", os.getenv("LANGFUSE_HOST", os.getenv("LANGFUSE_BASE_URL")))
else:
    print("Langfuse auth FAILED — check LANGFUSE_SECRET_KEY, LANGFUSE_PUBLIC_KEY, and LANGFUSE_HOST in .env")

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
                "Side-by-side comparison of 2-4 campaigns across all metrics. "
                "Use when user says 'compare X and Y', 'DTC vs HCP', or 'which is performing better'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "campaign_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of 2-4 campaign IDs or partial name strings.",
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




# ─── Observed LLM Call (GENERATION type) ─────────────────────────────────────

def _call_openai(messages: list, round_num: int = 1) -> object:
    """
    Single OpenAI chat completion, observed as a Langfuse GENERATION.

    Why GENERATION type?
      Langfuse treats "generation" observations specially — it uses them to display
      per-call token counts, compute cost estimates, and populate the LLM Analytics
      dashboard. Any LLM call should use as_type="generation".

    Why include round_num?
      The agent can call OpenAI multiple times per user turn (once to decide which
      tool to call, then again to synthesise the tool result). Naming them
      "round 1 / round 2" makes the trace timeline immediately readable.
    """
    # start_as_current_observation() opens a new child observation nested under
    # whatever observation is currently active (here: the "agent-run" agent span).
    # The `with` block defines the observation's time boundaries automatically.
    with langfuse.start_as_current_observation(
        as_type="generation",
        name=f"openai-call · round {round_num}",
        model="gpt-4o",
        # `input` is shown in the Langfuse trace detail — log the full message array
        # so you can see exactly what prompt + history was sent to the model.
        input=messages,
        model_parameters={"temperature": 0.2, "max_tokens": 450},
        metadata={"round": round_num},
    ) as gen:
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                tools=TOOL_DEFINITIONS,
                tool_choice="auto",
                temperature=0.2,
                max_tokens=450,
            )

            msg = response.choices[0].message

            # ── Build the output to log ──────────────────────────────────────
            # This is the root cause of the missing output bug:
            #   When the model decides to call a tool, msg.content is None.
            #   Logging msg.content directly shows nothing in the trace.
            #   Instead we check what the model actually decided to do:
            #     - "respond"    → model returned a plain text answer
            #     - "call_tools" → model chose to invoke one or more tools
            if msg.tool_calls:
                # Model is requesting tool calls — capture which tools and with what args.
                # This is what you actually want to see in the trace for tool-using turns.
                output = {
                    "decision": "call_tools",
                    "tool_calls": [
                        {
                            "tool": tc.function.name,
                            "arguments": json.loads(tc.function.arguments),
                        }
                        for tc in msg.tool_calls
                    ],
                }
                # Occasionally the model also returns reasoning text alongside tool calls
                if msg.content:
                    output["reasoning"] = msg.content
            else:
                # Model responded with a plain text answer (final answer turn)
                output = {"decision": "respond", "content": msg.content or ""}

            # gen.update() sets the output and usage AFTER the API call completes.
            # usage_details lets Langfuse automatically calculate cost per call.
            gen.update(
                output=output,
                usage_details={
                    "input": response.usage.prompt_tokens,
                    "output": response.usage.completion_tokens,
                    "total": response.usage.total_tokens,
                },
                metadata={
                    "round": round_num,
                    "finish_reason": response.choices[0].finish_reason,
                    "tools_requested": len(msg.tool_calls) if msg.tool_calls else 0,
                },
            )
        except Exception as e:
            # level="ERROR" highlights this observation in red in the Langfuse UI
            gen.update(
                output=f"OpenAI error: {type(e).__name__}: {e}",
                level="ERROR",
                metadata={"error_type": type(e).__name__, "round": round_num},
            )
            raise
    return response


# ─── Observed Tool Router (TOOL type) ─────────────────────────────────────────

def _execute_tool(
    name: str,
    args: dict,
    df: pd.DataFrame,
    current_filters: dict,
    round_num: int = 1,
) -> tuple[dict, list]:
    """
    Execute a single tool call, observed as a Langfuse TOOL observation.

    Why TOOL type?
      Langfuse renders "tool" observations as distinct nodes in the Agent Graph view,
      separate from generation nodes. This makes it easy to see: model called tool X
      with these args → got back this result → model then synthesised the answer.

    round_num is passed through so tool names like "tool:analyze_campaign · round 1"
    align with the corresponding "openai-call · round 1" that requested them.
    """
    # start_as_current_observation() nests this TOOL observation under the active
    # GENERATION observation (the openai-call that requested this tool).
    # input= logs exactly what arguments the model passed to the tool.
    with langfuse.start_as_current_observation(
        as_type="tool",
        name=f"tool:{name} · round {round_num}",
        input={"arguments": args},
        metadata={"tool_name": name, "round": round_num},
    ) as tool_obs:
        actions = []

        if name == "filter_dashboard":
            merged = {**current_filters, **args}
            result = filter_dashboard(df, **merged)
            # filter_dashboard produces a dashboard action that the frontend applies
            actions.append({"type": "filter_update", "filters": result["new_filters"]})

        elif name == "explain_metric":
            result = explain_metric(**args)

        elif name == "analyze_campaign":
            result = analyze_campaign(df, **args)

        elif name == "compare_campaigns":
            result = compare_campaigns(df, **args)

        elif name == "get_portfolio_summary":
            result = get_portfolio_summary(df, current_filters)

        elif name == "get_channel_analysis":
            result = get_channel_analysis(df, current_filters)

        else:
            result = {"error": f"Unknown tool: {name}"}

        # output= logs the full tool result — this is what gets fed back to the model
        # in the next round. Seeing it here lets you verify the tool returned sane data.
        tool_obs.update(
            output=result,
            metadata={
                "tool_name": name,
                "round": round_num,
                "produced_actions": len(actions),
            },
        )
    return result, actions


# ─── Main Agent Entry Point (AGENT type) ──────────────────────────────────────

def run_agent(
    user_message: str,
    current_filters: dict,
    history: list,
    df: pd.DataFrame,
    live_stats: dict,
    session_id: str | None = None,
    user_id: str | None = None,
) -> dict:
    """
    Full agent turn. Steps:
      1. Open an AGENT observation  → this is the root of the AI reasoning trace
      2. propagate_attributes()     → stamps session_id, user_id, tags, metadata onto
                                      every child observation (generation + tool) automatically
      3. Call OpenAI round 1        → model decides: respond directly OR call a tool
      4. If tool called: execute it, feed result back, call OpenAI round 2
      5. Close agent observation with the final answer as output
      6. langfuse.flush()           → ensures all buffered events are sent before returning
    """
    # Build tags from active filters — these appear as searchable labels in Langfuse,
    # making it easy to filter traces by campaign type, condition, or channel.
    tags = [ENVIRONMENT]
    if current_filters.get("campaign_type", "All") != "All":
        tags.append(current_filters["campaign_type"])
    if current_filters.get("condition", "All") != "All":
        tags.append(current_filters["condition"])
    if current_filters.get("channel", "All") != "All":
        tags.append(current_filters["channel"])

    # as_type="agent" unlocks the Agent Graph tab in Langfuse UI where you can
    # see the full reasoning flow as a visual node graph (agent → gen → tool → gen).
    # input= is shown at the top of the trace — log the user question + active filters
    # so the trace is self-contained and readable without needing external context.
    with langfuse.start_as_current_observation(
        as_type="agent",
        name="agent-run",
        input={
            "user_message": user_message,
            "active_filters": current_filters,
            "history_turns": len(history),
        },
    ) as agent_obs:

        # propagate_attributes() stamps these values onto the trace AND all child
        # observations (generations, tools) created inside this block.
        # This is what links multiple traces together into a "session" in Langfuse.
        attr_kwargs: dict = {
            "metadata": {
                "app_version": APP_VERSION,
                "environment": ENVIRONMENT,
                "active_filters": current_filters,
            },
            "tags": tags,
        }
        if session_id:
            # session_id groups all chat turns from one browser session together
            # in the Langfuse "Sessions" view — critical for multi-turn conversation analysis
            attr_kwargs["session_id"] = session_id
        if user_id:
            attr_kwargs["user_id"] = user_id

        with propagate_attributes(**attr_kwargs):
            # get_current_trace_id() returns the ID of the root trace created above.
            # We return this to the frontend so it can submit user feedback tied to this trace.
            trace_id = langfuse.get_current_trace_id()
            print(f"[Langfuse] trace_id={trace_id}  session_id={session_id}")

            try:
                # Build the message array: system prompt + conversation history + new user message
                context = build_context_message(current_filters, live_stats)
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT + "\n\n" + context},
                ]
                for turn in history[-20:]:
                    messages.append(turn)
                messages.append({"role": "user", "content": user_message})

                accumulated_actions: list = []
                tools_used: list[str] = []   # track which tools were called (for agent_obs metadata)
                max_tool_rounds = 3

                for round_num in range(1, max_tool_rounds + 1):
                    # ── OpenAI call (round N) ────────────────────────────────
                    # Each call is a child GENERATION observation named "openai-call · round N".
                    # The model either returns a plain text answer OR requests tool calls.
                    response = _call_openai(messages, round_num=round_num)
                    msg = response.choices[0].message

                    if not msg.tool_calls:
                        # ── Final answer: model responded with text ───────────
                        # Parse out any SUGGEST:[...] block the model appended
                        content, suggestions = _parse_suggestions(msg.content or "")

                        # Update the agent observation output with the final answer.
                        # This is what appears as the main "output" of the whole agent turn
                        # in the Langfuse trace detail view.
                        agent_obs.update(
                            output=content,
                            metadata={
                                "rounds_used": round_num,
                                "tools_used": tools_used,
                                "suggestions_count": len(suggestions),
                            },
                        )
                        # flush() sends all buffered Langfuse events immediately.
                        # Call this before returning so the trace is complete in Langfuse
                        # even if the server handles the next request before the async flush.
                        langfuse.flush()
                        return {
                            "message": content,
                            "actions": accumulated_actions,
                            "suggestions": suggestions,
                            "trace_id": trace_id,
                        }

                    # ── Tool call: model requested one or more tools ──────────
                    # Append the assistant's tool-call decision to the message history
                    # so the model remembers it called this tool in subsequent rounds.
                    messages.append({
                        "role": "assistant",
                        "content": msg.content,
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                            }
                            for tc in msg.tool_calls
                        ],
                    })

                    for tc in msg.tool_calls:
                        args = json.loads(tc.function.arguments)
                        tools_used.append(tc.function.name)

                        # Execute tool — each call is a child TOOL observation named
                        # "tool:<name> · round N", nested under the current agent-run.
                        result, actions = _execute_tool(
                            tc.function.name, args, df, current_filters, round_num=round_num
                        )
                        accumulated_actions.extend(actions)

                        # Append the tool result to message history so the model can
                        # read it in the next OpenAI call (round N+1).
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": json.dumps(result, default=str),
                        })

                # ── Safety exit: too many tool rounds ────────────────────────
                agent_obs.update(
                    output="Max tool rounds reached — returning partial result.",
                    metadata={"rounds_used": max_tool_rounds, "tools_used": tools_used},
                )
                langfuse.flush()
                return {
                    "message": "I've processed your request. The dashboard has been updated.",
                    "actions": accumulated_actions,
                    "suggestions": [],
                    "trace_id": trace_id,
                }

            except Exception as e:
                # level="ERROR" marks the entire agent-run observation red in Langfuse
                # so failed turns are immediately visible in the traces list.
                agent_obs.update(
                    output=f"Agent error: {type(e).__name__}: {e}",
                    level="ERROR",
                    metadata={"error_type": type(e).__name__},
                )
                langfuse.flush()
                raise


# ─── User Feedback Scoring ────────────────────────────────────────────────────

def score_trace(trace_id: str, value: int, comment: str = "") -> bool:
    """
    Attach a user-feedback score to a completed trace in Langfuse.

    Langfuse "scores" are numeric values linked to a trace_id. They appear in:
      - The trace detail view (score badge next to the trace)
      - The Scores dashboard (aggregate quality over time)
      - Evaluation exports (for fine-tuning datasets)

    value: 1 = thumbs up (positive), 0 = thumbs down (negative)
    The trace_id is returned by run_agent() and sent back to the frontend in the
    /api/chat response — the frontend calls /api/feedback to submit it here.
    """
    try:
        # create_score() links the numeric value to the specific trace so you can
        # filter traces in Langfuse by score and correlate quality with prompts/tools.
        langfuse.create_score(
            trace_id=trace_id,
            name="user-feedback",
            value=value,
            comment=comment,
        )
        # flush() ensures the score event is sent immediately, not held in the buffer
        langfuse.flush()
        return True
    except Exception as e:
        print(f"[Langfuse] Failed to record score: {e}")
        return False


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
