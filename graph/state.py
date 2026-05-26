"""Shared LangGraph state threaded through every InvestIQ agent node.

Each agent returns a *partial* update; LangGraph merges it into the running
state using the reducer declared on each field. List fields use
`Annotated[list, add]` so multiple nodes can append without overwriting each
other.
"""
from operator import add
from typing import Annotated, Any, TypedDict


class AgentState(TypedDict):
    """Workflow state. All nodes read from and return partial updates to this."""

    # --- Input ---
    ticker: str
    company_name: str

    # --- Research Agent output ---
    news_findings: Annotated[list[dict], add]
    market_context: str

    # --- Financial Agent output (next step) ---
    stock_price: dict
    financial_metrics: dict
    historical_data: dict

    # --- Analysis Agent output (later) ---
    strengths: list[str]
    risks: list[str]
    analyst_summary: str
    data_completeness: str

    # --- Report Agent output (later) ---
    final_report: str

    # --- Control ---
    errors: Annotated[list[str], add]


def initial_state(ticker: str) -> AgentState:
    """Build a fully-populated initial state so nodes never hit KeyError."""
    return AgentState(
        ticker=ticker.upper(),
        company_name="",
        news_findings=[],
        market_context="",
        stock_price={},
        financial_metrics={},
        historical_data={},
        strengths=[],
        risks=[],
        analyst_summary="",
        data_completeness="",
        final_report="",
        errors=[],
    )
