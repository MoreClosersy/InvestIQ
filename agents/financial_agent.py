"""Financial agent: fetches stock price, metrics, and historical data via yfinance.

Pure data-fetch node — no LLM. Calls the three independent finance helpers,
collects any per-call errors into the shared ``errors`` channel, and
backfills ``company_name`` from the fundamentals call if the upstream
research input left it blank.
"""
from __future__ import annotations

from graph.state import AgentState
from tools.finance_tool import (
    get_financial_metrics,
    get_historical_prices,
    get_stock_snapshot,
)


def financial_node(state: AgentState) -> dict:
    """Fetch yfinance data for the ticker. Returns partial state update."""
    ticker = state["ticker"]

    snapshot = get_stock_snapshot(ticker)
    metrics = get_financial_metrics(ticker)
    historical = get_historical_prices(ticker, period="3mo")

    errors: list[str] = []
    if "error" in snapshot:
        errors.append(f"[Financial] snapshot failed: {snapshot['error']}")
    if "error" in metrics:
        errors.append(f"[Financial] metrics failed: {metrics['error']}")
    if "error" in historical:
        errors.append(f"[Financial] historical failed: {historical['error']}")

    # Backfill company_name from metrics if the upstream caller didn't set it.
    company_name_update: dict = {}
    if not state.get("company_name") and metrics.get("company_name"):
        company_name_update["company_name"] = metrics["company_name"]

    print(
        f"[Financial] {ticker}: price=${snapshot.get('current_price', 'N/A')}, "
        f"mcap={metrics.get('market_cap', 0):,}"
    )

    return {
        "stock_price": snapshot,
        "financial_metrics": metrics,
        "historical_data": historical,
        "errors": errors,
        **company_name_update,
    }
