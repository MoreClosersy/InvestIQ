"""Report agent: synthesizes upstream outputs into a Markdown research report.

Free-form Markdown output (StrOutputParser, no structured schema). Temperature
bumped to 0.5 vs. the analysis node's 0.3 — writing tasks benefit from a bit
more variety, fact extraction does not.

Note: ``_format_market_cap`` and ``_format_volume`` are intentionally
duplicated from ``agents/analysis_agent.py``. Refactor to a shared
``tools/formatting.py`` after all agents are in and the workflow is wired.
"""
from __future__ import annotations

from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from graph.state import AgentState


_SYSTEM_PROMPT = """You are a senior equity research analyst writing a polished investment research report. Your job is to synthesize provided data into a well-structured Markdown report.

Strict rules:
- Use ONLY the data provided. Do not introduce external knowledge or speculation.
- Output VALID Markdown with the exact section structure specified.
- Maintain a professional, neutral analyst tone. Avoid hype words and marketing language.
- Do NOT provide buy/sell/hold recommendations. Describe facts and analysis only.
- When citing numbers, use the formatted versions provided (e.g., "$3.50T" not "3500000000000").
- The Disclaimer section MUST include the data_completeness note provided.
- Keep total report length under 800 words."""


_HUMAN_PROMPT = """Generate a Markdown investment research report for the following company.

# INPUT DATA

## Company
- Ticker: {ticker}
- Name: {company_name}
- Sector: {sector}
- Industry: {industry}

## Current Market Snapshot
- Price: {current_price}
- Day change: {change_pct}
- Volume: {volume}
- 52-week range: {week_52_low} - {week_52_high}

## Key Fundamentals
- Market cap: {market_cap}
- P/E ratio (trailing): {pe_ratio}
- Forward P/E: {forward_pe}
- EPS: {eps}
- Dividend yield: {dividend_yield_pct}
- Beta: {beta}

## 3-Month Performance
- Period return: {period_return_pct}
- Period high/low: {period_high} / {period_low}
- Annualized volatility: {volatility_pct}

## Recent Market Context (from news research)
{market_context}

## Analyst Strengths
{strengths_formatted}

## Analyst Risks
{risks_formatted}

## Analyst Summary
{analyst_summary}

## Data Completeness Note
{data_completeness}

# REQUIRED OUTPUT STRUCTURE

Generate Markdown with EXACTLY these sections, in this order (use ## for section headers):

## Executive Summary
A single paragraph (3-4 sentences) capturing the most important takeaways. Reference 2-3 specific data points.

## Key Metrics
A Markdown table with the most important metrics. Use this exact format:
| Metric | Value |
| --- | --- |
| Current Price | ... |
| Market Cap | ... |
| P/E Ratio | ... |
| 3-Month Return | ... |
| 52-Week Range | ... |

Pick the 5-7 most informative rows. If a metric value is exactly "N/A", OMIT that row entirely. Do not write rows with N/A values.

## Recent Developments
2-3 sentences summarizing the news context, with the most material recent events highlighted.

## Investment Strengths
Bullet list. Each bullet is one of the analyst strengths, optionally lightly rephrased for flow. Keep specific numbers intact.

## Key Risks
Bullet list. Each bullet is one of the analyst risks. Keep specific numbers intact.

## Synthesis
A 2-paragraph narrative integrating the strengths, risks, and current data. This is the "long form" prose section. Do not give buy/sell guidance.

## Disclaimer
A single paragraph that includes:
- A statement that this report is for informational purposes only and not investment advice
- A note that the analysis is based on publicly available data as of the report generation date
- The Disclaimer section MUST include this exact sentence verbatim: "Data completeness: {data_completeness}"

Generate the report now."""


def _format_bullets(items: list[str]) -> str:
    """Format a list of strings as Markdown bullets."""
    if not items:
        return "- (no items)"
    return "\n".join(f"- {item}" for item in items)


def _format_currency_amount(value: float, currency: str = "USD") -> str:
    """Format a price as e.g., ``$308.82 USD``; ``N/A`` if missing."""
    if value == 0:
        return "N/A"
    return f"${value:,.2f} {currency}"


def _format_pct(value: float) -> str:
    """Format a value already on a 0-100 scale (e.g. ``+1.50%``)."""
    if value == 0:
        return "N/A"
    sign = "+" if value > 0 else ""
    return f"{sign}{value:.2f}%"


def _format_decimal_as_pct(value: float) -> str:
    """Format a decimal value (e.g. 0.005) as a percentage string (``0.50%``)."""
    if value == 0:
        return "N/A"
    return f"{value * 100:.2f}%"


def _format_ratio(value: float, decimals: int = 2) -> str:
    """Format a numeric ratio. Returns ``N/A`` if zero (treated as data missing)."""
    if value == 0:
        return "N/A"
    return f"{value:.{decimals}f}"


def _format_market_cap(value: int) -> str:
    """Format market cap as ``$X.XXT``/``$X.XXB``/``$X.XXM``; ``N/A`` if missing."""
    if value == 0:
        return "N/A"
    if value >= 1_000_000_000_000:
        return f"${value / 1_000_000_000_000:.2f}T"
    if value >= 1_000_000_000:
        return f"${value / 1_000_000_000:.2f}B"
    if value >= 1_000_000:
        return f"${value / 1_000_000:.2f}M"
    return f"${value:,}"


def _format_volume(value: int) -> str:
    """Format volume with thousands separator; ``N/A`` if missing."""
    if value == 0:
        return "N/A"
    return f"{value:,}"


def _build_chain() -> Any:
    """Build the report generation LCEL chain (free-form Markdown output)."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)
    prompt = ChatPromptTemplate.from_messages(
        [("system", _SYSTEM_PROMPT), ("human", _HUMAN_PROMPT)]
    )
    return prompt | llm | StrOutputParser()


def report_node(state: AgentState) -> dict:
    """Generate the final Markdown investment report."""
    snapshot = state.get("stock_price", {})
    metrics = state.get("financial_metrics", {})
    historical = state.get("historical_data", {})

    # Defensive: skip the LLM if analysis produced nothing usable.
    if (
        not state.get("strengths")
        and not state.get("risks")
        and not state.get("analyst_summary")
    ):
        print("[Report] No analysis available, generating minimal report")
        return {
            "final_report": (
                f"# Investment Research Report: {state.get('ticker', 'N/A')}\n\n"
                f"**Status:** Insufficient data to generate full report.\n\n"
                f"**Errors:** {'; '.join(state.get('errors', []))}\n"
            ),
        }

    currency = snapshot.get("currency", "USD")
    chain = _build_chain()

    try:
        report_md = chain.invoke(
            {
                "ticker": state["ticker"],
                "company_name": state.get("company_name", "N/A"),
                "sector": metrics.get("sector", "N/A"),
                "industry": metrics.get("industry", "N/A"),
                "current_price": _format_currency_amount(snapshot.get("current_price", 0), currency),
                "change_pct": _format_pct(snapshot.get("change_pct", 0)),
                "volume": _format_volume(snapshot.get("volume", 0)),
                "week_52_low": _format_currency_amount(metrics.get("52_week_low", 0), currency),
                "week_52_high": _format_currency_amount(metrics.get("52_week_high", 0), currency),
                "market_cap": _format_market_cap(metrics.get("market_cap", 0)),
                "pe_ratio": _format_ratio(metrics.get("pe_ratio", 0)),
                "forward_pe": _format_ratio(metrics.get("forward_pe", 0)),
                "eps": _format_ratio(metrics.get("eps", 0)),
                "dividend_yield_pct": _format_decimal_as_pct(metrics.get("dividend_yield", 0)),
                "beta": _format_ratio(metrics.get("beta", 0)),
                "period_return_pct": _format_pct(historical.get("period_return_pct", 0)),
                "period_high": _format_currency_amount(historical.get("period_high", 0), currency),
                "period_low": _format_currency_amount(historical.get("period_low", 0), currency),
                "volatility_pct": _format_decimal_as_pct(historical.get("volatility", 0)),
                "market_context": state.get("market_context", "(no market context available)"),
                "strengths_formatted": _format_bullets(state.get("strengths", [])),
                "risks_formatted": _format_bullets(state.get("risks", [])),
                "analyst_summary": state.get("analyst_summary", "(no summary available)"),
                "data_completeness": state.get("data_completeness", "Data completeness status not recorded."),
            }
        )
    except Exception as exc:
        print(f"[Report] LLM call failed: {exc}")
        return {
            "final_report": f"# Report generation failed\n\nError: {exc!r}\n",
            "errors": [f"[Report] {exc!r}"],
        }

    print(f"[Report] Generated report for {state['ticker']}: {len(report_md)} chars")
    return {"final_report": report_md}
