"""Analysis agent: synthesizes research + financial data into structured insights.

Uses ``ChatOpenAI.with_structured_output(AnalysisOutput)`` so the LLM is
constrained to return validated fields directly — no free-text parsing.

Defensive: if both upstream agents produced nothing useful, the node skips
the LLM call entirely and emits an explicit error onto the shared channel.
"""
from __future__ import annotations

from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from graph.state import AgentState


class AnalysisOutput(BaseModel):
    """Structured output schema for the analysis agent."""

    strengths: list[str] = Field(
        description=(
            "3-5 specific investment strengths, each citing concrete data "
            "(e.g., 'Strong 3-month return of +16% outpacing sector peers') "
            "or news events. No vague claims."
        )
    )
    risks: list[str] = Field(
        description=(
            "3-5 specific risk factors, each citing concrete data or news "
            "events (e.g., 'High P/E ratio of 37 suggests valuation risk if "
            "growth slows'). No generic warnings."
        )
    )
    analyst_summary: str = Field(
        description=(
            "2-3 paragraph synthesis combining the strengths and risks into a "
            "coherent narrative. Reference specific numbers from the "
            "financial data and specific events from the news context. "
            "Keep under 400 words."
        )
    )
    data_completeness: str = Field(
        description=(
            "Brief note (1 sentence) on which data points are missing or "
            "incomplete, e.g., 'Forward P/E and beta unavailable.' If all "
            "data is present, return 'All key data points available.'"
        )
    )


_SYSTEM_PROMPT = """You are a senior equity research analyst. You analyze a company by synthesizing recent news and quantitative financial data into a structured assessment.

Strict rules:
- Use ONLY the data provided in the user message. Do not introduce external knowledge.
- Every strength and risk MUST cite a specific number or news event. No vague claims like "strong fundamentals" or "market uncertainty".
- When financial fields show 0 or empty values (e.g., forward_pe=0, beta=0), treat them as "data not available" — do not interpret 0 as a real value.
- Use the company's reporting currency (provided in stock_price.currency) when referencing dollar amounts.
- Do NOT provide buy/sell/hold recommendations. Only describe facts and analysis.
- Be balanced: strengths and risks should both be substantive, not one paragraph and one sentence."""


_HUMAN_PROMPT = """Analyze the following company.

Ticker: {ticker}
Company: {company_name}

=== RECENT NEWS CONTEXT ===
{market_context}

=== KEY NEWS ITEMS ===
{news_items_formatted}

=== FINANCIAL SNAPSHOT ===
Current price: {current_price} {currency}
Day change: {change_pct}%
Volume: {volume}

=== FUNDAMENTALS ===
Sector: {sector}
Industry: {industry}
Market cap: {market_cap}
P/E ratio (trailing): {pe_ratio}
Forward P/E: {forward_pe}
EPS: {eps}
Dividend yield: {dividend_yield} (decimal, e.g., 0.005 = 0.5%)
52-week range: {week_52_low} - {week_52_high}
Beta: {beta}

=== 3-MONTH PRICE HISTORY ===
Period return: {period_return_pct}%
Period high/low: {period_high} / {period_low}
Annualized volatility: {volatility} (decimal, e.g., 0.25 = 25% annualized vol)
Data points: {data_points} trading days

Produce strengths, risks, and a synthesized analyst summary per the schema."""


def _format_market_cap(value: int) -> str:
    """Render market cap as ``$X.XXT``/``$X.XXB``/``$X.XXM``; ``N/A`` if missing."""
    if not value:
        return "N/A"
    if value >= 1_000_000_000_000:
        return f"${value / 1_000_000_000_000:.2f}T"
    if value >= 1_000_000_000:
        return f"${value / 1_000_000_000:.2f}B"
    if value >= 1_000_000:
        return f"${value / 1_000_000:.2f}M"
    return f"${value:,}"


def _format_volume(value: int) -> str:
    """Render share volume with thousands separators; ``N/A`` if missing."""
    if not value:
        return "N/A"
    return f"{value:,}"


def _format_news_items(news_findings: list[dict]) -> str:
    """Format news list into a compact text block for the prompt."""
    if not news_findings:
        return "(no news items available)"
    lines: list[str] = []
    for i, item in enumerate(news_findings[:5], start=1):
        title = item.get("title", "(no title)")
        content = (item.get("content", "") or "")[:300]  # truncate to avoid prompt bloat
        lines.append(f"{i}. {title}\n   {content}")
    return "\n\n".join(lines)


def _build_chain() -> Any:
    """Build the structured-output LCEL chain."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    structured_llm = llm.with_structured_output(AnalysisOutput)
    prompt = ChatPromptTemplate.from_messages(
        [("system", _SYSTEM_PROMPT), ("human", _HUMAN_PROMPT)]
    )
    return prompt | structured_llm


def analysis_node(state: AgentState) -> dict:
    """Synthesize research + financial data into structured analysis."""
    snapshot = state.get("stock_price", {})
    metrics = state.get("financial_metrics", {})
    historical = state.get("historical_data", {})

    # Defensive: if both upstream agents produced nothing useful, skip the LLM.
    has_news = bool(state.get("news_findings"))
    has_financial = bool(metrics.get("company_name")) or snapshot.get("current_price", 0) > 0

    if not has_news and not has_financial:
        print("[Analysis] Insufficient data, skipping LLM call")
        return {
            "strengths": [],
            "risks": [],
            "analyst_summary": "Insufficient data to produce analysis.",
            "data_completeness": "Both research and financial data missing.",
            "errors": ["[Analysis] Both research and financial data missing"],
        }

    chain = _build_chain()

    try:
        result: AnalysisOutput = chain.invoke(
            {
                "ticker": state["ticker"],
                "company_name": state.get("company_name", "N/A"),
                "market_context": state.get("market_context", "(no market context)"),
                "news_items_formatted": _format_news_items(state.get("news_findings", [])),
                "current_price": snapshot.get("current_price", 0),
                "currency": snapshot.get("currency", "USD"),
                "change_pct": snapshot.get("change_pct", 0),
                "volume": _format_volume(snapshot.get("volume", 0)),
                "sector": metrics.get("sector", "N/A"),
                "industry": metrics.get("industry", "N/A"),
                "market_cap": _format_market_cap(metrics.get("market_cap", 0)),
                "pe_ratio": metrics.get("pe_ratio", 0),
                "forward_pe": metrics.get("forward_pe", 0),
                "eps": metrics.get("eps", 0),
                "dividend_yield": metrics.get("dividend_yield", 0),
                "week_52_low": metrics.get("52_week_low", 0),
                "week_52_high": metrics.get("52_week_high", 0),
                "beta": metrics.get("beta", 0),
                "period_return_pct": historical.get("period_return_pct", 0),
                "period_high": historical.get("period_high", 0),
                "period_low": historical.get("period_low", 0),
                "volatility": historical.get("volatility", 0),
                "data_points": historical.get("data_points", 0),
            }
        )
    except Exception as exc:
        print(f"[Analysis] LLM call failed: {exc}")
        return {
            "strengths": [],
            "risks": [],
            "analyst_summary": "Analysis failed due to LLM error.",
            "data_completeness": "Analysis could not be produced.",
            "errors": [f"[Analysis] {exc!r}"],
        }

    print(
        f"[Analysis] {state['ticker']}: "
        f"{len(result.strengths)} strengths, {len(result.risks)} risks"
    )

    return {
        "strengths": result.strengths,
        "risks": result.risks,
        "analyst_summary": result.analyst_summary,
        "data_completeness": result.data_completeness,
    }
