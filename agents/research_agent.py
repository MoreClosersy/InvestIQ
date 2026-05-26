"""Research agent: searches recent news and summarizes the market context.

Pipeline:
    1. Tavily web search for recent news on the ticker.
    2. LCEL chain (ChatPromptTemplate | ChatOpenAI | StrOutputParser) condenses
       the results into a short sentiment summary + key event list.

Returns a *partial* state update; LangGraph merges it with the running state
using the reducers defined on `AgentState`.
"""
from __future__ import annotations

from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from graph.state import AgentState
from tools.search_tool import search_company_news


_SYSTEM_PROMPT = (
    "You are a financial research analyst. Your job is to summarize recent "
    "news coverage for a publicly traded company.\n\n"
    "Strict rules:\n"
    "- Use ONLY the search results provided below. Do not invent facts, "
    "numbers, dates, or events that are not in the results.\n"
    "- If the results are thin or off-topic, say so explicitly rather than "
    "filling in from prior knowledge.\n"
    "- Be concise and neutral; avoid hype words and investment advice.\n\n"
    "Recency bias:\n"
    "- Prioritize the most recent developments. When items carry an explicit "
    "date or relative-time signal (e.g. 'yesterday', 'last week', 'Q3 2025'), "
    "weight those above items with no date signal.\n"
    "- If an older item has been superseded by a newer one in the results, "
    "use the newer item and ignore the stale one."
)

_HUMAN_PROMPT = (
    "Company: {company}\n"
    "Ticker: {ticker}\n\n"
    "Recent search results:\n"
    "{news_block}\n\n"
    "Produce the following, in plain text:\n"
    "1. Market sentiment (2-3 sentences) — the overall tone toward the "
    "company based on these results, emphasizing the most recent signals.\n"
    "2. Key events — up to 5 of the most material news items, deduplicated "
    "by topic (collapse multiple sources covering the same story into one "
    "bullet). Order from most recent to oldest where dates are available. "
    "End each bullet with the source domain in parentheses.\n"
)


def _format_news_for_prompt(items: list[dict[str, Any]]) -> str:
    """Render search results into a numbered block the LLM can read cleanly."""
    lines: list[str] = []
    for i, item in enumerate(items, start=1):
        title = item.get("title", "(no title)")
        url = item.get("url", "")
        content = item.get("content", "")
        lines.append(f"[{i}] {title}\n    URL: {url}\n    Snippet: {content}")
    return "\n\n".join(lines)


def _build_chain() -> Any:
    """Construct the LCEL summarization chain."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    prompt = ChatPromptTemplate.from_messages(
        [("system", _SYSTEM_PROMPT), ("human", _HUMAN_PROMPT)]
    )
    return prompt | llm | StrOutputParser()


def research_node(state: AgentState) -> dict:
    """Search news + summarize market context. Returns partial state update."""
    ticker = state["ticker"]
    company_name = state.get("company_name", "") or ticker

    raw_results = search_company_news(ticker, company_name)

    if not raw_results:
        print(f"[Research] No news found for {ticker}")
        return {
            "news_findings": [],
            "market_context": "No recent news available.",
            "errors": [f"[Research] No news found for {ticker}"],
        }

    chain = _build_chain()
    summary = chain.invoke(
        {
            "company": company_name,
            "ticker": ticker,
            "news_block": _format_news_for_prompt(raw_results),
        }
    )

    print(f"[Research] Found {len(raw_results)} news items for {ticker}")
    return {
        "news_findings": raw_results,
        "market_context": summary,
    }
