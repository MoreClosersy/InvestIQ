"""Tavily search tool wrapper for the research agent.

Note: `langchain_community.tools.tavily_search.TavilySearchResults` is marked
deprecated in favor of `langchain-tavily`'s `TavilySearch`. We keep
langchain-community here per project requirements; migration is a separate
task. The TAVILY_API_KEY env var is picked up automatically by the wrapper.
"""
from __future__ import annotations

import warnings
from typing import Any

# Suppress the noisy "langchain-community is being sunset" deprecation on import.
with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    from langchain_community.tools.tavily_search import TavilySearchResults


def get_tavily_tool(max_results: int = 5) -> TavilySearchResults:
    """Return a configured Tavily search tool instance."""
    return TavilySearchResults(max_results=max_results, search_depth="advanced")


def search_company_news(ticker: str, company_name: str = "") -> list[dict[str, Any]]:
    """Search recent news for a company. Returns cleaned Tavily results.

    Each item is a dict with keys: ``title``, ``url``, ``content``, ``score``.
    Returns an empty list on failure (errors are printed, not raised) so the
    LangGraph workflow can continue.
    """
    query = f"{company_name or ticker} stock news latest earnings"
    tool = get_tavily_tool(max_results=5)
    try:
        results = tool.invoke({"query": query})
    except Exception as exc:
        print(f"[search_company_news] Tavily search failed for {ticker}: {exc!r}")
        return []

    # `TavilySearchResults` uses response_format="content_and_artifact", which
    # makes BaseTool.invoke return only the content (already a list[dict]).
    if not isinstance(results, list):
        print(f"[search_company_news] Unexpected Tavily response type: {type(results)!r}")
        return []
    return results
