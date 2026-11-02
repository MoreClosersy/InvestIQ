"""Tavily search tool wrapper for the research agent.

Note: `langchain_community.tools.tavily_search.TavilySearchResults` is marked
deprecated in favor of `langchain-tavily`'s `TavilySearch`. We keep
langchain-community here per project requirements; migration is a separate
task. The TAVILY_API_KEY env var is picked up automatically by the wrapper.

Timeout: the upstream ``TavilySearchAPIWrapper.raw_results`` calls
``requests.post`` without a timeout, so a stalled Tavily response can block the
workflow indefinitely. The subclass below adds a configurable per-request
timeout (``TAVILY_TIMEOUT_SECONDS``, default 15).
"""
from __future__ import annotations

import os
import warnings
from typing import Any

import requests

# Suppress the noisy "langchain-community is being sunset" deprecation on import.
with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    from langchain_community.tools.tavily_search import TavilySearchResults
    from langchain_community.utilities.tavily_search import (
        TAVILY_API_URL,
        TavilySearchAPIWrapper,
    )

DEFAULT_TIMEOUT_SECONDS = 15.0


class TimeoutTavilySearchAPIWrapper(TavilySearchAPIWrapper):
    """Tavily API wrapper that enforces a per-request timeout."""

    timeout: float = DEFAULT_TIMEOUT_SECONDS

    def raw_results(
        self,
        query: str,
        max_results: int | None = 5,
        search_depth: str | None = "advanced",
        include_domains: list[str] | None = None,
        exclude_domains: list[str] | None = None,
        include_answer: bool | None = False,
        include_raw_content: bool | None = False,
        include_images: bool | None = False,
    ) -> dict:
        """Same request as the upstream wrapper, plus ``timeout``."""
        params = {
            "api_key": self.tavily_api_key.get_secret_value(),
            "query": query,
            "max_results": max_results,
            "search_depth": search_depth,
            "include_domains": include_domains or [],
            "exclude_domains": exclude_domains or [],
            "include_answer": include_answer,
            "include_raw_content": include_raw_content,
            "include_images": include_images,
        }
        response = requests.post(
            f"{TAVILY_API_URL}/search",
            json=params,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()


def get_tavily_tool(max_results: int | None = None) -> TavilySearchResults:
    """Return a configured Tavily search tool instance.

    Result count defaults to ``TAVILY_MAX_RESULTS`` (20). Twenty results raised
    strict slot coverage from 2.45 to 3.60 and removed the three ungrounded
    numbers at 1.77x cost (docs/experiments/2026-11-02-max-results/); set the
    variable to 5 to trade coverage back for latency and cost.
    """
    if max_results is None:
        max_results = int(os.getenv("TAVILY_MAX_RESULTS", "20"))
    wrapper = TimeoutTavilySearchAPIWrapper(
        timeout=float(os.getenv("TAVILY_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS))
    )
    return TavilySearchResults(
        max_results=max_results,
        search_depth="advanced",
        api_wrapper=wrapper,
    )


def search_company_news(ticker: str, company_name: str = "") -> list[dict[str, Any]]:
    """Search recent news for a company. Returns cleaned Tavily results.

    Each item is a dict with keys: ``title``, ``url``, ``content``, ``score``.
    Returns an empty list on failure (errors are printed, not raised) so the
    LangGraph workflow can continue.
    """
    query = f"{company_name or ticker} stock news latest earnings"
    tool = get_tavily_tool()
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
