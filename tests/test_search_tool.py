"""Tavily 取数条数的配置契约（不联网）。"""
from __future__ import annotations

from tools import search_tool


def test_default_max_results_is_twenty(monkeypatch) -> None:
    monkeypatch.delenv("TAVILY_MAX_RESULTS", raising=False)
    monkeypatch.setenv("TAVILY_API_KEY", "test")
    assert search_tool.get_tavily_tool().max_results == 20


def test_max_results_env_override(monkeypatch) -> None:
    monkeypatch.setenv("TAVILY_MAX_RESULTS", "5")
    monkeypatch.setenv("TAVILY_API_KEY", "test")
    assert search_tool.get_tavily_tool().max_results == 5


def test_search_company_news_uses_the_factory(monkeypatch) -> None:
    called: list[bool] = []

    class FakeTool:
        def invoke(self, payload: dict) -> list:
            return []

    def fake_get_tavily_tool(max_results=None):
        called.append(True)
        return FakeTool()

    monkeypatch.setattr(search_tool, "get_tavily_tool", fake_get_tavily_tool)
    assert search_tool.search_company_news("AAPL", "") == []
    assert called == [True]
