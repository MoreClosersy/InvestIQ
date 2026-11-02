"""Critic 的确定性检查与开关（不联网、不调 LLM）。"""
from __future__ import annotations

from graph.state import initial_state
from graph.workflow import build_graph
from agents.critic import critic_node


def _state_with(text: str, **overrides):
    state = initial_state("AAPL")
    state["stock_price"] = {"current_price": 100.0, "change_pct": 1.5}
    state["financial_metrics"] = {"market_cap": 1_000_000_000, "pe_ratio": 20.0}
    state["historical_data"] = {"period_return_pct": 5.0}
    state["news_findings"] = [{"title": "news", "content": "Revenue was 42.0 million."}]
    state["strengths"] = [text]
    state["risks"] = ["Risk with 20.0 P/E."]
    state["analyst_summary"] = "Summary."
    state.update(overrides)
    return state


def test_flags_ungrounded_number() -> None:
    report = critic_node(_state_with("Growth of 987.65% unsupported."))["critic_report"]
    kinds = {v["kind"] for v in report["violations"]}
    assert "ungrounded_number" in kinds
    assert report["grounding"]["ungrounded"] >= 1


def test_accepts_financial_number() -> None:
    report = critic_node(_state_with("P/E of 20.0 looks fair."))["critic_report"]
    assert report["grounding"]["ungrounded"] == 0


def test_flags_empty_sections_when_financial_data_exists() -> None:
    state = _state_with("ok")
    state["strengths"] = []
    report = critic_node(state)["critic_report"]
    assert any(v["kind"] == "empty_section" for v in report["violations"])


def test_does_not_flag_empty_sections_on_refusal_path() -> None:
    state = initial_state("ZZZZZZ")
    report = critic_node(state)["critic_report"]
    assert not any(v["kind"] == "empty_section" for v in report["violations"])


def test_critic_is_off_by_default(monkeypatch) -> None:
    monkeypatch.delenv("CRITIC_ENABLED", raising=False)
    assert "critic" not in build_graph().get_graph().nodes


def test_critic_can_be_enabled(monkeypatch) -> None:
    monkeypatch.setenv("CRITIC_ENABLED", "1")
    g = build_graph().get_graph()
    assert "critic" in g.nodes
    edges = {(e.source, e.target) for e in g.edges}
    assert ("analysis", "critic") in edges and ("critic", "report") in edges
