"""Critic node: deterministic checks on the analysis output.

Runs after ``analysis`` and before ``report`` when ``CRITIC_ENABLED=1``. This
first version is **shadow mode**: it records violations in state but does not
change routing, so the violation rate can be measured on the frozen set before
anything is blocked. Blocking without that measurement would trade a known
failure mode for an unknown one.

Checks, all deterministic (no LLM):

1. **Number grounding** — every number in strengths/risks/analyst_summary must
   trace back to a yfinance field or a raw news snippet. Reuses
   ``tools.grounding``, the same checker the offline eval uses, so runtime and
   eval cannot drift apart.
2. **Structural completeness** — empty strengths/risks/analysis on a ticker
   that did have financial data is a violation; on a ticker without financial
   data it is the expected refusal path.
"""
from __future__ import annotations

from typing import Any

from graph.state import AgentState
from tools.grounding import check_grounding

MAX_VIOLATIONS = 10


def _analysis_text(state: AgentState) -> str:
    parts: list[str] = []
    parts.extend(state.get("strengths") or [])
    parts.extend(state.get("risks") or [])
    summary = state.get("analyst_summary") or ""
    if summary:
        parts.append(summary)
    return "\n".join(parts)


def _grounding_record(state: AgentState) -> dict[str, Any]:
    """Flatten the nested state into the flat record ``check_grounding`` expects.

    Field names differ between the agents and the eval records (``52_week_high``
    vs ``week_52_high``); the mapping is explicit rather than a blind merge so a
    typo fails loudly here instead of silently shrinking the evidence pool.
    """
    price = state.get("stock_price") or {}
    metrics = state.get("financial_metrics") or {}
    history = state.get("historical_data") or {}
    return {
        "final_report": _analysis_text(state),
        "strengths": state.get("strengths") or [],
        "risks": state.get("risks") or [],
        "analyst_summary": state.get("analyst_summary") or "",
        "news_findings": state.get("news_findings") or [],
        "current_price": price.get("current_price", 0),
        "previous_close": price.get("previous_close", 0),
        "change_pct": price.get("change_pct", 0),
        "day_high": price.get("day_high", 0),
        "day_low": price.get("day_low", 0),
        "volume": price.get("volume", 0),
        "market_cap": metrics.get("market_cap", 0),
        "pe_ratio": metrics.get("pe_ratio", 0),
        "forward_pe": metrics.get("forward_pe", 0),
        "eps": metrics.get("eps", 0),
        "dividend_yield": metrics.get("dividend_yield", 0),
        "week_52_high": metrics.get("52_week_high", 0),
        "week_52_low": metrics.get("52_week_low", 0),
        "beta": metrics.get("beta", 0),
        "period_return_pct": history.get("period_return_pct", 0),
        "period_high": history.get("period_high", 0),
        "period_low": history.get("period_low", 0),
        "volatility": history.get("volatility", 0),
        "data_points": history.get("data_points", 0),
    }


def _has_financial_data(state: AgentState) -> bool:
    price = state.get("stock_price") or {}
    metrics = state.get("financial_metrics") or {}
    return bool(
        (price.get("current_price") or 0) > 0
        or (metrics.get("market_cap") or 0) > 0
    )


def evaluate_analysis(record: dict, *, has_financial: bool) -> dict:
    """Run the checks against a flat record.

    Split from ``critic_node`` so saved eval records can be re-scored offline:
    the same function then judges live state and recorded runs, which keeps the
    shadow numbers and the runtime behaviour from drifting apart.
    """
    violations: list[dict[str, Any]] = []

    grounding = check_grounding(record)
    for item in grounding["ungrounded_items"][:MAX_VIOLATIONS]:
        violations.append({
            "kind": "ungrounded_number",
            "detail": item["written"],
            "context": item.get("context", ""),
        })

    strengths = record.get("strengths") or []
    risks = record.get("risks") or []
    summary = record.get("analyst_summary") or ""

    if has_financial:
        for name, value in (("strengths", strengths), ("risks", risks),
                            ("analyst_summary", summary)):
            if not value:
                violations.append({"kind": "empty_section", "detail": name})

    items = [*strengths, *risks]
    with_numbers = sum(1 for item in items if any(ch.isdigit() for ch in item))
    return {
        "checked": True,
        "violations": violations[:MAX_VIOLATIONS],
        "grounding": {k: grounding[k] for k in
                      ("total", "financial", "news_only", "ungrounded")},
        "metrics": {
            "strengths": len(strengths),
            "risks": len(risks),
            "items": len(items),
            "items_with_numbers": with_numbers,
        },
    }


def critic_node(state: AgentState) -> dict:
    """Record grounding/shape violations for the current analysis. Shadow only."""
    report = evaluate_analysis(
        _grounding_record(state), has_financial=_has_financial_data(state)
    )
    if report["violations"]:
        print(f"[Critic] {state['ticker']}: {len(report['violations'])} violation(s) "
              f"(shadow mode, not blocking)")
    return {"critic_report": report}
