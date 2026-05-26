"""LangGraph workflow orchestration.

Builds the InvestIQ multi-agent graph:
- Research and Financial agents run in parallel from START.
- Analysis waits for both to complete (fan-in).
- Report generates final output sequentially after Analysis.

Parallel-write safety: ``research_node`` and ``financial_node`` both append
to ``errors`` (which has an ``add`` reducer in AgentState). They otherwise
touch disjoint single-value fields, so no merge conflicts occur.
"""
from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from agents.analysis_agent import analysis_node
from agents.financial_agent import financial_node
from agents.report_agent import report_node
from agents.research_agent import research_node
from graph.state import AgentState


def build_graph() -> Any:
    """Construct and compile the InvestIQ workflow graph."""
    workflow = StateGraph(AgentState)

    workflow.add_node("research", research_node)
    workflow.add_node("financial", financial_node)
    workflow.add_node("analysis", analysis_node)
    workflow.add_node("report", report_node)

    # Fan-out from START: research and financial run in parallel.
    workflow.add_edge(START, "research")
    workflow.add_edge(START, "financial")

    # Fan-in to analysis: both upstream nodes must complete first.
    workflow.add_edge("research", "analysis")
    workflow.add_edge("financial", "analysis")

    # Sequential tail: analysis -> report -> END.
    workflow.add_edge("analysis", "report")
    workflow.add_edge("report", END)

    return workflow.compile()


def run_research(ticker: str) -> dict:
    """Convenience function: build graph, invoke with ticker, return final state."""
    from graph.state import initial_state

    graph = build_graph()
    initial = initial_state(ticker)
    return graph.invoke(initial)
