"""Smoke tests: every module imports cleanly (no circular deps or syntax errors)."""
import importlib


MODULES = [
    "agents",
    "agents.research_agent",
    "agents.financial_agent",
    "agents.analysis_agent",
    "agents.report_agent",
    "tools",
    "tools.search_tool",
    "tools.finance_tool",
    "graph",
    "graph.state",
    "graph.workflow",
    "api",
    "api.main",
]


def test_imports() -> None:
    """Import every InvestIQ module to catch syntax errors and circular imports."""
    for name in MODULES:
        importlib.import_module(name)
