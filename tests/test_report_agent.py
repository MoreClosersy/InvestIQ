"""End-to-end manual test: runs all 4 agents and prints the final Markdown report.

Run directly:
    python tests/test_report_agent.py

NOT a pytest test — needs OPENAI_API_KEY, TAVILY_API_KEY, and network.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(override=True)

from agents.analysis_agent import analysis_node  # noqa: E402
from agents.financial_agent import financial_node  # noqa: E402
from agents.report_agent import report_node  # noqa: E402
from agents.research_agent import research_node  # noqa: E402
from graph.state import initial_state  # noqa: E402


def main() -> None:
    state = initial_state("AAPL")

    state.update(research_node(state))
    state.update(financial_node(state))
    state.update(analysis_node(state))
    state.update(report_node(state))

    print("\n" + "=" * 70)
    print(" FINAL REPORT")
    print("=" * 70 + "\n")
    print(state["final_report"])

    if state.get("errors"):
        print("\n" + "=" * 70)
        print(" Errors encountered along the way")
        print("=" * 70)
        for e in state["errors"]:
            print(f"  - {e}")


if __name__ == "__main__":
    main()
