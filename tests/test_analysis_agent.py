"""Manual integration test for analysis agent.

Runs research + financial nodes first to populate state with real data, then
feeds the result to analysis. End-to-end exercise of the LLM-heavy path.

Run directly:
    python tests/test_analysis_agent.py

NOT a pytest test — needs OPENAI_API_KEY, TAVILY_API_KEY, and network.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(override=True)

from agents.analysis_agent import analysis_node  # noqa: E402
from agents.financial_agent import financial_node  # noqa: E402
from agents.research_agent import research_node  # noqa: E402
from graph.state import initial_state  # noqa: E402


def main() -> None:
    state = initial_state("AAPL")

    state.update(research_node(state))
    state.update(financial_node(state))

    result = analysis_node(state)

    print("\n" + "=" * 60)
    print(" Analysis Agent Result")
    print("=" * 60)

    print("\n--- Strengths ---")
    for i, s in enumerate(result["strengths"], 1):
        print(f"{i}. {s}")

    print("\n--- Risks ---")
    for i, r in enumerate(result["risks"], 1):
        print(f"{i}. {r}")

    print("\n--- Analyst Summary ---")
    print(result["analyst_summary"])

    if result.get("errors"):
        print("\n--- Errors ---")
        for e in result["errors"]:
            print(f"  {e}")


if __name__ == "__main__":
    main()
