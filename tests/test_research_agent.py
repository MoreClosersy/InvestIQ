"""Manual integration test for the research agent. Requires real API keys.

Run directly:
    python tests/test_research_agent.py

NOT a pytest test — needs OPENAI_API_KEY and TAVILY_API_KEY in .env.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(override=True)

from agents.research_agent import research_node  # noqa: E402
from graph.state import initial_state  # noqa: E402


def main() -> None:
    state = initial_state("AAPL")
    state["company_name"] = "Apple Inc."

    result = research_node(state)

    print("\n=== Research Agent Result ===")
    print(f"News items: {len(result['news_findings'])}")
    for i, item in enumerate(result["news_findings"][:3], 1):
        print(f"\n{i}. {item.get('title', 'N/A')}")
        print(f"   {item.get('url', 'N/A')}")
    print(f"\n--- Market Context ---\n{result.get('market_context', '')}")
    if result.get("errors"):
        print(f"\n--- Errors ---\n{result['errors']}")


if __name__ == "__main__":
    main()
