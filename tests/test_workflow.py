"""End-to-end integration test: runs the compiled LangGraph workflow.

Run directly:
    python tests/test_workflow.py

NOT a pytest test — needs OPENAI_API_KEY, TAVILY_API_KEY, and network.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(override=True)

from graph.state import initial_state  # noqa: E402
from graph.workflow import build_graph  # noqa: E402


def main() -> None:
    ticker = "AAPL"

    print("Building graph...")
    graph = build_graph()

    print(f"Running workflow for {ticker}...\n")
    start_time = time.time()

    initial = initial_state(ticker)
    final_state = graph.invoke(initial)

    elapsed = time.time() - start_time
    print(f"\nWorkflow completed in {elapsed:.2f}s\n")

    print("=" * 70)
    print(" FINAL REPORT")
    print("=" * 70)
    print(final_state.get("final_report", "(no report generated)"))

    if final_state.get("errors"):
        print("\n" + "=" * 70)
        print(" Errors encountered")
        print("=" * 70)
        for e in final_state["errors"]:
            print(f"  - {e}")

    print("\n" + "=" * 70)
    print(" State Summary")
    print("=" * 70)
    print(f"Ticker: {final_state.get('ticker')}")
    print(f"Company: {final_state.get('company_name')}")
    print(f"News items found: {len(final_state.get('news_findings', []))}")
    print(f"Stock price: ${final_state.get('stock_price', {}).get('current_price', 0)}")
    print(f"Strengths: {len(final_state.get('strengths', []))}")
    print(f"Risks: {len(final_state.get('risks', []))}")
    print(f"Data completeness: {final_state.get('data_completeness')}")
    print(f"Report length: {len(final_state.get('final_report', ''))} chars")


if __name__ == "__main__":
    main()
