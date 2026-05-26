"""Manual integration test for financial agent. Uses real yfinance API.

Run directly:
    python tests/test_financial_agent.py

NOT a pytest test — needs network access to Yahoo Finance.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(override=True)

from agents.financial_agent import financial_node  # noqa: E402
from graph.state import initial_state  # noqa: E402


def main() -> None:
    state = initial_state("AAPL")
    result = financial_node(state)

    print("\n=== Financial Agent Result ===")
    print(f"\nCompany: {result.get('company_name', 'N/A')}")

    print("\n--- Stock Price ---")
    for k, v in result["stock_price"].items():
        print(f"  {k}: {v}")

    print("\n--- Financial Metrics ---")
    for k, v in result["financial_metrics"].items():
        print(f"  {k}: {v}")

    print("\n--- Historical (3mo) ---")
    for k, v in result["historical_data"].items():
        print(f"  {k}: {v}")

    if result.get("errors"):
        print("\n--- Errors ---")
        for e in result["errors"]:
            print(f"  {e}")


if __name__ == "__main__":
    main()
