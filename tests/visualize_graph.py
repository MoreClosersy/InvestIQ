"""Generate Mermaid diagram of the InvestIQ workflow graph.

Run directly:
    python tests/visualize_graph.py

No API keys or network required — only builds the graph structure.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from graph.workflow import build_graph  # noqa: E402


def main() -> None:
    graph = build_graph()
    mermaid_code = graph.get_graph().draw_mermaid()
    print("Mermaid diagram of InvestIQ workflow:\n")
    print(mermaid_code)
    print("\nPaste this into https://mermaid.live to view the graph visually.")


if __name__ == "__main__":
    main()
