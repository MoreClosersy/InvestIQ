"""Shadow-mode critic run: 24 tickers on the twenty-result cassette.

The critic is enabled but does not route. This script records what it *would*
have flagged, so blocking is a decision backed by a violation rate rather than
by intuition.

用法（需要 OPENAI_API_KEY；约 24 次流程调用）：
    CRITIC_ENABLED=1 python docs/experiments/2026-11-02-critic-shadow/run_shadow.py
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from eval import fixtures  # noqa: E402
from graph.state import initial_state  # noqa: E402
from graph.workflow import build_graph  # noqa: E402

OUT = Path(__file__).resolve().parent
CASSETTE = ROOT / "eval" / "cassettes" / "news20_20261102.json"
INVALID = {"ZZZZZZ", "INVALID", "NOTAREAL", "12345"}


def main() -> int:
    cassette = fixtures.Cassette.load(CASSETTE)
    fixtures.install(cassette, "replay")
    graph = build_graph()

    rows = []
    for ticker in cassette.sources:
        started = time.time()
        result = graph.invoke(initial_state(ticker))
        report = result.get("critic_report") or {}
        rows.append({
            "ticker": ticker,
            "invalid": ticker in INVALID,
            "elapsed_s": round(time.time() - started, 2),
            "violations": report.get("violations", []),
            "grounding": report.get("grounding", {}),
            "metrics": report.get("metrics", {}),
            "strengths": len(result.get("strengths") or []),
            "report_chars": len(result.get("final_report") or ""),
        })
        print(f"[shadow] {ticker:8s} violations={len(rows[-1]['violations'])} "
              f"ungrounded={rows[-1]['grounding'].get('ungrounded', 0)}")

    valid = [r for r in rows if not r["invalid"]]
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "cassette": str(CASSETTE.relative_to(ROOT)),
        "tickers": len(rows),
        "valid_tickers": len(valid),
        "valid_with_violations": sum(1 for r in valid if r["violations"]),
        "valid_with_ungrounded": sum(1 for r in valid if r["grounding"].get("ungrounded")),
        "total_ungrounded": sum(r["grounding"].get("ungrounded", 0) for r in valid),
        "empty_sections": sum(
            1 for r in valid for v in r["violations"] if v["kind"] == "empty_section"
        ),
        "items_with_numbers_ratio": (
            sum(r["metrics"].get("items_with_numbers", 0) for r in valid)
            / max(1, sum(r["metrics"].get("items", 0) for r in valid))
        ),
        "rows": rows,
    }
    (OUT / "shadow.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "report.md").write_text(_render(summary), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"},
                     ensure_ascii=False, indent=2))
    return 0


def _render(s: dict) -> str:
    lines = [
        "# Critic 影子运行：24 个标的（20 条新闻）",
        "",
        f"- 生成时间：{s['generated_at']}",
        f"- 磁带：`{s['cassette']}`",
        "- 模式：`CRITIC_ENABLED=1`，只记录不拦截",
        "",
        "## 总览（20 个有效标的）",
        "",
        "| 指标 | 数值 |",
        "| --- | ---: |",
        f"| 至少一条违规的标的 | {s['valid_with_violations']}/{s['valid_tickers']} |",
        f"| 含未接地数字的标的 | {s['valid_with_ungrounded']}/{s['valid_tickers']} |",
        f"| 未接地数字总数 | {s['total_ungrounded']} |",
        f"| 空章节违规 | {s['empty_sections']} |",
        f"| 带数字的条目占比 | {s['items_with_numbers_ratio']:.1%} |",
        "",
        "## 逐标的",
        "",
        "| ticker | 违规 | 未接地 | 接地（财务/新闻） | 条目 |",
        "| --- | ---: | ---: | --- | ---: |",
    ]
    for r in s["rows"]:
        g = r["grounding"]
        lines.append(
            f"| {r['ticker']} | {len(r['violations'])} | {g.get('ungrounded', 0)} | "
            f"{g.get('financial', 0)}/{g.get('news_only', 0)} | "
            f"{r['metrics'].get('items', 0)} |"
        )
    lines += [
        "",
        "## 判读",
        "",
        "- 若含未接地数字的标的 ≤ 2/20：拦截的误伤面小，可以进入“拦截/补数”路由。",
        "- 若 3–6/20：先改分析提示词（要求逐条给数字的口径），再谈拦截。",
        "- 若 > 6/20：接地检查对分析文本过严（派生值、四舍五入都会中），"
        "critic 只适合做“报告发布前的 warning”，不适合硬拦截。",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
