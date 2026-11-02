"""用已保存的 20 条臂评测结果离线重打分 critic 的检查项。

为什么不用在线跑：critic 的检查是纯确定性的，只要输入（分析文本 + 新闻 +
财务字段）相同，结论就相同。直接复用 `eval/results/20261102_news20/` 里
今天生成的分析结果，既不花钱，也不受网络影响。

用法：
    python docs/experiments/2026-11-02-critic-shadow/score_existing.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from agents.critic import evaluate_analysis  # noqa: E402

OUT = Path(__file__).resolve().parent
RESULTS = ROOT / "eval" / "results" / "20261102_news20"
INVALID = {"ZZZZZZ", "INVALID", "NOTAREAL", "12345"}


def main() -> int:
    result_file = sorted(RESULTS.glob("results_*.json"))[-1]
    records = json.loads(result_file.read_text(encoding="utf-8"))["records"]

    rows = []
    for rec in records:
        analysis_text = "\n".join(
            [*(rec.get("strengths") or []), *(rec.get("risks") or []),
             rec.get("analyst_summary") or ""]
        )
        flat = dict(rec)
        flat["final_report"] = analysis_text
        has_financial = bool(
            (rec.get("current_price") or 0) > 0 or (rec.get("market_cap") or 0) > 0
        )
        report = evaluate_analysis(flat, has_financial=has_financial)
        rows.append({
            "ticker": rec["ticker"],
            "invalid": rec["ticker"] in INVALID,
            "violations": report["violations"],
            "grounding": report["grounding"],
            "metrics": report["metrics"],
            "report_chars": len(rec.get("final_report") or ""),
        })

    valid = [r for r in rows if not r["invalid"]]
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": str(result_file.relative_to(ROOT)),
        "tickers": len(rows),
        "valid_tickers": len(valid),
        "valid_with_violations": sum(1 for r in valid if r["violations"]),
        "valid_with_ungrounded": sum(1 for r in valid if r["grounding"]["ungrounded"]),
        "total_ungrounded": sum(r["grounding"]["ungrounded"] for r in valid),
        "empty_sections": sum(
            1 for r in valid for v in r["violations"] if v["kind"] == "empty_section"
        ),
        "items_with_numbers_ratio": (
            sum(r["metrics"]["items_with_numbers"] for r in valid)
            / max(1, sum(r["metrics"]["items"] for r in valid))
        ),
        "rows": rows,
    }
    (OUT / "shadow_offline.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "report.md").write_text(_render(summary), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"},
                     ensure_ascii=False, indent=2))
    return 0


def _render(s: dict) -> str:
    lines = [
        "# Critic 影子评分：复用 20 条臂的分析结果",
        "",
        f"- 生成时间：{s['generated_at']}",
        f"- 数据源：`{s['source']}`（离线，确定性检查，不调 LLM）",
        "- 模式：只看“如果拦截会拦到谁”，当前代码仍不路由",
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
            f"| {r['ticker']} | {len(r['violations'])} | {g['ungrounded']} | "
            f"{g['financial']}/{g['news_only']} | {r['metrics']['items']} |"
        )
    lines += [
        "",
        "## 判读",
        "",
        "- 含未接地数字的标的 ≤ 2/20：拦截误伤面小，可以进入“拦截/补数”路由。",
        "- 3–6/20：先改分析提示词（逐条注明数字口径），再谈拦截。",
        "- > 6/20：接地检查对分析文本过严（派生值、四舍五入都会中），"
        "critic 只适合做发布前 warning，不适合硬拦截。",
        "",
        "> 分析文本里的派生值（P/E = 价格/EPS、增长率）本来就不会字面出现在输入里，"
        "会被判为未接地；这是检查器已知的边界，不是分析编造。",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
