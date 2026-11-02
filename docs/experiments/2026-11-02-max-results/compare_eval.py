"""对比两臂评测结果：news5（5 条）vs news20（20 条）。

用法：
    python docs/experiments/2026-11-02-max-results/compare_eval.py \
        eval/results/20261102_news5 eval/results/20261102_news20
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def latest_result(directory: str) -> dict:
    files = sorted(Path(directory).glob("results_*.json"))
    if not files:
        raise SystemExit(f"没有结果文件: {directory}")
    return json.loads(files[-1].read_text(encoding="utf-8"))


def arm_stats(d: dict) -> dict:
    s = d["summary"]
    records = d["records"]
    cost = sum(r.get("cost_usd") or 0 for r in records)
    return {
        "generated_at": s.get("generated_at"),
        "n": s.get("total"),
        "outcomes": s.get("outcomes", {}),
        "grounding": s.get("grounding", {}).get("totals", {}),
        "latency": s.get("latency", {}),
        "tokens": s.get("tokens", {}),
        "cost_total": cost,
        "cost_avg": cost / len(records) if records else 0,
        "report_chars": sum(r.get("report_chars") or 0 for r in records) / len(records),
        "news_items": sum(len(r.get("news_findings") or []) for r in records) / len(records),
        "records": {r["ticker"]: r for r in records},
    }


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        raise SystemExit("用法: compare_eval.py <news5_dir> <news20_dir>")
    a = arm_stats(latest_result(argv[1]))
    b = arm_stats(latest_result(argv[2]))

    lines = [
        "# 报告层对比：5 条 vs 20 条新闻",
        "",
        f"- 5 条臂：`{argv[1]}`（{a['generated_at']}）",
        f"- 20 条臂：`{argv[2]}`（{b['generated_at']}）",
        "- 两臂同一批标的、同一份财务输入、同一天、同一模型；只差新闻条数",
        "",
        "## 结果与质量",
        "",
        "| 指标 | 5 条 | 20 条 |",
        "| --- | ---: | ---: |",
        f"| 成功 / 正确拒答 / 幻觉 / 降级 / 失败 / 超时 | "
        f"{_fmt_outcomes(a['outcomes'])} | {_fmt_outcomes(b['outcomes'])} |",
        f"| 报告数字合计 | {a['grounding'].get('total', 0)} | {b['grounding'].get('total', 0)} |",
        f"| 有财务出处 | {a['grounding'].get('financial', 0)} | {b['grounding'].get('financial', 0)} |",
        f"| 仅新闻出处 | {a['grounding'].get('news_only', 0)} | {b['grounding'].get('news_only', 0)} |",
        f"| **未接地** | **{a['grounding'].get('ungrounded', 0)}** | "
        f"**{b['grounding'].get('ungrounded', 0)}** |",
        f"| 报告平均字符 | {a['report_chars']:.0f} | {b['report_chars']:.0f} |",
        "",
        "## 成本与延迟",
        "",
        "| 指标 | 5 条 | 20 条 | 倍数 |",
        "| --- | ---: | ---: | ---: |",
        f"| 平均 prompt tokens | {a['tokens'].get('avg_prompt', 0):.0f} | "
        f"{b['tokens'].get('avg_prompt', 0):.0f} | {_ratio(a['tokens'].get('avg_prompt', 0), b['tokens'].get('avg_prompt', 0))} |",
        f"| 平均 completion tokens | {a['tokens'].get('avg_completion', 0):.0f} | "
        f"{b['tokens'].get('avg_completion', 0):.0f} | {_ratio(a['tokens'].get('avg_completion', 0), b['tokens'].get('avg_completion', 0))} |",
        f"| 平均单次成本（$） | {a['cost_avg']:.5f} | {b['cost_avg']:.5f} | {_ratio(a['cost_avg'], b['cost_avg'])} |",
        f"| p50（s） | {a['latency'].get('p50', 0):.2f} | {b['latency'].get('p50', 0):.2f} | {_ratio(a['latency'].get('p50', 0), b['latency'].get('p50', 0))} |",
        f"| p95（s） | {a['latency'].get('p95', 0):.2f} | {b['latency'].get('p95', 0):.2f} | {_ratio(a['latency'].get('p95', 0), b['latency'].get('p95', 0))} |",
        "",
        "## 逐标的（未接地数字 / 成本 / 耗时）",
        "",
        "| ticker | 未接地 5→20 | 成本 5→20（$） | 耗时 5→20（s） |",
        "| --- | --- | --- | --- |",
    ]
    for ticker in a["records"]:
        ra, rb = a["records"][ticker], b["records"].get(ticker, {})
        ga = (ra.get("grounding") or {}).get("ungrounded", 0)
        gb = (rb.get("grounding") or {}).get("ungrounded", 0)
        lines.append(
            f"| {ticker} | {ga} → {gb} | {ra.get('cost_usd', 0):.5f} → {rb.get('cost_usd', 0):.5f} | "
            f"{ra.get('elapsed_s', 0):.1f} → {rb.get('elapsed_s', 0):.1f} |"
        )

    lines += ["", "## 判定", ""]
    lines += _verdict(a, b)
    lines += [
        "",
        "> 5 条臂的数字取自同一次 20 条响应的前 5 条，不是独立的 5 条调用；",
        "> 试点已见 Tavily 排序抖动，故 20 条的边际收益应视为上界。",
        "",
    ]
    out = Path(__file__).resolve().parent / "ab_news.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(out)
    return 0


def _fmt_outcomes(o: dict) -> str:
    return " / ".join(str(o.get(k, 0)) for k in
                      ("ok", "abstained_correctly", "hallucinated", "degraded", "failure", "timeout"))


def _ratio(a: float, b: float) -> str:
    return f"{b / a:.2f}×" if a else "n/a"


def _verdict(a: dict, b: dict) -> list[str]:
    un_a = a["grounding"].get("ungrounded", 0)
    un_b = b["grounding"].get("ungrounded", 0)
    cost_ratio = b["cost_avg"] / a["cost_avg"] if a["cost_avg"] else 0
    lat_ratio = b["latency"].get("p95", 0) / a["latency"].get("p95", 1) if a["latency"].get("p95") else 0
    out = []
    if un_b > un_a:
        out.append(f"- 未接地数字从 {un_a} 升到 {un_b} —— 触发“质量不回退”红线。")
    else:
        out.append(f"- 未接地数字 {un_a} → {un_b}，未回退。")
    out.append(f"- 单次成本 {cost_ratio:.2f}×，p95 {lat_ratio:.2f}×。")
    if un_b <= un_a and cost_ratio <= 2.0:
        out.append("- 按事先规则：默认改 20，并更新 README 的已知局限。")
    elif un_b <= un_a and cost_ratio <= 4.0:
        out.append("- 按事先规则：20 条作为可选开关（默认仍 5），只在覆盖优先的场景开启。")
    else:
        out.append("- 按事先规则：维持 5 条；扩量不划算，转去做来源替换。")
    return out


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
