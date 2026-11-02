"""对比三份磁带的证据层差异（不联网、不调 LLM）。

- `frozen.json`      2026-09-17 录的 5 条（旧基线）
- `news5_20261102`   2026-11-02 录的 20 条里取前 5（同一次响应）
- `news20_20261102`  2026-11-02 录的 20 条

frozen5 → news5 是时间漂移；news5 → news20 才是“取数条数”的效应。

用法：
    python docs/experiments/2026-11-02-max-results/analyze_evidence.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "docs" / "experiments"))

from slot_metric import (  # noqa: E402
    AGGREGATOR, INVALID_TICKERS, SLOTS, domain_of, loose_slots, strict_slots,
)

OUT = Path(__file__).resolve().parent
CASSETTES = {
    "frozen5": ROOT / "eval" / "cassettes" / "frozen.json",
    "news5": ROOT / "eval" / "cassettes" / "news5_20261102.json",
    "news20": ROOT / "eval" / "cassettes" / "news20_20261102.json",
}


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))["sources"]


def main() -> int:
    data = {name: load(path) for name, path in CASSETTES.items()}
    tickers = [t for t in data["frozen5"] if t not in INVALID_TICKERS]

    rows = []
    for t in tickers:
        row = {"ticker": t}
        for arm, sources in data.items():
            items = sources[t].get("search_company_news") or []
            row[arm] = {
                "docs": len(items),
                "chars": sum(len(i.get("content", "")) for i in items),
                "strict": sorted(strict_slots(items, t)),
                "loose": sorted(loose_slots(items)),
                "domains": [domain_of(i.get("url", "")) for i in items],
            }
        rows.append(row)

    summary = {"tickers": len(tickers), "arms": {}}
    for arm in data:
        docs = [r[arm]["docs"] for r in rows]
        chars = [r[arm]["chars"] for r in rows]
        strict = [len(r[arm]["strict"]) for r in rows]
        loose = [len(r[arm]["loose"]) for r in rows]
        domains = [d for r in rows for d in r[arm]["domains"]]
        agg = sum(1 for d in domains if d in AGGREGATOR)
        summary["arms"][arm] = {
            "avg_docs": sum(docs) / len(docs),
            "avg_chars": sum(chars) / len(chars),
            "approx_tokens_per_run": sum(chars) / len(chars) / 4,
            "avg_strict": sum(strict) / len(strict),
            "avg_loose": sum(loose) / len(loose),
            "all_five_strict": sum(1 for r in rows if len(r[arm]["strict"]) == 5),
            "slot_hits": {name: sum(1 for r in rows if name in r[arm]["strict"])
                          for name in SLOTS},
            "aggregator_share": agg / len(domains),
            "stories_le_3": None,
        }
        summary["arms"][arm]["stories_le_3"] = None  # 去重口径不稳定，不报

    summary["per_ticker"] = [
        {
            "ticker": r["ticker"],
            "docs": {a: r[a]["docs"] for a in data},
            "strict": {a: len(r[a]["strict"]) for a in data},
            "missing_news20": [s for s in SLOTS if s not in r["news20"]["strict"]],
        }
        for r in rows
    ]

    (OUT / "evidence_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "evidence_report.md").write_text(_render(summary), encoding="utf-8")
    print(json.dumps(summary["arms"], ensure_ascii=False, indent=2))
    return 0


def _render(s: dict) -> str:
    a = s["arms"]
    lines = [
        "# 证据层对比：5 条 vs 20 条新闻（24 标的，20 个有效）",
        "",
        "不联网、不调 LLM；只比较检索到的证据本身。报告层指标（接地、成本、p95）待 LLM 可用后补。",
        "",
        "## 总览",
        "",
        "| 指标 | 旧基线（09-17，5 条） | 新 5 条（11-02） | 新 20 条（11-02） |",
        "| --- | ---: | ---: | ---: |",
    ]
    for key, label, fmt in (
        ("avg_docs", "平均文档数", "{:.1f}"),
        ("avg_chars", "平均正文字符", "{:.0f}"),
        ("approx_tokens_per_run", "约合 prompt tokens", "{:.0f}"),
        ("avg_strict", "严格槽位", "{:.2f}/5"),
        ("avg_loose", "宽松槽位（对照）", "{:.2f}/5"),
        ("all_five_strict", "5 槽全满的标的数", "{}/20"),
    ):
        lines.append(f"| {label} | {fmt.format(a['frozen5'][key])} | "
                     f"{fmt.format(a['news5'][key])} | {fmt.format(a['news20'][key])} |")
    agg = a["news20"]["aggregator_share"]
    lines.append(f"| 聚合/导航页占比 | {a['frozen5']['aggregator_share']:.1%} | "
                 f"{a['news5']['aggregator_share']:.1%} | {agg:.1%} |")
    lines += [
        "",
        "## 各槽位命中标的数（严格口径）",
        "",
        "| 槽位 | 旧基线 | 新 5 条 | 新 20 条 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name in SLOTS:
        lines.append(f"| {name} | {a['frozen5']['slot_hits'][name]} | "
                     f"{a['news5']['slot_hits'][name]} | {a['news20']['slot_hits'][name]} |")
    lines += [
        "",
        "## 逐标的（严格槽位 /5）",
        "",
        "| ticker | 文档 旧/新5/新20 | 槽位 旧 | 槽位 新5 | 槽位 新20 | 20 条仍缺 |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for r in s["per_ticker"]:
        d = r["docs"]
        lines.append(
            f"| {r['ticker']} | {d['frozen5']}/{d['news5']}/{d['news20']} | "
            f"{r['strict']['frozen5']} | {r['strict']['news5']} | {r['strict']['news20']} | "
            f"{'、'.join(r['missing_news20']) or '—'} |")
    d20 = a["news20"]["avg_docs"]
    lines += [
        "",
        "## 判读规则（先写，后看数）",
        "",
        f"- 若新 20 条严格槽位显著高于新 5 条（≥ +1.0），说明截断是主因，"
        f"`max_results` 调到 20 值得；随后用完整评测确认未接地数字与成本。",
        "- 若差距 ≤ +0.5，说明 20 条只是把正文变长、槽位没实质变化，"
        "维持 5 条，转去做来源替换（IR/SEC/司法）。",
        "- 旧基线 → 新 5 条的差反映时间漂移与排序抖动，不能当作改动效果。",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
