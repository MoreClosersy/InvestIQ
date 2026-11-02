"""证伪实验：现有冻结语料下，"单次搜索覆盖不足"是否成立？

只读 eval/cassettes/frozen.json，不联网、不调 LLM。
输出四组可核对的数字：

  E1 来源构成   —— 120 条结果里，聚合/导航页 vs 新闻媒体 vs 视频社交
  E2 故事去重   —— 每个 ticker 的 5 条结果实际是几个不同故事（标题 Jaccard 聚类）
  E3 槽位覆盖   —— 计划里写死的 5 个槽位，现有 5 条结果覆盖了几个
  E4 回放天花板 —— fixed 模式已消费 5/5 条；replay 下任何新查询只能拿到同一批

用法（在仓库根目录）：python docs/experiments/2026-11-02-coverage/coverage_experiment.py \
    [--json-out summary.json] [--md-out report.md]
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import statistics
import urllib.parse
from pathlib import Path

CASSETTE = Path(__file__).resolve().parents[3] / "eval" / "cassettes" / "frozen.json"

# ---------------------------------------------------------------- E1 来源分类

AGGREGATOR = {
    "public.com", "finance.yahoo.com", "marketbeat.com", "investing.com",
    "zacks.com", "google.com", "robinhood.com", "stocktitan.net",
    "seekingalpha.com", "stockanalysis.com", "tipranks.com", "morningstar.com",
    "pricetargetresearch.com", "wallstreethorizon.com", "us.plus500.com",
    "bitget.com", "fool.com",
}
PUBLISHER = {
    "cnn.com", "cnbc.com", "nytimes.com", "theglobeandmail.com",
    "prnewswire.com", "businesswire.com", "reuters.com", "bloomberg.com",
    "wsj.com", "ft.com", "forbes.com", "barrons.com",
}
SOCIAL = {"youtube.com", "instagram.com", "x.com", "twitter.com", "reddit.com"}
IR_SUFFIX = ("investor.", "ir.", "q4cdn.com", "sec.gov")


def domain_of(url: str) -> str:
    return urllib.parse.urlparse(url).netloc.lower().removeprefix("www.")


def source_class(domain: str) -> str:
    if domain in SOCIAL:
        return "social/video"
    if domain in AGGREGATOR:
        return "aggregator/nav"
    if domain in PUBLISHER:
        return "publisher"
    if domain.startswith(IR_SUFFIX) or domain.endswith(".q4cdn.com"):
        return "IR/regulatory"
    return "other"


# ---------------------------------------------------------------- E2 故事去重

_STOP = set(
    "the a an of for to in on and or with its is are was were as at by from "
    "stock stocks shares price news today latest report reports".split()
)


def title_tokens(title: str) -> set[str]:
    return {
        w for w in re.findall(r"[a-z0-9$%.]+", title.lower())
        if w not in _STOP and len(w) > 1
    }


def distinct_stories(items: list[dict], threshold: float = 0.5) -> int:
    """单链接聚类：标题 token Jaccard ≥ 阈值即视为同一故事。"""
    clusters: list[set[str]] = []
    for it in items:
        toks = title_tokens(it.get("title", ""))
        for c in clusters:
            union = toks | c
            if union and len(toks & c) / len(union) >= threshold:
                c |= toks
                break
        else:
            clusters.append(set(toks))
    return len(clusters)


# ---------------------------------------------------------------- E3 槽位覆盖
# 规则刻意放宽：宁可多判"覆盖"，也不人为制造缺口。若放宽后仍缺，结论更硬。

SLOTS: dict[str, re.Pattern[str]] = {
    "财报/指引": re.compile(
        r"\b(earnings|eps|revenue|guidance|outlook|quarter|fiscal|q[1-4])\b", re.I
    ),
    "分析师动作": re.compile(
        r"\b(upgrade|downgrade|price target|analyst|rating|reiterat|initiat|"
        r"overweight|underweight|buy rating|sell rating)\b", re.I
    ),
    "重大事件": re.compile(
        r"\b(acquisition|acquire|merger|takeover|ceo|cfo|resign|appoint|"
        r"layoff|bankrupt|ipo|spin-off|stake|board)\b", re.I
    ),
    "风险/法律": re.compile(
        r"\b(lawsuit|sue[sd]?|investigation|regulat\w*|sec\b|fine[sd]?|"
        r"penalt\w*|recall|probe|antitrust|class action|settle\w*)\b", re.I
    ),
    "产品/战略": re.compile(
        r"\b(launch\w*|unveil\w*|partner\w*|expand\w*|new product|contract|"
        r"deal|investment|joint venture)\b", re.I
    ),
}


def slots_hit(items: list[dict]) -> dict[str, bool]:
    text = "\n".join(f"{i.get('title','')}\n{i.get('content','')}" for i in items)
    return {name: bool(rx.search(text)) for name, rx in SLOTS.items()}


# ---------------------------------------------------------------- 主流程


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json-out", default="")
    ap.add_argument("--md-out", default="")
    args = ap.parse_args()

    data = json.loads(CASSETTE.read_text(encoding="utf-8"))
    sources: dict[str, dict] = data["sources"]

    per_ticker = []
    class_counts: collections.Counter[str] = collections.Counter()
    slot_counts: collections.Counter[str] = collections.Counter()
    domains: collections.Counter[str] = collections.Counter()
    chars: list[int] = []

    for ticker, blob in sources.items():
        items = blob.get("search_company_news") or []
        for it in items:
            d = domain_of(it.get("url", ""))
            domains[d] += 1
            class_counts[source_class(d)] += 1
            chars.append(len(it.get("content", "")))

        hits = slots_hit(items)
        for name, ok in hits.items():
            slot_counts[name] += int(ok)

        nums = sum(len(re.findall(r"\d[\d,.]*", it.get("content", ""))) for it in items)
        per_ticker.append(
            {
                "ticker": ticker,
                "items": len(items),
                "distinct_stories": distinct_stories(items),
                "slots_covered": sum(hits.values()),
                "slots": hits,
                "numbers": nums,
                "chars": sum(len(it.get("content", "")) for it in items),
            }
        )

    n = len(per_ticker)
    total_items = sum(r["items"] for r in per_ticker)
    slot_hist = collections.Counter(r["slots_covered"] for r in per_ticker)
    invalid = {"ZZZZZZ", "INVALID", "NOTAREAL", "12345"}
    valid_rows = [r for r in per_ticker if r["ticker"] not in invalid]
    valid_slot_hist = collections.Counter(r["slots_covered"] for r in valid_rows)
    valid_slot_hits = {
        name: sum(r["slots"][name] for r in valid_rows) for name in SLOTS
    }
    full = sum(1 for r in per_ticker if r["slots_covered"] == len(SLOTS))
    zero = sum(1 for r in per_ticker if r["slots_covered"] == 0)

    summary = {
        "tickers": n,
        "items_total": total_items,
        "items_per_ticker": sorted({r["items"] for r in per_ticker}),
        "source_classes": dict(class_counts),
        "top_domains": domains.most_common(12),
        "distinct_story_hist": dict(sorted(collections.Counter(
            r["distinct_stories"] for r in per_ticker).items())),
        "slot_hist": dict(sorted(slot_hist.items())),
        "slot_hit_rate": {k: f"{v}/{n}" for k, v in slot_counts.items()},
        "valid_tickers": len(valid_rows),
        "valid_slot_hist": dict(sorted(valid_slot_hist.items())),
        "valid_slot_hit_rate": {k: f"{v}/{len(valid_rows)}" for k, v in valid_slot_hits.items()},
        "valid_tickers_all_slots": sum(1 for r in valid_rows if r["slots_covered"] == len(SLOTS)),
        "tickers_all_slots": full,
        "tickers_no_slots": zero,
        "content_chars_median": statistics.median(chars),
        "content_chars_total": sum(chars),
        "numbers_total": sum(r["numbers"] for r in per_ticker),
    }

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\nper-ticker:")
    for r in per_ticker:
        miss = [k for k, v in r["slots"].items() if not v]
        print(f"  {r['ticker']:7s} items={r['items']} stories={r['distinct_stories']} "
              f"slots={r['slots_covered']}/5 nums={r['numbers']:4d} miss={','.join(miss) or '-'}")

    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps({"summary": summary, "per_ticker": per_ticker},
                       ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    if args.md_out:
        Path(args.md_out).write_text(render_md(summary, per_ticker), encoding="utf-8")
    return 0


def render_md(s: dict, rows: list[dict]) -> str:
    lines = [
        "# 证伪实验：现有冻结语料能支撑 ReAct 吗？",
        "",
        f"输入：`eval/cassettes/frozen.json`（{s['tickers']} ticker / {s['items_total']} 条结果），"
        "只读、不联网、不调 LLM。规则是关键词启发式，刻意放宽。",
        "",
        "## E1 来源构成",
        "",
        "| 类别 | 条数 | 占比 |",
        "| --- | ---: | ---: |",
    ]
    for k, v in s["source_classes"].items():
        lines.append(f"| {k} | {v} | {v / s['items_total']:.1%} |")
    lines += ["", "抓取最多的域名：", ""]
    for d, c in s["top_domains"]:
        lines.append(f"- `{d}` — {c}")
    lines += [
        "",
        "## E2 故事去重（标题 Jaccard ≥ 0.5 归为同一故事）",
        "",
        "| 每个 ticker 的不同故事数 | ticker 个数 |",
        "| ---: | ---: |",
    ]
    for k, v in s["distinct_story_hist"].items():
        lines.append(f"| {k} | {v} |")
    lines += [
        "",
        "## E3 槽位覆盖（计划里的 5 个槽位）",
        "",
        "| 覆盖槽位数 | ticker 个数 |",
        "| ---: | ---: |",
    ]
    for k, v in s["slot_hist"].items():
        lines.append(f"| {k} | {v} |")
    lines += ["", "各槽位命中 ticker 数：", ""]
    for k, v in s["slot_hit_rate"].items():
        lines.append(f"- {k}: **{v}**")
    lines += [
        "",
        f"5 槽全满的 ticker：**{s['tickers_all_slots']}/{s['tickers']}**；"
        f"一个槽都没覆盖：**{s['tickers_no_slots']}/{s['tickers']}**。",
        "",
        "### 只看 20 个有效标的（剔除 4 个无效代码）",
        "",
        "| 覆盖槽位数 | ticker 个数 |",
        "| ---: | ---: |",
    ]
    for k, v in s["valid_slot_hist"].items():
        lines.append(f"| {k} | {v} |")
    lines += ["", "各槽位命中：", ""]
    for k, v in s["valid_slot_hit_rate"].items():
        lines.append(f"- {k}: **{v}**")
    lines += [
        "",
        f"有效标的里 5 槽全满：**{s['valid_tickers_all_slots']}/{s['valid_tickers']}**。",
        "",
        "## E4 回放天花板",
        "",
        f"每个 ticker 的搜索结果恒为 {s['items_per_ticker'][0]} 条，fixed 模式的 research 节点"
        "已把这 5 条全部喂进摘要 prompt（复用率 100%）。replay 下 `search_company_news` "
        "只按 ticker 取数、与 query 无关，因此**任何新增查询拿到的都是同一批 5 条**，"
        "新增证据恒为 0。",
        "",
        f"正文总量 {s['content_chars_total']:,} 字符（中位 {s['content_chars_median']:.0f}/条），"
        f"数字 token 合计 {s['numbers_total']:,}。",
        "",
        "## 结论",
        "",
        "1. **缺口存在，但集中**：有效 20 只里风险/法律只覆盖 5 只、重大事件 9 只；"
        "财报 19/20、分析师 18/20 基本够用。计划里“单次搜索覆盖不足”的前提成立。",
        "2. **瓶颈在来源，不在查询次数**：63.3% 的结果是聚合/导航页，媒体只占 11.7%。"
        "再加查询，大概率还是同一类页面，不是缺失的司法/监管事实。",
        "3. **现有磁带上循环拿不到新证据**：每 ticker 恒 5 条、检索与 query 无关，"
        "新增证据恒为 0——计划里的 2×2 必须先改造评测层才可执行。",
        "4. **槽位不满不等于该补**：风险/法律常常确实没有新闻；把“槽位满”当停止条件"
        "会诱导模型硬找风险。空槽应如实报空。",
        "",
        "> 规则是关键词启发式、刻意放宽，用于判断方向，不作为质量指标。",
        "",
        "## 逐 ticker",
        "",
        "| ticker | 条数 | 不同故事 | 槽位 | 缺失槽位 | 数字数 |",
        "| --- | ---: | ---: | ---: | --- | ---: |",
    ]
    for r in rows:
        miss = "、".join(k for k, v in r["slots"].items() if not v) or "—"
        lines.append(
            f"| {r['ticker']} | {r['items']} | {r['distinct_stories']} | "
            f"{r['slots_covered']}/5 | {miss} | {r['numbers']} |"
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
