"""S3 试点：单次搜索 vs 槽位定向查询 —— 多查 3 条，缺口补得上吗？

背景：24 ticker 的冻结语料里，风险/法律只覆盖 5/20、重大事件 9/20。
本脚本对 5 个低覆盖标的，用真实 Tavily 各发 4 条查询（1 条基线 + 3 条槽位
定向），把原始结果录进 JSON，再离线统计"每多一条查询带来多少新 URL / 新槽位"。

为什么不是 ReAct：这一步只测"查询内容"，不测"由谁决定查询"。
若固定的 3 条槽位查询就能补上缺口，LLM 规划循环的收益空间就被压掉了。

用法：
    # 1) 录制（联网，20 次搜索；需要 TAVILY_API_KEY）
    python docs/experiments/2026-11-02-slot-queries/slot_query_pilot.py record

    # 2) 离线分析
    python docs/experiments/2026-11-02-slot-queries/slot_query_pilot.py analyze
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))  # 从仓库根目录导入 tools/
RAW = OUT / "raw.json"
CONTROL = OUT / "control.json"

TICKERS = ["AAPL", "JPM", "LCID", "NIO", "SOFI"]

QUERIES: dict[str, str] = {
    "q0_generic": "{ticker} stock news latest earnings",
    "q1_earnings": "{ticker} earnings guidance revenue outlook",
    "q2_analyst": "{ticker} analyst upgrade downgrade price target",
    "q3_risk": "{ticker} lawsuit investigation regulatory risk",
}

# 与 docs/experiments/2026-11-02-coverage/coverage_experiment.py 同口径。
# 试点脚本刻意自带一份，避免为一个实验去改 eval/ 的生产代码。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # docs/experiments
from slot_metric import (  # noqa: E402
    AGGREGATOR,
    SLOTS,
    domain_of,
    loose_slots,
    strict_slots,
)

# ------------------------------------------------------------------ record

def record() -> int:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")

    from tools.search_tool import TimeoutTavilySearchAPIWrapper

    wrapper = TimeoutTavilySearchAPIWrapper(timeout=30)
    payload: dict = {
        "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tickers": TICKERS,
        "queries": QUERIES,
        "results": {},
    }

    for ticker in TICKERS:
        payload["results"][ticker] = {}
        for key, template in QUERIES.items():
            query = template.format(ticker=ticker)
            try:
                raw = wrapper.raw_results(query, max_results=5, search_depth="advanced")
                payload["results"][ticker][key] = {
                    "query": query,
                    "results": raw.get("results", []),
                }
                n = len(raw.get("results", []))
                print(f"[record] {ticker:5s} {key:12s} -> {n} results")
            except Exception as exc:  # 网络/额度问题不中断整轮
                payload["results"][ticker][key] = {"query": query, "error": repr(exc)}
                print(f"[record] {ticker:5s} {key:12s} -> ERROR {exc!r}")

    RAW.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {RAW}")
    return 0


def record_control() -> int:
    """控制组：同一条基线查询，一次取 20 条 —— 把"查询更多"和"文档更多"拆开。"""
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")

    from tools.search_tool import TimeoutTavilySearchAPIWrapper

    wrapper = TimeoutTavilySearchAPIWrapper(timeout=30)
    payload: dict = {
        "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "note": "q0_generic with max_results=20, same depth",
        "results": {},
    }
    for ticker in TICKERS:
        query = QUERIES["q0_generic"].format(ticker=ticker)
        try:
            raw = wrapper.raw_results(query, max_results=20, search_depth="advanced")
            payload["results"][ticker] = {"query": query, "results": raw.get("results", [])}
            print(f"[control] {ticker:5s} -> {len(raw.get('results', []))} results")
        except Exception as exc:
            payload["results"][ticker] = {"query": query, "error": repr(exc)}
            print(f"[control] {ticker:5s} -> ERROR {exc!r}")
    CONTROL.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {CONTROL}")
    return 0


# ------------------------------------------------------------------ analyze

def _dedupe(groups: list[list[dict]]) -> list[dict]:
    seen: set[str] = set()
    out: list[dict] = []
    for items in groups:
        for it in items:
            url = it.get("url", "")
            if url and url not in seen:
                seen.add(url)
                out.append(it)
    return out


def analyze() -> int:
    data = json.loads(RAW.read_text(encoding="utf-8"))
    ctl = json.loads(CONTROL.read_text(encoding="utf-8")) if CONTROL.exists() else None
    tickers = data["tickers"]
    order = list(QUERIES)

    per: dict[str, dict] = {}
    for t in tickers:
        q0 = (data["results"][t].get("q0_generic") or {}).get("results", [])
        four = _dedupe([(data["results"][t].get(k) or {}).get("results", []) for k in order])
        c20 = ((ctl or {}).get("results", {}).get(t) or {}).get("results", [])
        per[t] = {"q0_5": q0, "q0_20": c20, "four_q": four}

    def arm_stats(items_by_ticker: dict[str, list[dict]], searches: int) -> dict:
        docs = [len(v) for v in items_by_ticker.values()]
        loose = [len(loose_slots(v)) for v in items_by_ticker.values()]
        strict = [len(strict_slots(v, t)) for t, v in items_by_ticker.items()]
        return {
            "searches": searches,
            "avg_docs": sum(docs) / len(docs),
            "avg_loose": sum(loose) / len(loose),
            "avg_strict": sum(strict) / len(strict),
        }

    arms = {
        "q0_5": arm_stats({t: per[t]["q0_5"] for t in tickers}, 1),
        "q0_20": arm_stats({t: per[t]["q0_20"] for t in tickers}, 1),
        "four_q": arm_stats({t: per[t]["four_q"] for t in tickers}, 4),
    }

    # 文档数曲线（严格口径）
    curves = {}
    for arm, n_max in (("q0_20", 20), ("four_q", 20)):
        pts = []
        for n in (5, 10, 15, 20):
            if n > n_max:
                continue
            vals = [len(strict_slots(per[t][arm][:n], t)) for t in tickers]
            pts.append({"n": n, "avg_strict": sum(vals) / len(vals)})
        curves[arm] = pts

    # 宽松规则为什么不可信：抓几个"只有导航文字命中"的例子
    examples = []
    for t in tickers:
        for it in per[t]["four_q"]:
            text = (it.get("title", "") or "") + "\n" + (it.get("content", "") or "")
            for slot, rx in SLOTS.items():
                if not rx.search(text):
                    continue
                if slot in strict_slots([it], t):
                    continue
                m = rx.search(text)
                snippet = text[max(0, m.start() - 50):m.end() + 50].replace("\n", " ")
                examples.append({
                    "ticker": t, "slot": slot,
                    "domain": domain_of(it.get("url", "")),
                    "title": (it.get("title", "") or "")[:70],
                    "snippet": snippet,
                })
                break
            if len(examples) >= 4:
                break
        if len(examples) >= 4:
            break

    mix = {"aggregator": 0, "other": 0}
    for t in tickers:
        for it in per[t]["four_q"]:
            key = "aggregator" if domain_of(it.get("url", "")) in AGGREGATOR else "other"
            mix[key] += 1

    summary = {
        "recorded_at": data.get("recorded_at"),
        "tickers": tickers,
        "arms": arms,
        "curves": curves,
        "false_positive_examples": examples,
        "per_ticker": [
            {
                "ticker": t,
                "docs": {a: len(per[t][a]) for a in arms},
                "strict": {a: sorted(strict_slots(per[t][a], t)) for a in arms},
                "loose": {a: sorted(loose_slots(per[t][a])) for a in arms},
            }
            for t in tickers
        ],
        "source_mix": mix,
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "report.md").write_text(_render(summary), encoding="utf-8")
    print(json.dumps(summary["arms"], ensure_ascii=False, indent=2))
    return 0


def _render(s: dict) -> str:
    arms = s["arms"]
    lines = [
        "# S3 试点：缺口是“查询不够”还是“结果取少了”？",
        "",
        f"- 录制：{s['recorded_at']}；5 个标的 × 4 条查询（每条 5 结果）+ 1 条控制查询（20 结果）",
        "- 全部为真实 Tavily 调用，原始结果录在 `raw.json` / `control.json`，分析完全离线可复跑",
        "- 槽位规则是关键词启发式；本报告给出“宽松/严格”两种口径，并说明为什么只信严格口径",
        "",
        "## 先说指标问题：宽松口径会把导航栏当证据",
        "",
        "严格口径要求槽位关键词出现在标题，或与公司名同句。宽松口径只要求全文出现，",
        "于是 `SEC Filings`、`Risk Factors`、`IPO Calendar` 这类站点框架文字会同时点亮四个槽位。",
        "",
        "| ticker | 槽位 | 域名 | 被误判的文字 |",
        "| --- | --- | --- | --- |",
    ]
    for e in s["false_positive_examples"]:
        lines.append(f"| {e['ticker']} | {e['slot']} | `{e['domain']}` | …{e['snippet']}… |")
    lines += [
        "",
        "## 三种取数方案（同一批标的、同一天）",
        "",
        "| 方案 | 搜索次数 | 平均文档数 | 宽松槽位 | **严格槽位** |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, label in (("q0_5", "基线 q0 × 5（现状）"), ("q0_20", "q0 × 20（只取更多）"),
                        ("four_q", "4 条查询 × 5（多问几种）")):
        a = arms[name]
        lines.append(f"| {label} | {a['searches']} | {a['avg_docs']:.1f} | "
                     f"{a['avg_loose']:.2f}/5 | **{a['avg_strict']:.2f}/5** |")
    lines += [
        "",
        "## 文档数曲线（严格口径）",
        "",
        "| 文档数 | q0 单查询 | 4 条查询合计 |",
        "| ---: | ---: | ---: |",
    ]
    c20 = {p["n"]: p["avg_strict"] for p in s["curves"]["q0_20"]}
    c4 = {p["n"]: p["avg_strict"] for p in s["curves"]["four_q"]}
    for n in (5, 10, 15, 20):
        lines.append(f"| {n} | {c20.get(n, float('nan')):.2f} | {c4.get(n, float('nan')):.2f} |")
    lines += [
        "",
        "> 曲线的 n=5 点取的是控制组 20 条的前 5 条（1.80），与表格里独立调用的基线 5 条（2.00）"
        "不是同一次响应；0.2 的差来自 Tavily 排序抖动，也说明单次采样的精度有限。",
        "",
        "## 逐标的（严格口径）",
        "",
        "| ticker | 文档 q0×5 / q0×20 / 4q | 槽位 q0×5 | 槽位 q0×20 | 槽位 4q | 4q 仍缺 |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for r in s["per_ticker"]:
        d = r["docs"]
        st = r["strict"]
        missing = [k for k in SLOTS if k not in st["four_q"]]
        lines.append(
            f"| {r['ticker']} | {d['q0_5']} / {d['q0_20']} / {d['four_q']} | "
            f"{len(st['q0_5'])}/5 | {len(st['q0_20'])}/5 | {len(st['four_q'])}/5 | "
            f"{'、'.join(missing) or '—'} |")
    mix = s["source_mix"]
    total = mix["aggregator"] + mix["other"]
    lines += [
        "",
        "## 来源构成（4 条查询去重后）",
        "",
        f"- 聚合/导航页：{mix['aggregator']}/{total}（{mix['aggregator']/total:.1%}）",
        "",
        "## 判读规则（先写，后看数）",
        "",
        "- 若“只取更多”（q0×20）与“多问几种”（4q）在严格口径下差不多，说明缺口的成因是"
        "**结果被 max_results 截断**，不是查询不够聪明；先调条数，别急着上循环。",
        "- 若 4q 明显高于 q0×20，才说明查询多样性本身有价值，S3 的定向查询值得固化。",
        "- 若两者都上不去，说明瓶颈在来源质量，需要换源（IR/SEC/司法），加查询、加循环都没用。",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["record", "control", "analyze"])
    args = ap.parse_args()
    if args.mode == "record":
        raise SystemExit(record())
    if args.mode == "control":
        raise SystemExit(record_control())
    raise SystemExit(analyze())
