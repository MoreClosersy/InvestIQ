"""为 24 个标的录制 20 条新闻结果，并派生 5 条版本（同一次响应）。

为什么这样录：
- 只重录 news，yfinance 三块数据原样抄自 `frozen.json`，两臂的财务输入逐字节相同。
- 5 条版本取同一次 20 条响应的前 5 条，避免"今天的前 5"和"今天的前 20"来自两次调用、
  排序不同（试点里已看到 0.2 个槽位的抖动）。
- 代价：5 条版本不是真实 5 条调用的排序，结论里要注明。

用法（需要 TAVILY_API_KEY，24 次搜索）：
    python docs/experiments/2026-11-02-max-results/record_news.py
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

OUT = Path(__file__).resolve().parent
FROZEN = ROOT / "eval" / "cassettes" / "frozen.json"
CASSETTE_20 = ROOT / "eval" / "cassettes" / "news20_20261102.json"
CASSETTE_5 = ROOT / "eval" / "cassettes" / "news5_20261102.json"
MANIFEST = OUT / "recording.json"

QUERY = "{ticker} stock news latest earnings"
MAX_RESULTS = 20


def main() -> int:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    from tools.search_tool import TimeoutTavilySearchAPIWrapper

    frozen = json.loads(FROZEN.read_text(encoding="utf-8"))
    wrapper = TimeoutTavilySearchAPIWrapper(timeout=30)

    sources20: dict = {}
    sources5: dict = {}
    manifest: dict = {"recorded_at": "", "query": QUERY, "max_results": MAX_RESULTS,
                      "tickers": {}}

    for ticker, blob in frozen["sources"].items():
        query = QUERY.format(ticker=ticker)
        results: list[dict] = []
        error = None
        for attempt in (1, 2):
            try:
                raw = wrapper.raw_results(query, max_results=MAX_RESULTS,
                                          search_depth="advanced")
                results = raw.get("results", [])
                if results:
                    break
            except Exception as exc:
                error = repr(exc)
                time.sleep(1.0)
        if not results:
            print(f"[record] {ticker:8s} FAILED {error or 'empty'}")

        sources20[ticker] = dict(blob)          # 抄财务三块
        sources5[ticker] = dict(blob)
        sources20[ticker]["search_company_news"] = results
        sources5[ticker]["search_company_news"] = results[:5]

        manifest["tickers"][ticker] = {
            "query": query, "results": len(results), "error": error,
        }
        print(f"[record] {ticker:8s} -> {len(results):2d} results")

    recorded_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    manifest["recorded_at"] = recorded_at

    for path, sources in ((CASSETTE_20, sources20), (CASSETTE_5, sources5)):
        path.write_text(
            json.dumps({"version": 1, "recorded_at": recorded_at, "sources": sources},
                       ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"[write] {path} ({path.stat().st_size / 1024:.0f} KB)")

    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
