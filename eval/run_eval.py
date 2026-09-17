"""InvestIQ evaluation harness — an *external* driver, not part of the agents.

Runs the existing analysis interface (``graph.workflow.build_graph()``) over a
test set of tickers and records, per request:

  * start / end timestamps and total wall-clock latency,
  * status: ``success`` / ``failure`` / ``timeout`` (default timeout 60s),
  * token usage extracted from the LangChain/OpenAI callback events
    (``prompt_tokens`` / ``completion_tokens`` per LLM call, aggregated),
  * the key numeric outputs of the Financial Agent (price, P/E, dividend
    yield, beta, …) printed separately so they can be eyeballed against raw
    ``yfinance`` values,
  * for a handful of representative tickers, 3 repeated runs to check whether
    the key conclusions/numbers are stable.

Finally it prints a summary (Markdown) with totals, success rate, latency
p50/p95/mean, average token usage and cost (priced for GPT-4o-mini), and a
list of failures/timeouts with their concrete error messages.

The harness only *calls* the existing interface — it never changes any agent
core logic. It runs the compiled LangGraph **in-process** (not over HTTP),
which is required to collect token usage from the OpenAI callbacks; a
``POST /research`` response does not expose token usage.

Reproducibility (freeze / replay):
    Two runs of the same commit otherwise see different market data and news —
    yfinance moves every minute, Tavily returns fresh articles — so any output
difference is unattributable: you cannot tell a better prompt from a better
market day. ``--record`` captures the raw yfinance/Tavily responses for every
ticker into a JSON cassette; ``--replay`` serves them back with no network
calls at all, making the *inputs* byte-identical across runs. See
``eval/fixtures.py``. Only inputs are frozen; LLM sampling is deliberately
left alive because it is a property under measurement (see the consistency
section), not noise.

Outcome taxonomy:
    Each request gets exactly one ``outcome``, so a failure can never hide
    inside a success count:

      ``ok``                  有财务数据且产出了分析 —— 唯一算「成功」的一类
      ``abstained_correctly`` 无财务数据且正确地没产出分析(无效代码应有的行为)
      ``hallucinated``        无财务数据却产出了分析 —— 真 bug,必须为 0
      ``degraded``            有财务数据但没产出分析(LLM 被跳过或返回空)
      ``failure`` / ``timeout``

    Note ``abstained_correctly`` is *correct behaviour*, not a failure: an
    invalid ticker MUST be refused. Counting it against the system was the old
    harness's bug — it made the fix look like no improvement.

Usage:
    python eval/run_eval.py                                   # built-in 24-ticker set
    python eval/run_eval.py --tickers AAPL,MSFT,ZZZZZZ
    python eval/run_eval.py --test-set eval/test_set.example.json
    python eval/run_eval.py --timeout 60 --consistency-runs 3
    python eval/run_eval.py --no-consistency
    python eval/run_eval.py --out-dir eval/results
    python eval/run_eval.py --record --fixtures eval/cassettes/frozen.json
    python eval/run_eval.py --replay --fixtures eval/cassettes/frozen.json

Notes / limitations:
  * A timed-out request runs in a worker thread; Python cannot force-kill a
    thread, so the harness simply stops waiting. The orphaned thread finishes
    on its own in the background (it may keep the process alive until then).
  * ``dividend_yield`` is the value emitted by ``tools.finance_tool``, which
    normalizes yfinance's percentage ``dividendYield`` by ``/100`` (e.g.
    ``0.0044`` == ``0.44%``). Compare accordingly against raw yfinance values.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from statistics import mean
from typing import Any

# Allow running straight from the repo root without installing the package.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(override=True)

from langchain_core.callbacks import BaseCallbackHandler  # noqa: E402
from langchain_core.outputs import LLMResult  # noqa: E402

# Imported lazily inside get_graph() so env is loaded first.
# from graph.state import initial_state
# from graph.workflow import build_graph

# Freeze/replay layer for the two external data sources. Pure stdlib — importing
# it does not touch the network; patching only happens when install() is called.
from eval import fixtures  # noqa: E402
# Contract-layer check: does every number in the report trace back to an input?
from eval import grounding  # noqa: E402

# ---------------------------------------------------------------------------
# Pricing (verified against OpenAI's public pricing page, developers.openai.com,
# as of the run date — GPT-4o-mini: $0.15 / 1M input, $0.60 / 1M output.
# Cached input ($0.075/1M) is not separately modeled.)
# ---------------------------------------------------------------------------
MODEL_NAME = "gpt-4o-mini"
INPUT_PRICE_PER_1M = 0.15
OUTPUT_PRICE_PER_1M = 0.60

DEFAULT_TIMEOUT_S = 60.0
DEFAULT_CONSISTENCY_RUNS = 3
DEFAULT_CONSISTENCY_TARGETS = 3  # up to 5

# ---------------------------------------------------------------------------
# Built-in test set: large-cap, small/mid-cap, and invalid tickers.
# Override with --tickers or --test-set.
# ---------------------------------------------------------------------------
DEFAULT_TEST_SET: list[dict[str, str]] = [
    {"ticker": "AAPL", "label": "large-cap"},
    {"ticker": "MSFT", "label": "large-cap"},
    {"ticker": "GOOGL", "label": "large-cap"},
    {"ticker": "AMZN", "label": "large-cap"},
    {"ticker": "NVDA", "label": "large-cap"},
    {"ticker": "META", "label": "large-cap"},
    {"ticker": "TSLA", "label": "large-cap"},
    {"ticker": "JPM", "label": "large-cap"},
    {"ticker": "V", "label": "large-cap"},
    {"ticker": "WMT", "label": "large-cap"},
    {"ticker": "XOM", "label": "large-cap"},
    {"ticker": "JNJ", "label": "large-cap"},
    {"ticker": "SOFI", "label": "small-cap"},
    {"ticker": "ROKU", "label": "small-cap"},
    {"ticker": "PLUG", "label": "small-cap"},
    {"ticker": "LCID", "label": "small-cap"},
    {"ticker": "NIO", "label": "small-cap"},
    {"ticker": "BYND", "label": "small-cap"},
    {"ticker": "TLRY", "label": "small-cap"},
    {"ticker": "SPCE", "label": "small-cap"},
    {"ticker": "ZZZZZZ", "label": "invalid"},
    {"ticker": "INVALID", "label": "invalid"},
    {"ticker": "NOTAREAL", "label": "invalid"},
    {"ticker": "12345", "label": "invalid"},
]

# Fields written to the per-request CSV (long free-text lives in the JSON dump).
CSV_COLUMNS = [
    "ticker", "label", "status", "outcome", "started_at", "ended_at",
    "elapsed_s", "error", "company_name", "current_price", "currency",
    "market_cap", "pe_ratio", "forward_pe", "eps", "dividend_yield", "beta",
    "week_52_high", "week_52_low", "sector", "industry", "period_return_pct",
    "volatility", "data_points", "llm_calls", "prompt_tokens",
    "completion_tokens", "total_tokens", "cost_usd", "num_strengths",
    "num_risks", "report_chars", "data_completeness", "errors",
]

# Chinese display mappings (internal keys stay English for logic / JSON).
STATUS_CN = {"success": "成功", "failure": "失败", "timeout": "超时"}

# 结果分类,顺序即报告里的展示顺序。「正确拒答」是一项正确行为,不是失败。
OUTCOME_ORDER = (
    "ok",
    "abstained_correctly",
    "hallucinated",
    "degraded",
    "failure",
    "timeout",
)
OUTCOME_CN = {
    "ok": "成功",
    "abstained_correctly": "正确拒答",
    "hallucinated": "幻觉",
    "degraded": "降级",
    "failure": "失败",
    "timeout": "超时",
}
LABEL_CN = {
    "large-cap": "大盘股",
    "mid-cap": "中盘股",
    "small-cap": "小盘股",
    "invalid": "无效",
}
NUMERIC_CN = {
    "current_price": "现价",
    "pe_ratio": "市盈率(P/E)",
    "forward_pe": "远期市盈率",
    "eps": "每股收益(EPS)",
    "dividend_yield": "股息率(小数)",
    "beta": "Beta",
    "period_return_pct": "3个月收益率(%)",
    "volatility": "年化波动率",
}
CSV_HEADERS_CN = {
    "ticker": "股票代码",
    "label": "类别",
    "status": "状态",
    "outcome": "结果分类",
    "started_at": "开始时间",
    "ended_at": "结束时间",
    "elapsed_s": "耗时(秒)",
    "error": "错误信息",
    "company_name": "公司名称",
    "current_price": "现价",
    "currency": "货币",
    "market_cap": "市值",
    "pe_ratio": "市盈率(P/E)",
    "forward_pe": "远期市盈率",
    "eps": "每股收益(EPS)",
    "dividend_yield": "股息率(小数)",
    "beta": "Beta",
    "week_52_high": "52周最高",
    "week_52_low": "52周最低",
    "sector": "板块",
    "industry": "行业",
    "period_return_pct": "3个月收益率(%)",
    "volatility": "年化波动率",
    "data_points": "数据点数",
    "llm_calls": "LLM调用次数",
    "prompt_tokens": "输入token",
    "completion_tokens": "输出token",
    "total_tokens": "总token",
    "cost_usd": "成本(USD)",
    "num_strengths": "优势项数量",
    "num_risks": "风险项数量",
    "report_chars": "报告字符数",
    "data_completeness": "数据完整性说明",
    "errors": "错误列表",
}


def _cn_status(status: str) -> str:
    return STATUS_CN.get(status, status)


def _cn_label(label: str) -> str:
    return LABEL_CN.get(label, label)


def _cn_yesno(value: bool) -> str:
    return "是" if value else "否"


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Token usage collection via LangChain callbacks
# ---------------------------------------------------------------------------
def _extract_token_usage(response: LLMResult) -> tuple[int, int, int]:
    """Pull (prompt, completion, total) tokens out of an LLMResult.

    Prefers the standardized ``usage_metadata`` on each generated AIMessage
    (langchain-core >= 0.3), falling back to OpenAI's legacy
    ``llm_output["token_usage"]``.
    """
    prompt = completion = total = 0

    for gen_list in response.generations or []:
        for gen in gen_list:
            meta = getattr(getattr(gen, "message", None), "usage_metadata", None)
            if meta:
                prompt += int(meta.get("input_tokens") or 0)
                completion += int(meta.get("output_tokens") or 0)
                total += int(meta.get("total_tokens") or 0)

    if not prompt and not completion:
        token_usage = (
            getattr(response, "llm_output", None) or {}
        ).get("token_usage") or {}
        prompt = int(token_usage.get("prompt_tokens") or 0)
        completion = int(token_usage.get("completion_tokens") or 0)
        total = int(token_usage.get("total_tokens") or 0)

    if not total:
        total = prompt + completion
    return prompt, completion, total


class TokenUsageCollector(BaseCallbackHandler):
    """Accumulate token usage across every LLM call in a single graph run."""

    def __init__(self) -> None:
        super().__init__()
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.calls: list[dict[str, int]] = []

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        prompt, completion, total = _extract_token_usage(response)
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.total_tokens += total
        self.calls.append(
            {
                "prompt_tokens": prompt,
                "completion_tokens": completion,
                "total_tokens": total,
            }
        )


def compute_cost(prompt_tokens: int, completion_tokens: int) -> float:
    return (
        (prompt_tokens / 1_000_000) * INPUT_PRICE_PER_1M
        + (completion_tokens / 1_000_000) * OUTPUT_PRICE_PER_1M
    )


# ---------------------------------------------------------------------------
# Running a single request
# ---------------------------------------------------------------------------
def _get_graph() -> Any:
    from graph.state import initial_state  # noqa: F401
    from graph.workflow import build_graph

    return build_graph()


def _invoke(graph: Any, ticker: str, collector: TokenUsageCollector) -> dict:
    from graph.state import initial_state

    initial = initial_state(ticker)
    config = {"callbacks": [collector]}
    return graph.invoke(initial, config)


def _attach_state(record: dict, state: dict) -> None:
    """Copy the fields this eval cares about out of the final graph state."""
    sp = state.get("stock_price") or {}
    fm = state.get("financial_metrics") or {}
    hd = state.get("historical_data") or {}
    record.update(
        {
            "company_name": state.get("company_name") or fm.get("company_name", ""),
            "current_price": sp.get("current_price", 0.0),
            "currency": sp.get("currency", ""),
            # 快照/历史里其余字段也存下来:接地检查需要它们作为证据。少了
            # change_pct 就会把报告里的“日涨跌幅”误判成无出处;少了 period_high/
            # period_low 同理。缺失的证据只会变成假阳性,不会变成假阴性。
            "previous_close": sp.get("previous_close", 0.0),
            "change_pct": sp.get("change_pct", 0.0),
            "day_high": sp.get("day_high", 0.0),
            "day_low": sp.get("day_low", 0.0),
            "volume": sp.get("volume", 0),
            "market_cap": fm.get("market_cap", 0),
            "pe_ratio": fm.get("pe_ratio", 0.0),
            "forward_pe": fm.get("forward_pe", 0.0),
            "eps": fm.get("eps", 0.0),
            "dividend_yield": fm.get("dividend_yield", 0.0),
            "beta": fm.get("beta", 0.0),
            "week_52_high": fm.get("52_week_high", 0.0),
            "week_52_low": fm.get("52_week_low", 0.0),
            "sector": fm.get("sector", ""),
            "industry": fm.get("industry", ""),
            "period_return_pct": hd.get("period_return_pct", 0.0),
            "period_start_price": hd.get("period_start_price", 0.0),
            "period_end_price": hd.get("period_end_price", 0.0),
            "period_high": hd.get("period_high", 0.0),
            "period_low": hd.get("period_low", 0.0),
            "volatility": hd.get("volatility", 0.0),
            "data_points": hd.get("data_points", 0),
            "num_strengths": len(state.get("strengths") or []),
            "num_risks": len(state.get("risks") or []),
            "strengths": state.get("strengths") or [],
            "risks": state.get("risks") or [],
            "analyst_summary": state.get("analyst_summary", ""),
            "data_completeness": state.get("data_completeness", ""),
            "report_chars": len(state.get("final_report") or ""),
            "errors": state.get("errors") or [],
            # Retained verbatim so content-level checks (fact grounding, section
            # completeness) can run against an archived run instead of forcing a
            # re-run — and so a finding can be re-inspected months later.
            "final_report": state.get("final_report") or "",
            "news_findings": state.get("news_findings") or [],
            "market_context": state.get("market_context", ""),
        }
    )


def _attach_usage(record: dict, collector: TokenUsageCollector) -> None:
    record["llm_calls"] = len(collector.calls)
    record["prompt_tokens"] = collector.prompt_tokens
    record["completion_tokens"] = collector.completion_tokens
    record["total_tokens"] = collector.total_tokens
    record["cost_usd"] = round(
        compute_cost(collector.prompt_tokens, collector.completion_tokens), 6
    )


# ---------------------------------------------------------------------------
# Outcome classification
# ---------------------------------------------------------------------------
def _has_financial_data(record: dict) -> bool:
    """Mirror the Analysis agent's gate: price or market cap is the ground truth.

    Kept in sync deliberately — if the agent's rule changes, this must too, and
    the numbers reported here would otherwise silently stop meaning anything.
    """
    return bool(record.get("current_price")) or bool(record.get("market_cap"))


def _has_analysis(record: dict) -> bool:
    return bool(record.get("num_strengths")) or bool(record.get("num_risks"))


def classify_outcome(record: dict) -> str:
    """Map one request to exactly one outcome (see the module docstring).

    The two axes are independent and both matter:

        financial data?  analysis?   outcome
        ---------------- --------   --------------------
        yes              yes        ok
        no               no         abstained_correctly
        no               yes        hallucinated          <-- the bug we look for
        yes              no         degraded
    """
    if record["status"] != "success":
        return record["status"]

    funded = _has_financial_data(record)
    analysed = _has_analysis(record)
    if funded and analysed:
        return "ok"
    if not funded and not analysed:
        return "abstained_correctly"
    if not funded and analysed:
        return "hallucinated"
    return "degraded"


def run_one(graph: Any, entry: dict, timeout_s: float) -> dict:
    """Run a single ticker through the graph, capturing everything we need."""
    ticker = entry["ticker"]
    collector = TokenUsageCollector()
    record: dict[str, Any] = {
        "ticker": ticker,
        "label": entry.get("label", ""),
        "status": "success",
        "started_at": _iso_now(),
        "ended_at": "",
        "elapsed_s": None,
        "error": "",
    }

    wall_start = time.time()
    try:
        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(_invoke, graph, ticker, collector)
        try:
            state = future.result(timeout=timeout_s)
        except TimeoutError:
            record["status"] = "timeout"
            record["error"] = f"exceeded {timeout_s}s timeout"
            record["elapsed_s"] = round(time.time() - wall_start, 3)
            record["ended_at"] = _iso_now()
            _attach_usage(record, collector)
            executor.shutdown(wait=False, cancel_futures=True)
            return record
        executor.shutdown(wait=False)
        record["elapsed_s"] = round(time.time() - wall_start, 3)
        record["ended_at"] = _iso_now()
        _attach_state(record, state)
        _attach_usage(record, collector)
        return record
    except Exception as exc:  # noqa: BLE001
        record["status"] = "failure"
        record["error"] = f"{exc!r}\n{traceback.format_exc()}"
        record["elapsed_s"] = round(time.time() - wall_start, 3)
        record["ended_at"] = _iso_now()
        _attach_usage(record, collector)
        return record


# ---------------------------------------------------------------------------
# Consistency (repeat runs of representative tickers)
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Text similarity — two separate measures, deliberately
# ---------------------------------------------------------------------------
# The previous harness compared *raw sentence sets* with Jaccard set-overlap.
# That reports ~0.00 even when two runs state the identical fact in slightly
# different words (e.g. "...beating by 6.88%, indicating robust profitability"
# vs "...beating by 6.88%, indicating effective cost control"), so it measured
# the tokenizer rather than the model. Two measures replace it, answering two
# different questions:
#
#   * 词面相似度 (best-match, lexical) — 措辞/取舍稳不稳
#   * 事实重合度 (numeric-set Jaccard) — 引用的事实稳不稳   <-- 这才是重点
#
# Both are lexical by construction; neither is claimed to be semantic. A truly
# semantic score needs embeddings or a calibrated judge.
_STOPWORDS = frozenset(
    """a an the of and or to in for on with by is are was were be been being
    from as at that this these those it its their there here has have had
    will would can could should not no than then also more most other such
    only own same so too very s t""".split()
)

_WORD_RE = re.compile(r"[a-z0-9]+")
# 数字断言:要求带小数点或百分号,以此排除年份(2026)、季度(Q3)这类噪声
_FACT_RE = re.compile(r"\d+\.\d+|\d+\s*%")


def _content_token_list(text: str) -> list[str]:
    """保序的实词序列。用 list 而不是 set,因为顺序本身也是相似度的信号。"""
    return [
        tok
        for tok in _WORD_RE.findall((text or "").lower())
        if len(tok) > 1 and tok not in _STOPWORDS
    ]


def _content_tokens(text: str) -> set[str]:
    return set(_content_token_list(text))


def _jaccard_sets(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def lexical_similarity(a: str, b: str) -> float:
    """单句相似度:实词集合 Jaccard 与实词序列匹配比的较大者。

    序列匹配以 **token** 为元素,不是字符。按字符比较任何两段英文都有 ~0.3 的
    地板分(共享字母、冠词、词尾),会把「毫无关系的两句话」评成中度相似——
    那是另一个变相的假信号,只是这次是假高。
    """
    tokens_a = _content_token_list(a)
    tokens_b = _content_token_list(b)
    token_j = _jaccard_sets(set(tokens_a), set(tokens_b))
    seq = SequenceMatcher(None, tokens_a, tokens_b).ratio()
    return max(token_j, seq)


def _best_match(a_items: list[str], b_items: list[str]) -> float | None:
    """每条 a 在 b 中找最相似的一条取均值,再反向做一次取总平均。

    这是对 set-overlap Jaccard 的关键修正:句子不是集合元素,措辞一变就
    整条对不上,而最佳匹配让「同一事实换个说法」仍然得分。双向平均是为了
    不惩罚输出条数更多的那一次。
    """
    if not a_items or not b_items:
        return None
    forward = [max(lexical_similarity(x, y) for y in b_items) for x in a_items]
    backward = [max(lexical_similarity(y, x) for x in a_items) for y in b_items]
    return (mean(forward) + mean(backward)) / 2


def _facts(items: list[str]) -> set[str]:
    """抽出这批论点里引用的数字断言。"""
    out: set[str] = set()
    for item in items:
        out.update(m.replace(" ", "") for m in _FACT_RE.findall(item or ""))
    return out


def _pairwise_mean(items: list[Any], sim) -> float | None:
    scores = [
        score
        for i in range(len(items))
        for j in range(i + 1, len(items))
        if (score := sim(items[i], items[j])) is not None
    ]
    return mean(scores) if scores else None


def compute_consistency(runs: list[dict]) -> dict:
    """Compare the key conclusions/numbers across repeated runs of one ticker."""
    successful = [r for r in runs if r["status"] == "success"]
    if len(successful) < 2:
        return {"note": "成功运行不足 2 次，无法计算一致性"}

    strengths = [r["strengths"] for r in successful]
    risks = [r["risks"] for r in successful]
    summaries = [r["analyst_summary"] for r in successful]
    completeness = [r["data_completeness"] for r in successful]

    numeric_keys = [
        "current_price", "pe_ratio", "forward_pe", "eps", "dividend_yield",
        "beta", "period_return_pct", "volatility",
    ]
    numerics: dict[str, dict] = {}
    for key in numeric_keys:
        vals = [r[key] for r in successful if isinstance(r.get(key), (int, float))]
        if vals:
            mn, mx = min(vals), max(vals)
            mean_v = mean(vals)
            spread = ((mx - mn) / abs(mean_v)) * 100.0 if mean_v else 0.0
            numerics[key] = {
                "values": vals,
                "min": mn,
                "max": mx,
                "mean": mean_v,
                "spread_pct": round(spread, 4),
            }

    # 空串约定:两次运行都没有摘要时,相似度不是 1.00 —— 是没有东西可比。
    has_summary = any((s or "").strip() for s in summaries)

    facts_per_run = [_facts(s + r) for s, r in zip(strengths, risks)]
    all_facts: set[str] = set().union(*facts_per_run) if facts_per_run else set()

    # 空集不能算「完全重合」。Jaccard 对两个空集返回 1.0,直接用会在「一次运行
    # 压根没引用任何数字」时报出一个毫无信息量的 1.00 —— 和 N/A 必须区分开。
    facts_overlap = (
        _pairwise_mean(facts_per_run, _jaccard_sets) if all_facts else None
    )

    return {
        "runs": len(successful),
        # 措辞层:同一事实换个说法也应该高分(词面,最佳匹配)
        "strengths_similarity": _pairwise_mean(strengths, _best_match),
        "risks_similarity": _pairwise_mean(risks, _best_match),
        "strengths_identical": all(s == strengths[0] for s in strengths[1:]),
        "risks_identical": all(r == risks[0] for r in risks[1:]),
        "analyst_summary_similarity": (
            _pairwise_mean(summaries, lexical_similarity) if has_summary else None
        ),
        # 事实层:引用到的数字集合重合度——这才是稳定性的主指标
        "facts_overlap": facts_overlap,
        "facts_distinct": len(all_facts),
        "facts": sorted(all_facts),
        "data_completeness_values": sorted(set(completeness)),
        "data_completeness_identical": len(set(completeness)) == 1,
        "numerics": numerics,
    }


def resolve_consistency_targets(
    test_set: list[dict], explicit: list[str] | None, n: int
) -> list[str]:
    if explicit:
        return explicit[:n]
    picked: list[str] = []
    # Prefer one representative from each category, in order.
    for category in ("large", "small", "invalid"):
        for e in test_set:
            label = (e.get("label") or "").lower()
            if category == "large" and "large" in label and e["ticker"] not in picked:
                picked.append(e["ticker"])
                break
            if category == "small" and "small" in label and e["ticker"] not in picked:
                picked.append(e["ticker"])
                break
            if category == "invalid" and "invalid" in label and e["ticker"] not in picked:
                picked.append(e["ticker"])
                break
    for e in test_set:
        if len(picked) >= n:
            break
        if e["ticker"] not in picked:
            picked.append(e["ticker"])
    return picked[:n]


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
def _percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    k = (len(ordered) - 1) * (p / 100.0)
    lo = int(k)
    hi = lo + 1 if lo + 1 < len(ordered) else lo
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


def _cn_outcome(outcome: str) -> str:
    return OUTCOME_CN.get(outcome, outcome)


def build_summary(
    records: list[dict],
    consistency: list[dict],
    fixtures_info: dict | None = None,
) -> dict:
    total = len(records)
    by_status = {"success": 0, "failure": 0, "timeout": 0}
    for r in records:
        by_status[r["status"]] = by_status.get(r["status"], 0) + 1

    # Derived at report time on purpose, from the raw observed fields rather than
    # from any stored classification — so an archived run can be re-scored when
    # the rule itself changes, without paying for another LLM run. Storing a
    # derived value and then trusting it is how the old summary_*.md ended up
    # disagreeing with its own results_*.json.
    outcomes: dict[str, int] = {name: 0 for name in OUTCOME_ORDER}
    for r in records:
        key = classify_outcome(r)
        outcomes[key] = outcomes.get(key, 0) + 1

    # 两个**分母不同**的判定,分开统计——把它们合进一个百分比正是旧版
    # harness 的 bug:那会让「正确拒答无效代码」看起来像系统失败。
    succeeded = [r for r in records if r["status"] == "success"]
    with_data = [r for r in succeeded if _has_financial_data(r)]
    without_data = [r for r in succeeded if not _has_financial_data(r)]
    analysed = [r for r in with_data if _has_analysis(r)]
    abstained = [r for r in without_data if not _has_analysis(r)]
    hallucinated = [r for r in without_data if _has_analysis(r)]

    # Latency over completed requests (success + failure); timeouts are excluded
    # because they were cut off at the threshold, not a real elapsed time.
    completed = [r for r in records if r["status"] != "timeout"]
    latencies = [r["elapsed_s"] for r in completed if r["elapsed_s"] is not None]

    with_usage = [r for r in records if r.get("llm_calls", 0) > 0]
    avg = {
        "prompt": mean([r["prompt_tokens"] for r in with_usage]) if with_usage else 0.0,
        "completion": mean([r["completion_tokens"] for r in with_usage]) if with_usage else 0.0,
        "total": mean([r["total_tokens"] for r in with_usage]) if with_usage else 0.0,
        "cost_per_llm_request": mean([r["cost_usd"] for r in with_usage]) if with_usage else 0.0,
        "cost_per_request": mean([r["cost_usd"] for r in records]) if records else 0.0,
    }

    # 「需要关注」与「正确拒答」分开列。把正确行为归档到 problem 里,就是同一类
    # 范畴错误——只不过这次错在表格命名上。
    problem_rows = [
        r
        for r in records
        if classify_outcome(r) not in ("ok", "abstained_correctly")
    ]
    abstained_rows = [r for r in records if classify_outcome(r) == "abstained_correctly"]

    # 数字接地(契约层):报告里每个数字能不能对回输入。纯函数,只看归档字段,
    # 所以可以直接对历史 run 重算。
    grounding_runs = [(r, grounding.check_grounding(r)) for r in records]
    grounding_totals = {"total": 0, "financial": 0, "news_only": 0, "ungrounded": 0}
    grounding_rows: list[dict] = []
    ungrounded_items: list[dict] = []
    news_only_items: list[dict] = []
    for record, result in grounding_runs:
        if result["total"] == 0:
            continue
        for key in grounding_totals:
            grounding_totals[key] += result[key]
        grounding_rows.append(
            {
                "ticker": record["ticker"],
                "total": result["total"],
                "financial": result["financial"],
                "news_only": result["news_only"],
                "ungrounded": result["ungrounded"],
            }
        )
        ungrounded_items.extend(
            {"ticker": record["ticker"], **item} for item in result["ungrounded_items"]
        )
        news_only_items.extend(
            {"ticker": record["ticker"], **item} for item in result["news_only_items"]
        )

    return {
        "generated_at": _iso_now(),
        # 输入来源必须随结果一起存档,否则这份数字无法复现或对比
        "fixtures": fixtures_info or {"mode": "live", "path": None},
        "total": total,
        "success": by_status["success"],
        "failure": by_status["failure"],
        "timeout": by_status["timeout"],
        "process_success_rate": (by_status["success"] / total) if total else 0.0,
        "outcomes": outcomes,
        "analysis": {
            "requests_with_financial_data": len(with_data),
            "produced_analysis": len(analysed),
            "no_analysis": len(with_data) - len(analysed),
        },
        "abstention": {
            "requests_without_financial_data": len(without_data),
            "abstained_correctly": len(abstained),
            "hallucinated": len(hallucinated),
        },
        "latency": {
            "n": len(latencies),
            "p50": _percentile(latencies, 50),
            "p95": _percentile(latencies, 95),
            "mean": mean(latencies) if latencies else None,
            "min": min(latencies) if latencies else None,
            "max": max(latencies) if latencies else None,
        },
        "tokens": {
            "llm_requests": len(with_usage),
            "avg_prompt": round(avg["prompt"], 1),
            "avg_completion": round(avg["completion"], 1),
            "avg_total": round(avg["total"], 1),
        },
        "cost": {
            "avg_per_llm_request": round(avg["cost_per_llm_request"], 6),
            "avg_per_request": round(avg["cost_per_request"], 6),
        },
        "problem_rows": problem_rows,
        "abstained_rows": abstained_rows,
        "grounding": {
            "totals": grounding_totals,
            "rows": grounding_rows,
            "ungrounded_items": ungrounded_items,
            "news_only_unique": len(news_only_items),
        },
    }


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def _fmt(v: Any, digits: int = 2) -> str:
    if v is None:
        return "N/A"
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


def _fmt_sim(value: float | None) -> str:
    """一致性相似度。None = 样本不足无法计算,必须与 0.0 区分开。"""
    return "N/A" if value is None else f"{value:.2f}"


def _fmt_dy(record: dict) -> str:
    """Render dividend_yield as the decimal stored by the agent + a % hint."""
    dy = record.get("dividend_yield")
    if dy is None:
        return "N/A"
    try:
        return f"{float(dy):.4f} (≈{float(dy) * 100:.2f}%)"
    except (TypeError, ValueError):
        return "N/A"


def _financial_block(record: dict) -> str:
    lines = [
        f"    现价={record.get('current_price')} {record.get('currency') or ''}".rstrip(),
        f"    市值={record.get('market_cap'):,}",
        f"    市盈率(P/E)={_fmt(record.get('pe_ratio'))}  远期市盈率={_fmt(record.get('forward_pe'))}  每股收益(EPS)={_fmt(record.get('eps'))}",
        f"    股息率={_fmt_dy(record)}  （agent 存的是小数 = yfinance dividendYield/100）",
        f"    Beta={_fmt(record.get('beta'))}  52周最高/最低={_fmt(record.get('week_52_high'))} / {_fmt(record.get('week_52_low'))}",
        f"    板块={record.get('sector') or 'N/A'}  行业={record.get('industry') or 'N/A'}",
        f"    3个月收益率={_fmt(record.get('period_return_pct'))}%  年化波动率={_fmt(record.get('volatility'), 4)}  数据点数={record.get('data_points')}",
    ]
    return "\n".join(lines)


def print_request_result(index: int, total: int, record: dict) -> None:
    outcome = classify_outcome(record)
    status = OUTCOME_CN.get(outcome, outcome)
    elapsed = record["elapsed_s"] if record["elapsed_s"] is not None else "?"
    print(f"[{index}/{total}] {record['ticker']:>8}  {status:<12}  {elapsed:>8}s")
    if record["error"]:
        first_line = record["error"].splitlines()[0]
        print(f"          错误: {first_line}")
    if record["status"] == "success":
        print("  Financial Agent 关键数值:")
        print(_financial_block(record))
        print(f"  LLM 调用次数={record['llm_calls']}  "
              f"输入token={record['prompt_tokens']}  "
              f"输出token={record['completion_tokens']}  "
              f"总token={record['total_tokens']}  成本=${record['cost_usd']:.6f}")
        if record["errors"]:
            print(f"  错误({len(record['errors'])} 条): " + " | ".join(record["errors"][:3]))
        report_grounding = grounding.check_grounding(record)
        if report_grounding["total"]:
            print(
                f"  数字接地: 报告数字={report_grounding['total']}  "
                f"财务出处={report_grounding['financial']}  "
                f"仅新闻={report_grounding['news_only']}  "
                f"无出处={report_grounding['ungrounded']}"
            )
    print()


def render_markdown(summary: dict, consistency: list[dict]) -> str:
    s = summary
    lat = s["latency"]
    tok = s["tokens"]
    cost = s["cost"]
    lines: list[str] = []

    lines.append("# InvestIQ 评测汇总")
    lines.append("")
    lines.append(f"- 生成时间: {s['generated_at']}")
    lines.append(f"- 模型: `{MODEL_NAME}`")
    fx = s.get("fixtures") or {"mode": "live", "path": None}
    if fx.get("mode") == "replay":
        lines.append(f"- 输入来源: **回放冻结磁带** `{fx.get('path')}`（输入逐字节一致，结果可复现）")
    elif fx.get("mode") == "record":
        lines.append(f"- 输入来源: 实时拉取并录制到 `{fx.get('path')}`（本次结果本身不可复现）")
    else:
        lines.append(
            "- 输入来源: ⚠️ **实时拉取，未冻结**——行情与新闻随时间变化，"
            "本次结果不可复现、不可与其它运行直接对比"
        )
    lines.append(
        f"- 定价: 输入 ${INPUT_PRICE_PER_1M}/1M tokens，输出 "
        f"${OUTPUT_PRICE_PER_1M}/1M tokens（GPT-4o-mini，已按 OpenAI 官方定价页核实）"
    )
    lines.append("")
    lines.append("## 总览")
    lines.append("")
    lines.append(f"- 总请求数: {s['total']}")
    lines.append(
        f"- 流程完成率（正常返回，不含异常/超时）: {s['process_success_rate'] * 100:.1f}%"
    )
    lines.append("")
    lines.append("## 结果分类")
    lines.append("")
    lines.append("每一条请求只归入一类，所以失败无法藏在成功计数里。")
    lines.append("")
    lines.append("| 结果 | 数量 | 含义 |")
    lines.append("| --- | --- | --- |")
    o = s["outcomes"]
    lines.append(f"| 成功 | {o['ok']} | 有财务数据且产出了分析 |")
    lines.append(
        f"| 正确拒答 | {o['abstained_correctly']} | 无财务数据且未编造分析（**正确行为，不算失败**） |"
    )
    lines.append(f"| 幻觉 | {o['hallucinated']} | 无财务数据却生成了分析 —— 应为 0 |")
    lines.append(f"| 降级 | {o['degraded']} | 有财务数据但未产出分析 |")
    lines.append(f"| 失败 | {o['failure']} | 抛出异常 |")
    lines.append(f"| 超时 | {o['timeout']} | 超过阈值 |")
    lines.append("")

    an = s["analysis"]
    ab = s["abstention"]
    lines.append("## 有效请求（有财务数据）")
    lines.append("")
    lines.append("| 指标 | 数值 |")
    lines.append("| --- | --- |")
    lines.append(f"| 有财务数据的请求 | {an['requests_with_financial_data']} |")
    lines.append(f"| 其中产出分析 | {an['produced_analysis']} |")
    lines.append(f"| 其中未产出分析（降级） | {an['no_analysis']} |")
    if an["requests_with_financial_data"]:
        lines.append(
            f"| 分析产出率（本表分母） | "
            f"{an['produced_analysis'] / an['requests_with_financial_data'] * 100:.1f}% |"
        )
    lines.append("")
    lines.append("## 无效输入拒答（无财务数据）")
    lines.append("")
    lines.append("| 指标 | 数值 |")
    lines.append("| --- | --- |")
    lines.append(f"| 无财务数据的请求 | {ab['requests_without_financial_data']} |")
    lines.append(f"| 正确拒答 | {ab['abstained_correctly']} |")
    lines.append(f"| **幻觉（必须为 0）** | **{ab['hallucinated']}** |")
    if ab["requests_without_financial_data"]:
        lines.append(
            f"| 拒答正确率（本表分母） | "
            f"{ab['abstained_correctly'] / ab['requests_without_financial_data'] * 100:.1f}% |"
        )
    lines.append(
        "| 说明 | 数据源故障与无效代码在指标上不可区分，两者都归入「无财务数据」 |"
    )
    lines.append("")

    g = s.get("grounding") or {}
    gtot = g.get("totals") or {}
    if gtot.get("total"):
        lines.append("## 数字接地检查（契约层）")
        lines.append("")
        lines.append(
            "报告里出现的每个数字，对回**原始输入**：yfinance 字段 + 新闻正文。"
            "刻意**不把 Research 的 LLM 摘要当作出处**——那等于让模型给自己作证，"
            "幻觉可以经过一层摘要洗白。"
        )
        lines.append("")
        lines.append("| 分类 | 数量 | 含义 |")
        lines.append("| --- | --- | --- |")
        lines.append(
            f"| 有财务出处 | {gtot['financial']} | 在 yfinance 字段里找到（最硬的出处）|"
        )
        lines.append(
            f"| 仅新闻出处 | {gtot['news_only']} | 只在新闻正文里出现——"
            "**口径风险区**：数字有出处，但季度/年度之类口径可能与财务字段冲突 |"
        )
        lines.append(
            f"| **无出处** | **{gtot['ungrounded']}** | 两处都找不到——可能是编造，"
            "也可能是年份或单位换算 |"
        )
        lines.append(
            f"| 报告数字合计 | {gtot['total']} | 这三个数只是计数，**不给百分比**："
            "分母取决于模型写了多少字，一个比例会诱导人做无意义的横向比较 |"
        )
        lines.append("")
        if g.get("ungrounded_items"):
            lines.append("### 无出处的数字（需人工分诊）")
            lines.append("")
            lines.append("| 股票 | 数字 | 出现 | 上下文 |")
            lines.append("| --- | --- | --- | --- |")
            for item in g["ungrounded_items"]:
                mark = "（年份?）" if item.get("year_like") else ""
                context = item["context"].replace("|", "\\|")
                lines.append(
                    f"| {item['ticker']} | {item['written']}{mark} | "
                    f"{item['occurrences']} | {context} |"
                )
            lines.append("")

    lines.append("## 耗时（已完成请求，不含超时）")
    lines.append("")
    lines.append(f"- 样本数 = {lat['n']}")
    lines.append(f"- p50（中位数） = {_fmt(lat['p50'])} 秒")
    lines.append(f"- p95 = {_fmt(lat['p95'])} 秒")
    lines.append(f"- 均值 = {_fmt(lat['mean'])} 秒")
    lines.append(f"- 最小 / 最大 = {_fmt(lat['min'])} 秒 / {_fmt(lat['max'])} 秒")
    lines.append("")
    lines.append("## Token 用量与成本")
    lines.append("")
    lines.append("| 指标 | 数值 |")
    lines.append("| --- | --- |")
    lines.append(f"| 有 LLM 调用的请求数 | {tok['llm_requests']} |")
    lines.append(f"| 平均输入 token / 请求 | {tok['avg_prompt']} |")
    lines.append(f"| 平均输出 token / 请求 | {tok['avg_completion']} |")
    lines.append(f"| 平均总 token / 请求 | {tok['avg_total']} |")
    lines.append(f"| 平均单次成本（有 LLM 调用的请求） | ${cost['avg_per_llm_request']:.6f} |")
    lines.append(f"| 平均单次成本（全部请求） | ${cost['avg_per_request']:.6f} |")
    lines.append("")

    if consistency:
        lines.append("## 一致性与事实稳定性（重复运行）")
        lines.append("")
        lines.append(
            "回放同一份冻结输入、重复跑同一 ticker,区分两件事:**措辞**稳不稳、"
            "**引用的事实**稳不稳。两个指标都是词面/集合层面的度量,不是语义度量。"
        )
        lines.append("")
        for item in consistency:
            ticker = item["ticker"]
            m = item["metrics"]
            lines.append(f"### {ticker}")
            lines.append("")
            if "note" in m:
                lines.append(f"- {m['note']}")
            else:
                lines.append(f"- 运行次数: {m['runs']}")
                lines.append(
                    f"- 优势项词面相似度（最佳匹配）: {_fmt_sim(m['strengths_similarity'])}"
                    f"  （逐字一致: {_cn_yesno(m['strengths_identical'])}）"
                )
                lines.append(
                    f"- 风险项词面相似度（最佳匹配）: {_fmt_sim(m['risks_similarity'])}"
                    f"  （逐字一致: {_cn_yesno(m['risks_identical'])}）"
                )
                lines.append(
                    f"- 分析师摘要词面相似度: {_fmt_sim(m['analyst_summary_similarity'])}"
                )
                lines.append(
                    f"- **事实重合度（数字集合 Jaccard）**: {_fmt_sim(m['facts_overlap'])}"
                    f"  （共 {m['facts_distinct']} 个不同数字）"
                )
                lines.append(
                    f"- 数据完整性描述一致: {_cn_yesno(m['data_completeness_identical'])} "
                    f"（取值: {m['data_completeness_values']}）"
                )
                lines.append("")
                lines.append("| 指标 | 最小 | 最大 | 均值 | 偏差% |")
                lines.append("| --- | --- | --- | --- | --- |")
                for key, v in m["numerics"].items():
                    digits = 4 if key in ("dividend_yield", "volatility") else 2
                    label = NUMERIC_CN.get(key, key)
                    lines.append(
                        f"| {label} | {_fmt(v['min'], digits)} | {_fmt(v['max'], digits)} | "
                        f"{_fmt(v['mean'], digits)} | {v['spread_pct']}% |"
                    )
            lines.append("")

    if s["problem_rows"]:
        lines.append("## 需要关注的结果明细")
        lines.append("")
        lines.append("| 股票代码 | 结果 | 错误/说明 |")
        lines.append("| --- | --- | --- |")
        for r in s["problem_rows"]:
            status = OUTCOME_CN.get(classify_outcome(r), r["status"])
            err_lines = (r["error"] or "").splitlines()
            err = err_lines[0] if err_lines else ""
            if not err and r["errors"]:
                err = " | ".join(r["errors"])
            err = (err or "").replace("|", "\\|")
            lines.append(f"| {r['ticker']} | {status} | {err} |")
        lines.append("")

    if s.get("abstained_rows"):
        lines.append("## 已正确拒答的请求")
        lines.append("")
        lines.append(
            "这些请求没有拿到财务数据,系统选择**不生成分析**。列出来是为了可审计"
            "(看得到哪些代码被拒了),它们不是问题。"
        )
        lines.append("")
        lines.append("| 股票代码 | 抓到的错误 |")
        lines.append("| --- | --- |")
        for r in s["abstained_rows"]:
            err = " | ".join(r.get("errors") or []) or (r["error"] or "")
            lines.append(f"| {r['ticker']} | {err.replace('|', chr(92) + '|')} |")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# File output
# ---------------------------------------------------------------------------
def _csv_cells(record: dict) -> list[str]:
    cells: list[str] = []
    for col in CSV_COLUMNS:
        val = record.get(col, "")
        if col == "status":
            val = _cn_status(val)
        elif col == "outcome":
            val = _cn_outcome(classify_outcome(record))
        elif col == "label":
            val = _cn_label(val or "")
        elif isinstance(val, list):
            val = " | ".join(str(x) for x in val)
        elif val is None:
            val = ""
        cells.append(str(val))
    return cells


def write_outputs(
    out_dir: Path, records: list[dict], consistency: list[dict],
    summary: dict, markdown: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    csv_path = out_dir / f"results_{stamp}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow([CSV_HEADERS_CN.get(c, c) for c in CSV_COLUMNS])
        for r in records:
            writer.writerow(_csv_cells(r))

    json_path = out_dir / f"results_{stamp}.json"
    json_path.write_text(
        json.dumps(
            {
                "summary": summary,
                # outcome / grounding 都在导出时现算,不在 run 时存盘 —— 派生字段
                # 只保留一份真相,改指标定义后旧 run 能直接用新口径重新评分。
                "records": [
                    {
                        **r,
                        "outcome": classify_outcome(r),
                        "grounding": grounding.check_grounding(r),
                    }
                    for r in records
                ],
                "consistency": consistency,
            },
            indent=2,
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )

    # 三个产物共用同一个 stem,避免出现「json 说是降级 4、另一个 md 说是 0」这种
    # 同一次运行两份口径的历史遗留。
    md_path = out_dir / f"results_{stamp}.md"
    md_path.write_text(markdown, encoding="utf-8")

    print(f"\n已写入: {csv_path}")
    print(f"已写入: {json_path}")
    print(f"已写入: {md_path}")


# ---------------------------------------------------------------------------
# Test-set loading
# ---------------------------------------------------------------------------
def load_test_set(args: argparse.Namespace) -> list[dict]:
    if args.test_set:
        path = Path(args.test_set)
        raw = json.loads(path.read_text())
        return _normalize_test_set(raw)
    if args.tickers:
        return [{"ticker": t.strip().upper(), "label": ""} for t in args.tickers.split(",") if t.strip()]
    return list(DEFAULT_TEST_SET)


def _normalize_test_set(raw: Any) -> list[dict]:
    out: list[dict] = []
    for item in raw:
        if isinstance(item, str):
            out.append({"ticker": item.strip().upper(), "label": ""})
        elif isinstance(item, dict):
            out.append(
                {
                    "ticker": str(item["ticker"]).strip().upper(),
                    "label": item.get("label", ""),
                }
            )
        else:
            raise ValueError(f"Unsupported test-set entry: {item!r}")
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="InvestIQ evaluation harness (external driver; no core logic changes)."
    )
    p.add_argument("--tickers", help="Comma-separated tickers, e.g. 'AAPL,MSFT,ZZZZZZ'")
    p.add_argument("--test-set", help="Path to a JSON file: list of tickers or [{ticker,label}]")
    p.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S,
                   help=f"Per-request timeout in seconds (default {DEFAULT_TIMEOUT_S})")
    p.add_argument("--consistency-tickers",
                   help="Comma-separated tickers for the repeat-run consistency test")
    p.add_argument("--consistency-runs", type=int, default=DEFAULT_CONSISTENCY_RUNS,
                   help=f"Runs per consistency ticker (default {DEFAULT_CONSISTENCY_RUNS})")
    p.add_argument("--consistency-n", type=int, default=DEFAULT_CONSISTENCY_TARGETS,
                   help=f"Number of consistency tickers, max 5 (default {DEFAULT_CONSISTENCY_TARGETS})")
    p.add_argument("--no-consistency", action="store_true",
                   help="Skip the repeat-run consistency test")
    p.add_argument("--out-dir", default=str(REPO_ROOT / "eval" / "results"),
                   help="Directory for CSV/JSON/Markdown output")
    p.add_argument(
        "--fixtures",
        default=str(REPO_ROOT / "eval" / "cassettes" / "frozen.json"),
        help="Path to the frozen-input cassette (JSON)",
    )
    p.add_argument(
        "--record", action="store_true",
        help="Run live and record every yfinance/Tavily response into --fixtures",
    )
    p.add_argument(
        "--replay", action="store_true",
        help="Serve inputs from --fixtures instead of the network (needs a recorded cassette)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    test_set = load_test_set(args)

    if not test_set:
        print("测试集为空，无任务可执行。", file=sys.stderr)
        return 1

    if args.record and args.replay:
        print("--record 与 --replay 互斥。", file=sys.stderr)
        return 1

    # --- 冻结 / 回放输入 ---------------------------------------------------
    cassette: fixtures.Cassette | None = None
    fixtures_info: dict[str, Any] = {"mode": "live", "path": None}
    if args.record or args.replay:
        cassette_path = Path(args.fixtures)
        if args.replay:
            try:
                cassette = fixtures.Cassette.load(cassette_path)
            except (FileNotFoundError, ValueError) as exc:
                print(f"读取磁带失败: {exc}", file=sys.stderr)
                return 2
            missing = cassette.missing_for([e["ticker"] for e in test_set])
            if missing:
                print(
                    f"磁带 {cassette_path} 缺少 {len(missing)} 个 ticker 的输入: {missing}\n"
                    "回放模式不会静默回退到网络——那会无声地破坏复现性。"
                    "请先 --record 补齐,或用 --tickers 只跑已录制的部分。",
                    file=sys.stderr,
                )
                return 2
            fixtures.install(cassette, "replay")
            fixtures_info = {
                "mode": "replay",
                "path": str(cassette_path),
                "recorded_at": cassette.recorded_at,
            }
        else:
            cassette = fixtures.Cassette(cassette_path)
            fixtures.install(cassette, "record")
            fixtures_info = {"mode": "record", "path": str(cassette_path)}

    print("=" * 78)
    print(" InvestIQ 评测脚本")
    print(f" 模型={MODEL_NAME}  定价=输入 ${INPUT_PRICE_PER_1M}/1M、"
          f"输出 ${OUTPUT_PRICE_PER_1M}/1M  超时阈值={args.timeout}s")
    print(f" 测试集: {len(test_set)} 个 ticker")
    if fixtures_info["mode"] == "live":
        print(" 输入来源: 实时（未冻结）—— 行情与新闻会变,结果不可复现")
    else:
        print(f" 输入来源: {fixtures_info['mode']} -> {fixtures_info['path']}")
    print("=" * 78)

    print("\n正在构建图...")
    try:
        graph = _get_graph()
    except Exception as exc:  # noqa: BLE001
        print(f"构建图失败: {exc!r}", file=sys.stderr)
        print("请检查 .env 中的 OPENAI_API_KEY / TAVILY_API_KEY", file=sys.stderr)
        return 2

    records: list[dict] = []
    for i, entry in enumerate(test_set, 1):
        record = run_one(graph, entry, args.timeout)
        records.append(record)
        print_request_result(i, len(test_set), record)

    # 先把录制到的输入落盘,再往下算汇总 —— 汇总万一报错也不至于丢掉这次的输入。
    if cassette is not None and fixtures_info["mode"] == "record":
        cassette.save()
        print(f"\n已录制 {cassette.writes} 条外部输入到: {cassette.path}")

    consistency: list[dict] = []
    if not args.no_consistency:
        targets = resolve_consistency_targets(
            test_set,
            [t.strip().upper() for t in args.consistency_tickers.split(",")]
            if args.consistency_tickers else None,
            min(args.consistency_n, 5),
        )
        if targets:
            print("=" * 78)
            print(f" 一致性测试: {len(targets)} 个 ticker × 每个 {args.consistency_runs} 次")
            print("=" * 78)
            for ticker in targets:
                print(f"\n--- 一致性: {ticker} ---")
                runs = [
                    run_one(graph, {"ticker": ticker, "label": "consistency"}, args.timeout)
                    for _ in range(args.consistency_runs)
                ]
                for run in runs:
                    print(f"  运行: 状态={_cn_status(run['status'])} "
                          f"优势项={run['num_strengths']} 风险项={run['num_risks']} "
                          f"市盈率={_fmt(run.get('pe_ratio'))} 股息率={_fmt_dy(run)}")
                metrics = compute_consistency(runs)
                consistency.append({"ticker": ticker, "runs": runs, "metrics": metrics})
        else:
            print("\n未解析到一致性测试对象，跳过一致性测试。")

    summary = build_summary(records, consistency, fixtures_info)
    markdown = render_markdown(summary, consistency)

    print("\n" + "=" * 78)
    print(markdown)
    print("=" * 78)

    write_outputs(Path(args.out_dir), records, consistency, summary, markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
