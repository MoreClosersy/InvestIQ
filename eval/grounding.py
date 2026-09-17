"""契约层检查:报告里的数字断言能不能在输入数据里找到出处。

这是**契约层**检查,目的是发现报告里那些**追不回输入**的数字。

为什么值得做:Analysis 和 Report 都是 LLM,它们引用数字时可能

1. **改写** —— 把 `$308.82` 写成 `$308`;
2. **口径混用** —— 把新闻里的季度 EPS 和 yfinance 的滚动年度 EPS 并排放在一起
   而不标注口径(两个数都"有出处",但读者会误读);
3. **直接编** —— 输入里根本没这个数。

第 3 种是幻觉。这个检查器把"报告里出现的每个数字"对回"输入里真实存在的数字",
分成三类,把后两类挑出来给人看。

## 设计上刻意不做的事

* **不产出"接地率"百分比。** 分母(报告里有多少个数字)取决于 LLM 写了多少字,
  一个比例会诱导人拿它做横向比较。给的是**清单 + 计数**,分母天然清楚。
* **不拿 `market_context` 当出处。** 那是 Research LLM 自己的摘要,用它做校验
  等于让模型给自己作证——幻觉可以经过一层摘要洗白。只认**原始新闻正文** +
  **yfinance 字段**。
* **不静默过滤年份/季度号。** 它们通常确实无出处(报告引用财报期间),但直接
  过滤会掩盖真编造。做法是照常标为"无出处",额外打一个 `year_like` 标记,
  方便人分诊。
* **出处用白名单而不是黑名单。** 只有 agent 真正喂给 LLM 的那些字段算证据。
  黑名单会让将来新增的评测内部字段(耗时、token、成本)悄悄变成"合法出处"。
"""
from __future__ import annotations

import re
from typing import Any

# ---------------------------------------------------------------------------
# 证据来源:白名单
# ---------------------------------------------------------------------------
# 这里列的是 Financial Agent 从 yfinance 拿到、并最终进入 LLM 上下文的数值字段
# (对应 graph/state.py 的 stock_price / financial_metrics / historical_data)。
FINANCIAL_SOURCE_FIELDS: tuple[str, ...] = (
    # stock_price
    "current_price", "previous_close", "change_pct", "day_high", "day_low", "volume",
    # financial_metrics
    "market_cap", "pe_ratio", "forward_pe", "eps", "dividend_yield",
    "week_52_high", "week_52_low", "beta",
    # historical_data
    "period_start_price", "period_end_price", "period_return_pct",
    "period_high", "period_low", "volatility", "data_points",
)

# 容差判据:「字段值四舍五入到报告所写的小数位 × 报告所用的单位」。
#
# 早期版本用过固定的 0.5% 相对容差 —— 对大数字太松、小数字太紧,还把
# EPS 字段 8.74 与报告写的 "8.70"(差 0.46%)放过了。**按声明的精度判定才对。**
FLOAT_EPSILON = 1e-9

_SCALE = {"t": 1e12, "b": 1e9, "m": 1e6, "k": 1e3}
_SCALE_WORDS = {
    "trillion": 1e12,
    "billion": 1e9,
    "million": 1e6,
    "thousand": 1e3,
}

# 自然语言把符号写在**词**里:“a decline of 47.63%” 实际值是 -47.63。
# 判为这类负向语义时,额外试一下取负的候选值。
#
# 注意这**不是**把检查器变成不管符号:反方向仍然会抓到——“growth of 47%”
# 遇到字段 -47 依旧是“无出处”。只是当上下文明确说了方向时,不再要求数字
# 自己也把负号写出来。
_NEGATIVE_CUE_RE = re.compile(
    r"\b(?:decline[sd]?|declining|fell|fall(?:ing)?|drop(?:s|ped|ping)?|"
    r"loss(?:es)?|decrease[sd]?|decreasing|lower|bearish|contraction|"
    r"slid|slide|tumbl(?:e|ed|ing)|negative|shrank|shrink(?:ing)?|"
    r"underperform\w*|correction|sell-?off|pullback|receded|downward|down)\b",
    re.IGNORECASE,
)

# 数字 token。三条边界规则,都是**分词**决定而不是内容过滤:
#   * `(?<![\w.])` 排除 Q3、v2、P2 这类标签里的数字,以及小数点后的再匹配;
#   * `(?!\w|-)`    排除 3mo、5min 这类单位里的数字,也排除 "52-Week" 这种
#                    连字符复合标签(否则 "52" 会被当成一个独立数字断言);
#   * scale 字母只认大写 T/B/M/K,避免把 "5m" 当百万;单词形式另算。
_NUMBER_RE = re.compile(
    r"""
    (?<![\w.])
    (?P<sign>[-+])?
    (?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)
    \s*
    (?P<pct>%)?
    \s*
    (?P<scale>[TBMK])?
    (?P<scale_word>(?i:trillion|billion|million|thousand))?
    (?!\w|-)
    """,
    re.VERBOSE,
)

CONTEXT_CHARS = 40


def _to_float(raw: str) -> float:
    return float(raw.replace(",", ""))


def _decimals(raw: str) -> int:
    return len(raw.split(".")[1]) if "." in raw else 0


def _multiplier(scale: str | None, scale_word: str | None) -> float:
    """`4.77T` 和 `4.77 trillion` 必须被看成同一个量级。"""
    if scale:
        return _SCALE[scale.lower()]
    if scale_word:
        return _SCALE_WORDS[scale_word.lower()]
    return 1.0


def _candidates(
    signed: float, decimals: int, pct: bool, multiplier: float, negate: bool = False
) -> list[tuple[float, float]]:
    """返回 `(候选值, 容差)` 对。

    * `0.33%` 既可能是"0.33 这个数",也可能是十进制 0.0033(项目的股息率约定),
      所以两种都试,避免因为单位约定不同产生假阳性;
    * `negate` 用于上下文已用词表达了负向语义的情形(见 `_NEGATIVE_CUE_RE`)。

    容差必须**跟着每个候选值的量级走**:报告写 `47%` 时,对十进制解读 `0.47`
    的容差是 0.005(即 47±0.5 个百分点),而不是 0.5 —— 否则 0.47 会和无关的
    0.86(波动率)判成一致。
    """
    unit = 10 ** (-decimals) * multiplier
    base = signed * multiplier
    out: list[tuple[float, float]] = [(base, 0.5 * unit)]
    if pct:
        out.append((base / 100.0, 0.5 * unit / 100.0))
    if negate:
        out.extend((-value, tolerance) for value, tolerance in list(out))
    return out


def extract_number_tokens(text: str) -> list[dict[str, Any]]:
    """抽出文本里的数字 token,附带解读与上下文。"""
    tokens: list[dict[str, Any]] = []
    for match in _NUMBER_RE.finditer(text or ""):
        raw = match.group("num")
        sign = match.group("sign") or ""
        pct = bool(match.group("pct"))
        scale = match.group("scale")
        scale_word = match.group("scale_word")
        multiplier = _multiplier(scale, scale_word)
        value = _to_float(raw)
        signed = -value if sign == "-" else value

        start = max(0, match.start() - CONTEXT_CHARS)
        end = min(len(text), match.end() + CONTEXT_CHARS)
        context = re.sub(r"\s+", " ", text[start:end]).strip()
        negate = bool(_NEGATIVE_CUE_RE.search(context))

        tokens.append(
            {
                "written": f"{sign}{raw}{'%' if pct else ''}{scale or ''}"
                f"{(' ' + scale_word.lower()) if scale_word else ''}",
                "value": value,
                "decimals": _decimals(raw),
                "pct": pct,
                "multiplier": multiplier,
                "negated_by_context": negate,
                "candidates": _candidates(
                    signed, _decimals(raw), pct, multiplier, negate
                ),
                "context": f"…{context}…",
                "year_like": (
                    multiplier == 1.0
                    and not pct
                    and "." not in raw
                    and 1900 <= value <= 2100
                ),
            }
        )
    return tokens


def _is_grounded(target: float, tolerance: float, pool: list[float]) -> bool:
    """target 是否与 pool 里的某个值一致(容差由报告声明的精度决定)。"""
    tolerance += abs(target) * FLOAT_EPSILON  # 浮点表示余量
    return any(abs(value - target) <= tolerance for value in pool)


def _financial_pool(record: dict) -> list[float]:
    pool: list[float] = []
    for field in FINANCIAL_SOURCE_FIELDS:
        value = record.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        pool.append(float(value))
    return pool


def _news_pool(record: dict) -> list[float]:
    """原始新闻正文里的数字(yfinance 之外的另一类证据)。"""
    pool: list[float] = []
    for item in record.get("news_findings") or []:
        if not isinstance(item, dict):
            continue
        text = f"{item.get('title', '')} {item.get('content', '')}"
        for token in extract_number_tokens(text):
            pool.extend(value for value, _tolerance in token["candidates"])
    return pool


def check_grounding(record: dict) -> dict[str, Any]:
    """把报告里的每个数字 token 对回输入数据,返回分类结果。

    三类:
      * ``financial``  —— 在 yfinance 字段里找到(最硬的出处);
      * ``news_only``  —— 只在新闻正文里找到。**口径风险区**:数字有出处,但
        季度/年度之类口径可能和财务字段冲突;
      * ``ungrounded`` —— 两处都找不到。可能是编造,也可能是年份或单位换算。
    """
    report = record.get("final_report") or ""
    if not report.strip():
        return {
            "total": 0, "financial": 0, "news_only": 0, "ungrounded": 0,
            "news_only_items": [], "ungrounded_items": [],
        }

    financial_pool = _financial_pool(record)
    news_pool = _news_pool(record)

    buckets: dict[str, list[dict[str, Any]]] = {
        "financial": [], "news_only": [], "ungrounded": [],
    }
    for token in extract_number_tokens(report):
        in_financial = any(
            _is_grounded(value, tolerance, financial_pool)
            for value, tolerance in token["candidates"]
        )
        in_news = any(
            _is_grounded(value, tolerance, news_pool)
            for value, tolerance in token["candidates"]
        )
        if in_financial:
            bucket = "financial"
        elif in_news:
            bucket = "news_only"
        else:
            bucket = "ungrounded"
        buckets[bucket].append(token)

    def _summarise(tokens: list[dict]) -> list[dict]:
        """同一个数字可能重复出现,按写法去重并记出现次数。"""
        seen: dict[str, dict] = {}
        for token in tokens:
            key = token["written"]
            if key in seen:
                seen[key]["occurrences"] += 1
                continue
            seen[key] = {
                "written": key,
                "context": token["context"],
                "year_like": token["year_like"],
                "occurrences": 1,
            }
        return sorted(seen.values(), key=lambda item: item["written"])

    return {
        "total": sum(len(tokens) for tokens in buckets.values()),
        "financial": len(buckets["financial"]),
        "news_only": len(buckets["news_only"]),
        "ungrounded": len(buckets["ungrounded"]),
        "news_only_items": _summarise(buckets["news_only"]),
        "ungrounded_items": _summarise(buckets["ungrounded"]),
    }
