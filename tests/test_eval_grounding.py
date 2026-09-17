"""Offline tests for the contract-layer number-grounding check.

No network, no API keys. These lock down the two design decisions that make the
checker trustworthy rather than decorative:

1. **Only raw inputs count as evidence.** ``market_context`` is the Research
   LLM's own summary — grounding against it would let a hallucination launder
   itself through a summary. Neither may any eval-internal field (latency,
   tokens, cost) become a "source".
2. **Tokenisation is not filtering.** Year-like numbers are reported, merely
   annotated, so nothing can hide.
"""
from eval import grounding as g


def _rec(**kw):
    base = {
        "ticker": "TEST",
        "company_name": "Test Inc",
        "current_price": 332.41,
        "market_cap": 4_851_251_544_064,
        "eps": 8.74,
        "dividend_yield": 0.0033,
        "beta": 1.08,
        "period_return_pct": 12.41652,
        "final_report": "",
        "news_findings": [],
        "market_context": "",
        # eval-internal fields: must never become evidence
        "elapsed_s": 11.16,
        "cost_usd": 0.001603,
        "prompt_tokens": 5720,
        "num_strengths": 3,
        "num_risks": 2,
    }
    base.update(kw)
    return base


# ---------------------------------------------------------------------------
# Tokeniser
# ---------------------------------------------------------------------------
def _values(text: str, index: int = 0) -> list[float]:
    return [value for value, _tol in g.extract_number_tokens(text)[index]["candidates"]]


def _tolerances(text: str, index: int = 0) -> list[float]:
    return [tol for _value, tol in g.extract_number_tokens(text)[index]["candidates"]]


class TestTokeniser:
    def test_hyphenated_labels_are_not_numbers(self):
        """'52-Week Range' 里的 52 是标签的一部分,不是数字断言。"""
        assert g.extract_number_tokens("| 52-Week Range | $1 - $2 |") != []
        assert [t["written"] for t in g.extract_number_tokens("52-Week Range")] == []

    def test_unit_suffixes_are_not_numbers(self):
        """3mo / 5min 里的数字是单位的一部分。"""
        assert g.extract_number_tokens("over 3mo period") == []
        assert g.extract_number_tokens("5m and 10s") == []

    def test_letter_scale_is_applied(self):
        assert _values("$4.77T") == [4.77e12]

    def test_word_scale_is_applied(self):
        """报告写 'billion' 而新闻写 'B'——两边必须换算到同一量级。"""
        assert _values("$109.42 billion") == [109.42e9]
        assert _values("$109.42B") == [109.42e9]

    def test_percent_keeps_both_readings(self):
        """0.33% 可能是 0.33 这个数,也可能是十进制 0.0033(项目约定)。"""
        assert _values("yield of 0.33%") == [0.33, 0.0033]

    def test_negative_sign_is_captured(self):
        assert _values("a day change of -3.49%") == [-3.49, -0.0349]

    def test_thousands_separator(self):
        assert _values("$1,234.5 million") == [1234.5e6]

    def test_tolerance_shrinks_with_the_percent_reading(self):
        """容差跟着候选值量级走:47% 的十进制解读 0.47 容差是 0.005,
        而不是 0.5 —— 否则 0.47 会和无关的 0.86(波动率)判成一致。"""
        values = _values("47%")
        tolerances = _tolerances("47%")
        assert values == [47.0, 0.47]
        assert tolerances == [0.5, 0.005]

    def test_tolerance_scales_with_the_unit(self):
        """'4.85 trillion' 的精度是 0.01 万亿 = 1e10。"""
        assert _values("$4.85 trillion") == [4.85e12]
        assert _tolerances("$4.85 trillion") == [5e9]

    def test_year_is_reported_but_annotated(self):
        """年份通常确实无出处,但**不能静默过滤**——只打标记方便分诊。"""
        [token] = g.extract_number_tokens("Q3 2026 revenue")
        assert token["written"] == "2026"
        assert token["year_like"] is True

    def test_quarter_number_is_part_of_a_label(self):
        assert [t["written"] for t in g.extract_number_tokens("Q3 revenue")] == []

    def test_context_is_attached(self):
        [token] = g.extract_number_tokens("x" * 100 + " 42 " + "y" * 100)
        assert "42" in token["context"]
        assert token["context"].startswith("…")


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------
class TestClassification:
    def test_rounding_still_counts_as_grounded(self):
        """报告的 332.41 与字段的 332.4100036621094 是同一个数。"""
        result = g.check_grounding(_rec(final_report="Price is $332.41 today."))
        assert result["financial"] == 1
        assert result["ungrounded"] == 0

    def test_decimal_field_matches_percent_writing(self):
        """字段存 0.0033(十进制),报告写 0.33%。"""
        result = g.check_grounding(_rec(final_report="Dividend yield of 0.33%."))
        assert result["financial"] == 1

    def test_word_scale_matches_raw_field(self):
        result = g.check_grounding(_rec(final_report="Market cap of $4.85 trillion."))
        assert result["financial"] == 1
        assert result["ungrounded"] == 0

    def test_number_only_in_news_is_news_only(self):
        record = _rec(
            final_report="Revenue reached $109.42 billion.",
            news_findings=[{"title": "Earnings", "content": "revenue of $109.42 billion"}],
        )
        result = g.check_grounding(record)
        assert result["news_only"] == 1
        assert result["financial"] == 0

    def test_number_that_is_nowhere_is_ungrounded(self):
        result = g.check_grounding(_rec(final_report="The stock fell by 8.66% today."))
        assert result["ungrounded"] == 1
        assert result["ungrounded_items"][0]["written"] == "8.66%"

    def test_a_real_discrepancy_is_not_waved_through(self):
        """报告写 EPS 2.02,字段是 8.74 —— 相对差 77%,必须判无出处。"""
        result = g.check_grounding(_rec(final_report="EPS of $2.02 beat estimates."))
        assert result["ungrounded"] == 1

    def test_tolerance_follows_the_declared_precision(self):
        """容差按「报告写了多少位小数」定,不按相对百分比定。"""
        assert g.check_grounding(_rec(final_report="EPS 8.74"))["financial"] == 1
        # 8.74 四舍五入到 1 位 = 8.7,所以 "8.7" 算有出处
        assert g.check_grounding(_rec(final_report="EPS 8.7"))["financial"] == 1
        # 但 "8.70" 声称的是 2 位精度,8.74 不会舍入成 8.70 → 无出处
        assert g.check_grounding(_rec(final_report="EPS 8.70"))["ungrounded"] == 1

    def test_tolerance_scales_with_the_unit(self):
        """'4.85 trillion' 声称的精度是 0.01 万亿 = 1e10,容差必须跟着放大,
        否则四舍五入过的数十亿位数会被误报成无出处。"""
        record = _rec(final_report="Market cap of $4.85 trillion.", market_cap=4_851_251_544_064)
        assert g.check_grounding(record)["financial"] == 1

    def test_context_word_can_carry_the_sign(self):
        """'a decline of 47.63%' 实际值是 -47.63 —— 自然语言把符号写在词里。"""
        record = _rec(final_report="a drastic decline of 47.63% over three months",
                      period_return_pct=-47.63271215126123)
        assert g.check_grounding(record)["financial"] == 1

    def test_opposite_direction_is_still_caught(self):
        """但反方向不能放过:'growth of 47%' 遇到字段 -47 仍是无出处。"""
        record = _rec(final_report="impressive growth of 47% over three months",
                      period_return_pct=-47.63271215126123)
        assert g.check_grounding(record)["ungrounded"] == 1

    def test_no_cue_means_no_sign_flip(self):
        record = _rec(final_report="A return of 47.63% over three months",
                      period_return_pct=47.63271215126123)
        assert g.check_grounding(record)["financial"] == 1
        record = _rec(final_report="A return of 47.63% over three months",
                      period_return_pct=-47.63271215126123)
        assert g.check_grounding(record)["ungrounded"] == 1

    def test_empty_report_is_all_zeros(self):
        result = g.check_grounding(_rec(final_report=""))
        assert result == {
            "total": 0, "financial": 0, "news_only": 0, "ungrounded": 0,
            "news_only_items": [], "ungrounded_items": [],
        }

    def test_occurrences_are_deduplicated(self):
        report = "Beta 1.08. " * 4 + "Also beta 1.08."
        result = g.check_grounding(_rec(final_report=report))
        assert result["total"] >= 5
        assert len(result["ungrounded_items"]) == len({i["written"] for i in result["ungrounded_items"]})


# ---------------------------------------------------------------------------
# The design decisions that keep the checker honest
# ---------------------------------------------------------------------------
class TestEvidenceBoundary:
    def test_market_context_is_not_a_source(self):
        """Research 的 LLM 摘要不能当出处,否则幻觉可以经摘要洗白。

        这是整个检查器的关键设计:LLM 自己写的东西不能给自己作证。
        """
        record = _rec(
            final_report="Revenue reached $999.99 billion.",
            market_context="Revenue reached $999.99 billion this quarter.",
            news_findings=[],
        )
        assert g.check_grounding(record)["ungrounded"] == 1

    def test_research_summary_is_not_evidence_even_when_news_agrees(self):
        """反过来也要成立:只有 market_context 里有、原始新闻里没有 → 无出处。"""
        record = _rec(
            final_report="Growth of 77.7%.",
            market_context="Growth of 77.7% outpaced peers.",
            news_findings=[{"title": "t", "content": "something unrelated"}],
        )
        assert g.check_grounding(record)["ungrounded"] == 1

    def test_eval_internal_fields_are_not_evidence(self):
        """耗时/成本/token 是评测自己的数字,LLM 从没见过它们。

        用白名单而不是黑名单就是为了这个:将来新增的评测内部字段不会悄悄
        变成"合法出处"。
        """
        record = _rec(
            final_report="Total latency was 11.16 seconds and cost 0.001603 dollars.",
            elapsed_s=11.16,
            cost_usd=0.001603,
        )
        result = g.check_grounding(record)
        assert result["financial"] == 0
        assert result["ungrounded"] == 2

    def test_only_whitelisted_financial_fields_are_evidence(self):
        record = _rec(final_report="A made-up figure of 12345.6.")
        record["some_future_metric"] = 12345.6
        assert g.check_grounding(record)["ungrounded"] == 1
