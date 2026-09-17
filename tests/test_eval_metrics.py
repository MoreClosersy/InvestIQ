"""Offline unit tests for the eval harness's scoring logic.

Pure functions only — no API keys, no network, CI-friendly (same policy as
``tests/test_smoke.py``). These lock down the two things that were measurably
wrong in the first version of the harness:

1. ``classify_outcome`` — "correctly refused an invalid ticker" was scored as a
   failure, and "invented an analysis from generic news with zero financial
   data" was scored as a clean success.
2. the consistency metric — comparing *raw sentence sets* with Jaccard reported
   0.00 for runs that state the identical fact in different words, so the report
   announced non-determinism that did not exist.
"""
from eval import run_eval as r


def _rec(**kw):
    """A minimal well-formed record. Only the fields the scorer reads."""
    base = {
        "ticker": "TEST",
        "label": "",
        "status": "success",
        "current_price": 0.0,
        "market_cap": 0,
        "num_strengths": 0,
        "num_risks": 0,
        "strengths": [],
        "risks": [],
        "analyst_summary": "",
        "data_completeness": "",
        "elapsed_s": 1.0,
        "llm_calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "cost_usd": 0.0,
        "errors": [],
    }
    base.update(kw)
    return base


# ---------------------------------------------------------------------------
# Outcome classification
# ---------------------------------------------------------------------------
class TestClassifyOutcome:
    def test_funded_and_analysed_is_ok(self):
        assert r.classify_outcome(_rec(current_price=100.0, num_strengths=3)) == "ok"

    def test_market_cap_alone_counts_as_financial_data(self):
        """The agent's gate is `price OR market cap` — the scorer must match it."""
        assert r.classify_outcome(_rec(market_cap=10**12, num_risks=2)) == "ok"

    def test_invalid_ticker_correctly_refused_is_not_a_failure(self):
        """Post-fix behaviour: no data, no analysis. This is CORRECT, not degraded."""
        assert r.classify_outcome(_rec()) == "abstained_correctly"

    def test_invalid_ticker_with_invented_analysis_is_hallucinated(self):
        """The actual bug the harness exists to catch."""
        assert r.classify_outcome(_rec(num_risks=3)) == "hallucinated"

    def test_funded_but_no_analysis_is_degraded(self):
        assert r.classify_outcome(_rec(current_price=5.0)) == "degraded"

    def test_pipeline_failures_pass_through(self):
        assert r.classify_outcome(_rec(status="failure")) == "failure"
        assert r.classify_outcome(_rec(status="timeout")) == "timeout"


# ---------------------------------------------------------------------------
# Consistency metrics
# ---------------------------------------------------------------------------
_FACT_A = (
    "Strong Q3 2026 earnings with an EPS of $2.02, exceeding estimates of "
    "$1.89 by 6.88%, indicating robust profitability ahead."
)
_FACT_A_REWORDED = (
    "Strong Q3 2026 earnings with an EPS of $2.02, exceeding estimates of "
    "$1.89 by 6.88%, indicating effective cost control."
)


class TestConsistencyMetrics:
    def test_same_fact_different_wording_is_not_zero(self):
        """Regression: the old metric scored this 0.00 and the report printed
        「完全一致: 否」, converting a tokenizer artifact into a false finding."""
        assert r.lexical_similarity(_FACT_A, _FACT_A_REWORDED) > 0.5

    def test_best_match_survives_rephrasing(self):
        assert r._best_match([_FACT_A], [_FACT_A_REWORDED]) > 0.5

    def test_unrelated_text_scores_low(self):
        other = "The company refinanced its revolving credit facility in June."
        assert r.lexical_similarity(_FACT_A, other) < 0.3

    def test_identical_text_scores_one(self):
        assert r.lexical_similarity(_FACT_A, _FACT_A) == 1.0

    def test_empty_input_yields_none_not_zero(self):
        """样本不足必须与「确实不一致」区分开。"""
        assert r._best_match([], [_FACT_A]) is None

    def test_facts_keep_decimals_and_percentages(self):
        assert {"2.02", "1.89", "6.88"} <= r._facts([_FACT_A])

    def test_facts_drop_years_and_quarter_numbers(self):
        """年份/季度号不是被断言的事实;放进去会用噪声抬高重合度。"""
        facts = r._facts(["Q3 2026 revenue grew 22%"])
        assert "2026" not in facts
        assert "3" not in facts
        assert "22%" in facts

    def test_no_facts_at_all_is_na_not_perfect_agreement(self):
        """两条运行都没引用任何数字时,重合度必须是 N/A。Jaccard 对两个空集
        返回 1.0,直接用会报出「1.00,共 0 个数字」这种毫无信息量的满分。"""
        blank = _rec(strengths=[], risks=[])
        metrics = r.compute_consistency([dict(blank), dict(blank)])
        assert metrics["facts_overlap"] is None
        assert metrics["facts_distinct"] == 0

    def test_no_summary_at_all_is_na_not_perfect_agreement(self):
        """摘要为空(如正确拒答的无效代码)时,两个空串算出来是 1.00 ——
        读起来像「摘要高度一致」,实际是根本没有摘要。"""
        blank = _rec(analyst_summary="")
        metrics = r.compute_consistency([dict(blank), dict(blank)])
        assert metrics["analyst_summary_similarity"] is None

    def test_a_real_summary_is_still_compared(self):
        a = _rec(analyst_summary="Solid quarter with margin risk.")
        b = _rec(analyst_summary="Solid quarter with margin risk.")
        metrics = r.compute_consistency([dict(a), dict(b)])
        assert metrics["analyst_summary_similarity"] == 1.0

    def test_facts_overlap_detects_a_declining_report(self):
        """两次运行引用的事实完全不相交时,重合度必须掉到 0。"""
        run_a = [_rec(strengths=["EPS of $2.02 beat by 6.88%"], risks=[])["strengths"][0]]
        run_b = [_rec(strengths=["Beta sits at 1.08 with 22% upside"], risks=[])["strengths"][0]]
        assert r._facts(run_a).isdisjoint(r._facts(run_b))
        assert r._jaccard_sets(r._facts(run_a), r._facts(run_b)) == 0.0


# ---------------------------------------------------------------------------
# Summary assembly — the denominators
# ---------------------------------------------------------------------------
class TestBuildSummary:
    def test_abstention_is_not_counted_as_failure(self):
        records = [
            _rec(ticker="AAPL", current_price=100.0, num_strengths=3),
            _rec(ticker="ZZZZZZ", label="invalid"),
        ]
        summary = r.build_summary(records, [])
        assert summary["outcomes"]["ok"] == 1
        assert summary["outcomes"]["abstained_correctly"] == 1
        assert summary["outcomes"]["degraded"] == 0
        assert summary["abstention"] == {
            "requests_without_financial_data": 1,
            "abstained_correctly": 1,
            "hallucinated": 0,
        }
        # 只有 hallucinated 会让这条记录进「非成功明细」
        assert summary["problem_rows"] == []

    def test_hallucination_is_surfaced_as_a_count_not_hidden_in_a_rate(self):
        records = [
            _rec(ticker="AAPL", current_price=100.0, num_strengths=3),
            _rec(ticker="ZZZZZZ", label="invalid", num_risks=3),
        ]
        summary = r.build_summary(records, [])
        assert summary["outcomes"]["hallucinated"] == 1
        assert summary["abstention"]["hallucinated"] == 1
        assert len(summary["problem_rows"]) == 1

    def test_the_two_denominators_stay_separate(self):
        records = [
            _rec(ticker="A", current_price=1.0, num_strengths=1),
            _rec(ticker="B", current_price=1.0),  # has data, produced nothing
            _rec(ticker="INVALID"),               # no data, correctly silent
        ]
        summary = r.build_summary(records, [])
        assert summary["analysis"]["requests_with_financial_data"] == 2
        assert summary["analysis"]["produced_analysis"] == 1
        assert summary["abstention"]["requests_without_financial_data"] == 1

    def test_rescoring_an_archived_record_ignores_any_stored_outcome(self):
        """Changing the metric definition must re-score old runs correctly, so the
        summary reads only raw observed fields. Trusting a stored classification
        is exactly how the old ``summary_*.md`` ended up contradicting its own
        ``results_*.json``."""
        stale = _rec(
            ticker="AAPL", current_price=100.0, num_strengths=3, outcome="degraded"
        )
        summary = r.build_summary([stale], [])
        assert summary["outcomes"]["ok"] == 1
        assert summary["outcomes"]["degraded"] == 0

    def test_process_success_rate_counts_only_clean_returns(self):
        records = [
            _rec(ticker="A", current_price=1.0, num_strengths=1),
            _rec(ticker="B", status="failure"),
            _rec(ticker="C", status="timeout"),
        ]
        summary = r.build_summary(records, [])
        assert summary["success"] == 1
        assert summary["failure"] == 1
        assert summary["timeout"] == 1
        assert summary["process_success_rate"] == 1 / 3


# ---------------------------------------------------------------------------
# Rendering must not crash on empty / missing data
# ---------------------------------------------------------------------------
class TestRender:
    def test_renders_without_fixtures_provenance(self, tmp_path):
        summary = r.build_summary([_rec(current_price=1.0, num_strengths=1)], [])
        md = r.render_markdown(summary, [])
        assert "事实重合度" not in md  # no consistency block when there were no repeats
        assert "正确拒答" in md

    def test_live_run_is_flagged_as_not_reproducible(self):
        summary = r.build_summary([_rec(current_price=1.0, num_strengths=1)], [])
        assert "不可复现" in r.render_markdown(summary, [])

    def test_replay_run_is_labelled_reproducible(self):
        summary = r.build_summary(
            [_rec(current_price=1.0, num_strengths=1)],
            [],
            {"mode": "replay", "path": "eval/cassettes/frozen.json"},
        )
        md = r.render_markdown(summary, [])
        assert "回放冻结磁带" in md
        assert "eval/cassettes/frozen.json" in md
