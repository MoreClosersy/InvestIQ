"""Offline tests for the freeze/replay layer (``eval/fixtures.py``).

No network, no API keys, CI-friendly. ``record`` mode is exercised against fake
data sources injected through the public ``override_source`` seam, so these tests
prove the cassette machinery without touching yfinance or Tavily.
"""
import json

import pytest

import agents.financial_agent as fa
import agents.research_agent as ra
from eval import fixtures


def _snapshot(ticker: str) -> dict:
    return {"current_price": 123.0, "currency": "USD", "ticker": ticker}


def _metrics(ticker: str) -> dict:
    return {"market_cap": 42, "company_name": f"{ticker} Inc"}


def _historical(ticker: str, period: str = "3mo") -> dict:
    return {"period": period, "data_points": 63}


def _news(ticker: str, company_name: str = "") -> list[dict]:
    return [{"title": f"{ticker} news", "url": "https://example.test", "content": "..."}]


@pytest.fixture
def cassette_path(tmp_path):
    """Fake external sources installed + cleanup, yielding a cassette path."""
    fixtures.override_source("agents.financial_agent", "get_stock_snapshot", _snapshot)
    fixtures.override_source("agents.financial_agent", "get_financial_metrics", _metrics)
    fixtures.override_source("agents.financial_agent", "get_historical_prices", _historical)
    fixtures.override_source("agents.research_agent", "search_company_news", _news)
    path = tmp_path / "cassette.json"
    yield path
    fixtures.reset_overrides()


def _record(path) -> fixtures.Cassette:
    cassette = fixtures.Cassette(path)
    fixtures.install(cassette, "record")
    # 走一遍 agent 模块上的名字,而不是直接调工具函数——补丁打在导入方命名空间,
    # 正是这一层要验证的东西。
    fa.get_stock_snapshot("AAPL")
    fa.get_financial_metrics("AAPL")
    fa.get_historical_prices("AAPL", period="3mo")
    ra.search_company_news("AAPL", "")
    cassette.save()
    return cassette


class TestRecordThenReplay:
    def test_roundtrip_returns_identical_values(self, cassette_path):
        cassette = _record(cassette_path)
        assert cassette.writes == 4

        fixtures.uninstall()
        replayed = fixtures.Cassette.load(cassette_path)
        fixtures.install(replayed, "replay")

        assert fa.get_stock_snapshot("AAPL") == _snapshot("AAPL")
        assert fa.get_financial_metrics("AAPL") == _metrics("AAPL")
        assert fa.get_historical_prices("AAPL", period="3mo") == _historical("AAPL", "3mo")
        assert ra.search_company_news("AAPL", "") == _news("AAPL")
        assert replayed.reads == 4

    def test_replay_does_not_fall_back_to_the_source(self, cassette_path):
        """This is the whole point: replay must be airtight. If the wrapper still
        reached for the live source, the run would silently stop being
        reproducible — the failure mode freezes exist to prevent."""
        _record(cassette_path)

        def _boom(*args, **kwargs):
            raise AssertionError("回放模式下不应调用外部数据源")

        # 把「真身」也换成爆炸函数,然后回放。任何真实调用都会让测试失败。
        fixtures.override_source("agents.financial_agent", "get_stock_snapshot", _boom)
        fixtures.install(fixtures.Cassette.load(cassette_path), "replay")

        assert fa.get_stock_snapshot("AAPL")["current_price"] == 123.0

    def test_variant_is_part_of_the_key(self, cassette_path):
        """`period` 不同的历史窗口不能互相顶替。"""
        _record(cassette_path)
        cassette = fixtures.Cassette.load(cassette_path)
        assert "3mo" in cassette.sources["AAPL"]["get_historical_prices"]

        cassette.put("AAPL", "get_historical_prices", "1y", {"period": "1y", "data_points": 252})
        assert cassette.get("AAPL", "get_historical_prices", "1y")["data_points"] == 252
        assert cassette.get("AAPL", "get_historical_prices", "3mo")["data_points"] == 63


class TestIntegrity:
    def test_missing_fixture_raises_instead_of_going_live(self, cassette_path):
        cassette = fixtures.Cassette(cassette_path)
        fixtures.install(cassette, "replay")
        with pytest.raises(fixtures.MissingFixtureError):
            fa.get_stock_snapshot("ZZZZZZ")

    def test_missing_variant_is_reported(self, cassette_path):
        _record(cassette_path)
        cassette = fixtures.Cassette.load(cassette_path)
        fixtures.install(cassette, "replay")
        with pytest.raises(fixtures.MissingFixtureError) as excinfo:
            fa.get_historical_prices("AAPL", period="5y")
        assert "5y" in str(excinfo.value)

    def test_callers_cannot_corrupt_the_cassette(self, cassette_path):
        """回放返回深拷贝:被测代码改写返回值不能污染磁带(否则同一次回放里
        后续请求会拿到被改过的输入)。"""
        _record(cassette_path)
        cassette = fixtures.Cassette.load(cassette_path)
        fixtures.install(cassette, "replay")

        first = fa.get_stock_snapshot("AAPL")
        first["current_price"] = 999.0
        assert fa.get_stock_snapshot("AAPL")["current_price"] == 123.0

    def test_missing_for_drives_the_pre_flight_check(self, cassette_path):
        _record(cassette_path)
        cassette = fixtures.Cassette.load(cassette_path)
        assert cassette.missing_for(["AAPL"]) == []
        assert cassette.missing_for(["AAPL", "ZZZZZZ", "12345"]) == ["ZZZZZZ", "12345"]

    def test_version_mismatch_is_rejected(self, cassette_path):
        _record(cassette_path)
        raw = json.loads(cassette_path.read_text())
        raw["version"] = fixtures.CASSETTE_VERSION + 99
        cassette_path.write_text(json.dumps(raw))
        with pytest.raises(ValueError, match="版本不兼容"):
            fixtures.Cassette.load(cassette_path)

    def test_missing_file_says_how_to_make_one(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="--record"):
            fixtures.Cassette.load(tmp_path / "nope.json")


class TestInstallHygiene:
    def test_install_is_idempotent(self, cassette_path):
        """重复 install 不应把包装器再包一层(会自我递归/重复计数)。"""
        cassette = fixtures.Cassette(cassette_path)
        fixtures.install(cassette, "record")
        fixtures.install(cassette, "record")
        fa.get_stock_snapshot("AAPL")
        assert cassette.writes == 1

    def test_uninstall_restores_the_real_implementation(self, cassette_path):
        real = fa.get_stock_snapshot
        fixtures.install(fixtures.Cassette(cassette_path), "replay")
        assert fa.get_stock_snapshot is not real
        fixtures.uninstall()
        assert fa.get_stock_snapshot is real

    def test_unknown_mode_is_rejected(self, cassette_path):
        with pytest.raises(ValueError):
            fixtures.install(fixtures.Cassette(cassette_path), "sideways")
