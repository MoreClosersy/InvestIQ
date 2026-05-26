"""yfinance wrapper for financial data retrieval.

Three independent fetchers — snapshot (fast quote), metrics (.info-backed
fundamentals), and historical (price/volatility window). Each is wrapped in
try/except and returns a dict with default values plus an ``error`` key on
failure, so callers never see KeyError or uncaught exceptions.

Why split: ``.fast_info`` is a cheap quote endpoint, while ``.info`` triggers
a slower scrape — keep them separate so a slow/failing ``.info`` call doesn't
take the snapshot down with it.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import yfinance as yf


# ---------- defaults ----------

def _snapshot_defaults() -> dict[str, Any]:
    return {
        "current_price": 0.0,
        "previous_close": 0.0,
        "change_pct": 0.0,
        "day_high": 0.0,
        "day_low": 0.0,
        "volume": 0,
        "currency": "",
    }


def _metrics_defaults() -> dict[str, Any]:
    return {
        "company_name": "",
        "sector": "",
        "industry": "",
        "market_cap": 0,
        "pe_ratio": 0.0,
        "forward_pe": 0.0,
        "eps": 0.0,
        "dividend_yield": 0.0,
        "52_week_high": 0.0,
        "52_week_low": 0.0,
        "beta": 0.0,
    }


def _historical_defaults() -> dict[str, Any]:
    return {
        "period_start_price": 0.0,
        "period_end_price": 0.0,
        "period_return_pct": 0.0,
        "period_high": 0.0,
        "period_low": 0.0,
        "volatility": 0.0,
        "data_points": 0,
    }


# ---------- coercion helpers ----------

def _as_float(value: Any) -> float:
    """Coerce yfinance value to float, mapping None/NaN/bad inputs to 0.0."""
    if value is None:
        return 0.0
    try:
        out = float(value)
    except (TypeError, ValueError):
        return 0.0
    if out != out:  # NaN
        return 0.0
    return out


def _as_int(value: Any) -> int:
    """Coerce yfinance value to int, mapping None/NaN/bad inputs to 0."""
    if value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _as_str(value: Any) -> str:
    return "" if value is None else str(value)


# ---------- public API ----------

def get_stock_snapshot(ticker: str) -> dict[str, Any]:
    """Return current quote snapshot via yfinance ``fast_info`` (cheap call)."""
    out = _snapshot_defaults()
    try:
        fi = yf.Ticker(ticker).fast_info
        current = _as_float(getattr(fi, "last_price", None))
        previous = _as_float(getattr(fi, "previous_close", None))
        out["current_price"] = current
        out["previous_close"] = previous
        out["change_pct"] = (
            ((current - previous) / previous) * 100.0 if previous else 0.0
        )
        out["day_high"] = _as_float(getattr(fi, "day_high", None))
        out["day_low"] = _as_float(getattr(fi, "day_low", None))
        out["volume"] = _as_int(getattr(fi, "last_volume", None))
        out["currency"] = _as_str(getattr(fi, "currency", None))
    except Exception as exc:
        out["error"] = repr(exc)
    return out


def get_financial_metrics(ticker: str) -> dict[str, Any]:
    """Return fundamentals via ``.info``. Picks specific keys, no full-dict iteration."""
    out = _metrics_defaults()
    try:
        info = yf.Ticker(ticker).info or {}
        out["company_name"] = _as_str(info.get("longName") or info.get("shortName"))
        out["sector"] = _as_str(info.get("sector"))
        out["industry"] = _as_str(info.get("industry"))
        out["market_cap"] = _as_int(info.get("marketCap"))
        out["pe_ratio"] = _as_float(info.get("trailingPE"))
        out["forward_pe"] = _as_float(info.get("forwardPE"))
        out["eps"] = _as_float(info.get("trailingEps"))
        # yfinance 1.4.x returns dividendYield as a percentage value
        # (e.g. 0.35 means 0.35%); normalize to decimal per project spec.
        out["dividend_yield"] = _as_float(info.get("dividendYield")) / 100.0
        out["52_week_high"] = _as_float(info.get("fiftyTwoWeekHigh"))
        out["52_week_low"] = _as_float(info.get("fiftyTwoWeekLow"))
        out["beta"] = _as_float(info.get("beta"))
    except Exception as exc:
        out["error"] = repr(exc)
    return out


def get_historical_prices(ticker: str, period: str = "3mo") -> dict[str, Any]:
    """Return summary stats over ``period`` (no full time series).

    ``period`` accepts yfinance values: ``"1mo"``, ``"3mo"``, ``"6mo"``,
    ``"1y"``, ``"2y"``, ``"5y"``, ``"ytd"``, ``"max"``.

    ``volatility`` is annualized (std of daily returns * sqrt(252)), expressed
    as a decimal (e.g., 0.25 means 25% annualized vol).
    """
    out = _historical_defaults()
    try:
        hist = yf.Ticker(ticker).history(period=period)
        if hist is None or hist.empty:
            out["error"] = f"empty history for {ticker} over {period}"
            return out

        closes = hist["Close"].dropna()
        if closes.empty:
            out["error"] = f"no close prices for {ticker} over {period}"
            return out

        start = _as_float(closes.iloc[0])
        end = _as_float(closes.iloc[-1])
        out["period_start_price"] = start
        out["period_end_price"] = end
        out["period_return_pct"] = ((end - start) / start) * 100.0 if start else 0.0
        out["period_high"] = _as_float(hist["High"].max())
        out["period_low"] = _as_float(hist["Low"].min())

        daily_returns = closes.pct_change().dropna()
        out["volatility"] = (
            _as_float(daily_returns.std() * np.sqrt(252))
            if len(daily_returns) > 1
            else 0.0
        )
        out["data_points"] = int(len(hist))
    except Exception as exc:
        out["error"] = repr(exc)
    return out
