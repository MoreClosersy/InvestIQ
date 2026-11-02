"""槽位覆盖的共享度量（实验用）。

窄口径（strict）只在两种位置认槽位：
  1. 标题里出现槽位关键词；
  2. 正文的某个句子里，槽位关键词与公司名同时出现。
这会挡掉聚合页导航栏（`SEC Filings`、`Risk Factors`、`IPO Calendar`）造成的假阳性。

宽口径（loose）只要求全文出现，保留作对照——它会把站点框架文字当成证据，
2026-11-02 的试点就是被它误导的，详见 docs/experiments/2026-11-02-slot-queries/。

等指标稳定后，这个模块应迁到 eval/ 下，由评测 harness 直接引用。
"""
from __future__ import annotations

import re
import urllib.parse

SLOTS: dict[str, re.Pattern[str]] = {
    "财报/指引": re.compile(
        r"\b(earnings|eps|revenue|guidance|outlook|quarter|fiscal|q[1-4])\b", re.I),
    "分析师动作": re.compile(
        r"\b(upgrade|downgrade|price target|analyst|rating|reiterat|initiat|"
        r"overweight|underweight|buy rating|sell rating)\b", re.I),
    "重大事件": re.compile(
        r"\b(acquisition|acquire|merger|takeover|ceo|cfo|resign|appoint|"
        r"layoff|bankrupt|ipo|spin-off|stake|board)\b", re.I),
    "风险/法律": re.compile(
        r"\b(lawsuit|sue[sd]?|investigation|regulat\w*|sec\b|fine[sd]?|"
        r"penalt\w*|recall|probe|antitrust|class action|settle\w*)\b", re.I),
    "产品/战略": re.compile(
        r"\b(launch\w*|unveil\w*|partner\w*|expand\w*|new product|contract|"
        r"deal|investment|joint venture)\b", re.I),
}

NAMES: dict[str, str] = {
    "AAPL": r"apple",
    "MSFT": r"microsoft",
    "GOOGL": r"google|alphabet",
    "AMZN": r"amazon",
    "NVDA": r"nvidia",
    "META": r"meta|facebook",
    "TSLA": r"tesla",
    "JPM": r"jpmorgan|\bjpm\b",
    "V": r"\bvisa\b",
    "WMT": r"walmart",
    "XOM": r"exxon",
    "JNJ": r"johnson\s*&\s*johnson|johnson and johnson|\bj&j\b",
    "SOFI": r"sofi",
    "ROKU": r"roku",
    "PLUG": r"plug power|\bplug\b",
    "LCID": r"lucid",
    "NIO": r"\bnio\b",
    "BYND": r"beyond meat",
    "TLRY": r"tilray",
    "SPCE": r"virgin galactic",
    "ZZZZZZ": r"zzzzzz",
    "INVALID": r"invalid",
    "NOTAREAL": r"notareal",
    "12345": r"12345",
}

AGGREGATOR = {
    "public.com", "finance.yahoo.com", "marketbeat.com", "investing.com",
    "zacks.com", "google.com", "robinhood.com", "stocktitan.net",
    "seekingalpha.com", "stockanalysis.com", "tipranks.com", "morningstar.com",
    "pricetargetresearch.com", "wallstreethorizon.com", "us.plus500.com",
    "bitget.com", "fool.com",
}

INVALID_TICKERS = {"ZZZZZZ", "INVALID", "NOTAREAL", "12345"}


def domain_of(url: str) -> str:
    return urllib.parse.urlparse(url).netloc.lower().removeprefix("www.")


def sentences(text: str) -> list[str]:
    return [x.strip() for x in re.split(r"(?<=[.!?])\s+|\n+", text) if x.strip()]


def strict_slots(items: list[dict], ticker: str) -> set[str]:
    name_rx = re.compile(NAMES.get(ticker.upper(), ticker), re.I)
    hits: set[str] = set()
    for it in items:
        title = it.get("title", "") or ""
        for slot, rx in SLOTS.items():
            if rx.search(title):
                hits.add(slot)
                continue
            for sent in sentences(it.get("content", "") or ""):
                if rx.search(sent) and name_rx.search(sent):
                    hits.add(slot)
                    break
    return hits


def loose_slots(items: list[dict]) -> set[str]:
    text = "\n".join(f"{i.get('title', '')}\n{i.get('content', '')}" for i in items)
    return {name for name, rx in SLOTS.items() if rx.search(text)}
