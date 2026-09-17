"""冻结 / 回放外部数据源，让评测可复现。

环境是持续变化的:yfinance 报价每分钟都在动,Tavily 每次返回的新闻都不一样。
不冻结输入,同一个 commit 跑两次拿到的输入就不同,那么输出的差异**永远无法归因**
到被测对象(prompt / 模型 / 代码改动)上——你分不清是提示词改好了,还是今天大盘涨了。

本模块包装 agents 用到的四个外部取数函数::

    tools.finance_tool.get_stock_snapshot(ticker)
    tools.finance_tool.get_financial_metrics(ticker)
    tools.finance_tool.get_historical_prices(ticker, period)
    tools.search_tool.search_company_news(ticker, company_name)

两种模式:
  * ``record`` —— 真实调用一次,把返回值写进 JSON 磁带(cassette);
  * ``replay`` —— 从磁带读取,完全不碰网络,输入逐字节一致。

实现方式是在**导入方模块的命名空间**上打补丁(``agents.financial_agent.*`` /
``agents.research_agent.*``)。这两个 agent 用的是 ``from ... import fn`` 形式,
所以改 ``tools`` 模块里的名字不会生效。``agents/`` 与 ``tools/`` 的代码本身
不做任何改动——评测始终是外部驱动。

注意:这里冻结的是**输入**。LLM 调用刻意不冻结——模型的非确定性正是被测量的
性质(见一致性测试),不是要消除的噪声。回放模式下输入完全相同,于是任何残留
差异都可归因于模型采样。
"""
from __future__ import annotations

import copy
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

CASSETTE_VERSION = 1

# 被打补丁的目标:(模块名, 函数名, 变体参数名)
# 变体参数名非 None 时,该函数的返回值按变体(如 period="3mo")分开存储。
TARGETS: tuple[tuple[str, str, str | None], ...] = (
    ("agents.financial_agent", "get_stock_snapshot", None),
    ("agents.financial_agent", "get_financial_metrics", None),
    ("agents.financial_agent", "get_historical_prices", "period"),
    ("agents.research_agent", "search_company_news", None),
)

# 第一次打补丁前抓住的真实函数。只捕获一次,之后永不被覆盖 —— 所以 install()
# 可以重复调用而不发生「包装包装」的自我递归。
_ORIGINALS: dict[str, Callable[..., Any]] = {}

# 测试接缝:record 模式下用它替代真实实现,让录制流程可以在离线的单测里跑。
# 与 _ORIGINALS 分开存,是为了 uninstall() 永远能恢复成真身而不是假实现。
_OVERRIDES: dict[str, Callable[..., Any]] = {}


class MissingFixtureError(RuntimeError):
    """回放模式下磁带里缺数据。

    这是**评测本身没配好**,不是被测系统的问题——所以它必须显式报错,不能
    悄悄回退到真实网络调用(那样会静默破坏复现性)。
    """


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Cassette:
    """一份冻结输入磁带。结构::

        {
          "version": 1,
          "recorded_at": "...",
          "sources": {
            "AAPL": {
              "get_stock_snapshot": {...},
              "get_financial_metrics": {...},
              "get_historical_prices": {"3mo": {...}},
              "search_company_news": [...]
            }
          }
        }

    新闻按 **ticker 单独** 冻结(不带 company_name 变体):graph 里 research 与
    financial 并行,research 跑的时候 ``company_name`` 还没被回填,查询串恒为
    ``f"{ticker} stock news latest earnings"``,所以按 ticker 做键是精确的。
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.recorded_at: str = ""
        self.sources: dict[str, dict[str, Any]] = {}
        self.reads = 0
        self.writes = 0

    # ---------- 序列化 ----------

    @classmethod
    def load(cls, path: str | Path) -> "Cassette":
        c = cls(path)
        if not c.path.exists():
            raise FileNotFoundError(
                f"磁带不存在: {c.path}。先跑一次 --record 生成它。"
            )
        raw = json.loads(c.path.read_text(encoding="utf-8"))
        version = raw.get("version")
        if version != CASSETTE_VERSION:
            raise ValueError(
                f"磁带版本不兼容: 期望 {CASSETTE_VERSION}, 实际 {version!r}"
            )
        c.recorded_at = raw.get("recorded_at", "")
        c.sources = raw.get("sources") or {}
        return c

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.recorded_at = _iso_now()
        self.path.write_text(
            json.dumps(
                {
                    "version": CASSETTE_VERSION,
                    "recorded_at": self.recorded_at,
                    "sources": self.sources,
                },
                indent=2,
                ensure_ascii=False,
                default=str,
            ),
            encoding="utf-8",
        )

    # ---------- 读写 ----------

    def put(self, ticker: str, fn: str, variant: str, value: Any) -> None:
        bucket = self.sources.setdefault(ticker.upper(), {})
        if not variant:
            bucket[fn] = value
        else:
            bucket.setdefault(fn, {})[variant] = value
        self.writes += 1

    def get(self, ticker: str, fn: str, variant: str) -> Any:
        bucket = self.sources.get(ticker.upper())
        if bucket is None or fn not in bucket:
            raise MissingFixtureError(
                f"磁带缺少 {ticker.upper()} / {fn}。"
                f"已录制的 ticker: {sorted(self.sources)}"
            )
        entry = bucket[fn]
        if not variant:
            self.reads += 1
            return copy.deepcopy(entry)  # 防止被测代码改写磁带内容
        if not isinstance(entry, dict) or variant not in entry:
            available = sorted(entry) if isinstance(entry, dict) else entry
            raise MissingFixtureError(
                f"磁带缺少 {ticker.upper()} / {fn} 的变体 {variant!r}；"
                f"已有变体: {available}"
            )
        self.reads += 1
        return copy.deepcopy(entry[variant])

    def tickers(self) -> list[str]:
        return sorted(self.sources)

    def missing_for(self, tickers: list[str]) -> list[str]:
        have = set(self.sources)
        return [t for t in tickers if t.upper() not in have]


def _original(module: Any, name: str) -> Callable[..., Any]:
    """真实实现(真身),只在首次调用时捕获。"""
    key = f"{module.__name__}.{name}"
    if key not in _ORIGINALS:
        _ORIGINALS[key] = getattr(module, name)
    return _ORIGINALS[key]


def _effective(module: Any, name: str) -> Callable[..., Any]:
    """record 模式下真正被调用的实现:测试覆盖优先,否则是真身。"""
    return _OVERRIDES.get(f"{module.__name__}.{name}") or _original(module, name)


def _effective(module: Any, name: str) -> Callable[..., Any]:
    """record 模式下真正被调用的实现:测试覆盖优先,否则是真身。"""
    return _OVERRIDES.get(f"{module.__name__}.{name}") or _original(module, name)


def override_source(module_name: str, fn_name: str, fn: Callable[..., Any]) -> None:
    """测试接缝:把某个外部数据源换成假实现,供 record 模式调用。

    生产路径不用它——agent 模块始终只调用工具函数本身。
    """
    _OVERRIDES[f"{module_name}.{fn_name}"] = fn


def reset_overrides() -> None:
    """卸载补丁并清空假实现。测试收尾用(顺序不能反:先恢复再清)。"""
    uninstall()
    _OVERRIDES.clear()


def _make_wrapper(
    module: Any,
    name: str,
    cassette: Cassette,
    mode: str,
    variant_param: str | None,
) -> Callable[..., Any]:
    # 先无条件捕获真身:只要 install() 跑过,uninstall() 就必须能恢复。
    # （早期版本在「有测试覆盖」时跳过了这一步,导致 uninstall() 什么也恢复不了。）
    _original(module, name)
    real = _effective(module, name)

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        ticker = str(args[0] if args else kwargs.get("ticker", ""))
        variant = str(kwargs.get(variant_param, "")) if variant_param else ""

        if mode == "replay":
            return cassette.get(ticker, name, variant)

        value = real(*args, **kwargs)
        cassette.put(ticker, name, variant, value)
        return value

    wrapper.__name__ = name
    wrapper.__qualname__ = name
    wrapper.__doc__ = f"[{mode}] InvestIQ fixtures wrapper for {name}"
    wrapper.__wrapped__ = real  # type: ignore[attr-defined]
    return wrapper


def install(cassette: Cassette, mode: str) -> None:
    """把磁带包装器装到 agent 模块的命名空间上。

    ``mode`` 取 ``"record"`` 或 ``"replay"``。可重复调用(幂等)。
    """
    if mode not in ("record", "replay"):
        raise ValueError(f"未知模式: {mode!r}")

    import importlib

    for module_name, fn_name, variant_param in TARGETS:
        module = importlib.import_module(module_name)
        setattr(
            module,
            fn_name,
            _make_wrapper(module, fn_name, cassette, mode, variant_param),
        )


def uninstall() -> None:
    """把 agent 模块恢复成原始函数(测试用完清理,避免污染同进程的其它测试)。"""
    import importlib

    for module_name, fn_name, _variant in TARGETS:
        module = importlib.import_module(module_name)
        key = f"{module.__name__}.{fn_name}"
        if key in _ORIGINALS:
            setattr(module, fn_name, _ORIGINALS[key])
