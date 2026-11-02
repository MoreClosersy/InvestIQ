"""契约测试：把升级计划里的"红线"从提示词搬进 pytest。

今天应全绿；标 skip 的是后续切片的验收条件，实现前保持 skipped，不允许删。
背景与决策见 docs/adr/0002-investiq-upgrade.md。

运行：
    python -m pytest tests/test_contract.py -q
"""
from __future__ import annotations

import ast
import importlib
import re
import sys
from pathlib import Path
from typing import get_type_hints

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))  # 允许从任意 cwd 运行

from graph.state import AgentState, initial_state  # noqa: E402
from graph.workflow import build_graph  # noqa: E402


# ------------------------------------------------------------------ helpers

def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    mods: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            mods.add(node.module)
    return mods


def _reducer_fields() -> set[str]:
    hints = get_type_hints(AgentState, include_extras=True)
    return {k for k, v in hints.items() if getattr(v, "__metadata__", None)}


# ------------------------------------------------------------------ 默认拓扑

def test_default_graph_shape() -> None:
    """默认图就是 4 节点固定 DAG：并行取数 → 汇入分析 → 报告。"""
    g = build_graph().get_graph()
    assert {"research", "financial", "analysis", "report"} <= set(g.nodes)
    edges = {(e.source, e.target) for e in g.edges}
    assert edges == {
        ("__start__", "research"),
        ("__start__", "financial"),
        ("research", "analysis"),
        ("financial", "analysis"),
        ("analysis", "report"),
        ("report", "__end__"),
    }


def test_default_graph_has_no_checkpointer_or_store() -> None:
    """默认 build_graph() 不带 checkpointer / store —— 关闭开关时行为等价。"""
    g = build_graph()
    assert getattr(g, "checkpointer", None) is None
    assert getattr(g, "store", None) is None


# ------------------------------------------------------------------ state 契约

def test_state_fields_are_prefilled() -> None:
    """每个字段都必须被 initial_state 预填，节点才不会 KeyError。"""
    hints = get_type_hints(AgentState)
    state = initial_state("AAPL")
    assert set(hints) <= set(state)
    assert state["ticker"] == "AAPL"


def test_only_list_channels_use_add_reducer() -> None:
    """标量字段保持单写者；并行节点不得写同一个标量。

    加新字段时先在这里定 reducer，再写节点 —— 顺序反了就是随机覆盖。
    """
    reducers = _reducer_fields()
    assert {"news_findings", "errors"} <= reducers
    for scalar in (
        "ticker", "company_name", "market_context", "stock_price",
        "financial_metrics", "historical_data", "strengths", "risks",
        "analyst_summary", "data_completeness", "final_report",
    ):
        assert scalar not in reducers, f"{scalar} 不应有 reducer（应为单写者）"


# ------------------------------------------------------------------ 分层红线

def test_financial_agent_has_no_llm_dependency() -> None:
    """财务节点不调 LLM：确定性任务用确定性代码。"""
    mods = _imported_modules(REPO / "agents" / "financial_agent.py")
    banned = ("openai", "langchain_openai", "langchain_core", "tools.llm")
    assert not [m for m in mods if m.startswith(banned)]


def test_llm_agents_use_shared_factory() -> None:
    """所有 LLM 调用走 tools.llm.build_chat_model，便于统一超时/重试/预算。"""
    for name in ("analysis_agent.py", "report_agent.py", "research_agent.py"):
        mods = _imported_modules(REPO / "agents" / name)
        assert "tools.llm" in mods, f"{name} 应通过 tools.llm 建模型"
        assert not [m for m in mods if m.startswith("langchain_openai")], (
            f"{name} 不得直接构造 ChatOpenAI"
        )


def test_runtime_code_does_not_import_eval() -> None:
    """agents/ 与 tools/ 不得依赖 eval/ —— 否则评测和运行时互相污染。"""
    for path in list((REPO / "agents").glob("*.py")) + list((REPO / "tools").glob("*.py")):
        mods = _imported_modules(path)
        assert not [m for m in mods if m == "eval" or m.startswith("eval.")], path.name


# ------------------------------------------------------------------ 评测接缝

def test_eval_fixture_targets_still_exist() -> None:
    """冻结层按名字打补丁；research 换了检索函数名 = 回放静默走真网络。

    这条测试就是"Phase 2 不许改 search_company_news 名字"的机器版本。
    """
    from eval import fixtures

    for module_name, fn_name, _variant in fixtures.TARGETS:
        module = importlib.import_module(module_name)
        assert callable(getattr(module, fn_name, None)), f"{module_name}.{fn_name} 不存在"


def test_replay_seam_serves_frozen_data(tmp_path: Path) -> None:
    """装一次回放，检索函数就必须返回磁带里的数据，而不是联网。"""
    from eval import fixtures

    import agents.research_agent as research

    cassette = fixtures.Cassette(tmp_path / "cassette.json")
    cassette.sources = {
        "AAPL": {
            "search_company_news": [
                {"title": "frozen", "url": "https://example.test/a", "content": "body"}
            ]
        }
    }
    fixtures.install(cassette, "replay")
    try:
        got = research.search_company_news("AAPL", "")
        assert got and got[0]["url"] == "https://example.test/a"
        with pytest.raises(fixtures.MissingFixtureError):
            research.search_company_news("MSFT", "")
    finally:
        fixtures.uninstall()


def test_langgraph_version_is_pinned() -> None:
    """锁版本：store / checkpointer API 跨版本会变。"""
    text = (REPO / "requirements.txt").read_text(encoding="utf-8")
    assert re.search(r"^langgraph==", text, re.M), "langgraph 必须精确锁版本"


# ------------------------------------------------------------------ 后续切片验收（实现前 skip）

@pytest.mark.skip(reason="Phase 1 未实现：thread 记忆")
def test_same_thread_does_not_accumulate_lists() -> None:
    """同一 thread 二次 invoke，news_findings 不得把上一轮结果再追加一遍。

    已复现：checkpointer + Annotated[list, add] 会跨调用累加（['run'] → ['run','run']）。
    实现时必须显式隔离每次 run 的列表字段（例如 run 级命名空间或清空策略）。
    """


@pytest.mark.skip(reason="Phase 2 未实现：研究循环")
def test_research_loop_is_bounded() -> None:
    """fake LLM 永远说"还缺"时必须停在 max_steps，且 search_calls ≤ 上限。"""


@pytest.mark.skip(reason="Phase 3 未实现：critic 路由")
def test_critic_routes_cover_all_branches() -> None:
    """四条路由：正常 / 有违规可重试 / 拒答 / 重试后仍违规；analysis 最多跑 2 次。"""


@pytest.mark.skip(reason="Phase 4 未实现：长期记忆")
def test_memory_is_scoped_and_expires() -> None:
    """TTL 过期不复用、AAPL/MSFT namespace 隔离、supersede 后读不到旧记录。"""
