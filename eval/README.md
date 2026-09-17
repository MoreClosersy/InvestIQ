# `eval/` — 离线评测

**这是一个外部驱动目录。** 它只调用 `graph.workflow.build_graph()`,**从不修改
`agents/` 与 `tools/` 的任何代码**。评测侵入被测系统的那一天,评测本身就没人能验了。

本文件讲**这套东西怎么用、怎么扩展**,以及四层各自回答什么问题。
完整的结论与数字见下面各层的小节;产物的区别见 `results/README.md`。

---

## 四层,各回答一个问题

| 层 | 回答的问题 | 代码 | 需要网络? |
|---|---|---|---|
| 冻结输入 | 两次运行看到的是**同一份输入**吗 | `fixtures.py` | 录制要,回放不要 |
| 结果分类 | 这条请求是**成功、正确拒答、还是幻觉** | `run_eval.py::classify_outcome` | LLM 要 |
| 数字接地 | 报告里每个数字**能追回输入**吗 | `grounding.py` | 不要(纯函数) |
| 一致性 | 重复跑的**事实**和**措辞**稳不稳 | `run_eval.py::compute_consistency` | LLM 要 |

运维指标(延迟/token/成本)不属于质量,单独列,不需要 ground truth。

---

## 常用命令

```bash
# 完整评测,完全离线(不碰 yfinance / Tavily)
python eval/run_eval.py --replay --fixtures eval/cassettes/frozen.json

# 只跑几个 ticker,跳过一致性重跑(调试用)
python eval/run_eval.py --replay --fixtures eval/cassettes/frozen.json \
    --tickers AAPL,ZZZZZZ --no-consistency

# 重新录制磁带(需要梯子 + TAVILY_API_KEY;会覆盖)
python eval/run_eval.py --record --fixtures eval/cassettes/frozen.json

# 换一份测试集
python eval/run_eval.py --replay --fixtures eval/cassettes/frozen.json \
    --test-set eval/test_set.example.json

# 离线单测(不需要 key、不需要网络)
python -m pytest tests/ -q
```

### 回放是严格的

磁带缺某个 ticker 时,**harness 直接退出并列出缺哪些**,不会回退到网络。
理由:静默回退会无声地破坏复现性——你以为在跑回归测试,其实输入已经变了。

### 只冻结输入,不冻结 LLM

模型的非确定性**正是被测量的性质**(见一致性那一层),不是要消除的噪声。
所以回放同一条磁带两次,报告正文仍会不同——这是正常的。

---

## 磁带格式

`cassettes/frozen.json`(提交进仓库,理由见 `cassettes/README.md`):

```json
{
  "version": 1,
  "recorded_at": "...",
  "sources": {
    "AAPL": {
      "get_stock_snapshot":     { "...": "yfinance fast_info" },
      "get_financial_metrics":  { "...": "yfinance .info" },
      "get_historical_prices":  { "3mo": { "...": "yfinance .history 汇总" } },
      "search_company_news":    [ { "title": "...", "content": "..." } ]
    }
  }
}
```

`get_historical_prices` 按 `period` **变体**分开存,不同窗口不能互相顶替。

**补打补丁的位置**:`fixtures.py` 把补丁打在**导入方模块的命名空间**
(`agents.financial_agent.*` / `agents.research_agent.*`),因为两个 agent 用的是
`from ... import fn` 形式——改 `tools` 模块里的名字不会生效。这是个容易踩的坑。

---

## 加一个新检查:照着这个模式

四层里的后三层都遵循同一条设计规则:

> **检查逻辑必须是 `record` 的纯函数。**

原因:归档的 JSON 里已经存了报告正文和原始新闻,所以**改指标定义后可以直接对历史 run
重新评分**,不用再花一次 LLM 的钱。`grounding.py` 就是纯函数,所以接地检查是在
**没有重跑任何请求**的前提下先在归档数据上做出来的。

推论(也是踩过的坑):**派生字段不要存进 `record`**。存了再信任它,就会出现
"归档 JSON 说 A、报告说 B"的两份口径。`outcome` 和 `grounding` 都在导出时现算。

### 具体步骤

1. 写一个纯函数 `check_something(record) -> dict`(新的 `*.py`,或加进 `grounding.py`);
2. 在 `run_eval.py::build_summary` 里聚合成汇总段;
3. 在 `render_markdown` 里加一节;
4. 在 `write_outputs` 的 JSON 导出里注入(像 `outcome` / `grounding` 那样);
5. 在 `tests/` 里加**离线**测试——尤其是边界值和"空输入不能伪装成满分"这两类。

### 三条已经用血换来的原则

1. **出处用白名单,不用黑名单。** 只有 agent 真正喂给 LLM 的字段算证据
   (`grounding.FINANCIAL_SOURCE_FIELDS`)。黑名单会让将来新增的评测内部字段
   (耗时、token、成本)悄悄变成"合法出处"。
2. **不要拿 LLM 自己的输出当出处。** `market_context` 是 Research 的 LLM 摘要,
   用它做校验等于让模型给自己作证,幻觉可以经过一层摘要洗白。只认原始新闻正文。
3. **尽可能给计数和清单,少给百分比。** 一个"接地率"的分母取决于模型写了多少字,
   会诱导人做无意义的横向比较;清单和计数没有这个问题。

---

## 产物

跑一次会在 `--out-dir`(默认 `eval/results/`)写三个**同 stem** 的文件:

```
results_<UTC 时间戳>.csv    逐请求的关键字段,给人眼看
results_<UTC 时间戳>.json   全量(含报告正文、新闻原文、逐条接地结果)
results_<UTC 时间戳>.md     汇总报告
```

同 stem 是刻意的——早先 CSV/JSON 与 Markdown 用了不同命名,结果出现过
"同一次运行两份口径"。保留的几次运行及它们的区别见 `results/README.md`。

---

## 已知边界

- **没有人工标注的 golden set**,所以还测不了"报告漏了关键事实";
- **口径冲突测不了**:季度 EPS(新闻)和年度 EPS(yfinance)可以双双"有出处"
  却并排误导读者;
- 一致性相似度是**词面**指标,不是语义指标;
- 样本量小(24 个请求,无效代码只有 4 个且属同一类失效模式),所以报告里给的是
  **清单和计数**,不是百分比。

展开见上面「已知边界」一节。
