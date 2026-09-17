# 冻结输入磁带（cassette）

`frozen.json` 是 24 个 ticker 在**录制时刻**的原始外部输入快照：

- 每个 ticker 的 yfinance 三个返回值（`get_stock_snapshot` / `get_financial_metrics` /
  `get_historical_prices`，后者按 `period` 变体分开存）
- 每个 ticker 的 Tavily 原始搜索结果（`search_company_news`，含正文）

**这个文件要进 git。** 冻结输入是 golden set 的输入部分——没有它，别人 clone 下来
跑评测会拿到当天的行情和新闻，任何两次运行都不可比（详见 `eval/fixtures.py`
的模块说明）。

## 用法

```bash
# 离线回放（不碰网络，输入逐字节一致）
python eval/run_eval.py --replay --fixtures eval/cassettes/frozen.json

# 重新录制（会覆盖，需要梯子 + TAVILY_API_KEY）
python eval/run_eval.py --record --fixtures eval/cassettes/frozen.json
```

回放模式下如果磁带缺某个 ticker，harness 会直接报错退出，**不会**静默回退到网络。

## 注意

- 磁带里包含新闻正文，是录制时刻的公开网页内容，会留在 git 历史里。
- `recorded_at` 字段记录了录制时间；跨时间的两次录制**不能**直接对比指标。
- LLM 采样**故意不冻结**：模型输出的非确定性正是被测量的性质，不是要消除的噪声。
  所以回放同一条磁带两次，报告正文仍会不同——这正常。
