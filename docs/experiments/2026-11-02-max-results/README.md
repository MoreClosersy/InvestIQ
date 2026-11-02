# `max_results` 5 → 20：证据层与报告层对比

- 日期：2026-11-02
- 成本：24 次 Tavily 搜索（录制）+ 48 次 LLM 运行（两臂评测，约 $0.06）
- 设计：同一次 20 条响应派生 5 条臂，两臂财务输入逐字节相同、同一天、同一模型
- 原始数据：`eval/cassettes/news{5,20}_20261102.json`；评测结果：`eval/results/20261102_news{5,20}/`

## 结论

**按事先写好的规则，默认值改为 20。**

| 指标 | 5 条 | 20 条 | 变化 |
|---|---:|---:|---:|
| 严格槽位（20 有效标的） | 2.45/5 | **3.60/5** | +1.15 |
| 聚合/导航页占比 | 66% | 41% | −25pp |
| 6 类结果 | 20/4/0/0/0/0 | 20/4/0/0/0/0 | 不变 |
| 未接地数字 | 3 | **0** | 未回退 |
| 仅新闻出处数字 | 494 | 500 | +6 |
| 平均单次成本 | $0.00165 | $0.00292 | 1.77× |
| p50 / p95 | 21.2 / 24.3s | 22.4 / 29.9s | 1.06× / 1.23× |
| 平均 prompt tokens | 6,032 | 14,347 | 2.38× |

判定规则在跑之前写在 `evidence_report.md` / ADR 里：未接地不回退且成本 ≤2× → 默认改 20。两条都满足。

## 需要打折的地方

- **5 条臂不是独立的 5 条调用**，而是同一次 20 条响应的前 5 条。试点见过 Tavily 排序抖动（0.2 个槽位），所以 +1.15 应视为上界。
- **绝对延迟不可与 9 月基线比**：本机走美国 VPN，5 条臂 p50 已从 9 月的 11.6s 涨到 21.2s。两臂在同一网络下，相对差（+1.2s / +5.6s）才有意义。
- **未接地 3 → 0 不能说明 20 条更“真”**：3 个里有 2 个是派生值（AMZN 的 P/E 19.99、TLRY 的增长率 14.4%），接地检查只认字面出现在输入里的数字。样本太小，只能说“没有回退”。
- **拒答路径成本涨了 4 倍**（无效代码 $0.0004 → $0.0017）：research 仍然为 20 条新闻做摘要，然后 analysis 才拒答。绝对金额小，但这是结构上可以省掉的一步。
- 单日、n=24、单次采样；没有重复运行一致性检查（`--no-consistency`）。

## 复跑

```bash
# 录制（24 次搜索）
python docs/experiments/2026-11-02-max-results/record_news.py

# 两臂评测（离线回放，需要 OPENAI_API_KEY）
python eval/run_eval.py --replay --fixtures eval/cassettes/news5_20261102.json \
  --no-consistency --out-dir eval/results/20261102_news5
python eval/run_eval.py --replay --fixtures eval/cassettes/news20_20261102.json \
  --no-consistency --out-dir eval/results/20261102_news20

# 对比
python docs/experiments/2026-11-02-max-results/compare_eval.py \
  eval/results/20261102_news5 eval/results/20261102_news20
```

## 下一步

1. 默认改 20 已落地（`TAVILY_MAX_RESULTS`，可回退到 5）。
2. 拒答路径可以省掉 research 摘要：先判 ticker 有无财务数据，或把摘要推迟到 analysis 之后——这是本次顺带发现的成本点。
3. 下一个 ADR 切片：S4 运行时 critic（复用接地检查），不再动取数。
