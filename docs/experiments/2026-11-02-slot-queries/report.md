# S3 试点：缺口是“查询不够”还是“结果取少了”？

- 录制：2026-11-02T13:03:19+00:00；5 个标的 × 4 条查询（每条 5 结果）+ 1 条控制查询（20 结果）
- 全部为真实 Tavily 调用，原始结果录在 `raw.json` / `control.json`，分析完全离线可复跑
- 槽位规则是关键词启发式；本报告给出“宽松/严格”两种口径，并说明为什么只信严格口径

## 先说指标问题：宽松口径会把导航栏当证据

严格口径要求槽位关键词出现在标题，或与公司名同句。宽松口径只要求全文出现，
于是 `SEC Filings`、`Risk Factors`、`IPO Calendar` 这类站点框架文字会同时点亮四个槽位。

| ticker | 槽位 | 域名 | 被误判的文字 |
| --- | --- | --- | --- |
| AAPL | 风险/法律 | `finance.yahoo.com` | …M EST  All  News  Earnings Calls  Press Releases  SEC Filings   Apple to report Q4 earnings Monday, wit… |
| AAPL | 分析师动作 | `cmelitegroup.com` | …ns. A result near the top would point to a strong upgrade cycle. 2. iPhone sales. The quarter includes the … |
| AAPL | 分析师动作 | `chartmill.com` | …w consensus estimates. Hover or focus a value for analyst count and low–high range when supplied. Past peri… |
| AAPL | 财报/指引 | `marketwatch.com` | …[]( |  View All Upgrades & Downgrades  ## FY 2026 EPS Estimate Trends  Current$8.84 1 Month Ago$8.84 3 … |

## 三种取数方案（同一批标的、同一天）

| 方案 | 搜索次数 | 平均文档数 | 宽松槽位 | **严格槽位** |
| --- | ---: | ---: | ---: | ---: |
| 基线 q0 × 5（现状） | 1 | 5.0 | 3.00/5 | **2.00/5** |
| q0 × 20（只取更多） | 1 | 20.0 | 5.00/5 | **4.00/5** |
| 4 条查询 × 5（多问几种） | 4 | 19.8 | 4.80/5 | **4.40/5** |

## 文档数曲线（严格口径）

| 文档数 | q0 单查询 | 4 条查询合计 |
| ---: | ---: | ---: |
| 5 | 1.80 | 2.00 |
| 10 | 3.00 | 3.00 |
| 15 | 3.40 | 3.20 |
| 20 | 4.00 | 4.40 |

> 曲线的 n=5 点取的是控制组 20 条的前 5 条（1.80），与表格里独立调用的基线 5 条（2.00）不是同一次响应；0.2 的差来自 Tavily 排序抖动，也说明单次采样的精度有限。

## 逐标的（严格口径）

| ticker | 文档 q0×5 / q0×20 / 4q | 槽位 q0×5 | 槽位 q0×20 | 槽位 4q | 4q 仍缺 |
| --- | --- | ---: | ---: | ---: | --- |
| AAPL | 5 / 20 / 20 | 2/5 | 3/5 | 4/5 | 重大事件 |
| JPM | 5 / 20 / 20 | 1/5 | 4/5 | 3/5 | 重大事件、产品/战略 |
| LCID | 5 / 20 / 19 | 1/5 | 5/5 | 5/5 | — |
| NIO | 5 / 20 / 20 | 3/5 | 3/5 | 5/5 | — |
| SOFI | 5 / 20 / 20 | 3/5 | 5/5 | 5/5 | — |

## 来源构成（4 条查询去重后）

- 聚合/导航页：38/99（38.4%）

## 判读规则（先写，后看数）

- 若“只取更多”（q0×20）与“多问几种”（4q）在严格口径下差不多，说明缺口的成因是**结果被 max_results 截断**，不是查询不够聪明；先调条数，别急着上循环。
- 若 4q 明显高于 q0×20，才说明查询多样性本身有价值，S3 的定向查询值得固化。
- 若两者都上不去，说明瓶颈在来源质量，需要换源（IR/SEC/司法），加查询、加循环都没用。
