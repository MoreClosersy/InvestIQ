"""InvestIQ 评测工具包(外部驱动,不属于 agents/ 与 tools/)。

* ``run_eval`` —— 评测主程序:跑测试集、汇总指标、产出 CSV/JSON/Markdown;
* ``fixtures`` —— 冻结/回放 yfinance 与 Tavily 的输入,让评测可复现。
"""
