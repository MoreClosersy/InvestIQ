# InvestIQ

> Multi-agent investment research system: ticker in, structured Markdown report out.

InvestIQ uses [LangGraph](https://github.com/langchain-ai/langgraph) to orchestrate four specialized agents — Research, Financial, Analysis, and Report — that together turn a single stock ticker into a fully-structured Markdown investment research report in 20–30 seconds.

The pipeline is covered by an offline evaluation harness ([`eval/`](eval/)) that freezes every external input into a replayable cassette, so runs are reproducible and differences can actually be attributed to a code change rather than to a moving market.

![Workflow](docs/workflow.png)

## What it does

```
Input:  "AAPL"

Output: 7-section Markdown report
        ├── Executive Summary
        ├── Key Metrics            (live yfinance data)
        ├── Recent Developments    (Tavily-sourced news)
        ├── Investment Strengths   (data-cited)
        ├── Key Risks              (data-cited)
        ├── Synthesis              (2-paragraph narrative)
        └── Disclaimer             (compliance-aware)
```

## Demo

![Demo](docs/screenshot.png?v=2)

## Architecture

| Agent     | Tool/Model                | LLM? | Responsibility                                |
| --------- | ------------------------- | ---- | --------------------------------------------- |
| Research  | Tavily + GPT-4o-mini      | ✅   | Search and summarize recent news              |
| Financial | yfinance                  | ❌   | Fetch price, fundamentals, volatility         |
| Analysis  | GPT-4o-mini + Pydantic    | ✅   | Synthesize strengths, risks, narrative        |
| Report    | GPT-4o-mini               | ✅   | Generate final Markdown report                |

Key design decisions:

- **Financial Agent intentionally does not use an LLM.** yfinance returns structured numeric data — passing those numbers through an LLM only adds latency, cost, and hallucination risk. The agent emits clean dicts that downstream LLM agents consume as ground truth.
- **Parallel fan-out, then fan-in.** Research and Financial run concurrently from `START`; Analysis waits for both to complete (LangGraph's implicit barrier on multiple inbound edges) before running. This roughly halves end-to-end latency vs. a sequential pipeline.
- **Soft failure mode.** Each agent catches its own exceptions and appends to a shared `errors` channel (with an `add` reducer) instead of raising. One agent's failure never kills the pipeline; downstream agents see the missing data and degrade gracefully. The Analysis agent emits an explicit `data_completeness` note that the Report agent surfaces in the Disclaimer.
- **Compliance-aware design.** The Disclaimer requires a verbatim "Data completeness: …" sentence (grep-able for automated audits), and the long-form narrative section is named "Synthesis" rather than "Outlook" to avoid implying forward-looking statements under SEC framing.

## Tech stack

- **LangGraph 1.2** — workflow orchestration with typed shared state
- **OpenAI GPT-4o-mini** (via LangChain) — Research / Analysis / Report LLM calls
- **Tavily API** — recent news search
- **yfinance** — live price, fundamentals, and historical price series
- **FastAPI + uvicorn** — HTTP API and ASGI server
- **Single-file HTML + marked.js** (CDN) — minimal browser UI, no build step
- **Docker + docker-compose** — one-command local deployment

## Quick start

### Prerequisites

- Python 3.11+
- An OpenAI API key
- A Tavily API key (free tier: 1,000 searches/month)

### Local setup

```bash
git clone https://github.com/MoreClosersy/InvestIQ.git
cd InvestIQ

python3.11 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

cp .env.example .env
# Edit .env and fill in OPENAI_API_KEY and TAVILY_API_KEY.

uvicorn api.main:app --reload
```

Then open http://127.0.0.1:8000 in your browser.

### Docker setup

```bash
cp .env.example .env
# Edit .env and fill in OPENAI_API_KEY and TAVILY_API_KEY.

docker compose up --build
```

Then open http://localhost:8000 in your browser.

## Project structure

```
InvestIQ/
├── agents/              # LangGraph nodes, one per agent role
│   ├── research_agent.py    # Tavily search + LLM news summarization
│   ├── financial_agent.py   # yfinance data fetch (no LLM)
│   ├── analysis_agent.py    # Structured Pydantic output (strengths/risks/...)
│   └── report_agent.py      # Final Markdown report generation
├── tools/               # External integrations
│   ├── search_tool.py       # Tavily wrapper
│   └── finance_tool.py      # yfinance wrapper
├── graph/               # Shared state and workflow assembly
│   ├── state.py             # AgentState TypedDict + reducers
│   └── workflow.py          # StateGraph wiring + compile
├── api/                 # FastAPI app
│   └── main.py              # /research endpoint + minimal HTML UI
├── eval/                # Offline evaluation harness (external driver)
│   ├── fixtures.py          # Freeze/replay yfinance + Tavily inputs (record/replay cassette)
│   ├── grounding.py         # Contract layer: trace every number in a report back to its input
│   ├── run_eval.py          # Test set runner, outcome classification, report rendering
│   ├── cassettes/           # Frozen inputs — committed so runs are reproducible
│   └── results/             # Run artifacts + controlled A/B evidence
├── tests/               # Smoke tests + offline regression tests + manual integration runners
├── docs/                # Workflow diagram + sample report
├── Dockerfile           # Production image (python:3.11-slim)
├── docker-compose.yml   # One-command local deployment
├── .dockerignore        # Keeps .venv / tests / .git out of the image
├── requirements.txt     # Pinned dependencies
└── .env.example         # Template for API key configuration
```

## Sample output

See [docs/sample-report.md](docs/sample-report.md) for a full Apple Inc. report generated by InvestIQ.

## Evaluation

`eval/` is an **external driver**: it only calls `graph.workflow.build_graph()` and never modifies agent code.

```bash
python eval/run_eval.py --replay --fixtures eval/cassettes/frozen.json   # 24 tickers, offline
python -m pytest tests/ -q                                              # 67 offline tests, no keys, no network
```

- **Frozen inputs.** yfinance quotes move every minute and Tavily returns fresh articles, so two runs of the same commit otherwise see different inputs and any output difference is unattributable. `--record` captures the raw responses into a JSON cassette; `--replay` serves them back with no network access at all. Replay is strict — a missing ticker aborts rather than silently going live. Only *inputs* are frozen; LLM sampling is deliberately left alive because it is a property under measurement, not noise.
- **Mutually-exclusive outcome classification.** Every request lands in exactly one bucket — `ok` / `abstained_correctly` / `hallucinated` / `degraded` / `failure` / `timeout` — with separate denominators. A single success rate hid both directions of failure: refusing an invalid ticker was counted as a failure, and inventing an analysis with zero financial data was counted as a success.
- **Number grounding (contract layer).** Every number token in a report is traced back to a raw input (yfinance fields or news text). Across 24 tickers: **700 numbers — 286 financially grounded, 414 news-only, 0 ungrounded.** The risk surface is not fabrication but *provenance*: roughly six in ten numbers come from news rather than the structured data pipeline.
- **Frozen-input A/B.** Replaying one cassette against two revisions of the code isolated a real bug: invalid tickers with zero financial data were producing full analyses assembled from generic news. **Hallucinated reports went 3 → 0, correct refusals 1 → 4.**

Full design notes, evidence and the honest limitations of this harness are in [`eval/README.md`](eval/README.md).

## What I'd do next

- **RetryPolicy on Research and Financial nodes** — Tavily and yfinance both have occasional transient failures; LangGraph's `RetryPolicy` on `add_node` would absorb these without changing the graph shape.
- **Caching layer** — same ticker re-requested within 5 minutes currently re-runs the full ~$0.01 pipeline. An in-memory TTL cache (or Redis for multi-instance deployment) keyed on ticker would cut both cost and latency dramatically.
- **Multi-ticker batch analysis** — LangGraph's `Send` API supports dynamic fan-out; a single request like `["AAPL", "MSFT", "GOOGL"]` could run N parallel workflows and return a combined comparison report.
- **Sector baseline injection** — currently the Analysis agent has no comparative anchor (it can't say "P/E of 37 is high *for this sector*"). Pulling sector-average fundamentals into the prompt would unlock relative valuation reasoning.
- **Cross-source fact verification** — numbers that appear in news snippets (revenue, growth %, earnings) should be cross-checked against yfinance fundamentals before being cited as strengths/risks, to catch reporter errors and stale figures. The grounding check measures the exposure (59% of cited numbers are news-only) but does not yet reconcile them.
- **Caliber-conflict detection** — an AAPL report cites a quarterly EPS of `$2.02` from news alongside a trailing annual `eps` of `8.74` from yfinance. Both are grounded, but nothing flags that two different bases are being compared. This needs a semantic layer (assertion + unit/period) on top of the grounding check.
- **Golden set with hand-labelled facts** — the current checks catch *fabrication* and *loss of provenance*, but not *omission*. Asserting that each report mentions its expected key facts requires a human-labelled set, which is the missing half of the evaluation.

## License

MIT
