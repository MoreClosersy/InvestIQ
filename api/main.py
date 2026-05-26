"""FastAPI entry point for InvestIQ.

Endpoints:
    GET  /          -> Serves a minimal HTML UI for entering a ticker
    POST /research  -> Runs the LangGraph workflow and returns the final report
    GET  /health    -> Health check

The compiled LangGraph is built once at startup and reused across requests
(avoids re-compiling the graph on every call).
"""
from __future__ import annotations

import time

from dotenv import load_dotenv

load_dotenv(override=True)

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.responses import HTMLResponse  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from graph.state import initial_state  # noqa: E402
from graph.workflow import build_graph  # noqa: E402


app = FastAPI(
    title="InvestIQ",
    description="Multi-agent investment research system",
    version="1.0.0",
)

# Build graph once at startup (avoid rebuilding on every request).
_graph = build_graph()


class ResearchRequest(BaseModel):
    ticker: str = Field(
        ..., min_length=1, max_length=10, description="Stock ticker symbol, e.g., 'AAPL'"
    )


class ResearchResponse(BaseModel):
    ticker: str
    company_name: str
    final_report: str
    strengths: list[str]
    risks: list[str]
    data_completeness: str
    elapsed_seconds: float
    errors: list[str]


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/research", response_model=ResearchResponse)
async def research(request: ResearchRequest) -> ResearchResponse:
    """Run the multi-agent research workflow for the given ticker."""
    ticker = request.ticker.upper().strip()

    if not ticker.isalnum():
        raise HTTPException(status_code=400, detail="Ticker must be alphanumeric")

    start = time.time()
    try:
        initial = initial_state(ticker)
        result = _graph.invoke(initial)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Workflow failed: {exc!r}")

    elapsed = time.time() - start

    return ResearchResponse(
        ticker=ticker,
        company_name=result.get("company_name", "N/A"),
        final_report=result.get("final_report", ""),
        strengths=result.get("strengths", []),
        risks=result.get("risks", []),
        data_completeness=result.get("data_completeness", ""),
        elapsed_seconds=round(elapsed, 2),
        errors=result.get("errors", []),
    )


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    """Minimal frontend: ticker input + Markdown rendering."""
    return _INDEX_HTML


_INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>InvestIQ - Multi-Agent Investment Research</title>
  <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; max-width: 900px; margin: 2em auto; padding: 0 1em; color: #222; }
    h1 { font-size: 1.5em; margin-bottom: 0.2em; }
    .subtitle { color: #666; margin-bottom: 1.5em; }
    .input-row { display: flex; gap: 8px; margin-bottom: 1.5em; }
    input { flex: 1; padding: 10px 12px; font-size: 16px; border: 1px solid #ccc; border-radius: 6px; }
    button { padding: 10px 20px; font-size: 16px; background: #2563eb; color: white; border: none; border-radius: 6px; cursor: pointer; }
    button:hover { background: #1d4ed8; }
    button:disabled { background: #999; cursor: not-allowed; }
    .status { color: #666; margin: 1em 0; }
    .error { color: #dc2626; background: #fee; padding: 1em; border-radius: 6px; }
    #report { border-top: 1px solid #eee; padding-top: 1.5em; margin-top: 1em; }
    #report table { border-collapse: collapse; margin: 1em 0; }
    #report th, #report td { border: 1px solid #ddd; padding: 6px 12px; }
    #report th { background: #f5f5f5; }
    .meta { color: #888; font-size: 0.9em; margin-bottom: 1em; }
  </style>
</head>
<body>
  <h1>InvestIQ</h1>
  <p class="subtitle">Multi-agent investment research powered by LangGraph</p>

  <div class="input-row">
    <input id="ticker" type="text" placeholder="Enter ticker (e.g., AAPL, MSFT, GOOGL)" value="AAPL" />
    <button id="run-btn" onclick="runResearch()">Generate Report</button>
  </div>

  <div id="status" class="status"></div>
  <div id="report"></div>

  <script>
    async function runResearch() {
      const ticker = document.getElementById('ticker').value.trim().toUpperCase();
      if (!ticker) return;

      const btn = document.getElementById('run-btn');
      const status = document.getElementById('status');
      const report = document.getElementById('report');

      btn.disabled = true;
      btn.textContent = 'Generating...';
      status.textContent = `Running multi-agent workflow for ${ticker}... (this takes 15-25 seconds)`;
      report.innerHTML = '';

      try {
        const response = await fetch('/research', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ ticker })
        });

        if (!response.ok) {
          const err = await response.json();
          throw new Error(err.detail || 'Request failed');
        }

        const data = await response.json();
        status.textContent = '';

        const meta = `<div class="meta">Generated in ${data.elapsed_seconds}s | ${data.errors.length} warnings</div>`;
        report.innerHTML = meta + marked.parse(data.final_report);
      } catch (e) {
        status.innerHTML = `<div class="error">Error: ${e.message}</div>`;
      } finally {
        btn.disabled = false;
        btn.textContent = 'Generate Report';
      }
    }

    document.getElementById('ticker').addEventListener('keydown', (e) => {
      if (e.key === 'Enter') runResearch();
    });
  </script>
</body>
</html>"""
