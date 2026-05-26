FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Dependencies first so code-only edits don't bust the install cache.
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy only the source dirs we actually need; tests/, .venv/, .git/, etc.
# are excluded both by .dockerignore and by being absent from this list.
COPY agents/ ./agents/
COPY api/ ./api/
COPY graph/ ./graph/
COPY tools/ ./tools/

EXPOSE 8000

# Healthcheck hits /health using stdlib urllib (no need to install curl).
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=4)" || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
