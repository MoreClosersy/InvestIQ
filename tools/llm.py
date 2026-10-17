"""Shared ChatOpenAI factory.

Centralizes model selection, timeout, and retry policy so every agent is
subject to the same failure bounds. The OpenAI client defaults to a 600s
timeout, which lets a single stalled call block the whole workflow; the
bounds below cap each node's worst case at
``timeout * (max_retries + 1)``.

Overridable via environment variables so deployment can tune without a code
change:
    LLM_MODEL            (default: gpt-4o-mini)
    LLM_TIMEOUT_SECONDS  (default: 20)
    LLM_MAX_RETRIES      (default: 2)
"""
from __future__ import annotations

import os

from langchain_openai import ChatOpenAI

DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_TIMEOUT_SECONDS = 20.0
DEFAULT_MAX_RETRIES = 2


def build_chat_model(temperature: float) -> ChatOpenAI:
    """Build a ChatOpenAI client with explicit timeout and retry bounds.

    The OpenAI SDK retries connection errors, 408/409/429, and 5xx responses,
    which covers the transient failures worth retrying. Non-retryable errors
    (4xx other than 408/409/429) fail fast.
    """
    return ChatOpenAI(
        model=os.getenv("LLM_MODEL", DEFAULT_MODEL),
        temperature=temperature,
        timeout=float(os.getenv("LLM_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS)),
        max_retries=int(os.getenv("LLM_MAX_RETRIES", DEFAULT_MAX_RETRIES)),
    )
