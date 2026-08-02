"""
groq.py
-------
Groq (OpenAI-compatible) LLM provider — a drop-in alternative to gemini.py.

Driven by the LangGraph state exactly like gemini.py: the workflow passes its
state dict to generate_from_state(), which reads `provider`, `model`,
`temperature`, `system_prompt` and `conversation_text` from it and dispatches
to Groq. Nothing in this module knows about FastAPI, sessions, or the graph.

Config (all optional, sensible defaults):
    GROQ_API_KEY                 - required to actually reach the API
    GROQ_MODEL                   - default: llama-3.3-70b-versatile
    GROQ_TEMPERATURE             - default: 0.0
    GROQ_RETRY_ATTEMPTS          - default: 3
    GROQ_RETRY_DELAY_SECONDS     - default: 3
    GROQ_MAX_RETRY_DELAY_SECONDS - cap on backoff, default: 60
"""

import os
import re
import time

from openai import OpenAI

DEFAULT_MODEL = "llama-3.3-70b-versatile"

# Server-side errors that resolve on their own in seconds (per-minute token
# or request windows) and are worth retrying.
_TRANSIENT_MARKERS = (
    "429",
    "503",
    "rate limit",
    "unavailable",
    "overloaded",
    "high demand",
)


class QuotaExceededError(RuntimeError):
    """Groq daily request quota exhausted (resets at midnight UTC)."""


def generate_from_state(state: dict) -> str:
    """
    Generate a code response from the LangGraph state.

    `provider`, `model` and `temperature` come from the state first, falling
    back to env vars.
    """
    provider = (state.get("provider") or "groq").lower()
    if provider != "groq":
        raise ValueError(
            f"Unknown LLM provider '{provider}'. This module only handles 'groq'."
        )

    model = state.get("model") or os.environ.get("GROQ_MODEL") or DEFAULT_MODEL
    temperature = float(
        state.get("temperature")
        if state.get("temperature") is not None
        else os.environ.get("GROQ_TEMPERATURE", "0.0")
    )

    return call_groq(
        system_prompt=state["system_prompt"],
        conversation_text=state["conversation_text"],
        model=model,
        temperature=temperature,
    )


def call_groq(
    system_prompt: str,
    conversation_text: str,
    model: str | None = None,
    temperature: float = 0.0,
) -> str:
    """A single Groq chat completion call with retry/backoff.

    Backoff honors the Retry-After header when Groq sends one (capped by
    GROQ_MAX_RETRY_DELAY_SECONDS). A hard daily-cap 429 fails fast with a
    clear QuotaExceededError — retrying cannot reset a daily window.
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set — add it to the environment to use Groq "
            "as the LLM provider."
        )
    client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
    model = model or os.environ.get("GROQ_MODEL") or DEFAULT_MODEL
    retry_attempts = int(os.environ.get("GROQ_RETRY_ATTEMPTS", "3"))
    retry_delay = float(os.environ.get("GROQ_RETRY_DELAY_SECONDS", "3"))
    max_retry_delay = float(os.environ.get("GROQ_MAX_RETRY_DELAY_SECONDS", "60"))

    last_exception = None
    for attempt in range(retry_attempts):
        try:
            completion = client.chat.completions.create(
                model=model,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": conversation_text},
                ],
            )
            content = completion.choices[0].message.content
            return content.strip() if content else ""
        except Exception as e:
            last_exception = e
            if _is_daily_cap(e):
                raise QuotaExceededError(_quota_message(e, model))
            if _is_transient(e) and attempt < retry_attempts - 1:
                delay = min(_retry_after_seconds(e) or retry_delay, max_retry_delay)
                time.sleep(delay)
                continue
            raise

    raise last_exception


def _is_transient(err: Exception) -> bool:
    text = str(err).lower()
    return any(marker in text for marker in _TRANSIENT_MARKERS)


def _is_daily_cap(err: Exception) -> bool:
    text = str(err).lower()
    return "daily limit" in text or "requests per day" in text or "per day" in text


def _retry_after_seconds(err: Exception) -> float | None:
    """Retry-After header from the API error, or a delay parsed from its message."""
    headers = getattr(err, "headers", None)
    if headers and hasattr(headers, "get"):
        val = headers.get("retry-after")
        if val:
            try:
                return float(val)
            except (TypeError, ValueError):
                pass
    text = str(err)
    match = re.search(r"retry.?after[:\s]+(\d+)", text, re.IGNORECASE)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            pass
    match = re.search(r"(\d+)\s*seconds", text, re.IGNORECASE)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            pass
    return None


def _quota_message(err: Exception, model: str) -> str:
    return (
        f"Groq daily quota exhausted for model '{model}' (resets at midnight UTC). "
        f"Try again tomorrow or switch models via GROQ_MODEL. Provider response: {err}"
    )
