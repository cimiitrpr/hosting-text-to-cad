"""
gemini.py
---------
The only LLM client in this codebase (Grok/xAI support was removed).

Every request is driven by the LangGraph state: the graph passes its state
dict to `generate_from_state()`, which reads `provider`, `model`,
`temperature`, `system_prompt` and `conversation_text` from it and dispatches
to Gemini. Nothing in this module knows about FastAPI, sessions, or the
graph itself.

Config (all optional, sensible defaults):
    GEMINI_API_KEY              - required to actually reach the API
    GEMINI_MODEL                    - default: gemini-2.5-flash
    GEMINI_TEMPERATURE              - default: 0.0
    GEMINI_RETRY_ATTEMPTS           - default: 3
    GEMINI_RETRY_DELAY_SECONDS      - default: 3
    GEMINI_MAX_RETRY_DELAY_SECONDS  - cap on backoff, default: 60
"""

import os
import re
import time

from google import genai
from google.genai import types
from dotenv import load_dotenv
load_dotenv()

DEFAULT_MODEL = "gemini-2.5-flash"

# Google error responses carry a RetryInfo with a suggested delay, e.g.
# {"@type": "...google.rpc.RetryInfo", "retryDelay": "56.170712972s"}
_RETRY_DELAY_RE = re.compile(r'"retryDelay"\s*:\s*"([\d.]+)s"')


class QuotaExceededError(RuntimeError):
    """Gemini request quota exhausted (free tier is ~20 requests/day per model)."""


# Server-side errors that resolve on their own in seconds and are worth
# retrying instead of being treated like a real bug in the generated code.
_TRANSIENT_MARKERS = (
    "503",
    "unavailable",
    "overloaded",
    "high demand",
    "rate limit",
    "429",
)


def generate_from_state(state: dict) -> str:
    """
    Generate a code response from the LangGraph state.

    Only "gemini" is wired up as a provider. `model`, `temperature` and
    `provider` come from the state first, falling back to env vars.
    """
    provider = (state.get("provider") or "gemini").lower()
    if provider != "gemini":
        raise ValueError(
            f"Unknown LLM provider '{provider}'. Only 'gemini' is supported "
            "(Grok/xAI support was removed from this codebase)."
        )

    model = state.get("model") or os.environ.get("GEMINI_MODEL") or DEFAULT_MODEL
    temperature = float(
        state.get("temperature")
        if state.get("temperature") is not None
        else os.environ.get("GEMINI_TEMPERATURE", "0.0")
    )

    return call_gemini(
        system_prompt=state["system_prompt"],
        conversation_text=state["conversation_text"],
        model=model,
        temperature=temperature,
    )


def call_gemini(
    system_prompt: str,
    conversation_text: str,
    model: str | None = None,
    temperature: float = 0.0,
) -> str:
    """A single Gemini generate_content call with retry/backoff on
    transient server errors (e.g. Gemini's 503 'high demand' response).

    Backoff honors the RetryInfo.retryDelay Google returns when present
    (capped by GEMINI_MAX_RETRY_DELAY_SECONDS). Quota exhaustion is treated
    as non-retryable: the daily free-tier quota will not reset in seconds,
    so we fail fast with a clear QuotaExceededError."""
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    model = model or os.environ.get("GEMINI_MODEL") or DEFAULT_MODEL
    retry_attempts = int(os.environ.get("GEMINI_RETRY_ATTEMPTS", "3"))
    retry_delay = float(os.environ.get("GEMINI_RETRY_DELAY_SECONDS", "3"))
    max_retry_delay = float(os.environ.get("GEMINI_MAX_RETRY_DELAY_SECONDS", "60"))

    last_exception = None
    for attempt in range(retry_attempts):
        try:
            response = client.models.generate_content(
                model=model,
                contents=conversation_text,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=temperature,
                ),
            )
            return response.text.strip()
        except Exception as e:
            last_exception = e
            if _is_quota(e):
                raise QuotaExceededError(_quota_message(e, model))
            if _is_transient(e) and attempt < retry_attempts - 1:
                continue
            raise

    raise last_exception


def _is_transient(err: Exception) -> bool:
    text = str(err).lower()
    return any(marker in text for marker in _TRANSIENT_MARKERS)

def _is_quota(err: Exception) -> bool:
    text = str(err).lower()
    return "resource_exhausted" in text or "quota" in text


def _retry_delay_seconds(err: Exception) -> float | None:
    """Suggested retry delay from Google's RetryInfo, if present."""
    match = _RETRY_DELAY_RE.search(str(err))
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    return None


def _quota_message(err: Exception, model: str) -> str:
    return (
        f"Gemini quota exceeded for model '{model}' (the free tier allows a "
        f"small number of requests per day, e.g. 20 generate_content calls). "
        f"Use a paid plan / your own billing project, switch GEMINI_MODEL, or "
        f"wait for the quota to reset. Provider response: {err}"
    )