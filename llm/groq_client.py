"""Groq chat-completions client for GPT-OSS (default: openai/gpt-oss-20b).

All Groq-specific behaviour lives here: SDK initialisation, error mapping, retries with
backoff, and secret redaction. Agent code only sees the neutral types in ``llm.provider``.

Note: GPT-OSS in this configuration is hosted by Groq and reached over the network.
It is NOT a local model.
"""
from __future__ import annotations

import logging
import os
import random
import re
import time
from typing import Any, Callable

from llm.provider import (
    SETUP_MESSAGE,
    ChatMessage,
    LLMAuthError,
    LLMConfigError,
    LLMError,
    LLMRateLimitError,
    LLMResponse,
    LLMResponseError,
    LLMUnavailableError,
)

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "openai/gpt-oss-20b"
_PLACEHOLDER_KEYS = {"", "your_groq_api_key_here", "changeme"}
_KEY_PATTERN = re.compile(r"gsk_[A-Za-z0-9]{8,}")


def _redact(text: str, secret: str | None) -> str:
    text = str(text)
    if secret:
        text = text.replace(secret, "[REDACTED]")
    return _KEY_PATTERN.sub("[REDACTED]", text)


class GroqClient:
    """Reusable Groq client with bounded retries and safe error handling."""

    name = "groq"

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        *,
        timeout_s: float = 30.0,
        max_retries: int = 3,
        backoff_base_s: float = 1.0,
        backoff_max_s: float = 20.0,
        reasoning_effort: str | None = "low",
        client: Any | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        key = (api_key if api_key is not None else os.getenv("GROQ_API_KEY", "")).strip()
        if key in _PLACEHOLDER_KEYS:
            raise LLMConfigError(SETUP_MESSAGE)
        self.__api_key = key  # name-mangled; never logged or rendered
        self.model = (model or os.getenv("GROQ_MODEL") or DEFAULT_MODEL).strip()
        self.timeout_s = float(timeout_s)
        self.max_retries = max(0, int(max_retries))
        self.backoff_base_s = backoff_base_s
        self.backoff_max_s = backoff_max_s
        self.reasoning_effort = reasoning_effort
        self._sleep = sleep
        if client is not None:
            self._client = client  # injected (tests)
        else:
            from groq import Groq
            # SDK retries disabled: this class owns retry policy so it is bounded and observable.
            self._client = Groq(api_key=key, timeout=self.timeout_s, max_retries=0)

    def __repr__(self) -> str:  # never include the key
        return f"GroqClient(model={self.model!r}, timeout_s={self.timeout_s}, max_retries={self.max_retries})"

    @classmethod
    def from_env(cls, **overrides: Any) -> "GroqClient":
        from core.config import env_float, env_int, load_env

        load_env()
        effort = os.getenv("GROQ_REASONING_EFFORT", "low").strip().lower() or None
        kwargs: dict[str, Any] = dict(
            timeout_s=env_float("GROQ_TIMEOUT_SECONDS", 30.0),
            max_retries=env_int("GROQ_MAX_RETRIES", 3),
            reasoning_effort=None if effort in (None, "none") else effort,
        )
        kwargs.update(overrides)
        return cls(**kwargs)

    # ------------------------------------------------------------------ public API
    def chat(
        self,
        messages: list[ChatMessage],
        *,
        json_mode: bool = False,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        """Send system/user messages to the configured model and return text + metadata."""
        if not messages:
            raise ValueError("messages must not be empty")
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature,
            "max_completion_tokens": max_tokens,
            "timeout": self.timeout_s,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        if self.reasoning_effort and "gpt-oss" in self.model:
            payload["reasoning_effort"] = self.reasoning_effort

        attempt = 0
        while True:
            attempt += 1
            started = time.monotonic()
            try:
                raw = self._client.chat.completions.create(**payload)
                return self._to_response(raw, time.monotonic() - started, attempt)
            except LLMError:
                raise
            except Exception as exc:  # map SDK errors to safe, typed errors
                err = self._map_error(exc)
            if not err.retryable or attempt > self.max_retries:
                if err.retryable:
                    err = type(err)(f"{err} (gave up after {attempt} attempts)")  # type: ignore[call-arg]
                raise err from None  # `from None`: don't chain SDK objects that hold request details
            delay = self._backoff(attempt, getattr(err, "retry_after", None))
            logger.warning("Groq transient error (attempt %d/%d): %s; retrying in %.1fs",
                           attempt, self.max_retries + 1, err, delay)
            self._sleep(delay)

    # ------------------------------------------------------------------ internals
    def _backoff(self, attempt: int, retry_after: float | None) -> float:
        if retry_after is not None and retry_after >= 0:
            return min(retry_after, self.backoff_max_s)
        exp = self.backoff_base_s * (2 ** (attempt - 1))
        return min(self.backoff_max_s, exp) * (0.8 + 0.4 * random.random())

    def _to_response(self, raw: Any, latency: float, attempt: int) -> LLMResponse:
        try:
            choice = raw.choices[0]
            text = choice.message.content or ""
        except (AttributeError, IndexError, TypeError) as exc:
            raise LLMResponseError("Groq returned a response without any message content") from exc
        usage = {}
        u = getattr(raw, "usage", None)
        if u is not None:
            for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
                v = getattr(u, k, None)
                if isinstance(v, int):
                    usage[k] = v
        if not text.strip():
            raise LLMResponseError(f"Groq returned empty content (finish_reason={getattr(choice, 'finish_reason', None)})")
        return LLMResponse(
            text=text, model=getattr(raw, "model", None) or self.model, provider=self.name,
            request_id=getattr(raw, "id", None), finish_reason=getattr(choice, "finish_reason", None),
            usage=usage, latency_s=round(latency, 3), attempts=attempt,
        )

    def _map_error(self, exc: Exception) -> LLMError:
        import groq

        msg = _redact(getattr(exc, "message", None) or str(exc), self.__api_key)
        code = ""
        body = getattr(exc, "body", None)
        if isinstance(body, dict):
            inner = body.get("error", body)
            if isinstance(inner, dict):
                code = str(inner.get("code") or "")

        if isinstance(exc, groq.AuthenticationError):
            return LLMAuthError("Groq rejected the API key (401). Check GROQ_API_KEY in your .env file.")
        if isinstance(exc, groq.PermissionDeniedError):
            return LLMAuthError(f"Groq denied access (403) to model '{self.model}' for this key: {msg}")
        if isinstance(exc, groq.RateLimitError):
            retry_after = None
            resp = getattr(exc, "response", None)
            if resp is not None:
                try:
                    retry_after = float(resp.headers.get("retry-after"))
                except (TypeError, ValueError):
                    retry_after = None
            return LLMRateLimitError(
                "Groq rate limit reached (free-tier quotas are limited). "
                "Wait a moment and try again; deterministic results are still available.",
                retry_after=retry_after,
            )
        if isinstance(exc, groq.NotFoundError) or code in ("model_not_found", "model_decommissioned"):
            return LLMConfigError(
                f"Model '{self.model}' is not available to this Groq account. "
                "Set GROQ_MODEL in .env to a model listed in the Groq Console."
            )
        if isinstance(exc, groq.APITimeoutError):
            return LLMUnavailableError(f"Groq request timed out after {self.timeout_s:.0f}s")
        if isinstance(exc, groq.APIConnectionError):
            return LLMUnavailableError("Could not reach the Groq API (network error)")
        if isinstance(exc, groq.InternalServerError):
            return LLMUnavailableError(f"Groq service error: {msg}")
        if isinstance(exc, groq.APIStatusError):
            status = getattr(exc, "status_code", None)
            if status in (502, 503, 504):
                return LLMUnavailableError(f"Groq temporarily unavailable ({status})")
            if code == "json_validate_failed":
                return LLMResponseError("The model did not produce valid JSON")
            return LLMResponseError(f"Groq API error ({status}): {msg}")
        return LLMResponseError(f"Unexpected Groq client error: {type(exc).__name__}: {msg}")
