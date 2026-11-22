"""Provider-neutral LLM interface.

Agent code depends only on this module. Groq-specific code lives in ``groq_client.py``;
adding another OpenAI-compatible provider means writing one more class that implements
``LLMProvider`` and registering it in ``get_provider``.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

SETUP_MESSAGE = (
    "Groq is not configured: GROQ_API_KEY is missing. "
    "Create an API key in the Groq Console (https://console.groq.com/keys), copy .env.example to .env, "
    "set GROQ_API_KEY there, and restart the app. Running in deterministic fallback mode until then."
)


# ----------------------------------------------------------------------------- errors
class LLMError(Exception):
    """Base class. Messages are always safe to show (never contain secrets)."""
    retryable = False


class LLMConfigError(LLMError):
    """Missing key, disabled provider, unknown provider, or invalid model ID. Not retryable."""


class LLMAuthError(LLMError):
    """Key rejected or lacks permission. Not retryable."""


class LLMRateLimitError(LLMError):
    retryable = True

    def __init__(self, message: str, retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


class LLMUnavailableError(LLMError):
    """Network error, timeout or provider-side 5xx."""
    retryable = True


class LLMResponseError(LLMError):
    """The provider answered but the request or response was unusable."""


# ----------------------------------------------------------------------------- data types
@dataclass
class ChatMessage:
    role: str  # "system" | "user" | "assistant"
    content: str

    def __post_init__(self) -> None:
        if self.role not in ("system", "user", "assistant"):
            raise ValueError(f"Unsupported role: {self.role}")


@dataclass
class LLMResponse:
    text: str
    model: str
    provider: str
    request_id: str | None = None
    finish_reason: str | None = None
    usage: dict[str, int] = field(default_factory=dict)
    latency_s: float = 0.0
    attempts: int = 1
    cached: bool = False

    def metadata(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if k != "text"}


@runtime_checkable
class LLMProvider(Protocol):
    name: str
    model: str

    def chat(self, messages: list[ChatMessage], *, json_mode: bool = False,
             temperature: float = 0.2, max_tokens: int = 1024) -> LLMResponse: ...


# ----------------------------------------------------------------------------- cache
class ResponseCache:
    """Small SQLite-backed cache keyed by (provider, model, messages, options).

    Stored in its own file so it can never interfere with the experiment registry.
    """

    def __init__(self, path: Path | str | None = None, ttl_s: float = 24 * 3600):
        self.ttl_s = ttl_s
        self._lock = threading.Lock()
        target = str(path) if path else ":memory:"
        if path:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(target, check_same_thread=False)
        self._conn.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, created REAL, payload TEXT)")

    @staticmethod
    def key(provider: str, model: str, messages: list[ChatMessage], **opts: Any) -> str:
        blob = json.dumps({"p": provider, "m": model, "msgs": [m.__dict__ for m in messages], "o": opts},
                          sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()

    def get(self, key: str) -> LLMResponse | None:
        with self._lock:
            row = self._conn.execute("SELECT created, payload FROM cache WHERE key = ?", (key,)).fetchone()
        if not row or time.time() - row[0] > self.ttl_s:
            return None
        data = json.loads(row[1])
        data["cached"] = True
        return LLMResponse(**data)

    def put(self, key: str, resp: LLMResponse) -> None:
        with self._lock:
            self._conn.execute("INSERT OR REPLACE INTO cache VALUES (?, ?, ?)",
                               (key, time.time(), json.dumps(resp.__dict__)))
            self._conn.commit()


class CachedProvider:
    """Wraps any provider so identical requests don't spend free-tier quota twice."""

    def __init__(self, inner: LLMProvider, cache: ResponseCache):
        self.inner = inner
        self.cache = cache
        self.name = inner.name
        self.model = inner.model

    def chat(self, messages: list[ChatMessage], *, json_mode: bool = False,
             temperature: float = 0.2, max_tokens: int = 1024) -> LLMResponse:
        k = ResponseCache.key(self.name, self.model, messages, json_mode=json_mode,
                              temperature=temperature, max_tokens=max_tokens)
        hit = self.cache.get(k)
        if hit is not None:
            return hit
        resp = self.inner.chat(messages, json_mode=json_mode, temperature=temperature, max_tokens=max_tokens)
        self.cache.put(k, resp)
        return resp


# ----------------------------------------------------------------------------- factory
@dataclass
class ProviderStatus:
    provider: LLMProvider | None
    enabled: bool
    message: str

    @property
    def available(self) -> bool:
        return self.provider is not None


def get_provider(cache_path: Path | str | None = None, *, use_cache: bool = True) -> ProviderStatus:
    """Build the configured provider, or explain why the app is in fallback mode.

    Never raises: configuration problems become a ProviderStatus with a setup message.
    """
    from core.config import env_bool, load_env

    load_env()
    if not env_bool("LLM_ENABLED", True):
        return ProviderStatus(None, False, "LLM disabled (LLM_ENABLED=false). Using deterministic mode.")
    name = os.getenv("LLM_PROVIDER", "groq").strip().lower()
    if name != "groq":
        return ProviderStatus(None, False, f"Unknown LLM_PROVIDER '{name}'. Supported: groq. Using deterministic mode.")
    from llm.groq_client import GroqClient

    try:
        client = GroqClient.from_env()
    except LLMConfigError as exc:
        return ProviderStatus(None, False, str(exc))
    provider: LLMProvider = CachedProvider(client, ResponseCache(cache_path)) if use_cache else client
    return ProviderStatus(provider, True,
                          f"Using {client.model} hosted by Groq (remote API, not a local model).")
