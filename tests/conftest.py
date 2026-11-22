"""Shared fixtures. The whole suite runs without a Groq API key and without network access."""
from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from core.config import Settings
from core.registry import Registry
from demo.seed import seed_demo
from llm.provider import ChatMessage, LLMResponse

FAKE_KEY = "gsk_TESTKEY1234567890abcdefSECRET"


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch):
    """No real key, no .env loading, and any real Groq SDK construction fails loudly."""
    for var in ("GROQ_API_KEY", "GROQ_MODEL", "LLM_PROVIDER", "LLM_ENABLED", "GROQ_REASONING_EFFORT"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr("core.config.load_env", lambda *a, **k: None)

    class _Blocked:
        def __init__(self, *a, **k):
            raise AssertionError("Tests must never construct a real Groq client")

    monkeypatch.setattr("groq.Groq", _Blocked)


@pytest.fixture
def settings(tmp_path) -> Settings:
    s = Settings.from_env(tmp_path / "data")
    s.ensure_dirs()
    return s


@pytest.fixture
def registry(settings) -> Registry:
    reg = Registry(settings.db_path)
    seed_demo(reg, settings)
    yield reg
    reg.close()


class ScriptedProvider:
    """Provider double: returns queued texts or raises queued exceptions; records prompts."""
    name = "scripted"
    model = "openai/gpt-oss-20b"

    def __init__(self, *outputs: Any):
        self.outputs = list(outputs)
        self.prompts: list[list[ChatMessage]] = []

    def chat(self, messages, *, json_mode=False, temperature=0.2, max_tokens=1024):
        self.prompts.append(messages)
        out = self.outputs.pop(0) if len(self.outputs) > 1 else self.outputs[0]
        if isinstance(out, BaseException):
            raise out
        text = out if isinstance(out, str) else json.dumps(out)
        return LLMResponse(text=text, model=self.model, provider=self.name, request_id="req-1")


@pytest.fixture
def scripted():
    return ScriptedProvider


# ----------------------------------------------------------------------------- fake Groq SDK
def sdk_completion(text: str, model: str = "openai/gpt-oss-20b") -> Any:
    return SimpleNamespace(
        id="chatcmpl-123", model=model,
        choices=[SimpleNamespace(message=SimpleNamespace(content=text), finish_reason="stop")],
        usage=SimpleNamespace(prompt_tokens=11, completion_tokens=7, total_tokens=18),
    )


class FakeGroqSDK:
    """Mimics groq.Groq().chat.completions.create with queued results."""

    def __init__(self, *results: Any):
        self.results = list(results)
        self.calls: list[dict[str, Any]] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        r = self.results.pop(0)
        if isinstance(r, BaseException):
            raise r
        return r


def sdk_error(cls_name: str, status: int, message: str = "error", body: Any = None, headers=None):
    import groq

    req = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    if cls_name == "APIConnectionError":
        return groq.APIConnectionError(request=req)
    if cls_name == "APITimeoutError":
        return groq.APITimeoutError(request=req)
    resp = httpx.Response(status, request=req, headers=headers or {})
    return getattr(groq, cls_name)(message, response=resp, body=body)
