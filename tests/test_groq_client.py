"""Groq client: configuration, response handling, error mapping, retries, secret hygiene."""
from __future__ import annotations

import logging

import pytest

from llm.groq_client import DEFAULT_MODEL, GroqClient
from llm.provider import (
    SETUP_MESSAGE, ChatMessage, LLMAuthError, LLMConfigError, LLMRateLimitError,
    LLMResponseError, LLMUnavailableError, get_provider,
)
from tests.conftest import FAKE_KEY, FakeGroqSDK, sdk_completion, sdk_error

MSGS = [ChatMessage("system", "be brief"), ChatMessage("user", "hello")]


def make(sdk, **kw):
    sleeps = []
    client = GroqClient(api_key=FAKE_KEY, client=sdk, sleep=sleeps.append, backoff_base_s=0.01, **kw)
    return client, sleeps


# ------------------------------------------------------------------ setup / configuration
def test_missing_api_key_gives_clear_setup_message():
    with pytest.raises(LLMConfigError) as exc:
        GroqClient(client=FakeGroqSDK())
    assert "GROQ_API_KEY" in str(exc.value) and ".env" in str(exc.value)
    assert str(exc.value) == SETUP_MESSAGE


def test_placeholder_key_is_treated_as_missing(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "your_groq_api_key_here")
    with pytest.raises(LLMConfigError):
        GroqClient(client=FakeGroqSDK())


def test_get_provider_without_key_falls_back_with_setup_message():
    status = get_provider()
    assert status.provider is None and not status.available
    assert "GROQ_API_KEY" in status.message


def test_get_provider_respects_llm_enabled_false(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", FAKE_KEY)
    monkeypatch.setenv("LLM_ENABLED", "false")
    status = get_provider()
    assert status.provider is None and "disabled" in status.message.lower()


def test_unknown_provider_name_is_reported(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "somethingelse")
    assert "Unknown LLM_PROVIDER" in get_provider().message


def test_default_model_is_gpt_oss_20b():
    client, _ = make(FakeGroqSDK())
    assert client.model == DEFAULT_MODEL == "openai/gpt-oss-20b"


def test_model_is_configurable_via_env(monkeypatch):
    monkeypatch.setenv("GROQ_MODEL", "openai/gpt-oss-120b")
    sdk = FakeGroqSDK(sdk_completion("hi", model="openai/gpt-oss-120b"))
    client, _ = make(sdk)
    client.chat(MSGS)
    assert client.model == "openai/gpt-oss-120b"
    assert sdk.calls[0]["model"] == "openai/gpt-oss-120b"


def test_model_and_timeout_configurable_via_from_env(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", FAKE_KEY)
    monkeypatch.setenv("GROQ_MODEL", "custom/model")
    monkeypatch.setenv("GROQ_TIMEOUT_SECONDS", "12.5")
    monkeypatch.setenv("GROQ_MAX_RETRIES", "1")
    client = GroqClient.from_env(client=FakeGroqSDK())
    assert (client.model, client.timeout_s, client.max_retries) == ("custom/model", 12.5, 1)


# ------------------------------------------------------------------ responses
def test_successful_response_returns_text_and_metadata():
    sdk = FakeGroqSDK(sdk_completion('{"ok": true}'))
    client, _ = make(sdk, timeout_s=9)
    resp = client.chat(MSGS, json_mode=True, max_tokens=50)
    assert resp.text == '{"ok": true}'
    assert resp.request_id == "chatcmpl-123" and resp.finish_reason == "stop"
    assert resp.usage == {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}
    assert resp.attempts == 1 and resp.provider == "groq"
    call = sdk.calls[0]
    assert call["messages"] == [{"role": "system", "content": "be brief"}, {"role": "user", "content": "hello"}]
    assert call["response_format"] == {"type": "json_object"}
    assert call["timeout"] == 9 and call["max_completion_tokens"] == 50
    assert call["reasoning_effort"] == "low"


def test_empty_content_is_an_error():
    client, _ = make(FakeGroqSDK(sdk_completion("")))
    with pytest.raises(LLMResponseError):
        client.chat(MSGS)


# ------------------------------------------------------------------ errors and retries
def test_rate_limit_is_retried_with_backoff_then_succeeds():
    sdk = FakeGroqSDK(sdk_error("RateLimitError", 429, headers={"retry-after": "2"}),
                      sdk_error("APIConnectionError", 0), sdk_completion("done"))
    client, sleeps = make(sdk, max_retries=3)
    resp = client.chat(MSGS)
    assert resp.text == "done" and resp.attempts == 3
    assert sleeps[0] == 2.0 and len(sleeps) == 2


def test_rate_limit_retries_are_bounded_and_message_is_clear():
    sdk = FakeGroqSDK(*[sdk_error("RateLimitError", 429) for _ in range(5)])
    client, sleeps = make(sdk, max_retries=2)
    with pytest.raises(LLMRateLimitError) as exc:
        client.chat(MSGS)
    assert len(sdk.calls) == 3 and len(sleeps) == 2
    assert "rate limit" in str(exc.value).lower() and "3 attempts" in str(exc.value)


@pytest.mark.parametrize("err,expected", [
    (lambda: sdk_error("AuthenticationError", 401), LLMAuthError),
    (lambda: sdk_error("PermissionDeniedError", 403), LLMAuthError),
    (lambda: sdk_error("NotFoundError", 404, body={"error": {"code": "model_not_found"}}), LLMConfigError),
    (lambda: sdk_error("BadRequestError", 400, body={"error": {"code": "model_decommissioned"}}), LLMConfigError),
    (lambda: sdk_error("BadRequestError", 400, body={"error": {"code": "json_validate_failed"}}), LLMResponseError),
])
def test_permanent_errors_are_not_retried(err, expected):
    sdk = FakeGroqSDK(err(), sdk_completion("never reached"))
    client, sleeps = make(sdk, max_retries=3)
    with pytest.raises(expected):
        client.chat(MSGS)
    assert len(sdk.calls) == 1 and sleeps == []


def test_invalid_model_message_names_the_model_and_fix():
    client, _ = make(FakeGroqSDK(sdk_error("NotFoundError", 404)), model="bad/model")
    with pytest.raises(LLMConfigError) as exc:
        client.chat(MSGS)
    assert "bad/model" in str(exc.value) and "GROQ_MODEL" in str(exc.value)


@pytest.mark.parametrize("err", [
    lambda: sdk_error("APITimeoutError", 0),
    lambda: sdk_error("InternalServerError", 500),
    lambda: sdk_error("APIStatusError", 503),
])
def test_transient_errors_become_unavailable_after_retries(err):
    sdk = FakeGroqSDK(*[err() for _ in range(3)])
    client, sleeps = make(sdk, max_retries=2)
    with pytest.raises(LLMUnavailableError):
        client.chat(MSGS)
    assert len(sdk.calls) == 3


def test_api_key_never_appears_in_errors_logs_or_repr(caplog):
    leaky = sdk_error("InternalServerError", 500, message=f"upstream failed for key {FAKE_KEY}")
    sdk = FakeGroqSDK(leaky, sdk_error("BadRequestError", 400, message=f"bad header {FAKE_KEY}"))
    client, _ = make(sdk, max_retries=1)
    with caplog.at_level(logging.DEBUG), pytest.raises(LLMResponseError) as exc:
        client.chat(MSGS)
    assert FAKE_KEY not in str(exc.value)
    assert exc.value.__cause__ is None and exc.value.__suppress_context__
    assert FAKE_KEY not in caplog.text and "[REDACTED]" in caplog.text
    assert FAKE_KEY not in repr(client) and FAKE_KEY not in str(vars(client).get("model"))


def test_auth_error_does_not_echo_key():
    client, _ = make(FakeGroqSDK(sdk_error("AuthenticationError", 401, message=f"Invalid API Key {FAKE_KEY}")))
    with pytest.raises(LLMAuthError) as exc:
        client.chat(MSGS)
    assert FAKE_KEY not in str(exc.value)
