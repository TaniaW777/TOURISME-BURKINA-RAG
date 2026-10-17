# -*- coding: utf-8 -*-
"""Tests Phase 4 / PR 3 — adaptateur Groq.

Aucun réseau, aucun sommeil réel : client factice + sleep injecté.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import groq as groq_sdk
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.config import Settings  # noqa: E402
from src.llm import (  # noqa: E402
    LLMAuthError,
    LLMConfigurationError,
    LLMProviderError,
    LLMRateLimitError,
    LLMTimeoutError,
    GroqService,
)

MSG = [{"role": "user", "content": "Question test"}]


def _settings(**over):
    base = dict(groq_api_key="gsk_test_key", groq_model="openai/gpt-oss-20b",
                groq_timeout_s=5.0, groq_max_retries=2)
    base.update(over)
    return Settings(**base)


# ── doublures du SDK ──
@dataclass
class _Msg:
    content: str


@dataclass
class _Choice:
    message: _Msg


@dataclass
class _Resp:
    choices: list


def _ok(content="Réponse de test"):
    return _Resp([_Choice(_Msg(content))])


def _err(exc):
    import httpx

    req = httpx.Request("POST", "https://api.groq.com/chat/completions")
    resp = httpx.Response(exc_status(exc), request=req,
                          headers=exc_headers(exc))
    return exc("erreur simulée", response=resp, body=None)


def exc_status(exc):
    return {groq_sdk.RateLimitError: 429, groq_sdk.AuthenticationError: 401,
            groq_sdk.InternalServerError: 500}.get(exc, 500)


def exc_headers(exc):
    if exc is groq_sdk.RateLimitError:
        return {"retry-after": "3"}
    return {}


class FakeClient:
    """Client Groq factice : scénario d'exceptions puis succès."""

    def __init__(self, failures=None, content="• [Source 1] Fait test."):
        self.failures = list(failures or [])
        self.content = content
        self.calls = 0

    @property
    def chat(self):
        return self

    @property
    def completions(self):
        return self

    def create(self, **kwargs):
        self.calls += 1
        if self.failures:
            raise self.failures.pop(0)
        return _ok(self.content)

    @property
    def models(self):
        return _FakeModels(["openai/gpt-oss-20b", "autre/modele"])


class _FakeModels:
    def __init__(self, ids):
        from types import SimpleNamespace

        self.data = [SimpleNamespace(id=i) for i in ids]

    def list(self):
        return self  # API: models.list() -> objet avec .data


class TestConfiguration:
    def test_missing_key_without_client_raises(self):
        with pytest.raises(LLMConfigurationError, match="GROQ_API_KEY"):
            GroqService(settings=_settings(groq_api_key=""))

    def test_injected_client_allows_missing_key(self):
        s = GroqService(settings=_settings(groq_api_key=""), client=FakeClient())
        assert s.generate(MSG)

    def test_name_reflects_model(self):
        assert GroqService(settings=_settings(), client=FakeClient()).name \
            == "groq:openai/gpt-oss-20b"


class TestGeneration:
    def test_successful_generation(self):
        c = FakeClient(content="• [Source 2] Fait.")
        s = GroqService(settings=_settings(), client=c)
        assert s.generate(MSG) == "• [Source 2] Fait."
        assert c.calls == 1
        # paramètres réellement passés au SDK
        s.generate(MSG, max_tokens=99, temperature=0.2)

    def test_empty_response_maps_to_provider_error(self):
        c = FakeClient(content="   ")
        with pytest.raises(LLMProviderError, match="vide"):
            GroqService(settings=_settings(), client=c).generate(MSG)

    def test_invalid_messages_rejected_before_any_call(self):
        c = FakeClient()
        s = GroqService(settings=_settings(), client=c)
        with pytest.raises(LLMConfigurationError):
            s.generate([])
        assert c.calls == 0


class TestErrorMapping:
    def test_auth_error(self):
        c = FakeClient(failures=[_err(groq_sdk.AuthenticationError)])
        with pytest.raises(LLMAuthError):
            GroqService(settings=_settings(), client=c).generate(MSG)

    def test_persistent_rate_limit_maps_with_capped_retry_after(self):
        c = FakeClient(failures=[groq_sdk.RateLimitError] and
                       [_err(groq_sdk.RateLimitError)] * 3)
        sleeps: list[float] = []
        s = GroqService(settings=_settings(), client=c,
                        sleep_fn=sleeps.append)
        with pytest.raises(LLMRateLimitError) as ei:
            s.generate(MSG)
        assert ei.value.retry_after_s == 3.0  # lu depuis Retry-After
        assert len(sleeps) >= 1              # on a attendu avant de réessayer

    def test_rate_limit_then_success_retries_with_wait(self):
        c = FakeClient(failures=[_err(groq_sdk.RateLimitError)])
        sleeps: list[float] = []
        s = GroqService(settings=_settings(), client=c,
                        sleep_fn=sleeps.append)
        out = s.generate(MSG)
        assert out and c.calls == 2 and sleeps == [3.0]

    def test_retry_after_capped_at_30s(self):
        import httpx

        req = httpx.Request("POST", "https://api.groq.com/x")
        resp = httpx.Response(429, request=req, headers={"retry-after": "600"})
        exc = groq_sdk.RateLimitError("rl", response=resp, body=None)
        c = FakeClient(failures=[exc, exc, exc])
        sleeps: list[float] = []
        s = GroqService(settings=_settings(), client=c, sleep_fn=sleeps.append)
        with pytest.raises(LLMRateLimitError):
            s.generate(MSG)
        assert all(w <= 30.0 for w in sleeps)

    def test_timeout_retried_then_fails_typed(self):
        c = FakeClient(failures=[groq_sdk.APITimeoutError("t")] * 3)
        sleeps: list[float] = []
        s = GroqService(settings=_settings(), client=c, sleep_fn=sleeps.append)
        with pytest.raises(LLMTimeoutError):
            s.generate(MSG)
        assert c.calls == 3  # 1 + 2 retries = budget borné

    def test_server_error_retried_then_provider_error(self):
        c = FakeClient(failures=[_err(groq_sdk.InternalServerError)] * 3)
        with pytest.raises(LLMProviderError):
            GroqService(settings=_settings(), client=c,
                        sleep_fn=lambda s: None).generate(MSG)
        assert c.calls == 3

    def test_auth_never_retried(self):
        c = FakeClient(failures=[_err(groq_sdk.AuthenticationError)] * 3)
        with pytest.raises(LLMAuthError):
            GroqService(settings=_settings(), client=c,
                        sleep_fn=lambda s: None).generate(MSG)
        assert c.calls == 1


class TestAvailability:
    def test_model_listed(self):
        assert GroqService(settings=_settings(), client=FakeClient()).is_available()

    def test_model_missing(self):
        assert not GroqService(
            settings=_settings(groq_model="inconnu/modele"),
            client=FakeClient()).is_available()

    def test_provider_down_returns_false_not_raise(self):
        class Down(FakeClient):
            @property
            def models(self):
                raise RuntimeError("injoignable")

        assert not GroqService(settings=_settings(), client=Down()).is_available()


class TestNoSecretLeak:
    def test_key_never_in_exceptions_or_name(self):
        c = FakeClient(failures=[_err(groq_sdk.AuthenticationError)])
        s = GroqService(settings=_settings(groq_api_key="gsk_SECRET_VALUE"),
                        client=c)
        try:
            s.generate(MSG)
            raise AssertionError("devait lever")
        except LLMAuthError as e:
            assert "gsk_SECRET_VALUE" not in str(e)
        assert "gsk_SECRET_VALUE" not in s.name
