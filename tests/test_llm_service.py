# -*- coding: utf-8 -*-
"""Tests Phase 4 / PR 1 — contrat LLMService + MockLLMService.

Aucun réseau, aucun modèle, aucune dépendance provider.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.llm import (  # noqa: E402
    LLMAuthError,
    LLMConfigurationError,
    LLMError,
    LLMProviderError,
    LLMRateLimitError,
    LLMService,
    LLMTimeoutError,
    MockLLMService,
)
from src.llm.base import validate_messages  # noqa: E402

MSG = [{"role": "user", "content": "Quels sites visiter à Banfora ?"}]


class TestContractShape:
    def test_mock_satisfies_llm_service_protocol(self):
        # runtime_checkable : le mock EST un LLMService structurellement
        assert isinstance(MockLLMService(), LLMService)

    def test_interface_module_imports_no_provider(self):
        # le contrat ne doit IMPORTER aucun SDK provider (les mentions en
        # commentaires restent autorisées — c'est l'absence de dépendance
        # d'exécution qui est contractualisée)
        import src.llm.base as base
        import src.llm.mock as mock

        for mod in (base, mock):
            tree = ast.parse(Path(mod.__file__).read_text(encoding="utf-8"))
            imported = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported.update(a.name.split(".")[0] for a in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imported.add(node.module.split(".")[0])
            assert "groq" not in imported, f"{mod.__name__} importe le SDK groq"
            assert "ollama" not in imported, f"{mod.__name__} importe le SDK ollama"


class TestMockGeneration:
    def test_successful_generation_returns_string(self):
        out = MockLLMService().generate(MSG)
        assert isinstance(out, str) and out

    def test_deterministic_same_input_same_output(self):
        a, b = MockLLMService(), MockLLMService()
        assert a.generate(MSG) == b.generate(MSG)
        assert a.generate(MSG) == a.generate(MSG)  # stable en interne aussi

    def test_configurable_response(self):
        out = MockLLMService(response="• [Source 1] Réponse de test.").generate(MSG)
        assert out == "• [Source 1] Réponse de test."

    def test_scripted_sequence_then_repeat(self):
        m = MockLLMService(responses=["première", "deuxième"])
        assert m.generate(MSG) == "première"
        assert m.generate(MSG) == "deuxième"
        assert m.generate(MSG) == "deuxième"  # dernière répétée

    def test_generation_records_calls_for_assertions(self):
        m = MockLLMService()
        m.generate(MSG)
        m.generate([{"role": "system", "content": "s"}, {"role": "user", "content": "q"}])
        assert m.calls[0] == MSG
        assert m.calls[1][0]["role"] == "system"

    def test_optional_kwargs_accepted(self):
        MockLLMService().generate(MSG, max_tokens=100, temperature=0.1)

    def test_empty_responses_list_rejected_at_construction(self):
        with pytest.raises(LLMConfigurationError):
            MockLLMService(responses=[])


class TestErrorContract:
    def test_fail_with_raises_configured_error(self):
        m = MockLLMService(fail_with=LLMTimeoutError("délai dépassé"))
        with pytest.raises(LLMTimeoutError, match="délai"):
            m.generate(MSG)

    def test_error_hierarchy(self):
        for exc in (LLMConfigurationError, LLMAuthError, LLMRateLimitError,
                    LLMTimeoutError, LLMProviderError):
            assert issubclass(exc, LLMError)
        assert issubclass(LLMError, Exception)

    def test_rate_limit_carries_retry_after(self):
        e = LLMRateLimitError("quota dépassé", retry_after_s=12.5)
        assert e.retry_after_s == 12.5
        assert LLMRateLimitError("x").retry_after_s is None

    def test_invalid_messages_fail_before_any_provider_call(self):
        m = MockLLMService(fail_with=LLMProviderError("ne doit pas être levée"))
        for bad in ([], [{"role": "intrus", "content": "x"}],
                    [{"role": "user", "content": "  "}], ["pas un dict"]):
            with pytest.raises(LLMConfigurationError):
                m.generate(bad)
        assert m.calls == []  # l'erreur d'entrée n'a jamais atteint la "génération"
