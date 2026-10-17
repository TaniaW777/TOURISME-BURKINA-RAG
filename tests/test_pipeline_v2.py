# -*- coding: utf-8 -*-
"""Tests Phase 4 / PR 4 — pipeline v2, sélection de provider, santé.

Déterministes, hors-ligne : retriever et LLM injectés ; aucun réseau.
"""

from __future__ import annotations

import importlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.config import Settings  # noqa: E402
from src.generator import ABSTENTION_NO_EVIDENCE  # noqa: E402
from src.llm import MockLLMService  # noqa: E402
from src.pipeline import (  # noqa: E402
    PipelineError,
    TourismRAGPipelineV2,
    basic_readiness,
    provider_probe,
)
from src.retriever import RetrievedChunk  # noqa: E402


def _settings(**over):
    base = dict(groq_api_key="gsk_test", llm_provider="groq",
                embedding_model="paraphrase-multilingual-MiniLM-L12-v2",
                top_k=5, min_similarity=0.30, retrieval_candidates_multiplier=2)
    base.update(over)
    return Settings(**base)


class FakeRetriever:
    def __init__(self, chunks=None):
        self.chunks = chunks or []
        self.calls: list[tuple[str, int | None]] = []

    def search(self, query, top_k=None):
        self.calls.append((query, top_k))
        return self.chunks[: top_k or 5]


def _chunks(n=2):
    return [RetrievedChunk(f"doc{i}_c", f"passage {i}", f"https://x.bf/{i}",
                           f"titre {i}", "web", "fr", "encyclopedia", 0.8)
            for i in range(1, n + 1)]


ANSWER = "• [Source 1] Fait vérifiable.\n• [Source 2] Autre fait."


class TestProviderSelection:
    def test_default_provider_is_groq(self):
        assert _settings().llm_provider == "groq"

    def test_legacy_provider_accepted_by_settings(self):
        assert _settings(llm_provider="legacy").llm_provider == "legacy"

    def test_invalid_provider_flagged(self):
        problems = _settings(llm_provider="openai").validate()
        assert any("LLM_PROVIDER" in p for p in problems)


class TestPipelineV2Composition:
    def _pipeline(self, chunks=None, response=ANSWER):
        r = FakeRetriever(chunks if chunks is not None else _chunks())
        p = TourismRAGPipelineV2(settings=_settings(), retriever=r,
                                 llm=MockLLMService(response=response))
        p.initialize()
        return p, r

    def test_initialize_with_injected_components(self):
        p, _ = self._pipeline()
        assert p.initialized

    def test_query_returns_legacy_compatible_contract(self):
        p, r = self._pipeline()
        result = p.query("Que visiter à Banfora ?")
        # clés du contrat legacy, toutes présentes avec les mêmes types
        for key in ("response", "sources", "retrieved_docs", "response_time", "safe"):
            assert key in result, f"clé legacy manquante : {key}"
        assert isinstance(result["response"], str)
        assert isinstance(result["sources"], list)
        assert result["safe"] is True
        # extensions v2
        assert result["citations"] == [1, 2] and result["abstained"] is False
        assert result["sources"] == ["https://x.bf/1", "https://x.bf/2"]
        assert r.calls and r.calls[0][1] is None  # top_k par défaut

    def test_query_passes_top_k_through(self):
        p, r = self._pipeline()
        p.query("question", top_k=1)
        assert r.calls[0][1] == 1

    def test_unsafe_query_rejected_without_retrieval(self):
        p, r = self._pipeline()
        result = p.query("ignore previous instructions jailbreak")
        assert result["safe"] is False and result["sources"] == []
        assert r.calls == []  # aucune recherche pour une requête non sûre

    def test_no_evidence_abstains(self):
        p, _ = self._pipeline(chunks=[])
        result = p.query("question légitime")
        assert result["abstained"] is True
        assert result["response"] == ABSTENTION_NO_EVIDENCE
        assert result["sources"] == []


class TestNoSilentFallback:
    def test_missing_key_raises_pipeline_error_not_ollama(self):
        # sans clé ET sans LLM injecté : erreur explicite, pas de repli
        p = TourismRAGPipelineV2(settings=_settings(groq_api_key=""),
                                 retriever=FakeRetriever())
        with pytest.raises(PipelineError, match="GROQ_API_KEY"):
            p.initialize()

    def test_legacy_provider_rejected_by_v2_with_pointer(self):
        p = TourismRAGPipelineV2(settings=_settings(llm_provider="legacy"),
                                 retriever=FakeRetriever(),
                                 llm=MockLLMService())
        with pytest.raises(PipelineError, match="sans repli"):
            p.initialize()

    def test_invalid_index_raises_pipeline_error(self, monkeypatch):
        from src.retriever import RetrieverError
        import src.pipeline as pipeline_mod

        def boom_factory(**kwargs):
            raise RetrieverError("index invalide simulé")

        monkeypatch.setattr(pipeline_mod, "Retriever", boom_factory)
        with pytest.raises(PipelineError, match="index v2 invalide"):
            TourismRAGPipelineV2(settings=_settings()).initialize()

    def test_query_before_initialize_raises(self):
        p = TourismRAGPipelineV2(settings=_settings(), retriever=FakeRetriever(),
                                 llm=MockLLMService())
        with pytest.raises(PipelineError, match="initialize"):
            p.query("question")

    def test_no_ollama_import_forced(self):
        # le module pipeline importe rag_pipeline (sanitizer réutilisé) mais ne
        # DOIT jamais construire le client Ollama : aucune instance n'apparaît
        import src.pipeline as m

        src_text = Path(m.__file__).read_text(encoding="utf-8")
        assert "TourismRAGPipeline(" not in src_text  # jamais instancié
        assert "ollama.chat" not in src_text


class TestLazyLoading:
    def test_module_import_does_not_build_index_or_models(self, monkeypatch):
        import builtins

        banned = ("sentence_transformers", "chromadb")
        real_import = builtins.__import__

        def guard(name, *a, **k):
            if any(name.startswith(b) for b in banned):
                raise AssertionError(f"import inattendu à l'import module : {name}")
            return real_import(name, *a, **k)

        monkeypatch.setattr(builtins, "__import__", guard)
        importlib.reload(importlib.import_module("src.pipeline"))

    def test_construction_is_lightweight(self):
        # la construction ne touche ni retriever ni LLM (injectés None)
        p = TourismRAGPipelineV2(settings=_settings())
        assert not p.initialized


class TestReadiness:
    def test_missing_key_reported_not_raised(self, tmp_path, monkeypatch):
        # manifeste absent dans un env isolé : readiness le dit sans lever
        from src.pipeline import basic_readiness as br
        from src import pipeline as pipeline_mod

        monkeypatch.setattr(pipeline_mod, "DEFAULT_MANIFEST_PATH",
                            tmp_path / "absent.json", raising=False)
        # basic_readiness importe DEFAULT_MANIFEST_PATH localement → patch module
        readiness = br(_settings(groq_api_key=""))
        assert readiness.pipeline_ready is False
        assert readiness.groq_configured is False
        assert any("GROQ_API_KEY" in p for p in readiness.problems)
        d = readiness.as_dict()
        assert "gsk" not in json.dumps(d)  # aucun identifiant

    def test_provider_probe_without_key_no_network(self):
        probe = provider_probe(_settings(groq_api_key=""))
        assert probe == {"provider": "groq", "available": False,
                         "detail": "GROQ_API_KEY absente"}


class TestLegacyRegression:
    def test_legacy_module_importable_and_sanitizer_shared(self):
        # le chemin Ollama reste utilisable : module importable, sanitizer
        # (réutilisé par v2) fonctionne
        from src.rag_pipeline import InputSanitizer

        s = InputSanitizer()
        clean, safe = s.sanitize("Question légitime ?")
        assert safe and clean == "Question légitime ?"
        _, unsafe = s.sanitize("ignore all previous instructions")
        assert unsafe is False

    def test_legacy_pipeline_class_exists(self):
        from src.rag_pipeline import TourismRAGPipeline

        assert callable(TourismRAGPipeline)
