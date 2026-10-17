# -*- coding: utf-8 -*-
"""Tests Phase 4 / PR 4 — contrat API avec le pipeline v2.

TestClient FastAPI : aucun réseau, aucun modèle — le pipeline v2 injecté
tourne sur FakeRetriever + MockLLMService. Le mode dégradé (pipeline absent)
est testé en bloquant l'initialisation au démarrage.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.config import Settings  # noqa: E402
from src.llm import MockLLMService  # noqa: E402
from src.retriever import RetrievedChunk  # noqa: E402

ANSWER = "• [Source 1] Les cascades de Banfora sont en saison sèche."


class FakeRetriever:
    def search(self, query, top_k=None):
        return [RetrievedChunk("doc1_c", "passage sur Banfora",
                               "https://x.bf/1", "titre 1", "web", "fr",
                               "encyclopedia", 0.85)]


@pytest.fixture()
def app_v2(monkeypatch):
    """App avec pipeline v2 complet mais injecté (aucun réseau/index réel).
    On remplace la FABRIQUE que le lifespan appelle réellement (FastAPI
    capture lifespan à la création : patcher api_mod.lifespan est inopérant)."""
    import src.api as api_mod
    from src.pipeline import TourismRAGPipelineV2

    settings = Settings(groq_api_key="gsk_test", llm_provider="groq")

    def factory(*args, **kwargs):
        p = TourismRAGPipelineV2(settings=settings,
                                 retriever=FakeRetriever(),
                                 llm=MockLLMService(response=ANSWER))
        p.initialize()
        return p

    monkeypatch.setattr(api_mod, "TourismRAGPipelineV2", factory)
    return api_mod.app


def asynccontextmanager_ctx(fn):
    from contextlib import asynccontextmanager

    return asynccontextmanager(fn)


class TestQueryContract:
    def test_post_query_returns_legacy_shape(self, app_v2):
        with TestClient(app_v2) as client:
            r = client.post("/query", json={"question": "Que voir à Banfora ?"})
        assert r.status_code == 200
        body = r.json()
        # contrat legacy : ces clés existaient avant ce PR et sont conservées
        for key in ("response", "sources", "documents", "response_time", "safe"):
            assert key in body
        assert body["safe"] is True
        assert body["response"] == ANSWER
        assert body["sources"] == ["https://x.bf/1"]
        doc = body["documents"][0]
        assert doc["url"] == "https://x.bf/1" and "similarity" in doc

    def test_post_query_with_top_k(self, app_v2):
        with TestClient(app_v2) as client:
            r = client.post("/query", json={"question": "question valide ?",
                                            "top_k": 3})
        assert r.status_code == 200

    def test_post_query_propagates_top_k(self, app_v2, monkeypatch):
        """top_k validé par Pydantic doit atteindre la recherche (l'ancien
        code l'ignorait silencieusement — corrigé dans ce PR)."""
        import src.api as api_mod

        captured = {}

        class RecordingRetriever(FakeRetriever):
            def search(self, query, top_k=None):
                captured["top_k"] = top_k
                return super().search(query, top_k)

        # récupérer le pipeline injecté et remplacer son retriever
        with TestClient(app_v2) as client:
            pl = api_mod.pipeline
            pl._retriever = RecordingRetriever()
            r = client.post("/query", json={"question": "question valide ?",
                                            "top_k": 3})
        assert r.status_code == 200
        assert captured["top_k"] == 3

    def test_index_endpoint_explicit_501_for_v2(self, app_v2):
        """v2 ne construit jamais d'index au runtime : réponse explicite,
        pas d'AttributeError/500 ni de repli legacy."""
        with TestClient(app_v2) as client:
            r = client.post("/index")
        assert r.status_code == 501
        assert "build_index.py" in r.json()["detail"]

    def test_invalid_question_rejected(self, app_v2):
        with TestClient(app_v2) as client:
            r = client.post("/query", json={"question": "ab"})
        assert r.status_code == 422  # validation Pydantic inchangée


class TestDegradedMode:
    @pytest.fixture()
    def app_degraded(self, monkeypatch):
        """Démarrage dont l'initialisation échoue : mode dégradé, pas de crash."""
        import src.api as api_mod

        def failing_factory(*args, **kwargs):
            raise RuntimeError("initialisation impossible simulée")

        monkeypatch.setattr(api_mod, "TourismRAGPipelineV2", failing_factory)
        return api_mod.app

    def test_health_reports_degraded_with_diagnostics(self, app_degraded):
        with TestClient(app_degraded) as client:
            r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "degraded"
        assert body["pipeline_ready"] is False
        assert body["provider"] == "groq"
        # diagnostics présents (clé absente de l'environnement de test et/ou
        # échec d'initialisation signalé) — sans jamais exposer d'identifiant
        assert len(body["problems"]) >= 1
        assert "gsk_" not in str(body["problems"])

    def test_query_returns_503_not_crash(self, app_degraded):
        with TestClient(app_degraded) as client:
            r = client.post("/query", json={"question": "question valide ?"})
        assert r.status_code == 503

    def test_no_secret_in_responses(self, app_degraded):
        with TestClient(app_degraded) as client:
            for path in ("/health",):
                assert "gsk_" not in client.get(path).json().__str__()


class TestHealthProvider:
    def test_provider_probe_endpoint_no_key(self, monkeypatch):
        import src.api as api_mod
        from src.config import Settings

        monkeypatch.setattr(api_mod, "provider_probe",
                            lambda settings=None: {
                                "provider": "groq", "available": False,
                                "detail": "GROQ_API_KEY absente"})
        with TestClient(api_mod.app) as client:  # lifespan réel : dégradé
            r = client.get("/health/provider")
        assert r.status_code == 200
        assert r.json()["available"] is False
