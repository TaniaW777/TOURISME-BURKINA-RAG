# -*- coding: utf-8 -*-
"""
tests/test_acceptance_v2.py
==========================
Tests d'acceptance de bout en bout — Phase 4, PR 5.

Contrairement aux tests unitaires des PR 1-4, ces tests traversent la
PILE v2 COMPLÈTE avec de vrais composants autant que possible hors-ligne :

  pipeline v2 → sanitizer réel → RETRIEVER RÉEL (index ChromaDB factice +
  manifeste complet dans tmp_path, embedder déterministe) → générateur réel
  → MockLLMService.  Et via TestClient pour l'API.

Aucun réseau, aucune clé Groq réelle, aucun poids de modèle, aucun index
local de l'utilisateur. La procédure Groq RÉELLE est manuelle :
docs/phase4-groq-smoke-test.md (jamais en CI).
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import chromadb
import pytest
from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.config import Settings  # noqa: E402
from src.llm import MockLLMService  # noqa: E402
from src.pipeline import PipelineError, TourismRAGPipelineV2  # noqa: E402
from src.retriever import Retriever, RetrieverError  # noqa: E402

import hashlib as _hashlib
import re as _re

MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
DIM = 64


class AcceptanceEmbedder:
    """Embeddings déterministes par tokens (md5) : deux textes sont proches
    s'ils partagent des mots — l'ordre de retrieval suit l'intuition, et le
    hash md5 garantit la stabilité inter-exécutions (hash() Python est salé)."""

    def __init__(self):
        self.query_calls = 0

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * DIM
        for tok in _re.findall(r"\w+", text.lower()):
            h = int(_hashlib.md5(tok.encode()).hexdigest(), 16)
            v[h % DIM] += 1.0 if h & 1 else -1.0
        norm = sum(x * x for x in v) ** 0.5
        return [x / norm for x in v] if norm else v

    def embed_query(self, query: str) -> list[float]:
        self.query_calls += 1
        return self._vec(query)


CHUNKS = [
    {"id": f"acc{i}_chunk_0000", "text": t, "url": u, "title": ti,
     "source_type": "web", "language": "fr", "trust": "encyclopedia",
     "parent_id": f"acc{i}", "chunk_idx": 0, "token_count": 12}
    for i, (t, u, ti) in enumerate([
        ("Les cascades de Karfiguéla se visitent près de Banfora en saison sèche.",
         "https://fr.wikipedia.org/wiki/Cascades_de_Banfora", "Cascades de Banfora"),
        ("Le FESPACO est le festival panafricain du cinéma à Ouagadougou.",
         "https://fr.wikipedia.org/wiki/FESPACO", "FESPACO"),
        ("Les ruines de Loropéni sont classées au patrimoine mondial de l'UNESCO.",
         "https://fr.wikipedia.org/wiki/Ruines_de_Lorop%C3%A9ni", "Ruines de Loropéni"),
    ], 1)
]

# La requête d'acceptation partage des tokens avec les passages Cascades et
# Loropéni (aucun mot plein avec FESPACO) : les deux documents pertinents
# occupent les deux premières positions. Avec un seuil à 0.0, les trois
# preuves sont récupérées : les citations [Source 1] et [Source 2] mappent
# sur les deux documents pertinents, quel que soit leur ordre interne.
GROUND_ANSWER = (
    "• [Source 1] Premier fait appuyé par la première preuve.\n"
    "• [Source 2] Second fait appuyé par la seconde preuve."
)
EXPECTED_TOP2_URLS = {
    "https://fr.wikipedia.org/wiki/Cascades_de_Banfora",
    "https://fr.wikipedia.org/wiki/Ruines_de_Lorop%C3%A9ni"}


def _settings(**over):
    base = dict(groq_api_key="gsk_test", llm_provider="groq",
                embedding_model=MODEL, top_k=5, min_similarity=0.30,
                retrieval_candidates_multiplier=2)
    base.update(over)
    return Settings(**base)


def build_acceptance_stack(tmp_path: Path, chunks=None, settings=None):
    """Index ChromaDB factice + manifeste vérifiable + retriever RÉEL +
    pipeline v2 complet. Retourne (pipeline, embedder, llm)."""
    chunks = chunks if chunks is not None else CHUNKS
    emb = AcceptanceEmbedder()
    index_dir = tmp_path / "chroma_db_v2"
    col = chromadb.PersistentClient(path=str(index_dir)).create_collection(
        name="tourisme_burkina_v2", metadata={"hnsw:space": "cosine"})
    col.add(ids=[c["id"] for c in chunks],
            embeddings=[emb._vec(c["text"]) for c in chunks],
            documents=[c["text"] for c in chunks],
            metadatas=[{k: c[k] for k in ("url", "title", "source_type",
                                          "language", "trust")}
                       for c in chunks])
    corpus = tmp_path / "chunks.jsonl"
    corpus.write_text("\n".join(json.dumps(c, ensure_ascii=False)
                                for c in chunks), encoding="utf-8")
    manifest = {
        "embedding_model": MODEL, "collection_name": "tourisme_burkina_v2",
        "index_path": str(index_dir), "corpus_path": str(corpus),
        "corpus_sha256": hashlib.sha256(corpus.read_bytes()).hexdigest(),
        "source_chunk_count": len(chunks), "indexed_count": len(chunks),
        "embedding_dim": DIM, "built_at": "2026-01-01T00:00:00Z",
        "distance_metric": "cosine", "normalized_embeddings": True,
    }
    manifest_path = tmp_path / "index_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    st = settings or _settings()
    retriever = Retriever(embedder=emb, manifest_path=manifest_path,
                          settings=st)
    llm = MockLLMService(response=GROUND_ANSWER)
    pipeline = TourismRAGPipelineV2(settings=st, retriever=retriever,
                                    llm=llm)
    pipeline.initialize()
    return pipeline, emb, llm


# ──────────────────────────────────────────────
# A. REQUÊTE DE BOUT EN BOUT RÉUSSIE
# ──────────────────────────────────────────────
class TestEndToEndQuery:
    def test_full_query_answer_evidence_and_citations(self, tmp_path):
        pipeline, emb, llm = build_acceptance_stack(
            tmp_path, settings=_settings(min_similarity=0.0))
        result = pipeline.query("cascades banfora ruines loropeni tourisme")

        assert result["response"] == GROUND_ANSWER
        assert result["abstained"] is False
        # citations mappées sur les documents RÉELLEMENT récupérés :
        # la requête place Cascades et Loropéni en tête (tokens partagés)
        assert result["citations"] == [1, 2]
        docs = result["retrieved_docs"]
        assert len(docs) >= 2
        for n in result["citations"]:
            assert 1 <= n <= len(docs)
        cited_urls = {docs[n - 1]["url"] for n in result["citations"]}
        assert cited_urls == EXPECTED_TOP2_URLS
        # sources = les preuves CITÉES (pas toutes les récupérées)
        assert set(result["sources"]) == cited_urls
        # métadonnées de preuve complètes + similarité bornée
        for d in docs:
            assert d["url"].startswith("https://") and d["language"] == "fr"
            assert d["source_type"] == "web" and d["title"]
            assert 0.0 <= d["similarity"] <= 1.0001
        # requête réellement embeddée ; LLM appelé UNE fois
        assert emb.query_calls >= 1 and len(llm.calls) == 1
        assert result["model"] == "mock"
        assert result["safe"] is True and result["response_time"] >= 0

    def test_evidence_connected_to_llm_input(self, tmp_path):
        """Les numéros cités par le mock ne sont valides QUE parce que les
        passages correspondants ont été envoyés au LLM : la boucle
        preuve→prompt→citation est fermée."""
        pipeline, _, llm = build_acceptance_stack(tmp_path)
        result = pipeline.query("cascades Banfora ruines Loropéni")
        sent = llm.calls[0][1]["content"]
        for n in result["citations"]:
            assert result["retrieved_docs"][n - 1]["text"] in sent


# ──────────────────────────────────────────────
# B. ABSENCE DE PREUVES → ABSTENTION SANS APPEL LLM
# ──────────────────────────────────────────────
class TestNoEvidence:
    def test_threshold_eliminating_all_evidence_abstains(self, tmp_path):
        """Retriever réel, seuil inatteignable (1.0) : zéro preuve retenue,
        abstention, et LE point clé — aucun appel LLM."""
        pipeline, _, llm = build_acceptance_stack(
            tmp_path, settings=_settings(min_similarity=1.0))
        result = pipeline.query("cascades de Banfora")
        assert result["retrieved_docs"] == []
        assert result["abstained"] is True
        assert result["response"].startswith("Je n'ai pas trouvé")
        assert llm.calls == []

    def test_empty_corpus_index_abstains(self, tmp_path):
        pipeline, _, llm = build_acceptance_stack(tmp_path, chunks=CHUNKS[:1])
        # remplace le retriever par un index VIDE cohérent
        empty_dir = tmp_path / "empty_index"
        col = chromadb.PersistentClient(path=str(empty_dir)).create_collection(
            name="tourisme_burkina_v2", metadata={"hnsw:space": "cosine"})
        corpus = tmp_path / "empty_corpus.jsonl"
        corpus.write_text("", encoding="utf-8")
        manifest = {
            "embedding_model": MODEL, "collection_name": "tourisme_burkina_v2",
            "index_path": str(empty_dir), "corpus_path": str(corpus),
            "corpus_sha256": hashlib.sha256(corpus.read_bytes()).hexdigest(),
            "source_chunk_count": 0, "indexed_count": 0, "embedding_dim": DIM,
            "built_at": "2026-01-01T00:00:00Z", "distance_metric": "cosine",
            "normalized_embeddings": True,
        }
        m_path = tmp_path / "empty_manifest.json"
        m_path.write_text(json.dumps(manifest), encoding="utf-8")
        pipeline._retriever = Retriever(embedder=AcceptanceEmbedder(),
                                        manifest_path=m_path,
                                        settings=_settings(),
                                        deep_verify=False)  # index vide :
        # manifeste 0-chunk dégénéré (un vrai build refuse un corpus vide) —
        # on teste l'abstention, pas la chaîne de vérification
        result = pipeline.query("n'importe quelle question")
        assert result["abstained"] is True and llm.calls == []


# ──────────────────────────────────────────────
# C. VALIDATION DES CITATIONS (mappage, PAS de grounding sémantique)
# ──────────────────────────────────────────────
class TestCitationMapping:
    def test_invalid_id_never_becomes_valid_citation(self, tmp_path):
        pipeline, _, llm = build_acceptance_stack(
            tmp_path, settings=_settings(min_similarity=0.0))
        llm._responses = ["• [Source 99] fait inventé\n• [Source 2] fait réel"]
        result = pipeline.query("FESPACO cinéma Ouagadougou")
        assert 2 in result["citations"]
        assert 99 not in result["citations"]
        # source citée = document récupéré correspondant, rien d'autre
        assert result["sources"] == [result["retrieved_docs"][1]["url"]]

    def test_zero_id_rejected(self, tmp_path):
        pipeline, _, llm = build_acceptance_stack(tmp_path)
        llm._responses = ["• [Source 0] invalide"]
        result = pipeline.query("FESPACO")
        assert result["citations"] == [] or 0 not in result["citations"]


# ──────────────────────────────────────────────
# D. ÉCHECS D'INITIALISATION ET DE CONFIGURATION (jamais de repli Ollama)
# ──────────────────────────────────────────────
class TestInitializationFailures:
    def test_missing_manifest_surfaces_pipeline_error(self, tmp_path, monkeypatch):
        import src.pipeline as pipeline_mod

        def boom(**kwargs):
            raise RetrieverError("manifeste introuvable : simulé")

        monkeypatch.setattr(pipeline_mod, "Retriever", boom)
        with pytest.raises(PipelineError, match="index v2 invalide"):
            TourismRAGPipelineV2(settings=_settings()).initialize()

    def test_model_mismatch_surfaces_pipeline_error(self, tmp_path, monkeypatch):
        import src.pipeline as pipeline_mod

        def boom(**kwargs):
            raise RetrieverError("modèle d'embeddings incohérent : simulé")

        monkeypatch.setattr(pipeline_mod, "Retriever", boom)
        with pytest.raises(PipelineError, match="index v2 invalide"):
            TourismRAGPipelineV2(settings=_settings()).initialize()

    def test_missing_groq_key_no_ollama_fallback(self, tmp_path, monkeypatch):
        import src.pipeline as pipeline_mod

        # manifeste valide simulé : le retriever passe, la clé manque
        monkeypatch.setattr(pipeline_mod, "Retriever",
                            lambda **k: build_acceptance_stack(tmp_path)[0]._retriever)
        src = Path(pipeline_mod.__file__).read_text(encoding="utf-8")
        assert "TourismRAGPipeline(" not in src  # aucune instanciation legacy
        with pytest.raises(PipelineError, match="GROQ_API_KEY"):
            TourismRAGPipelineV2(settings=_settings(groq_api_key="")).initialize()

    def test_invalid_provider_rejected(self):
        with pytest.raises(PipelineError, match="sans repli"):
            TourismRAGPipelineV2(settings=_settings(llm_provider="legacy")).initialize()


# ──────────────────────────────────────────────
# E. INTÉGRATION API (TestClient, provider probe UNIQUEMENT mocké)
# ──────────────────────────────────────────────
class TestApiIntegration:
    @pytest.fixture()
    def client_app(self, tmp_path, monkeypatch):
        """App réelle ; on patche SEULEMENT initialize() pour brancher le
        pipeline v2 sur la stack d'acceptance factice (le reste — lifespan,
        routes, validation — est le vrai code)."""
        import src.api as api_mod

        stack_pipeline, _, stack_llm = build_acceptance_stack(
            tmp_path, settings=_settings(min_similarity=0.0))

        def patched_initialize(self):
            self._retriever = stack_pipeline._retriever
            self._llm = stack_llm
            self._generator = stack_pipeline._generator
            self._initialized = True

        monkeypatch.setattr(TourismRAGPipelineV2, "initialize", patched_initialize)

        from src.pipeline import Readiness

        monkeypatch.setattr(api_mod, "basic_readiness",
                            lambda settings=None: Readiness(
                                pipeline_ready=True, provider="groq",
                                groq_configured=True, index_manifest_ok=True,
                                problems=[]))
        return api_mod.app

    def test_query_through_api_preserves_contract(self, client_app):
        with TestClient(client_app) as client:
            r = client.post("/query", json={"question": "cascades de Banfora ?"})
        assert r.status_code == 200
        body = r.json()
        for key in ("response", "sources", "documents", "response_time", "safe"):
            assert key in body
        assert body["response"] == GROUND_ANSWER
        assert body["safe"] is True
        assert set(body["sources"]) == EXPECTED_TOP2_URLS

    def test_top_k_propagated_through_api(self, client_app):
        import src.api as api_mod

        captured = {}
        with TestClient(client_app) as client:
            pl = api_mod.pipeline
            original_search = pl._retriever.search

            def recording(query, top_k=None):
                captured["top_k"] = top_k
                return original_search(query, top_k)

            pl._retriever.search = recording
            r = client.post("/query", json={"question": "cascades ?", "top_k": 2})
        assert r.status_code == 200 and captured["top_k"] == 2

    def test_health_reports_ready_without_secrets(self, client_app):
        with TestClient(client_app) as client:
            r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["pipeline_ready"] is True
        assert body["provider"] == "groq"
        assert "gsk_" not in json.dumps(body)

    def test_provider_probe_only_via_mock(self, client_app, monkeypatch):
        import src.api as api_mod

        monkeypatch.setattr(api_mod, "provider_probe",
                            lambda settings=None: {"provider": "groq",
                                                   "available": True,
                                                   "detail": ""})
        with TestClient(client_app) as client:
            r = client.get("/health/provider")
        assert r.status_code == 200 and r.json()["available"] is True
