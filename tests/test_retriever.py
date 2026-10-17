# -*- coding: utf-8 -*-
"""Tests Phase 4 / PR 2 — src/retriever.py.

Aucun réseau, aucun poids de modèle : embedder factice déterministe +
index ChromaDB factice construit dans un répertoire temporaire, avec un
manifeste synthétique qui satisfait la vérification complète
(verify_manifest : sha du corpus, comptes, dimension).
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import chromadb
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.config import Settings  # noqa: E402
from src.retriever import Retriever, RetrieverError  # noqa: E402

MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
DIM = 8

# vocabulaire miniature : chaque requête/chunk est embeddé par la moyenne
# des codes de ses lettres — déterministe, distinction suffisante pour
# les tests de tri/threshold.
class FakeEmbedder:
    def embed_query(self, query: str) -> list[float]:
        vals = [(ord(ch) % 17) + 1.0 for ch in query.lower()[:DIM]]
        vals += [1.0] * (DIM - len(vals))
        norm = sum(v * v for v in vals) ** 0.5
        return [v / norm for v in vals]

    def embed_corpus(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(t) for t in texts]


CHUNKS = [
    {"id": "doc0000000000000_chunk_0000", "text": "Les cascades de Banfora sont magnifiques en saison sèche.",
     "url": "https://fr.wikipedia.org/wiki/Cascades_de_Banfora", "title": "Cascades de Banfora",
     "source_type": "web", "language": "fr", "trust": "encyclopedia", "token_count": 10,
     "parent_id": "doc0000000000000", "chunk_idx": 0},
    {"id": "doc0000000000001_chunk_0000", "text": "Le FESPACO célèbre le cinéma africain à Ouagadougou.",
     "url": "https://fr.wikipedia.org/wiki/FESPACO", "title": "FESPACO",
     "source_type": "web", "language": "fr", "trust": "encyclopedia", "token_count": 10,
     "parent_id": "doc0000000000001", "chunk_idx": 0},
    {"id": "doc0000000000002_chunk_0000", "text": "Les ruines de Loropéni sont classées au patrimoine UNESCO.",
     "url": "https://fr.wikipedia.org/wiki/Ruines_de_Lorop%C3%A9ni", "title": "Ruines de Loropéni",
     "source_type": "web", "language": "fr", "trust": "encyclopedia", "token_count": 10,
     "parent_id": "doc0000000000002", "chunk_idx": 0},
]


def _build_fixture(tmp_path: Path, corpus_chunks=None, model=MODEL, count=None,
                   collection="tourisme_burkina_v2"):
    """Construit index factice + manifeste vérifiable. Retourne manifest_path."""
    corpus_chunks = corpus_chunks if corpus_chunks is not None else CHUNKS
    emb = FakeEmbedder()
    index_dir = tmp_path / "chroma_db_v2"
    client = chromadb.PersistentClient(path=str(index_dir))
    col = client.create_collection(name=collection, metadata={"hnsw:space": "cosine"})
    col.add(
        ids=[c["id"] for c in corpus_chunks],
        embeddings=emb.embed_corpus([c["text"] for c in corpus_chunks]),
        documents=[c["text"] for c in corpus_chunks],
        metadatas=[{k: c[k] for k in ("url", "title", "source_type", "language", "trust")}
                   for c in corpus_chunks],
    )
    corpus_file = tmp_path / "chunks.jsonl"
    corpus_file.write_text(
        "\n".join(json.dumps(c, ensure_ascii=False) for c in corpus_chunks), encoding="utf-8")
    sha = hashlib.sha256(corpus_file.read_bytes()).hexdigest()
    n = count if count is not None else len(corpus_chunks)
    manifest = {
        "embedding_model": model,
        "collection_name": collection,
        "index_path": str(index_dir),
        "corpus_path": str(corpus_file),
        "corpus_sha256": sha,
        "source_chunk_count": n,
        "indexed_count": len(corpus_chunks),
        "embedding_dim": DIM,
        "built_at": "2026-01-01T00:00:00Z",
        "distance_metric": "cosine",
        "normalized_embeddings": True,
    }
    manifest_path = tmp_path / "index_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


def _settings(**over):
    base = dict(embedding_model=MODEL, top_k=5, min_similarity=0.30,
                retrieval_candidates_multiplier=2)
    base.update(over)
    return Settings(**base)


class TestLoadingAndValidation:
    def test_valid_manifest_and_index_load(self, tmp_path):
        r = Retriever(embedder=FakeEmbedder(), manifest_path=_build_fixture(tmp_path),
                      settings=_settings())
        assert r.manifest["indexed_count"] == len(CHUNKS)

    def test_missing_manifest_raises(self, tmp_path):
        with pytest.raises(RetrieverError, match="manifeste introuvable"):
            Retriever(embedder=FakeEmbedder(), manifest_path=tmp_path / "absent.json",
                      settings=_settings())

    def test_malformed_manifest_raises(self, tmp_path):
        p = tmp_path / "index_manifest.json"
        p.write_text("{pas du json", encoding="utf-8")
        with pytest.raises(RetrieverError, match="JSON invalide"):
            Retriever(embedder=FakeEmbedder(), manifest_path=p, settings=_settings())

    def test_incomplete_manifest_raises(self, tmp_path):
        m_path = _build_fixture(tmp_path)
        m = json.loads(m_path.read_text(encoding="utf-8"))
        del m["embedding_dim"]
        m_path.write_text(json.dumps(m), encoding="utf-8")
        with pytest.raises(RetrieverError, match="champs manquants"):
            Retriever(embedder=FakeEmbedder(), manifest_path=m_path, settings=_settings())

    def test_model_mismatch_raises(self, tmp_path):
        m_path = _build_fixture(tmp_path, model="BAAI/bge-m3")
        with pytest.raises(RetrieverError, match="modèle d'embeddings incohérent"):
            Retriever(embedder=FakeEmbedder(), manifest_path=m_path, settings=_settings())

    def test_incomplete_build_manifest_raises(self, tmp_path):
        m_path = _build_fixture(tmp_path, count=99)
        with pytest.raises(RetrieverError, match="source=99"):
            Retriever(embedder=FakeEmbedder(), manifest_path=m_path, settings=_settings())

    def test_missing_index_dir_raises(self, tmp_path):
        m_path = _build_fixture(tmp_path)
        m = json.loads(m_path.read_text(encoding="utf-8"))
        m["index_path"] = str(tmp_path / "inexistant")
        m_path.write_text(json.dumps(m), encoding="utf-8")
        with pytest.raises(RetrieverError, match="index v2 introuvable"):
            Retriever(embedder=FakeEmbedder(), manifest_path=m_path, settings=_settings())

    def test_wrong_collection_raises(self, tmp_path):
        m_path = _build_fixture(tmp_path)
        m = json.loads(m_path.read_text(encoding="utf-8"))
        m["collection_name"] = "collection_inexistante"  # diverge de l'index
        m_path.write_text(json.dumps(m), encoding="utf-8")
        with pytest.raises(RetrieverError, match="collection"):
            Retriever(embedder=FakeEmbedder(), manifest_path=m_path, settings=_settings())

    def test_corpus_drift_detected_by_deep_verify(self, tmp_path):
        m_path = _build_fixture(tmp_path)
        # le corpus change APRÈS le build → sha mismatch
        corpus = tmp_path / "chunks.jsonl"
        corpus.write_text(corpus.read_text(encoding="utf-8") + "\n" + json.dumps(CHUNKS[0]),
                          encoding="utf-8")
        with pytest.raises(RetrieverError, match="corpus source a changé"):
            Retriever(embedder=FakeEmbedder(), manifest_path=m_path, settings=_settings())


def _retriever(tmp_path, **settings_over):
    return Retriever(embedder=FakeEmbedder(), manifest_path=_build_fixture(tmp_path),
                     settings=_settings(**settings_over))


class TestSearch:

    def test_query_returns_structured_results_with_metadata(self, tmp_path):
        r = _retriever(tmp_path)
        results = r.search("cascades de Banfora")
        assert results, "au moins un résultat attendu"
        for c in results:
            assert c.text and c.url.startswith("https://")
            assert c.language == "fr" and c.trust == "encyclopedia"
            assert c.chunk_id and 0.0 <= c.similarity <= 1.0001

    def test_top_k_respected(self, tmp_path):
        r = _retriever(tmp_path)
        assert len(r.search("Burkina", top_k=2)) <= 2
        assert len(r.search("Burkina", top_k=1)) == 1

    def test_threshold_filters_results(self, tmp_path):
        r = _retriever(tmp_path)
        base = r.search("cascades", min_similarity=0.0)
        strict = r.search("cascades", min_similarity=0.99)
        assert len(strict) < len(base)
        for c in strict:
            assert c.similarity >= 0.99

    def test_threshold_one_can_empty_results(self, tmp_path):
        r = _retriever(tmp_path)
        assert r.search("cascades", min_similarity=1.0) == []  # abstention

    def test_results_sorted_by_similarity_desc(self, tmp_path):
        results = _retriever(tmp_path).search("Burkina tourisme")
        sims = [c.similarity for c in results]
        assert sims == sorted(sims, reverse=True)

    def test_empty_query_raises(self, tmp_path):
        r = _retriever(tmp_path)
        with pytest.raises(RetrieverError, match="requête vide"):
            r.search("   ")
        with pytest.raises(RetrieverError):
            r.search("")

    def test_invalid_params_raise(self, tmp_path):
        r = _retriever(tmp_path)
        with pytest.raises(RetrieverError, match="top_k"):
            r.search("x", top_k=0)
        with pytest.raises(RetrieverError, match="min_similarity"):
            r.search("x", min_similarity=1.5)


class TestLazyEmbedding:
    def test_import_does_not_load_model(self):
        # le module est importable sans sentence_transformers initialisé :
        # l'attribut de classe reste un adaptateur non chargé tant que
        # search() n'est pas appelé.
        from src.retriever import MiniLMEmbedder

        e = MiniLMEmbedder(MODEL)
        assert e._model is None  # rien chargé à la construction

    def test_injected_embedder_used_without_lazy_load(self, tmp_path, monkeypatch):
        # si le lazy embedder était sollicité, l'import de ST échouerait ici
        import builtins

        real_import = builtins.__import__

        def guard(name, *a, **k):
            if "sentence_transformers" in name:
                raise AssertionError("sentence_transformers ne doit pas être importé "
                                     "quand l'embedder est injecté")
            return real_import(name, *a, **k)

        monkeypatch.setattr(builtins, "__import__", guard)
        r = Retriever(embedder=FakeEmbedder(),
                      manifest_path=_build_fixture(tmp_path), settings=_settings())
        assert r.search("cascades")  # fonctionne sans jamais charger le modèle


class TestSimilaritySemantics:
    def test_cosine_similarity_contract(self, tmp_path):
        """similarity = 1 - distance (vecteurs normalisés, espace cosine) —
        et NON 1 - dist/2 (convention legacy, non applicable au v2)."""
        r = _retriever(tmp_path)
        emb = FakeEmbedder()
        query = "cascades de Banfora"
        raw = r._collection.query(query_embeddings=[emb.embed_query(query)],
                                  n_results=1, include=["distances"])
        dist = raw["distances"][0][0]
        (result,) = [c for c in r.search(query, top_k=1)]
        assert result.similarity == pytest.approx(1 - dist, abs=1e-3)
