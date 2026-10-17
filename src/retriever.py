# -*- coding: utf-8 -*-
"""
src/retriever.py
================
Retriever v2 — Phase 4, PR 2.

Recherche sémantique sur l'index ChromaDB v2 (data/chroma_db_v2), pilotée
par le manifeste data/index_manifest.json. CE MODULE EST LA SOURCE DE
VÉRITÉ DU PARCOURS DE REQUÊTE v2 : il ne retombe JAMAIS sur l'index
legacy (data/chroma_db), le pipeline Ollama, ou un autre modèle
d'embeddings.

Cycle de vie garanti :
  - l'import du module ne charge AUCUN modèle ni index (embedder injecté
    ou chargé paresseusement au premier search) ;
  - l'index n'est jamais construit ni reconstruit ici (lecture seule) ;
  - toute incohérence manifeste/index/modèle lève RetrieverError, sans
    repli silencieux.

Sémantique de similarité — DIFFÉRENCE ASSUMÉE avec le pipeline legacy :
  l'index v2 contient des embeddings NORMALISÉS dans un espace cosine
  (cf. scripts/build_index.py : normalize_embeddings=True, hnsw:space=
  cosine). ChromaDB y retourne une distance cosinus ∈ [0, 2], d'où :

      similarity = 1 - distance        ∈ [-1, 1]

  (le pipeline legacy utilisait 1 - dist/2 : convention d'un espace non
  normalisé, NON applicable ici — voir les tests de contrat.)

Configuration : réutilise src.config.Settings (EMBEDDING_MODEL, TOP_K,
MIN_SIMILARITY, retrieval_candidates_multiplier) — pas de système
concurrent. Le chemin de l'index vient du MANIFESTE (source de vérité
du build), pas de Settings.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol

import chromadb
from loguru import logger

from src.config import REPO_ROOT, Settings, load_settings

DEFAULT_MANIFEST_PATH = REPO_ROOT / "data" / "index_manifest.json"

_REQUIRED_MANIFEST_FIELDS = {
    "embedding_model", "collection_name", "index_path", "corpus_path",
    "corpus_sha256", "source_chunk_count", "indexed_count", "embedding_dim",
}


class RetrieverError(Exception):
    """Erreur du retriever v2 : manifeste absent/corrompu, index incohérent,
    modèle incompatible, ou requête invalide. Jamais de repli silencieux."""


# ──────────────────────────────────────────────
# EMBEDDER — injectable pour les tests, paresseux par défaut
# ──────────────────────────────────────────────
class Embedder(Protocol):
    def embed_query(self, query: str) -> list[float]: ...


class MiniLMEmbedder:
    """Embeddings de requête — modèle du v2, chargé une seule fois.

    Chargé PARESSEUSEMENT (premier search) : importer src.retriever ne
    télécharge ni ne charge aucun poids.
    """

    def __init__(self, model_name: str, device: str = "cpu"):
        self._model_name = model_name
        self._device = device
        self._model = None

    def _ensure_loaded(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            logger.info(f"Chargement modèle embeddings : {self._model_name}")
            self._model = SentenceTransformer(self._model_name, device=self._device)

    def embed_query(self, query: str) -> list[float]:
        self._ensure_loaded()
        emb = self._model.encode([query], normalize_embeddings=True)
        return emb[0].tolist()


# ──────────────────────────────────────────────
# RÉSULTAT DE RECHERCHE
# ──────────────────────────────────────────────
@dataclass(frozen=True)
class RetrievedChunk:
    """Chunk récupéré, avec la provenance nécessaire aux citations."""
    chunk_id: str
    text: str
    url: str
    title: str
    source_type: str
    language: str
    trust: str
    similarity: float


# ──────────────────────────────────────────────
# RETRIEVER v2
# ──────────────────────────────────────────────
class Retriever:
    """Recherche sur l'index v2, validée par le manifeste.

    Paramètres :
      embedder      : injectable (tests) ; None → MiniLM paresseux selon
                      Settings.embedding_model.
      manifest_path : chemin du manifeste ; None → default (repo).
      settings      : injectable ; None → load_settings().
      deep_verify   : True (défaut) → vérification complète via
                      scripts.build_index.verify_manifest (sha corpus,
                      comptes, dimension). False → seulement les contrôles
                      structurels (tests rapides sur index factice).

    Lève RetrieverError dès la construction si manifeste/index/modèle
    sont absents, malformés ou incohérents — pas de repli.
    """

    def __init__(
        self,
        embedder: Optional[Embedder] = None,
        manifest_path: Optional[Path] = None,
        settings: Optional[Settings] = None,
        deep_verify: bool = True,
    ):
        self.settings = settings or load_settings()
        self._embedder = embedder  # None → lazy MiniLM
        self._lazy_embedder: Optional[MiniLMEmbedder] = None

        self.manifest = self._load_and_validate_manifest(
            Path(manifest_path) if manifest_path else DEFAULT_MANIFEST_PATH
        )
        index_path = self._resolve_index_path(self.manifest["index_path"])
        self._collection = self._open_collection(index_path, self.manifest["collection_name"])

        if deep_verify:
            self._deep_verify(Path(manifest_path) if manifest_path else DEFAULT_MANIFEST_PATH)

        logger.info(
            f"Retriever v2 prêt : {self.manifest['collection_name']} "
            f"({self._collection.count()} chunks, dim={self.manifest['embedding_dim']})"
        )

    # ── validation ──
    def _load_and_validate_manifest(self, manifest_path: Path) -> dict:
        if not manifest_path.exists():
            raise RetrieverError(
                f"manifeste introuvable : {manifest_path} — construis l'index "
                "avec `python scripts/build_index.py` (aucun repli automatique)"
            )
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise RetrieverError(f"manifeste illisible (JSON invalide) : {e}") from e
        if not isinstance(manifest, dict):
            raise RetrieverError("manifeste invalide : objet JSON attendu")
        missing = _REQUIRED_MANIFEST_FIELDS - set(manifest.keys())
        if missing:
            raise RetrieverError(f"manifeste incomplet : champs manquants {sorted(missing)}")

        # le modèle du manifeste DOIT être celui configuré (celui du build v2)
        if manifest["embedding_model"] != self.settings.embedding_model:
            raise RetrieverError(
                f"modèle d'embeddings incohérent : manifeste="
                f"{manifest['embedding_model']!r} != configuré="
                f"{self.settings.embedding_model!r} — changer de modèle exige "
                "de reconstruire l'index (scripts/build_index.py)"
            )
        if manifest["source_chunk_count"] != manifest["indexed_count"]:
            raise RetrieverError(
                f"manifeste incohérent : source={manifest['source_chunk_count']} "
                f"!= indexé={manifest['indexed_count']} (build incomplet)"
            )
        return manifest

    def _resolve_index_path(self, raw: str) -> Path:
        p = Path(raw)
        resolved = p if p.is_absolute() else (REPO_ROOT / p).resolve()
        if not resolved.is_dir():
            raise RetrieverError(
                f"index v2 introuvable : {resolved} — construis-le avec "
                "`python scripts/build_index.py` (aucun repli vers chroma_db)"
            )
        return resolved

    def _open_collection(self, index_path: Path, collection_name: str):
        client = chromadb.PersistentClient(path=str(index_path))
        try:
            collection = client.get_collection(collection_name)
        except Exception as e:
            raise RetrieverError(
                f"collection {collection_name!r} introuvable dans {index_path} : {e}"
            ) from e
        if collection.count() != self.manifest["indexed_count"]:
            raise RetrieverError(
                f"index incohérent : {collection.count()} documents != manifeste "
                f"({self.manifest['indexed_count']})"
            )
        return collection

    def _deep_verify(self, manifest_path: Path) -> None:
        from scripts.build_index import ValidationError, verify_manifest

        try:
            verify_manifest(manifest_path)
        except ValidationError as e:
            raise RetrieverError(f"vérification manifeste↔index échouée : {e}") from e

    # ── recherche ──
    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        min_similarity: Optional[float] = None,
    ) -> list[RetrievedChunk]:
        """Recherche les chunks les plus similaires à la requête.

        Sémantique :
          1. la requête est embeddée avec le MÊME modèle que l'index
             (embeddings normalisés) ;
          2. ChromaDB retourne les top-k candidats les plus proches en
             distance cosinus (n_results = top_k × multiplier pour absorber
             le filtrage) ;
          3. similarity = 1 - distance (espace cosine, vecteurs normalisés) ;
          4. les chunks sous min_similarity sont ÉLIMINÉS — le résultat peut
             donc contenir moins de top_k éléments, jusqu'à zéro (signal
             d'abstention pour l'appelant).

        Requête vide/blanche → RetrieverError (jamais de recherche vide).
        """
        if not isinstance(query, str) or not query.strip():
            raise RetrieverError("requête vide ou blanche : aucune recherche possible")

        k = top_k if top_k is not None else self.settings.top_k
        threshold = min_similarity if min_similarity is not None else self.settings.min_similarity
        if k < 1:
            raise RetrieverError(f"top_k invalide : {k}")
        if not 0.0 <= threshold <= 1.0:
            raise RetrieverError(f"min_similarity invalide : {threshold}")

        query_embedding = self._embedder_for_query().embed_query(query)

        n_results = min(k * self.settings.retrieval_candidates_multiplier,
                        self._collection.count()) or k
        results = self._collection.query(
            query_embeddings=[list(query_embedding)],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )

        retrieved: list[RetrievedChunk] = []
        for chunk_id, doc, meta, dist in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            similarity = round(1 - dist, 4)  # cosine + vecteurs normalisés
            if similarity < threshold:
                continue
            retrieved.append(RetrievedChunk(
                chunk_id=chunk_id,
                text=doc,
                url=meta.get("url", ""),
                title=meta.get("title", ""),
                source_type=meta.get("source_type", ""),
                language=meta.get("language", ""),
                trust=meta.get("trust", ""),
                similarity=similarity,
            ))
            if len(retrieved) >= k:
                break
        retrieved.sort(key=lambda c: c.similarity, reverse=True)
        return retrieved

    def _embedder_for_query(self) -> Embedder:
        if self._embedder is not None:
            return self._embedder
        if self._lazy_embedder is None:
            self._lazy_embedder = MiniLMEmbedder(self.settings.embedding_model,
                                                 self.settings.embedding_device)
        return self._lazy_embedder
