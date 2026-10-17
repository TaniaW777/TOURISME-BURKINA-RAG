# -*- coding: utf-8 -*-
"""
src/pipeline.py
===============
Pipeline RAG v2 — Phase 4, PR 4 (orchestration).

Compose les briques validées des PR 1-3 :
  InputSanitizer (réutilisé du legacy) → Retriever v2 (index manifesté)
  → RAGGenerator (LLMService) → résultat compatible avec le contrat
  historique {response, sources, retrieved_docs, response_time, safe}.

Règles d'or :
  - AUCUNE construction à l'import : initialize() est explicite ;
  - AUCUN repli silencieux Groq→Ollama : configuration absente ou
    initialisation échouée → erreur claire (mode dégradé côté API) ;
  - le pipeline legacy (src/rag_pipeline.py, Ollama) reste utilisable
    tel quel jusqu'aux tests d'acceptation Groq (sélection via
    LLM_PROVIDER=legacy) ;
  - la santé de base (readiness) ne fait AUCUN appel réseau : la sonde
    provider (probe) est séparée, explicite et bornée.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Protocol

from loguru import logger

from src.config import Settings, load_settings
from src.generator import RAGGenerator
from src.llm.base import LLMError, LLMService
from src.retriever import RetrievedChunk, Retriever, RetrieverError

# Le sanitizer est réutilisé TEL QUEL du pipeline legacy : une seule
# implémentation de la politique de sécurité des entrées.
from src.rag_pipeline import InputSanitizer


class PipelineError(Exception):
    """Initialisation ou configuration du pipeline v2 invalide."""


# ──────────────────────────────────────────────
# ÉTAT DE SANTÉ (sans réseau)
# ──────────────────────────────────────────────
@dataclass(frozen=True)
class Readiness:
    """Santé de base : processus + configuration. Aucun appel réseau."""
    pipeline_ready: bool
    provider: str
    groq_configured: bool
    index_manifest_ok: bool
    problems: list[str]

    def as_dict(self) -> dict:
        # aucun identifiant ni erreur brute n'est exposé
        return {
            "pipeline_ready": self.pipeline_ready,
            "provider": self.provider,
            "groq_configured": self.groq_configured,
            "index_manifest_ok": self.index_manifest_ok,
            "problems": list(self.problems),
        }


def basic_readiness(settings: Optional[Settings] = None) -> Readiness:
    """Vérifie la configuration SANS réseau ni modèle :
    validité Settings, présence/cohérence du manifeste (parse + champs),
    clé Groq. Le pipeline n'est prêt que si tout est OK."""
    s = settings or load_settings()
    problems = list(s.validate())  # diagnostics, jamais de crash

    manifest_ok = False
    try:
        from src.retriever import DEFAULT_MANIFEST_PATH, _REQUIRED_MANIFEST_FIELDS
        import json

        m = json.loads(DEFAULT_MANIFEST_PATH.read_text(encoding="utf-8"))
        missing = _REQUIRED_MANIFEST_FIELDS - set(m)
        if missing:
            problems.append(f"manifeste index incomplet : {sorted(missing)}")
        elif m["embedding_model"] != s.embedding_model:
            problems.append("manifeste index : modèle d'embeddings incohérent")
        else:
            manifest_ok = True
    except FileNotFoundError:
        problems.append("manifeste index absent (python scripts/build_index.py)")
    except Exception as e:
        problems.append(f"manifeste index illisible : {e.__class__.__name__}")

    return Readiness(
        pipeline_ready=(not problems),
        provider=s.llm_provider,
        groq_configured=s.groq_configured,
        index_manifest_ok=manifest_ok,
        problems=problems,
    )


def provider_probe(settings: Optional[Settings] = None) -> dict:
    """Sonde provider EXPLICITE et bornée (un appel réel models.list).

    À appeler à la demande (pas à chaque /health) et avec mise en cache
    côté appelant. Ne lève jamais, n'expose ni identifiants ni erreurs
    brutes.
    """
    s = settings or load_settings()
    if not s.groq_configured:
        return {"provider": "groq", "available": False,
                "detail": "GROQ_API_KEY absente"}
    try:
        from src.llm.groq_service import GroqService

        return {"provider": "groq",
                "available": GroqService(settings=s).is_available(),
                "detail": ""}
    except LLMError as e:
        return {"provider": "groq", "available": False,
                "detail": str(e)[:120]}  # message typé, sans identifiant


# ──────────────────────────────────────────────
# PIPELINE v2
# ──────────────────────────────────────────────
class TourismRAGPipelineV2:
    """Pipeline v2 : sanitizer → retriever v2 → générateur (LLMService).

    Construction légère (aucun modèle/index) ; initialize() est EXPLICITE :
    il charge le retriever (manifeste + index existants — jamais de build)
    et le service LLM Groq. Toute erreur lève PipelineError : PAS de repli
    silencieux vers Ollama ou vers l'index legacy.

    `retriever` et `llm` sont injectables pour les tests.
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        retriever: Optional[Retriever] = None,
        llm: Optional[LLMService] = None,
    ):
        self.settings = settings or load_settings()
        self.sanitizer = InputSanitizer()
        self._retriever = retriever
        self._llm = llm
        self._generator: Optional[RAGGenerator] = None
        self._initialized = False

    @property
    def initialized(self) -> bool:
        return self._initialized

    def initialize(self) -> None:
        """Charge index (lecture seule) + provider Groq. Erreur → PipelineError."""
        if self._initialized:
            return
        if self.settings.llm_provider != "groq":
            raise PipelineError(
                f"pipeline v2 : provider non supporté "
                f"{self.settings.llm_provider!r} (v2 = groq uniquement ; "
                "le chemin legacy Ollama reste src/rag_pipeline.py, sans repli)"
            )
        if self._retriever is None:
            try:
                self._retriever = Retriever(settings=self.settings)
            except RetrieverError as e:
                raise PipelineError(f"index v2 invalide : {e}") from e
        if self._llm is None:
            if not self.settings.groq_configured:
                raise PipelineError(
                    "GROQ_API_KEY absente : pipeline v2 non configurable "
                    "(mode dégradé — pas de repli Ollama)"
                )
            from src.llm.groq_service import GroqService

            try:
                self._llm = GroqService(settings=self.settings)
            except LLMError as e:
                raise PipelineError(f"service Groq non initialisable : {e}") from e
        self._generator = RAGGenerator(llm=self._llm)
        self._initialized = True
        logger.info("Pipeline v2 initialisé (retriever manifesté + Groq).")

    def query(self, user_query: str, top_k: Optional[int] = None) -> dict:
        """Contrat compatible avec le pipeline legacy (clés identiques +
        extensions). Voir tests/test_api_v2.py pour la compatibilité."""
        start = time.time()
        clean, is_safe = self.sanitizer.sanitize(user_query)
        if not is_safe:
            return {
                "response": "Requête non autorisée.", "sources": [],
                "retrieved_docs": [], "response_time": 0.0, "safe": False,
                "citations": [], "abstained": True,
            }
        if not self._initialized:
            raise PipelineError(
                "pipeline v2 non initialisé — appeler initialize() "
                "(aucune exécution implicite)"
            )

        chunks: list[RetrievedChunk] = self._retriever.search(clean, top_k=top_k)
        result = self._generator.answer(clean, chunks)
        return {
            "response": result.text,
            "sources": [c.url for c in result.cited_chunks],
            "retrieved_docs": [
                {"text": c.text, "url": c.url, "title": c.title,
                 "source_type": c.source_type, "language": c.language,
                 "trust": c.trust, "similarity": c.similarity}
                for c in chunks
            ],
            "response_time": round(time.time() - start, 2),
            "safe": True,
            # extensions v2 (sur-ensemble du contrat legacy)
            "citations": result.citations,
            "abstained": result.abstained,
            "model": result.model_name,
        }
