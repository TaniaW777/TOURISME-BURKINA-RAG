# -*- coding: utf-8 -*-
"""
src/config.py
=============
Configuration centralisée — Tourisme Burkina RAG.

Sources de configuration, par ordre de priorité :
  1. Variables d'environnement réelles (docker, CI, hébergement)
  2. Fichier .env à la racine du repo (développement local uniquement)
  3. Valeurs par défaut ci-dessous

Tous les chemins sont ancrés à la racine du repo, indépendamment du
répertoire courant d'où l'application est lancée.

Aucune dépendance lourde ici (pas de torch/chromadb) : ce module doit
pouvoir être importé par les tests et par tout point d'entrée léger.

Introduit en Phase 1 (préparation Groq). Le pipeline existant
(rag_pipeline.Config) n'est PAS encore migré — voir roadmap.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# ──────────────────────────────────────────────
# RACINE DU REPO — ancrage indépendant du CWD
# ──────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_dotenv_if_present() -> None:
    """Charge .env à la racine du repo s'il existe (dev local).

    Ne remplace JAMAIS une variable d'environnement déjà définie
    (comportement load_dotenv par défaut) — en production les vraies
    variables d'environnement priment toujours.
    """
    dotenv_path = REPO_ROOT / ".env"
    if not dotenv_path.exists():
        return
    try:
        from dotenv import load_dotenv

        load_dotenv(dotenv_path=dotenv_path, override=False)
    except ImportError:
        # python-dotenv absent : on utilise uniquement l'environnement réel
        pass


# ──────────────────────────────────────────────
# VALEURS PAR DÉFAUT
# ──────────────────────────────────────────────
DEFAULT_GROQ_MODEL = "openai/gpt-oss-20b"
DEFAULT_EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"


@dataclass(frozen=True)
class Settings:
    """Configuration immuable de l'application.

    Construite via `load_settings()` qui lit l'environnement.
    """

    # ── LLM (Groq) ──
    groq_api_key: str = ""          # vide = mode dégradé (voir groq_configured)
    groq_model: str = DEFAULT_GROQ_MODEL
    groq_timeout_s: float = 20.0    # timeout par requête, borné aussi côté client
    groq_max_retries: int = 2       # retries SDK ; les 429 ne sont JAMAIS rejoués à l'identique

    # ── Embeddings (local, sentence-transformers) ──
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    embedding_device: str = "cpu"

    # ── Stockage vectoriel ──
    chroma_path: Path = REPO_ROOT / "data" / "chroma_db"
    collection_name: str = "tourisme_burkina"
    corpus_path: Path = REPO_ROOT / "data" / "corpus.json"

    # ── RAG ──
    top_k: int = 5
    min_similarity: float = 0.30
    max_query_length: int = 500
    retrieval_candidates_multiplier: int = 2  # candidats = top_k × mult, avant filtrage

    # ── API / exploitation ──
    rate_limit_per_minute: int = 10
    log_retrieved_chunks: bool = False
    audit_log_path: Path = REPO_ROOT / "logs" / "audit.log"

    # ──────────────────────────────────────────
    # ÉTAT DÉRIVÉ
    # ──────────────────────────────────────────
    @property
    def groq_configured(self) -> bool:
        """True si une clé Groq est présente.

        Mode dégradé : sans clé, l'UI et /health démarrent normalement,
        mais /query doit renvoyer 503 explicite plutôt qu'un crash.
        """
        return bool(self.groq_api_key.strip())

    def validate(self) -> list[str]:
        """Retourne la liste des problèmes de configuration (vide = OK).

        Ne lève jamais : permet un démarrage dégradé avec diagnostic.
        """
        problems: list[str] = []
        if not self.groq_api_key.strip():
            problems.append(
                "GROQ_API_KEY absente : l'application démarre en mode dégradé "
                "(réponses LLM indisponibles)."
            )
        if self.groq_timeout_s <= 0 or self.groq_timeout_s > 60:
            problems.append(f"GROQ_TIMEOUT_S invalide : {self.groq_timeout_s} (attendu 0 < t <= 60)")
        if self.groq_max_retries < 0 or self.groq_max_retries > 5:
            problems.append(f"GROQ_MAX_RETRIES invalide : {self.groq_max_retries} (attendu 0-5)")
        if self.top_k < 1 or self.top_k > 10:
            problems.append(f"TOP_K invalide : {self.top_k} (attendu 1-10)")
        if not 0.0 <= self.min_similarity <= 1.0:
            problems.append(f"MIN_SIMILARITY invalide : {self.min_similarity} (attendu 0.0-1.0)")
        if not self.embedding_model.strip():
            problems.append("EMBEDDING_MODEL vide.")
        return problems


def load_settings(env: dict[str, str] | None = None) -> Settings:
    """Construit les Settings depuis l'environnement (+ .env si présent).

    `env` permet aux tests d'injecter un environnement sans le modifier
    globalement (monkeypatch des variables réelles inutile).
    """
    if env is None:
        _load_dotenv_if_present()
        source: dict[str, str] = dict(os.environ)
    else:
        source = env

    get  = source.get
    gint = lambda name, default: _to_int(get(name, ""), default)
    gflt = lambda name, default: _to_float(get(name, ""), default)
    gbool = lambda name, default: get(name, "").strip().lower() in ("1", "true", "yes", "on") if get(name, "") else default

    return Settings(
        # Groq — jamais loguée, jamais exposée côté client
        groq_api_key   =get("GROQ_API_KEY", ""),
        groq_model     =get("GROQ_MODEL", DEFAULT_GROQ_MODEL).strip(),
        groq_timeout_s =gflt("GROQ_TIMEOUT_S", 20.0),
        groq_max_retries=gint("GROQ_MAX_RETRIES", 2),
        # Embeddings
        embedding_model=get("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL).strip(),
        embedding_device=get("EMBEDDING_DEVICE", "cpu").strip(),
        # ChromaDB
        chroma_path    =Path(get("CHROMA_PATH", str(REPO_ROOT / "data" / "chroma_db"))),
        corpus_path    =Path(get("CORPUS_PATH", str(REPO_ROOT / "data" / "corpus.json"))),
        collection_name=get("COLLECTION_NAME", "tourisme_burkina").strip(),
        # RAG
        top_k          =gint("TOP_K", 5),
        min_similarity =gflt("MIN_SIMILARITY", 0.30),
        max_query_length=gint("MAX_QUERY_LENGTH", 500),
        # Exploitation
        rate_limit_per_minute=gint("RATE_LIMIT_PER_MINUTE", 10),
        log_retrieved_chunks=gbool("LOG_RETRIEVED_CHUNKS", False),
    )


def _to_int(raw: str, default: int) -> int:
    try:
        return int(raw) if raw else default
    except ValueError:
        return default


def _to_float(raw: str, default: float) -> float:
    try:
        return float(raw) if raw else default
    except ValueError:
        return default


def resolve_repo_relative(path: Path) -> Path:
    """Résout un chemin relatif contre la racine du repo, pas le CWD.

    Utilisé pour les chemins passés par l'environnement : un chemin
    relatif comme 'data/chroma_db' doit toujours désigner le même
    répertoire, quel que soit l'endroit d'où le serveur est lancé.
    """
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()
