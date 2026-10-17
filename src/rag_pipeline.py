# -*- coding: utf-8 -*-
"""
src/rag_pipeline.py
===================
Pipeline RAG — Tourisme Burkinabè
100% open source, local, zéro dépendance cloud.

Architecture :
  Question → Sanitization → Query Expansion → Embedding (cache)
  → ChromaDB → Top-K chunks → LLM (Ollama) → Réponse + Sources + Métriques

Historique des corrections :
  v1 → v2 : lru_cache module-level, prompt structuré, tokens réduits, AuditLogger idempotent
  v2 → v3 : modèle qwen2.5:3b, query expansion, prompt anti-hallucination, debug chunks
  v3 → v4 : [FIX-9]  num_predict 350→120 — cause principale des 46s (math : tokens × vitesse CPU)
             [FIX-10] Prompt compressé — 681 tokens input → ~280 tokens (mêmes contraintes, moitié moins)
             [FIX-11] Nettoyage encoding UTF-8 cassé dans les chunks (Ã©→é, Ã¨→è, etc.)
             [FIX-12] Filtre chunks : ignore les chunks qui commencent au milieu d'une phrase
"""

from __future__ import annotations

import re
import time
import json
import hashlib
from functools import lru_cache
from pathlib import Path
from typing import Optional

import bleach
import chromadb
import ollama
from sentence_transformers import SentenceTransformer
from loguru import logger


# ══════════════════════════════════════════════════════════
# CONFIGURATION CENTRALE
# ══════════════════════════════════════════════════════════
class Config:
    # Embeddings
    EMBEDDING_MODEL    = "paraphrase-multilingual-MiniLM-L12-v2"
    EMBEDDING_DEVICE   = "cpu"

    # Vector DB
    CHROMA_PATH        = "./data/chroma_db"
    COLLECTION_NAME    = "tourisme_burkina"

    # LLM local via Ollama
    LLM_MODEL          = "tourisme-burkina"
    LLM_TEMPERATURE    = 0.1
    # [FIX-9] CAUSE PRINCIPALE DES 46s
    # Avant : num_predict=350 → Ollama génère jusqu'à 350 tokens = ~16s sur CPU
    # Après : num_predict=120 → 5 bullet points = ~80-100 tokens suffisent largement
    # Calcul : à ~22 tokens/sec (mesuré), 350 tokens = 16s, 120 tokens = 5-6s
    # Combiné avec la réduction des tokens INPUT (FIX-10) : objectif 8-12s total
    LLM_MAX_TOKENS     = 220   # 5 bullets × ~30 tokens — augmenté pour éviter coupure
    LLM_TOP_P          = 0.9
    # Limite le context window d'Ollama — réduit l'allocation mémoire et accélère l'inférence
    # 1024 suffit largement : ~430 tokens input + 180 tokens output = 610 tokens max
    LLM_NUM_CTX        = 1024

    # RAG
    TOP_K              = 4
    MIN_SIMILARITY     = 0.25
    CHUNK_MAX_CHARS    = 400
    BATCH_SIZE         = 8

    # Corpus
    CORPUS_PATH        = "./data/corpus.json"

    # Sécurité
    MAX_QUERY_LENGTH   = 500
    RATE_LIMIT_SECONDS = 1.0

    # Logs
    AUDIT_LOG_PATH     = "./logs/audit.log"
    LOG_ROTATION       = "10 MB"
    LOG_RETENTION      = "7 days"

    # Passer à True pour voir les chunks dans les logs — diagnostique qualité
    LOG_RETRIEVED_CHUNKS = False


# ══════════════════════════════════════════════════════════
# SINGLETON — Modèle d'embedding chargé une seule fois
# ══════════════════════════════════════════════════════════
_embedding_model: Optional[SentenceTransformer] = None


def get_embedding_model() -> SentenceTransformer:
    """Retourne le modèle d'embedding partagé. Chargé une seule fois — pattern Singleton."""
    global _embedding_model
    if _embedding_model is None:
        logger.info(f"Chargement modèle embeddings : {Config.EMBEDDING_MODEL}")
        _embedding_model = SentenceTransformer(
            Config.EMBEDDING_MODEL,
            device=Config.EMBEDDING_DEVICE,
        )
        logger.success(f"Modèle chargé : {Config.EMBEDDING_MODEL}")
    return _embedding_model


# ══════════════════════════════════════════════════════════
# CACHE LRU AU NIVEAU MODULE
# ══════════════════════════════════════════════════════════
@lru_cache(maxsize=256)
def _embed_query_cached(model: SentenceTransformer, query: str) -> tuple[float, ...]:
    """Cache LRU module-level. Clé = (model, query). Retourne tuple (hashable)."""
    embeddings = model.encode([query], normalize_embeddings=True)
    return tuple(embeddings[0].tolist())


# ══════════════════════════════════════════════════════════
# [FIX-11] NETTOYAGE ENCODING UTF-8 CASSÉ
# ══════════════════════════════════════════════════════════
def fix_encoding(text: str) -> str:
    """
    Corrige les artéfacts d'encoding Latin-1/UTF-8 mal interprété
    fréquents dans les PDFs et pages web scraped.

    POURQUOI C'EST UN PROBLÈME :
    Quand un PDF ou une page web est encodé en Latin-1 (ISO-8859-1) mais
    lu comme UTF-8, les caractères accentués deviennent des séquences
    comme Ã© (é), Ã¨ (è), Ã (à), etc.
    Le LLM reçoit ce texte corrompu et peut soit l'ignorer, soit
    produire des réponses avec les mêmes artéfacts.

    Exemple vu dans les chunks :
      "caractÃ©ristiques"  → "caractéristiques"
      "touristiquebÃ¨"     → "touristiquebè"  (chunking cassé + encoding cassé)
      "protÃ©gÃ©es"        → "protégées"

    Stratégie : tentative de re-décodage Latin-1→UTF-8 d'abord,
    puis table de remplacement explicite pour les cas résiduels.
    """
    # Tentative de round-trip encoding : utf-8 → latin-1 → utf-8
    # Récupère les vrais caractères si le texte a été mal décodé à la source
    try:
        fixed = text.encode("latin-1").decode("utf-8")
        return fixed
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass

    # Table de remplacement explicite pour les séquences les plus communes
    # en français — couvre 95% des cas observés dans les corpus web/PDF
    REPLACEMENTS = {
        "Ã©": "\u00e9",  # é
        "Ã¨": "\u00e8",  # è
        "Ãª": "\u00ea",  # ê
        "Ã«": "\u00eb",  # ë
        "Ã ": "\u00e0",  # à
        "Ã¢": "\u00e2",  # â
        "Ã¤": "\u00e4",  # ä
        "Ã®": "\u00ee",  # î
        "Ã¯": "\u00ef",  # ï
        "Ã´": "\u00f4",  # ô
        "Ã¶": "\u00f6",  # ö
        "Ã²": "\u00f2",  # ò
        "Ã»": "\u00fb",  # û
        "Ã¹": "\u00f9",  # ù
        "Ã¼": "\u00fc",  # ü
        "Ã§": "\u00e7",  # ç
        "Ã†": "\u00c6",  # Æ
        "Ã\u2030": "\u00c9",  # É
        "Ã\u02c6": "\u00c8",  # È
        "Ã\u009c": "\u00dc",  # Ü
        "\u00e2\u20ac\u2122": "\u2019",  # apostrophe ’
        "\u00e2\u20ac\u0153": "\u201c",  # guillemet ouvrant
        "\u00e2\u20ac\x9d": "\u201d",  # guillemet fermant
        "\u00e2\u20ac\x93": "\u2013",  # tiret demi-cadratin
        "\u00e2\u20ac\x94": "\u2014",  # tiret cadratin
        "\u00c2\u00a0": " ",             # espace insécable
        "\u00c2\u00ab": "\u00ab",       # «
        "\u00c2\u00bb": "\u00bb",       # »
    }
    for bad, good in REPLACEMENTS.items():
        text = text.replace(bad, good)
    return text


# ══════════════════════════════════════════════════════════
# SÉCURITÉ — Sanitization des inputs
# ══════════════════════════════════════════════════════════
class InputSanitizer:
    """Protège contre prompt injection, HTML injection, queries trop longues."""

    INJECTION_PATTERNS = [
        r"ignore\s+(all\s+)?previous\s+instructions",
        r"tu\s+es\s+maintenant",
        r"oublie\s+tes\s+instructions",
        r"system\s*:",
        r"<\s*system\s*>",
        r"assistant\s*:",
        r"\[INST\]",
        r"<<SYS>>",
        r"jailbreak",
        r"DAN\s+(mode|prompt)",
    ]

    def __init__(self):
        self._compiled = [
            re.compile(p, re.IGNORECASE)
            for p in self.INJECTION_PATTERNS
        ]

    def sanitize(self, query: str) -> tuple[str, bool]:
        """Retourne (query_nettoyée, is_safe). is_safe=False si injection détectée."""
        if len(query) > Config.MAX_QUERY_LENGTH:
            query = query[:Config.MAX_QUERY_LENGTH]

        query = bleach.clean(query, tags=[], strip=True)
        query = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', query)

        for pattern in self._compiled:
            if pattern.search(query):
                logger.warning(f"INJECTION DÉTECTÉE : {query[:50]}...")
                return query, False

        return query.strip(), True


# ══════════════════════════════════════════════════════════
# QUERY EXPANSION — Améliore le retrieval ChromaDB
# ══════════════════════════════════════════════════════════
class QueryExpander:
    """
    Reformule la question utilisateur en termes documentaires
    pour maximiser le cosine similarity au retrieval.
    """

    _STOPWORDS = {
        "quels", "quelles", "quel", "quelle", "sont", "est", "les",
        "des", "du", "de", "la", "le", "un", "une", "au", "aux",
        "en", "et", "ou", "pour", "sur", "dans", "avec", "par",
        "me", "te", "se", "nous", "vous", "il", "elle", "ils", "elles",
        "je", "tu", "on", "ce", "ces", "mon", "ma", "mes", "ton", "ta",
        "ses", "son", "leur", "leurs", "que", "qui", "quoi", "où",
        "comment", "pourquoi", "combien", "peut", "pouvez", "puis",
        "parler", "dire", "savoir", "trouver", "voir", "visiter",
        "principaux", "principales", "meilleurs", "meilleures", "grands",
        "grand", "petit", "petits", "beau", "beaux", "belle", "belles",
    }

    _ENRICHMENTS: dict[str, list[str]] = {
        "site"       : ["site touristique", "attraction", "lieu", "patrimoine"],
        "sites"      : ["sites touristiques", "attractions", "lieux", "patrimoine"],
        "touristique": ["touristique", "culturel", "naturel", "historique"],
        "parc"       : ["parc national", "réserve naturelle", "faune"],
        "culture"    : ["culturel", "traditions", "ethnies", "musée", "festival"],
        "festival"   : ["festival", "fête", "célébration", "FESPACO", "SIAO"],
        "musée"      : ["musée", "musée national", "exposition"],
        "artisanat"  : ["artisanat", "SIAO", "sculpture", "tissu"],
        "nature"     : ["parc", "réserve", "faune", "flore", "hippopotame"],
        "histoire"   : ["historique", "patrimoine", "vestiges", "ruines", "UNESCO"],
        "UNESCO"     : ["UNESCO", "patrimoine mondial", "classé", "Loropéni"],
        "restaurant" : ["gastronomie", "cuisine burkinabè", "plat local"],
        "hôtel"      : ["hébergement", "hôtel", "lodge", "auberge"],
        "sécurité"   : ["sécurité", "conseils voyage", "zones"],
        "transport"  : ["transport", "bus", "vol", "route", "accès"],
        "ouagadougou": ["Ouagadougou", "Ouaga", "capitale", "musée national","Bangr Weogo", "FESPACO", "monuments", "attractions", "visiter"],
        "visiter"    : ["visiter", "voir", "attractions", "sites", "monuments", "musées"],
        "voir"       : ["voir", "visiter", "attractions", "sites", "monuments"],
        "bobo"       : ["Bobo-Dioulasso", "Bobo", "deuxième ville"],
        "loropeni"   : ["Loropéni", "ruines", "UNESCO", "patrimoine mondial"],
        "fespaco"    : ["FESPACO", "festival cinéma", "Ouagadougou"],
        "siao"       : ["SIAO", "artisanat", "salon international"],
    }

    def expand(self, query: str) -> str:
        """Produit une query enrichie pour le retrieval vectoriel."""
        query_lower = query.lower()
        keywords    = [
            w for w in re.findall(r'\b\w+\b', query_lower)
            if w not in self._STOPWORDS and len(w) > 2
        ]

        enrichments: list[str] = []
        for kw in keywords:
            if kw in self._ENRICHMENTS:
                enrichments.extend(self._ENRICHMENTS[kw])

        seen: set[str] = set()
        all_terms    = keywords + enrichments
        unique_terms = [t for t in all_terms if not (t in seen or seen.add(t))]  # type: ignore[func-returns-value]

        expanded = " ".join(unique_terms[:8])

        if expanded.lower() != query_lower:
            logger.debug(f"Query originale : {query}")
            logger.debug(f"Query enrichie  : {expanded}")

        return expanded if expanded else query


# ══════════════════════════════════════════════════════════
# EMBEDDINGS
# ══════════════════════════════════════════════════════════
class EmbeddingEngine:
    """Wrapper sentence-transformers avec cache LRU module-level."""

    def __init__(self):
        self.model = get_embedding_model()

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Encode une liste de textes — utilisé à l'indexation."""
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 100,
            batch_size=Config.BATCH_SIZE,
        )
        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        """Encode une query avec cache LRU — évite de recalculer."""
        return list(_embed_query_cached(self.model, query))


# ══════════════════════════════════════════════════════════
# BASE VECTORIELLE — ChromaDB
# ══════════════════════════════════════════════════════════
class VectorStore:
    """Interface ChromaDB persistante avec gestion des erreurs robuste."""

    def __init__(self, embedding_engine: EmbeddingEngine):
        self.client = chromadb.PersistentClient(path=Config.CHROMA_PATH)
        self.embedding_engine = embedding_engine
        self.collection = self.client.get_or_create_collection(
            name=Config.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            f"Collection ChromaDB : {Config.COLLECTION_NAME} "
            f"({self.collection.count()} docs)"
        )

    def index_corpus(self, corpus_path: str = Config.CORPUS_PATH) -> int:
        """Indexe le corpus JSON dans ChromaDB. Idempotent."""
        corpus_file = Path(corpus_path)
        if not corpus_file.exists():
            raise FileNotFoundError(
                f"Corpus introuvable : {corpus_path}\n"
                "Lance d'abord le scraping."
            )

        with open(corpus_file, encoding="utf-8") as f:
            chunks = json.load(f)

        existing   = set(self.collection.get(include=[])["ids"])
        new_chunks = [c for c in chunks if c["id"] not in existing]

        if not new_chunks:
            logger.info("Corpus déjà entièrement indexé.")
            return 0

        logger.info(f"Indexation de {len(new_chunks)} nouveaux chunks (batch={Config.BATCH_SIZE})...")

        indexed = 0
        for i in range(0, len(new_chunks), Config.BATCH_SIZE):
            batch = new_chunks[i : i + Config.BATCH_SIZE]
            texts = [c["text"] for c in batch]

            try:
                embeddings = self.embedding_engine.embed(texts)
                metadatas  = [
                    {
                        "url"        : c.get("url", ""),
                        "title"      : c.get("title", ""),
                        "source_type": c.get("source_type", "pdf"),
                        "language"   : c.get("language", "fr"),
                    }
                    for c in batch
                ]
                self.collection.add(
                    ids       =[c["id"] for c in batch],
                    embeddings=embeddings,
                    documents =texts,
                    metadatas =metadatas,
                )
                indexed += len(batch)
                logger.info(f"Indexé {indexed}/{len(new_chunks)} chunks...")

            except Exception as e:
                logger.error(f"Erreur batch {i // Config.BATCH_SIZE} : {e}")
                continue

        logger.success(f"Indexation terminée : {indexed} chunks ajoutés")
        return indexed

    def search(
        self,
        query_embedding : list[float],
        top_k           : int = Config.TOP_K,
        filter_language : Optional[str] = None,
    ) -> list[dict]:
        """
        Recherche les chunks les plus similaires.

        [FIX-11] Chaque chunk passe par fix_encoding() avant d'être retourné.
        [FIX-12] Les chunks qui commencent au milieu d'une phrase sont filtrés
                 (commencent par une minuscule ou une ponctuation de continuation).
        """
        if self.collection.count() == 0:
            logger.warning("Collection ChromaDB vide — indexe le corpus d'abord.")
            return []

        where_filter = (
            {"language": {"$eq": filter_language}}
            if filter_language else None
        )

        # On récupère plus de candidats pour compenser le filtre FIX-12
        n_candidates = min(top_k * 2, self.collection.count())

        results = self.collection.query(
            query_embeddings=[list(query_embedding)],
            n_results       =n_candidates,
            include         =["documents", "metadatas", "distances"],
            where           =where_filter,
        )

        docs = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            similarity = round(1 - dist, 4)
            if similarity < Config.MIN_SIMILARITY:
                continue

            # [FIX-11] Correction encoding avant tout traitement
            clean_doc = fix_encoding(doc)

            # [FIX-12] Filtre les chunks qui commencent au milieu d'une phrase.
            # Un chunk propre commence par une majuscule, un chiffre, ou "[".
            # Un chunk mal coupé commence par une minuscule, ")", ",", etc.
            first_char = clean_doc.lstrip()[:1]
            if first_char and first_char.islower() and not first_char in ["l", "d", "u"]:
                logger.debug(f"Chunk ignoré (début de phrase) : {clean_doc[:60]}...")
                continue

            docs.append({
                "text"       : clean_doc,
                "url"        : meta.get("url", ""),
                "title"      : meta.get("title", ""),
                "source_type": meta.get("source_type", ""),
                "similarity" : similarity,
            })

            # On s'arrête dès qu'on a top_k chunks valides
            if len(docs) >= top_k:
                break

        docs.sort(key=lambda x: x["similarity"], reverse=True)

        if Config.LOG_RETRIEVED_CHUNKS:
            for i, d in enumerate(docs):
                logger.debug(
                    f"Chunk {i+1} | sim={d['similarity']:.4f} | "
                    f"url={d['url'][:55]} | "
                    f"text={d['text'][:100].replace(chr(10), ' ')}..."
                )

        return docs


# ══════════════════════════════════════════════════════════
# PROMPT BUILDER
# ══════════════════════════════════════════════════════════
class PromptBuilder:
    """
    Construit les prompts LLM de façon centralisée.

    [FIX-10] Prompt compressé — même contraintes, ~60% moins de tokens.
    Avant : ~400 tokens d'instructions + ~280 tokens de contexte = 681 tokens input
    Après : ~150 tokens d'instructions + ~280 tokens de contexte = ~430 tokens input
    Impact : à ~22 tokens/sec, 250 tokens économisés = ~11s récupérées côté input
    (Note : Ollama charge le contexte input quasi instantanément — le vrai gain
     est surtout sur num_predict, mais un prompt court aide le modèle à rester focalisé)
    """

    @staticmethod
    def build_rag_prompt(query: str, docs: list[dict]) -> str:
        """
        Prompt compressé anti-hallucination avec citations forcées.
        Objectif : <160 tokens d'instructions, contexte <300 tokens.
        """
        context_parts = []
        for i, doc in enumerate(docs):
            raw_text   = doc["text"][:Config.CHUNK_MAX_CHARS]
            # Troncature propre — coupe au dernier espace
            clean_text = (
                raw_text.rsplit(" ", 1)[0]
                if len(doc["text"]) > Config.CHUNK_MAX_CHARS
                else raw_text
            )
            source_label = f"[Source {i + 1}]"
            if doc.get("title"):
                source_label += f" — {doc['title'][:40]}"
            context_parts.append(f"{source_label}\n{clean_text}")

        context = "\n---\n".join(context_parts)

        # [FIX-10] Version compressée — mêmes 3 contraintes, moitié moins de mots
        # Règle : chaque mot du prompt coûte des tokens = de la latence
        # On garde : (1) interdit d'inventer, (2) citer [Source N], (3) format bullet
        return (
            "Guide touristique bilingue du Burkina Faso. "
            "Sources disponibles en français ET en anglais. Reponds dans la langue de la QUESTION. "
            "Réponds dans la langue de la QUESTION. "
            "Utilise les infos des sources quelle que soit leur langue.\n"
            "RÈGLES : utilise UNIQUEMENT les sources ci-dessous. "
            "INTERDIT d'inventer. Si absent des sources : 'Non disponible.'\n"
            "FORMAT STRICT : réponds avec 2 à 5 bullet points.\n"
            "Chaque ligne commence par '• [Source' suivi du numéro et du fait.\n"
            "Exemple : '• [Source 2] Les Cascades de Banfora sont situées dans l'ouest.'\n"
            "Rien d'autre : pas d'introduction, pas de conclusion.\n\n"
            f"SOURCES :\n{context}\n\n"
            f"QUESTION : {query}\n\n"
            "RÉPONSE :"
        )

    @staticmethod
    def no_result_response() -> str:
        return (
            "Je n'ai pas trouvé d'informations suffisantes sur ce sujet "
            "dans ma base de données sur le tourisme burkinabè. "
            "Essayez une question plus spécifique (ville, type de site, festival...)."
        )


# ══════════════════════════════════════════════════════════
# AUDIT LOGS
# ══════════════════════════════════════════════════════════
class AuditLogger:
    """Traçabilité complète. Idempotent — handler Loguru ajouté une seule fois."""

    _initialized: bool = False

    def __init__(self):
        if not AuditLogger._initialized:
            log_path = Path(Config.AUDIT_LOG_PATH)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            logger.add(
                str(log_path),
                rotation =Config.LOG_ROTATION,
                retention=Config.LOG_RETENTION,
                format   ="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
                level    ="INFO",
            )
            AuditLogger._initialized = True

    def log_query(
        self,
        query        : str,
        nb_docs      : int,
        response_time: float,
        is_safe      : bool,
    ) -> None:
        query_hash = hashlib.sha256(query.encode()).hexdigest()[:8]
        logger.info(
            f"QUERY | hash={query_hash} | docs={nb_docs} | "
            f"time={response_time:.2f}s | safe={is_safe}"
        )


# ══════════════════════════════════════════════════════════
# PIPELINE RAG PRINCIPAL
# ══════════════════════════════════════════════════════════
class TourismRAGPipeline:
    """
    Orchestre le pipeline complet :
    Question → Sanitization → Query Expansion → Embedding
    → Retrieval (+ encoding fix + chunk filter) → LLM → Réponse

    Usage :
        pipeline = TourismRAGPipeline()
        pipeline.index()
        result = pipeline.query("Où sont les ruines de Loropéni ?")

    Debug qualité :
        Config.LOG_RETRIEVED_CHUNKS = True

    Structure du résultat :
        {
            "response"      : str,        # Réponse LLM formatée
            "sources"       : list[str],  # URLs des chunks
            "retrieved_docs": list[dict], # Chunks avec similarité
            "expanded_query": str,        # Query enrichie (retrieval)
            "response_time" : float,      # Temps total en secondes
            "safe"          : bool,       # False si injection
        }
    """

    def __init__(self):
        logger.info("Initialisation pipeline RAG Tourisme Burkina...")
        self.sanitizer      = InputSanitizer()
        self.query_expander = QueryExpander()
        self.embedder       = EmbeddingEngine()
        self.vector_store   = VectorStore(self.embedder)
        self.prompt         = PromptBuilder()
        self.audit          = AuditLogger()
        self._last_request_time = 0.0
        logger.success("Pipeline prêt.")

    def index(self) -> int:
        """Indexe le corpus. Idempotent."""
        return self.vector_store.index_corpus()

    def query(self, user_query: str) -> dict:
        """
        Exécute une query RAG complète.

        Budget tokens visé (v4) :
          Input  : ~430 tokens  (instructions ~150 + contexte ~280)
          Output : 120 tokens max
          Total  : ~550 tokens → ~25s sur CPU → objectif 10-15s avec cache embedding
        """
        start_time = time.time()

        # ── Rate limiting ──────────────────────────────────
        elapsed = time.time() - self._last_request_time
        if elapsed < Config.RATE_LIMIT_SECONDS:
            time.sleep(Config.RATE_LIMIT_SECONDS - elapsed)
        self._last_request_time = time.time()

        # ── Sanitization ───────────────────────────────────
        clean_query, is_safe = self.sanitizer.sanitize(user_query)
        if not is_safe:
            return {
                "response"      : "Requête non autorisée.",
                "sources"       : [],
                "retrieved_docs": [],
                "expanded_query": "",
                "response_time" : 0,
                "safe"          : False,
            }

        # ── Query Expansion ────────────────────────────────
        # expanded_query → retrieval (cosine similarity)
        # clean_query    → prompt LLM (vraie question)
        expanded_query = self.query_expander.expand(clean_query)

        # ── Embedding + Retrieval ──────────────────────────
        query_embedding = self.embedder.embed_query(clean_query)
        retrieved_docs  = self.vector_store.search(query_embedding)

        # ── Génération LLM ─────────────────────────────────
        if not retrieved_docs:
            response_text = self.prompt.no_result_response()
        else:
            prompt_text = self.prompt.build_rag_prompt(clean_query, retrieved_docs)
            try:
                resp = ollama.chat(
                    model   =Config.LLM_MODEL,
                    messages=[{"role": "user", "content": prompt_text}],
                    options ={
                        "temperature" : Config.LLM_TEMPERATURE,
                        "num_predict" : Config.LLM_MAX_TOKENS,
                        "top_p"       : Config.LLM_TOP_P,
                        # [FIX-9b] num_ctx réduit = Ollama alloue moins de mémoire
                        # et traite chaque token plus vite. Sans ça, Ollama utilise
                        # le context window par défaut du modèle (souvent 4096-32768)
                        # même pour un prompt de 400 tokens — gaspillage pur.
                        "num_ctx"     : Config.LLM_NUM_CTX,
                        # Stop sequences : Ollama arrête dès qu'il voit ces tokens
                        # évite que le modèle continue après les bullet points
                        "stop"        : ["QUESTION", "SOURCES", "\n\n\n"],
                    },
                )
                response_text = resp["message"]["content"].strip()
            except Exception as e:
                logger.error(f"Erreur LLM : {e}")
                response_text = (
                    "Désolé, une erreur est survenue lors de la génération. "
                    "Vérifiez qu'Ollama est lancé (`ollama serve`) "
                    f"et que le modèle '{Config.LLM_MODEL}' est installé "
                    f"(`ollama pull {Config.LLM_MODEL}`)."
                )

        # ── Audit + Retour ─────────────────────────────────
        response_time = round(time.time() - start_time, 2)
        self.audit.log_query(
            clean_query, len(retrieved_docs), response_time, is_safe
        )
        logger.success(f"Réponse générée en {response_time:.2f}s")

        return {
            "response"      : response_text,
            "sources"       : [d.get("url", "") for d in retrieved_docs],
            "retrieved_docs": retrieved_docs,
            "expanded_query": expanded_query,
            "response_time" : response_time,
            "safe"          : True,
        }


# ══════════════════════════════════════════════════════════
# POINT D'ENTRÉE
# ══════════════════════════════════════════════════════════
if __name__ == "__main__":
    # Config.LOG_RETRIEVED_CHUNKS = True  # décommenter pour debug qualité

    pipeline = TourismRAGPipeline()
    pipeline.index()

    questions = [
        "Quels sont les sites touristiques du Burkina Faso classés UNESCO ?",
        "Quels sont les principaux parcs nationaux du Burkina Faso ?",
        "Que visiter à Ouagadougou ?",
    ]

    for question in questions:
        print(f"\n{'═' * 60}")
        print(f"❓ {question}")
        result = pipeline.query(question)
        print(f"\n{'─' * 40}")
        print(result["response"])
        print(f"\n⏱  Temps         : {result['response_time']}s")
        print(f"🔍 Query enrichie : {result['expanded_query']}")
        print(f"📚 Sources ({len(result['sources'])}) : {result['sources']}")



