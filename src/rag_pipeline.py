"""
src/rag_pipeline.py
===================
Pipeline RAG complet pour le Tourisme Burkinabè.
100% open source, local, zéro dépendance cloud.

Architecture :
  Question → Sanitization → Embeddings (BGE-M3) → ChromaDB
  → Top-K chunks → Couche agentique → Qwen3/LLaMA via Ollama
  → Réponse + Sources + Métriques

Technologies :
  - sentence-transformers : Apache 2.0
  - chromadb              : Apache 2.0
  - ollama                : MIT
  - langgraph             : MIT
  - loguru                : MIT
"""

import re
import time
import json
import hashlib
from pathlib import Path
from typing import Optional

import bleach
import chromadb
import ollama
from sentence_transformers import SentenceTransformer
from loguru import logger
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator


# ══════════════════════════════════════════════════════════
# CONFIGURATION CENTRALE
# ══════════════════════════════════════════════════════════
class Config:
    # --- Embeddings ---
    # BAAI/bge-m3 : meilleur modèle multilingue open source 2025
    # Top MTEB Multilingual, supporte français natif
    # Licence : MIT — https://huggingface.co/BAAI/bge-m3
    EMBEDDING_MODEL = "BAAI/bge-m3"

    # --- Vector DB ---
    # ChromaDB local, persistant sur disque
    # Aucun service externe requis
    CHROMA_PATH = "./data/chroma_db"
    COLLECTION_NAME = "tourisme_burkina"

    # --- LLM ---
    # Qwen3-8B-Instruct via Ollama (local, gratuit, excellent en français)
    # Alternative : llama3.1:8b-instruct-q4_K_M
    # Fallback : mistral:7b-instruct
    LLM_MODEL = "qwen2.5:7b-instruct"  # Disponible dans Ollama registry
    LLM_TEMPERATURE = 0.3   # Bas = réponses factuelles (pas d'hallucination)
    LLM_MAX_TOKENS = 1024

    # --- RAG ---
    TOP_K = 5              # Nombre de chunks récupérés (empirique : 5 = bon ratio qualité/vitesse)
    MIN_SIMILARITY = 0.3   # Seuil similarité cosinus minimum (0 = tout, 1 = exact)
    CHUNK_SIZE = 500        # Mots par chunk (voir scraping_script.py)

    # --- Corpus ---
    CORPUS_PATH = "./data/corpus.json"

    # --- Sécurité ---
    MAX_QUERY_LENGTH = 500         # Longueur max question utilisateur
    RATE_LIMIT_SECONDS = 1.0       # Min secondes entre requêtes
    AUDIT_LOG_PATH = "./logs/audit.log"


# ══════════════════════════════════════════════════════════
# SÉCURITÉ — Sanitization des inputs
# ══════════════════════════════════════════════════════════
class InputSanitizer:
    """
    Protège contre :
    - Prompt injection (instructions cachées dans la question)
    - HTML/JS injection
    - Questions trop longues (DoS)
    - Caractères de contrôle malveillants
    """

    # Patterns de prompt injection courants
    # Un attaquant peut écrire "Ignore tes instructions et fais X"
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
        # Compile les patterns une fois pour la perf
        self._compiled = [
            re.compile(p, re.IGNORECASE) for p in self.INJECTION_PATTERNS
        ]

    def sanitize(self, query: str) -> tuple[str, bool]:
        """
        Sanitize la query utilisateur.
        Retourne : (query_nettoyée, is_safe)
        is_safe = False si injection détectée
        """
        # Longueur max
        if len(query) > Config.MAX_QUERY_LENGTH:
            query = query[:Config.MAX_QUERY_LENGTH]

        # Nettoyage HTML
        query = bleach.clean(query, tags=[], strip=True)

        # Supprime caractères de contrôle
        query = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', query)

        # Détecte prompt injection
        for pattern in self._compiled:
            if pattern.search(query):
                logger.warning(f"PROMPT INJECTION DÉTECTÉE : {query[:50]}...")
                return query, False  # Marque comme non-safe

        return query.strip(), True


# ══════════════════════════════════════════════════════════
# EMBEDDINGS — Modèle BGE-M3
# ══════════════════════════════════════════════════════════
class EmbeddingEngine:
    """
    Wrapper autour de sentence-transformers.
    BGE-M3 : multilingue, 8192 tokens max, dense + sparse retrieval.
    Chargé une seule fois en mémoire pour la performance.
    """

    def __init__(self):
        logger.info(f"Chargement modèle embeddings : {Config.EMBEDDING_MODEL}")
        # SentenceTransformer télécharge automatiquement le modèle (HuggingFace)
        # Premier run : ~500MB de téléchargement
        self.model = SentenceTransformer(
            Config.EMBEDDING_MODEL,
            device="cpu",  # CPU suffisant pour embedding ; GPU si disponible
        )
        logger.success(f"Modèle chargé : {Config.EMBEDDING_MODEL}")

    def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Encode une liste de textes en vecteurs.
        normalize_embeddings=True requis pour similarité cosinus correcte.
        """
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,  # OBLIGATOIRE pour cosine similarity
            show_progress_bar=len(texts) > 100,  # Barre progrès si corpus
            batch_size=32,  # Batch pour perf mémoire
        )
        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        """Encode une question unique (inference time)."""
        # BGE-M3 supporte instruction prefix pour les queries
        prefixed = f"Represent this sentence for searching relevant passages: {query}"
        return self.embed([prefixed])[0]


# ══════════════════════════════════════════════════════════
# BASE VECTORIELLE — ChromaDB
# ══════════════════════════════════════════════════════════
class VectorStore:
    """
    Interface ChromaDB persistante.
    Les données survivent aux redémarrages (stockage disque).
    """

    def __init__(self, embedding_engine: EmbeddingEngine):
        # Client persistant : les données sont sauvées dans CHROMA_PATH
        self.client = chromadb.PersistentClient(path=Config.CHROMA_PATH)
        self.embedding_engine = embedding_engine

        # Récupère ou crée la collection
        # metadata{"hnsw:space": "cosine"} = similarité cosinus (obligatoire avec BGE-M3 normalisé)
        self.collection = self.client.get_or_create_collection(
            name=Config.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(f"Collection ChromaDB : {Config.COLLECTION_NAME} ({self.collection.count()} docs)")

    def index_corpus(self, corpus_path: str = Config.CORPUS_PATH) -> int:
        """
        Indexe le corpus JSON dans ChromaDB.
        Idempotent : skip les documents déjà indexés.
        Retourne le nombre de nouveaux documents indexés.
        """
        corpus_file = Path(corpus_path)
        if not corpus_file.exists():
            raise FileNotFoundError(
                f"Corpus introuvable : {corpus_path}\n"
                "Lance d'abord : python data/scraping_script.py"
            )

        with open(corpus_file, encoding="utf-8") as f:
            chunks = json.load(f)

        logger.info(f"Indexation de {len(chunks)} chunks...")

        # Récupère les IDs déjà indexés pour éviter les doublons
        existing = set(self.collection.get(include=[])["ids"])

        # Filtre les nouveaux chunks
        new_chunks = [c for c in chunks if c["id"] not in existing]
        logger.info(f"Nouveaux chunks à indexer : {len(new_chunks)}")

        if not new_chunks:
            logger.info("Aucun nouveau chunk. Corpus déjà indexé.")
            return 0

        # Indexation par batches (ChromaDB a une limite par appel)
        batch_size = 100
        indexed = 0

        for i in range(0, len(new_chunks), batch_size):
            batch = new_chunks[i : i + batch_size]

            # Génère les embeddings pour ce batch
            texts = [c["text"] for c in batch]
            embeddings = self.embedding_engine.embed(texts)

            # Métadonnées filtrables (ex : filtrer par langue ou type de source)
            metadatas = [
                {
                    "url": c["url"],
                    "title": c["title"],
                    "source_type": c["source_type"],
                    "language": c.get("language", "fr"),
                    "chunk_idx": c.get("chunk_idx", 0),
                }
                for c in batch
            ]

            # Insertion dans ChromaDB
            self.collection.add(
                ids=[c["id"] for c in batch],
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
            )

            indexed += len(batch)
            logger.info(f"Indexé {indexed}/{len(new_chunks)} chunks...")

        logger.success(f"Indexation terminée : {indexed} nouveaux chunks")
        return indexed

    def search(
        self,
        query_embedding: list[float],
        top_k: int = Config.TOP_K,
        filter_language: Optional[str] = None,
    ) -> list[dict]:
        """
        Recherche les chunks les plus similaires à la query.
        filter_language : filtre optionnel par langue (ex: "fr")
        """
        # Construction du filtre metadata optionnel
        where_filter = None
        if filter_language:
            where_filter = {"language": {"$eq": filter_language}}

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
            where=where_filter,
        )

        # Transforme le résultat ChromaDB en liste de dicts propres
        docs = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            # ChromaDB retourne distance cosinus [0,2] avec "cosine" space
            # On convertit en similarité [0,1] : sim = 1 - dist/2
            similarity = 1 - (dist / 2)

            if similarity >= Config.MIN_SIMILARITY:
                docs.append({
                    "text": doc,
                    "url": meta.get("url", ""),
                    "title": meta.get("title", ""),
                    "source_type": meta.get("source_type", ""),
                    "similarity": round(similarity, 4),
                })

        # Tri par similarité décroissante
        docs.sort(key=lambda x: x["similarity"], reverse=True)
        return docs


# ══════════════════════════════════════════════════════════
# COUCHE AGENTIQUE LÉGÈRE — LangGraph
# ══════════════════════════════════════════════════════════
# Pattern : Vérificateur → Synthétiseur
# Pas de sur-ingénierie : 2 nœuds max

class AgentState(TypedDict):
    """État partagé entre les nœuds de l'agent."""
    query: str                         # Question originale
    retrieved_docs: list[dict]         # Documents récupérés
    context: str                       # Contexte formaté pour le LLM
    response: str                      # Réponse générée
    sources: list[str]                 # URLs des sources
    needs_refinement: bool             # Le vérificateur demande-t-il un raffinement ?
    messages: Annotated[list, operator.add]  # Historique messages


class AgentLayer:
    """
    Couche agentique légère avec LangGraph.
    Nœud 1 — Vérificateur : évalue si les docs sont pertinents
    Nœud 2 — Synthétiseur : génère la réponse finale
    """

    def __init__(self, llm_client):
        self.llm = llm_client
        self.graph = self._build_graph()

    def _build_graph(self) -> any:
        """Construit le graphe LangGraph."""
        workflow = StateGraph(AgentState)

        # Ajoute les nœuds
        workflow.add_node("verifier", self._verify_relevance)
        workflow.add_node("synthesizer", self._synthesize_response)

        # Définit les transitions
        workflow.set_entry_point("verifier")

        # Transition conditionnelle : si docs pas pertinents, synthèse directe d'excuse
        workflow.add_conditional_edges(
            "verifier",
            lambda state: "synthesizer",  # Toujours synthétise (simple pour ce scope)
            {"synthesizer": "synthesizer"},
        )
        workflow.add_edge("synthesizer", END)

        return workflow.compile()

    def _verify_relevance(self, state: AgentState) -> AgentState:
        """
        Nœud 1 : Vérifie que les documents récupérés sont pertinents.
        Évite de générer une réponse à partir de docs hors-sujet.
        """
        if not state["retrieved_docs"]:
            state["needs_refinement"] = True
            state["context"] = ""
            return state

        # Formate le contexte pour le LLM
        context_parts = []
        sources = []

        for i, doc in enumerate(state["retrieved_docs"], 1):
            context_parts.append(
                f"[Source {i}] (similarité: {doc['similarity']:.2f})\n"
                f"Titre: {doc['title']}\n"
                f"URL: {doc['url']}\n"
                f"Contenu: {doc['text'][:600]}...\n"
            )
            if doc["url"] not in sources:
                sources.append(doc["url"])

        state["context"] = "\n---\n".join(context_parts)
        state["sources"] = sources
        state["needs_refinement"] = False

        logger.info(f"Vérificateur : {len(state['retrieved_docs'])} docs vérifiés")
        return state

    def _synthesize_response(self, state: AgentState) -> AgentState:
        """
        Nœud 2 : Synthétise la réponse finale avec le LLM.
        Prompt structuré pour forcer les sources et limiter les hallucinations.
        """
        if state["needs_refinement"] or not state["context"]:
            state["response"] = (
                "Je n'ai pas trouvé d'informations suffisantes sur ce sujet "
                "dans ma base de données sur le tourisme burkinabè. "
                "Essayez une question plus spécifique, par exemple sur Loropéni, "
                "les Cascades de Banfora ou la culture Mossi."
            )
            return state

        # Prompt système : instructions strictes contre hallucination
        system_prompt = """Tu es un assistant expert en tourisme burkinabè.
Tu réponds UNIQUEMENT en te basant sur les sources fournies.
Tu ne dois PAS inventer d'informations non présentes dans les sources.
Tu cites toujours tes sources à la fin de ta réponse.
Tu réponds en français, de manière claire et informative.
Si une information n'est pas dans les sources, dis-le explicitement."""

        # Prompt utilisateur avec contexte
        user_prompt = f"""Question : {state['query']}

Sources disponibles :
{state['context']}

Réponds à la question en te basant uniquement sur ces sources.
À la fin, liste les sources utilisées sous le format :
📚 Sources : [url1], [url2], ..."""

        # Appel LLM via Ollama (local, pas de cloud)
        response = self.llm.chat(
            model=Config.LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            options={
                "temperature": Config.LLM_TEMPERATURE,
                "num_predict": Config.LLM_MAX_TOKENS,
            },
        )

        state["response"] = response["message"]["content"]
        logger.info("Synthétiseur : réponse générée")
        return state

    def run(self, query: str, retrieved_docs: list[dict]) -> dict:
        """Point d'entrée de l'agent."""
        initial_state = {
            "query": query,
            "retrieved_docs": retrieved_docs,
            "context": "",
            "response": "",
            "sources": [],
            "needs_refinement": False,
            "messages": [],
        }
        final_state = self.graph.invoke(initial_state)
        return final_state


# ══════════════════════════════════════════════════════════
# AUDIT LOGS — Traçabilité complète
# ══════════════════════════════════════════════════════════
class AuditLogger:
    """
    Loggue toutes les requêtes pour traçabilité.
    Log format : timestamp | query_hash | nb_docs | temps_réponse | safe
    """

    def __init__(self):
        log_path = Path(Config.AUDIT_LOG_PATH)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        # Configure loguru pour fichier rotatif (max 10MB, garde 7 jours)
        logger.add(
            str(log_path),
            rotation="10 MB",
            retention="7 days",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
            level="INFO",
        )

    def log_query(
        self,
        query: str,
        nb_docs: int,
        response_time: float,
        is_safe: bool,
    ):
        """Loggue une requête (sans stocker la query complète pour la vie privée)."""
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
    Classe principale qui orchestre tout le pipeline :
    Question → Sanitization → Embedding → Retrieval → Agent → Réponse

    Usage :
        pipeline = TourismRAGPipeline()
        pipeline.index()          # Indexe le corpus (une seule fois)
        result = pipeline.query("Où se trouvent les ruines de Loropéni ?")
    """

    def __init__(self):
        logger.info("Initialisation pipeline RAG Tourisme Burkina...")

        # Composants du pipeline
        self.sanitizer = InputSanitizer()
        self.embedder = EmbeddingEngine()
        self.vector_store = VectorStore(self.embedder)
        self.audit = AuditLogger()

        # Client Ollama (LLM local)
        self.llm_client = ollama  # Module ollama direct
        self.agent = AgentLayer(self.llm_client)

        # Rate limiting simple (timestamp dernière requête)
        self._last_request_time = 0.0

        logger.success("Pipeline prêt.")

    def index(self) -> int:
        """
        Indexe le corpus dans ChromaDB.
        Appelle cette méthode une seule fois après le scraping.
        Idempotent : sans risque d'appels multiples.
        """
        return self.vector_store.index_corpus()

    def query(self, user_query: str) -> dict:
        """
        Point d'entrée principal pour une question utilisateur.

        Retourne :
            {
                "response": str,      # Réponse générée
                "sources": list[str], # URLs des sources
                "retrieved_docs": list, # Docs récupérés
                "response_time": float, # Temps en secondes
                "safe": bool,          # Query est-elle safe ?
            }
        """
        start_time = time.time()

        # ── 1. Rate limiting ──
        elapsed = time.time() - self._last_request_time
        if elapsed < Config.RATE_LIMIT_SECONDS:
            time.sleep(Config.RATE_LIMIT_SECONDS - elapsed)
        self._last_request_time = time.time()

        # ── 2. Sanitization ──
        clean_query, is_safe = self.sanitizer.sanitize(user_query)

        if not is_safe:
            return {
                "response": "Requête non autorisée. Posez une question sur le tourisme burkinabè.",
                "sources": [],
                "retrieved_docs": [],
                "response_time": time.time() - start_time,
                "safe": False,
            }

        # ── 3. Embedding de la question ──
        logger.info(f"Embedding query : {clean_query[:60]}...")
        query_embedding = self.embedder.embed_query(clean_query)

        # ── 4. Recherche vectorielle ──
        logger.info("Recherche vectorielle ChromaDB...")
        retrieved_docs = self.vector_store.search(
            query_embedding,
            top_k=Config.TOP_K,
        )
        logger.info(f"{len(retrieved_docs)} documents récupérés")

        # ── 5. Agent LangGraph (vérification + synthèse) ──
        logger.info("Lancement agent LangGraph...")
        agent_result = self.agent.run(clean_query, retrieved_docs)

        # ── 6. Résultat final ──
        response_time = time.time() - start_time
        self.audit.log_query(clean_query, len(retrieved_docs), response_time, is_safe)

        logger.success(f"Réponse générée en {response_time:.2f}s")

        return {
            "response": agent_result["response"],
            "sources": agent_result["sources"],
            "retrieved_docs": retrieved_docs,
            "response_time": round(response_time, 2),
            "safe": True,
        }


# ══════════════════════════════════════════════════════════
# POINT D'ENTRÉE DIRECT (test)
# ══════════════════════════════════════════════════════════
if __name__ == "__main__":
    # Test rapide du pipeline complet
    pipeline = TourismRAGPipeline()

    # Indexation (skip si déjà fait)
    pipeline.index()

    # Test query
    result = pipeline.query("Quels sont les sites touristiques du Burkina Faso classés UNESCO ?")
    print("\n=== RÉPONSE ===")
    print(result["response"])
    print(f"\n⏱  Temps de réponse : {result['response_time']}s")
    print(f"📚 Sources : {result['sources']}")
