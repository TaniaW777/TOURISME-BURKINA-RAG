"""
src/api.py
==========
API REST FastAPI pour le système RAG Tourisme Burkina.
Sécurité : rate limiting, CORS strict, validation Pydantic, headers sécurité.

Technologies :
  - FastAPI  : MIT — https://github.com/tiangolo/fastapi
  - slowapi  : MIT — https://github.com/laurentS/slowapi
  - pydantic : MIT — https://github.com/pydantic/pydantic

Usage :
    uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload

Endpoints :
    POST /query   — Question → Réponse RAG
    GET  /health  — Health check
    POST /index   — (Re)indexer le corpus
    GET  /stats   — Statistiques base vectorielle
"""

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from loguru import logger

# Import du pipeline RAG
import sys
sys.path.insert(0, ".")
from src.rag_pipeline import TourismRAGPipeline


# ──────────────────────────────────────────────
# RATE LIMITER — SlowAPI (MIT)
# Protège contre les abus : max 10 requêtes/minute par IP
# ──────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address)

# Variable globale du pipeline (singleton, chargé au démarrage)
pipeline: TourismRAGPipeline | None = None


# ──────────────────────────────────────────────
# LIFESPAN — Chargement du pipeline au démarrage
# ──────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Charge le pipeline RAG une seule fois au démarrage.
    Évite de recharger le modèle d'embeddings à chaque requête.
    """
    global pipeline
    logger.info("Chargement pipeline RAG au démarrage...")
    pipeline = TourismRAGPipeline()
    logger.success("Pipeline RAG prêt.")
    yield
    # Cleanup au shutdown (rien de spécial ici)
    logger.info("Arrêt API.")


# ──────────────────────────────────────────────
# APPLICATION FASTAPI
# ──────────────────────────────────────────────
app = FastAPI(
    title="Assistant IA Tourisme Burkina Faso",
    description="Système RAG 100% open source sur le tourisme burkinabè",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",      # Swagger UI
    redoc_url="/redoc",    # ReDoc
)

# ── Rate limiter ──
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ── CORS — Autorise uniquement les origines connues ──
# En production : remplace par ton domaine exact
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:7860",  # Gradio local
        "http://localhost:3000",  # Dev frontend
        "https://huggingface.co", # HF Spaces
        # Ajoute ton domaine de prod ici
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],  # Seulement les méthodes nécessaires
    allow_headers=["Content-Type"],
)


# ──────────────────────────────────────────────
# MIDDLEWARE — Security headers
# ──────────────────────────────────────────────
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    """
    Ajoute des headers de sécurité à chaque réponse.
    Content-Security-Policy, X-Frame-Options, etc.
    """
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    return response


# ──────────────────────────────────────────────
# SCHÉMAS PYDANTIC — Validation stricte des données
# ──────────────────────────────────────────────
class QueryRequest(BaseModel):
    """Requête de l'utilisateur."""
    question: str = Field(
        ...,
        min_length=3,
        max_length=500,
        description="Question sur le tourisme burkinabè",
        examples=["Où se situent les ruines de Loropéni ?"],
    )
    top_k: int = Field(
        default=5,
        ge=1,          # Minimum 1
        le=10,         # Maximum 10 (évite les requêtes abusives)
        description="Nombre de documents à récupérer",
    )

    @field_validator("question")
    @classmethod
    def question_not_empty(cls, v: str) -> str:
        """Valide que la question n'est pas juste des espaces."""
        if not v.strip():
            raise ValueError("La question ne peut pas être vide")
        return v.strip()


class DocumentResult(BaseModel):
    """Document récupéré par le RAG."""
    title: str
    url: str
    similarity: float
    excerpt: str  # 200 premiers caractères du texte


class QueryResponse(BaseModel):
    """Réponse du système RAG."""
    response: str
    sources: list[str]
    documents: list[DocumentResult]
    response_time: float
    safe: bool


class IndexResponse(BaseModel):
    """Résultat de l'indexation."""
    indexed_count: int
    message: str


class StatsResponse(BaseModel):
    """Statistiques de la base vectorielle."""
    total_documents: int
    collection_name: str
    embedding_model: str
    llm_model: str


class HealthResponse(BaseModel):
    """Health check."""
    status: str
    pipeline_ready: bool
    timestamp: float


# ──────────────────────────────────────────────
# ENDPOINTS
# ──────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
async def health_check():
    """
    Vérifie que l'API et le pipeline sont opérationnels.
    Utilisé par les health checks de déploiement.
    """
    return HealthResponse(
        status="ok" if pipeline is not None else "loading",
        pipeline_ready=pipeline is not None,
        timestamp=time.time(),
    )


@app.post("/query", response_model=QueryResponse, tags=["RAG"])
@limiter.limit("10/minute")  # Max 10 questions par minute par IP
async def query_rag(request: Request, body: QueryRequest):
    """
    Endpoint principal : soumet une question au système RAG.

    Pipeline :
    1. Validation Pydantic de la question
    2. Sanitization + détection injection
    3. Embedding de la question (BGE-M3)
    4. Recherche vectorielle (ChromaDB)
    5. Agent LangGraph (vérification + synthèse)
    6. Retour réponse + sources
    """
    if pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Pipeline en cours de chargement. Réessayez dans quelques secondes.",
        )

    try:
        # Appel au pipeline RAG
        result = pipeline.query(body.question)

        # Formate les documents pour la réponse API
        docs_formatted = [
            DocumentResult(
                title=doc["title"],
                url=doc["url"],
                similarity=doc["similarity"],
                excerpt=doc["text"][:200] + "...",
            )
            for doc in result["retrieved_docs"]
        ]

        return QueryResponse(
            response=result["response"],
            sources=result["sources"],
            documents=docs_formatted,
            response_time=result["response_time"],
            safe=result["safe"],
        )

    except Exception as e:
        logger.error(f"Erreur query : {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur interne : {str(e)}",
        )


@app.post("/index", response_model=IndexResponse, tags=["Administration"])
async def index_corpus(request: Request):
    """
    (Re)indexe le corpus dans ChromaDB.
    Appelle cet endpoint après un nouveau scraping.
    Note : En production, protège cet endpoint avec authentification.
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline non prêt")

    try:
        count = pipeline.index()
        return IndexResponse(
            indexed_count=count,
            message=f"{count} nouveaux documents indexés avec succès.",
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Erreur indexation : {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats", response_model=StatsResponse, tags=["Monitoring"])
async def get_stats():
    """Statistiques de la base vectorielle."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline non prêt")

    count = pipeline.vector_store.collection.count()
    return StatsResponse(
        total_documents=count,
        collection_name=pipeline.vector_store.collection.name,
        embedding_model="BAAI/bge-m3",
        llm_model="qwen2.5:7b-instruct",
    )
