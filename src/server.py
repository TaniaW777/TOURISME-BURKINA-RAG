# -*- coding: utf-8 -*-
"""
src/server.py
=============
Serveur FastAPI persistant pour le pipeline RAG Tourisme Burkina.
Le pipeline est chargé UNE SEULE FOIS au démarrage.

Lancement : uvicorn src.server:app --port 8000
"""

from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager
from src.rag_pipeline import TourismRAGPipeline, Config, PromptBuilder
import ollama
import time
import json

_pipeline: TourismRAGPipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pipeline
    print("Chargement du pipeline RAG...")
    _pipeline = TourismRAGPipeline()
    _pipeline.index()
    print("Pipeline prêt — serveur opérationnel.")
    yield
    print("Arrêt du serveur.")


app = FastAPI(
    title       = "Tourisme Burkina RAG API",
    description = "Pipeline RAG bilingue FR/EN — 100% local",
    version     = "2.0.0",
    lifespan    = lifespan,
)


@app.get("/")
def root():
    return {"status": "ok", "message": "Tourisme Burkina RAG API v2"}


@app.get("/health")
def health():
    return {"status": "healthy", "pipeline": _pipeline is not None}


@app.get("/query")
def query(q: str = Query(..., description="Question en français ou anglais")):
    """Endpoint standard — attend la réponse complète."""
    if _pipeline is None:
        return {"error": "Pipeline non initialisé"}

    result = _pipeline.query(q)
    return {
        "question"      : q,
        "response"      : result["response"],
        "sources"       : result["sources"],
        "expanded_query": result["expanded_query"],
        "response_time" : result["response_time"],
        "safe"          : result["safe"],
    }


@app.get("/stream")
def stream(q: str = Query(..., description="Question en français ou anglais")):
    """
    Endpoint streaming — affiche les tokens au fur et à mesure.
    L'utilisateur voit la réponse apparaître en temps réel
    sans attendre les 42s de génération complète.

    Compatible avec :
      - curl : curl "http://localhost:8000/stream?q=Que+visiter+a+Ouagadougou"
      - JavaScript EventSource / fetch avec ReadableStream
      - Python requests avec stream=True
    """
    if _pipeline is None:
        def error_gen():
            yield "data: Pipeline non initialisé\n\n"
        return StreamingResponse(error_gen(), media_type="text/event-stream")

    def generate():
        # Sanitization + expansion (rapide — <1s)
        clean_query, is_safe = _pipeline.sanitizer.sanitize(q)
        if not is_safe:
            yield f"data: {json.dumps({'token': 'Requête non autorisée.'})}\n\n"
            return

        expanded = _pipeline.query_expander.expand(clean_query)
        embedding = _pipeline.embedder.embed_query(expanded)
        docs      = _pipeline.vector_store.search(embedding)

        if not docs:
            msg = _pipeline.prompt.no_result_response()
            yield f"data: {json.dumps({'token': msg})}\n\n"
            yield "data: [DONE]\n\n"
            return

        prompt_text = _pipeline.prompt.build_rag_prompt(clean_query, docs)
        sources     = [d.get("url", "") for d in docs]

        # Streaming Ollama — chaque token est envoyé immédiatement au client
        start = time.time()
        try:
            stream = ollama.chat(
                model   = Config.LLM_MODEL,
                messages= [{"role": "user", "content": prompt_text}],
                options = {
                    "temperature" : Config.LLM_TEMPERATURE,
                    "num_predict" : Config.LLM_MAX_TOKENS,
                    "num_ctx"     : int(Config.LLM_NUM_CTX),
                    "stop"        : ["QUESTION", "SOURCES", "\n\n\n"],
                },
                stream=True,   # <-- clé du streaming
            )
            for chunk in stream:
                token = chunk["message"]["content"]
                if token:
                    yield f"data: {json.dumps({'token': token})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'token': f'Erreur LLM: {str(e)}'})}\n\n"

        # Métadonnées finales
        response_time = round(time.time() - start, 2)
        yield f"data: {json.dumps({'done': True, 'sources': sources, 'response_time': response_time})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")