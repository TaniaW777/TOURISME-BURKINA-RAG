# -*- coding: utf-8 -*-
"""
scripts/ingest.py
=================
Orchestrateur d'ingestion Phase 2.

  data/sources.yaml ─► fetch (robots, SSL, limites)
        ─► normalize (encodage, langue) ─► documents.jsonl
        ─► chunker (tokenizer réel, ≤128 tokens) ─► chunks.jsonl
        ─► data/ingest_report.json (succès/échecs par source — auditable)

DESTINATIONS SÉPARÉES : data/corpus.json, corpus_v1.backup.json et
data/chroma_db/ ne sont NI lus pour écriture, NI modifiés.

Usage :
    python scripts/ingest.py [--max-sources N] [--offline-fixture F]

Le rapport d'ingestion rend chaque échec explicite : une source ratée
n'est jamais traitée comme du contenu valide vide.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import requests
import yaml
from loguru import logger

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ingestion.chunker import (  # noqa: E402
    Chunk,
    TransformersTokenCounter,
    chunk_document,
    dedup_exact,
    is_boilerplate_fragment,
)
from ingestion.fetch import (  # noqa: E402
    FetchResult,
    RobotsCache,
    fetch_local_pdf,
    fetch_pdf,
    fetch_web,
)
from ingestion.normalize import (  # noqa: E402
    detect_language,
    doc_id_for,
    normalize_text,
)

DOCUMENTS_OUT = REPO_ROOT / "data" / "documents.jsonl"
CHUNKS_OUT = REPO_ROOT / "data" / "chunks.jsonl"
REPORT_OUT = REPO_ROOT / "data" / "ingest_report.json"

# Fragments plus courts que ce plancher ne sont PAS publiés tels quels.
# Ils ne sont abandonnés QUE s'ils matchent la classification de boilerplate
# (ingestion.chunker.is_boilerplate_fragment) — chaque drop est audité
# individuellement dans le rapport. Un fragment meaningful trop court pour
# être publié et préservé fait ÉCHOUER l'ingestion : jamais de drop aveugle.
MIN_PUBLISHED_CHUNK_CHARS = 25


def filter_fragments(
    chunks: list[Chunk],
    doc: dict,
) -> tuple[list[Chunk], list[dict]]:
    """Sépare les chunks publiables des fragments abandonnés (audités).

    Retourne (publiés, enregistrements_d'audit).
    Lève ValueError si un fragment court meaningful ne peut être classé
    boilerplate — c'est un bug de chunking à corriger, pas un drop.
    """
    kept: list[Chunk] = []
    dropped: list[dict] = []
    for c in chunks:
        if len(c.text) >= MIN_PUBLISHED_CHUNK_CHARS:
            kept.append(c)
            continue
        if not is_boilerplate_fragment(c.text):
            raise ValueError(
                f"fragment court meaningful non préservé (bug chunker) : "
                f"{c.text!r} — document {doc['url']} (chunk {c.chunk_id})"
            )
        dropped.append({
            "fragment": c.text,
            "chunk_id": c.chunk_id,
            "document_id": c.parent_id,
            "url": c.url,
            "reason": "fragment court sans valeur informationnelle",
            "rule": "is_boilerplate_fragment (motif de citation/initiale/ponctuation)",
        })
    return kept, dropped


def load_sources(path: Path = REPO_ROOT / "data" / "sources.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_document(result: FetchResult, source_type: str, trust: str) -> dict | None:
    """Normalise un FetchResult réussi en document canonique. Échec → None."""
    if not result.ok:
        return None
    text = normalize_text(result.raw_text)
    if len(text) < 150:
        return None  # explicitement trop court — sera noté "too_short" par l'appelant
    url = result.canonical_url or result.url
    return {
        "id": doc_id_for(url),
        "url": url,
        "requested_url": result.url,        # provenance complète
        "title": result.title[:200] or url,
        "text": text,
        "source_type": source_type,
        "language": detect_language(text),
        "trust": trust,
        "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def run(max_sources: int | None = None, documents_path: Path | None = None) -> dict:
    """Avec documents_path : repart de documents.jsonl déjà normalisés —
    AUCUN réseau, seulement chunking + dedup + écriture (reconstruction
    déterministe après correction du chunker)."""
    if documents_path is not None:
        documents = [json.loads(l) for l in
                     documents_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        logger.info(f"Mode hors-ligne : {len(documents)} documents depuis {documents_path}")
        report_entries = [{"url": d["url"], "status": "ok", "language": d["language"],
                           "trust": d.get("trust", ""), "offline_rechunk": True}
                          for d in documents]
        return _chunk_and_write(documents, report_entries, sources_requested=len(documents))

    config = load_sources()
    sources = config["sources"]
    if max_sources:
        sources = sources[:max_sources]

    session = requests.Session()
    robots = RobotsCache()

    documents: list[dict] = []
    report_entries: list[dict] = []
    canonical_seen: set[str] = set()

    # ── Sources distantes ──
    for src in sources:
        url, stype = src["url"], src.get("type", "web")
        trust = src.get("trust", "blog")
        logger.info(f"→ {url[:70]}")

        result = fetch_pdf(url, session, robots) if stype == "pdf" else fetch_web(url, session, robots)

        if not result.ok:
            report_entries.append({"url": url, "status": result.status, "error": result.error[:300]})
            logger.warning(f"  ✗ {result.status} : {result.error[:80]}")
            continue

        canonical = result.canonical_url or result.url
        if canonical in canonical_seen:
            report_entries.append({"url": url, "status": "duplicate_canonical", "error": canonical})
            continue
        canonical_seen.add(canonical)

        doc = build_document(result, stype, trust)
        if doc is None:
            report_entries.append({"url": url, "status": "too_short",
                                   "error": "contenu normalisé < 150 caractères"})
            continue

        documents.append(doc)
        report_entries.append({"url": url, "status": "ok", "language": doc["language"],
                               "chunks_pending": True, "trust": trust})

    # ── PDFs locaux ──
    local_cfg = config.get("local_pdfs") or {}
    pdf_dir = REPO_ROOT / local_cfg.get("dir", "data/pdfs")
    excludes = set(local_cfg.get("exclude_subdirs", []))
    if pdf_dir.exists():
        for pdf_path in sorted(pdf_dir.glob("*.pdf")):
            result = fetch_local_pdf(pdf_path)
            if not result.ok:
                report_entries.append({"url": str(pdf_path), "status": result.status, "error": result.error[:300]})
                continue
            doc = build_document(result, "pdf_local", local_cfg.get("trust", "institutional"))
            if doc is None:
                report_entries.append({"url": str(pdf_path), "status": "too_short", "error": ""})
                continue
            documents.append(doc)
            report_entries.append({"url": str(pdf_path), "status": "ok",
                                   "language": doc["language"], "trust": doc["trust"]})

    return _chunk_and_write(documents, report_entries, sources_requested=len(sources))


def _chunk_and_write(documents: list[dict], report_entries: list[dict],
                     sources_requested: int) -> dict:
    # ── Chunking avec le tokenizer réel ──
    logger.info("Chargement du tokenizer MiniLM (fichiers cache HF)...")
    counter = TransformersTokenCounter()

    all_chunks: list[Chunk] = []
    drop_audit: list[dict] = []
    for doc in documents:
        doc_chunks = chunk_document(doc, counter)
        doc_chunks, dropped = filter_fragments(doc_chunks, doc)
        drop_audit.extend(dropped)
        all_chunks.extend(doc_chunks)

    chunks, dup_removed = dedup_exact(all_chunks)
    total_dup = dup_removed

    # ── Écriture (destinations Phase 2 uniquement) ──
    DOCUMENTS_OUT.write_text(
        "\n".join(json.dumps(d, ensure_ascii=False) for d in documents), encoding="utf-8"
    )
    CHUNKS_OUT.write_text(
        "\n".join(json.dumps(c.to_record(), ensure_ascii=False) for c in chunks), encoding="utf-8"
    )

    # ── Statistiques et rapport ──
    lang_counts = Counter(d["language"] for d in documents)
    status_counts = Counter(e["status"] for e in report_entries)
    token_counts = [c.token_count for c in chunks]
    char_counts = [len(c.text) for c in chunks]

    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sources_requested": sources_requested,
        "documents": len(documents),
        "chunks": len(chunks),
        "exact_duplicates_removed": total_dup,
        "unmergeable_fragments_dropped": len(drop_audit),
        "dropped_fragments": drop_audit,
        "languages": dict(lang_counts),
        "source_status": dict(status_counts),
        "token_stats": {
            "max": max(token_counts) if token_counts else 0,
            "avg": round(sum(token_counts) / len(token_counts), 1) if token_counts else 0,
            "limit": 126,
        },
        "char_stats": {
            "min": min(char_counts) if char_counts else 0,
            "avg": round(sum(char_counts) / len(char_counts)) if char_counts else 0,
            "max": max(char_counts) if char_counts else 0,
        },
        "entries": report_entries,
    }
    REPORT_OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingestion Phase 2 — Tourisme Burkina RAG")
    parser.add_argument("--max-sources", type=int, default=None,
                        help="limite le nombre de sources distantes (tests bornés)")
    parser.add_argument("--from-documents", type=Path, default=None,
                        help="rechunker documents.jsonl existant — aucun réseau")
    args = parser.parse_args()

    final_report = run(max_sources=args.max_sources, documents_path=args.from_documents)

    print(f"\n{'═' * 60}")
    print(f"Documents     : {final_report['documents']}")
    print(f"Chunks        : {final_report['chunks']} (doublons exacts supprimés : "
          f"{final_report['exact_duplicates_removed']}, fragments inmergeables abandonnés : "
          f"{final_report['unmergeable_fragments_dropped']}")
    print(f"Langues       : {final_report['languages']}")
    print(f"Statuts       : {final_report['source_status']}")
    print(f"Tokens/chunk  : max {final_report['token_stats']['max']} / "
          f"moy {final_report['token_stats']['avg']} (limite {final_report['token_stats']['limit']})")
    print(f"Rapport       : {REPORT_OUT}")
