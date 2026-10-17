# -*- coding: utf-8 -*-
"""Pipeline d'ingestion — Tourisme Burkina RAG (Phase 2).

Modules :
  fetch     — récupération HTTP/PDF respectueuse (robots, SSL, limites)
  normalize — nettoyage, correction d'encodage, détection de langue
  chunker   — découpage conscient des phrases, limité par le tokenizer réel
"""
