# -*- coding: utf-8 -*-
"""Couche LLM provider-neutre — Tourisme Burkina RAG (Phase 4, PR 1).

Modules :
  base  — contrat LLMService + exceptions typées (aucune dépendance provider)
  mock  — implémentation déterministe pour les tests (aucun réseau)
"""

from src.llm.base import (  # noqa: F401
    LLMAuthError,
    LLMConfigurationError,
    LLMError,
    LLMProviderError,
    LLMRateLimitError,
    LLMService,
    LLMTimeoutError,
)
from src.llm.mock import MockLLMService  # noqa: F401
