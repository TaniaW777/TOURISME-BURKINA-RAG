# -*- coding: utf-8 -*-
"""
src/llm/base.py
===============
Contrat provider-neutre pour la génération LLM (Phase 4, PR 1).

Volontairement INDÉPENDANT de Groq comme d'Ollama : ce module ne connaît
ni n'importe aucun SDK provider. Les implémentations concrètes
(GroqService en PR 3 ; le pipeline Ollama legacy reste en parallèle)
dépendent de ce contrat, jamais l'inverse.

Format des messages : liste de dicts {"role": ..., "content": ...}
(rôles "system" / "user" / "assistant") — format commun aux API
OpenAI-compatibles, déjà utilisé par le pipeline existant.

Sévérité des erreurs — hiérarchie typée pour que l'appelant puisse
réagir finement SANS attraper des exceptions SDK :

  LLMError                      (base, ne jamais raiser directement)
  ├── LLMConfigurationError     clé/paramétrage absent → 503 côté API
  ├── LLMAuthError              clé invalide/révoquée → 503
  ├── LLMRateLimitError         429, porte retry_after_s → 429 côté API
  ├── LLMTimeoutError           délai dépassé → 502
  └── LLMProviderError          panne/5xx/réponse illisible → 502
"""

from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable


@runtime_checkable
class LLMService(Protocol):
    """Contrat minimal : générer du texte à partir de messages.

    Implémentations : MockLLMService (tests), GroqService (PR 3).
    Le pipeline doit dépendre de CE contrat uniquement.
    """

    @property
    def name(self) -> str:
        """Nom lisible de l'implémentation (logs, /health, métadonnées)."""
        ...

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """Génère la réponse du modèle pour la liste de messages.

        Paramètres :
          messages    : [{"role": "system"|"user"|"assistant",
                         "content": str}, ...] — non vide, rôles valides.
          max_tokens  : borne de sortie (None = défaut de l'implémentation).
          temperature : None = défaut ; les implémentations peuvent bornes
                        leurs propres plages valides.

        Retour : le texte généré (str, non vide en cas de succès).

        Erreurs : lève l'une des exceptions typées ci-dessous — jamais une
        exception SDK brute — pour tout échec (configuration, auth,
        rate limit, timeout, panne provider, entrée invalide).
        """
        ...


def validate_messages(messages: list[dict[str, str]]) -> None:
    """Valide le format des messages — partagé par toutes les implémentations.

    Lève LLMConfigurationError (entrée appelant invalide) — contrat commun :
    une entrée non conforme est détectée AVANT tout appel provider.
    """
    if not messages:
        raise LLMConfigurationError("messages vide : au moins un message requis")
    for i, m in enumerate(messages):
        if not isinstance(m, dict):
            raise LLMConfigurationError(f"message[{i}] : dict attendu, {type(m).__name__} reçu")
        role = m.get("role")
        content = m.get("content")
        if role not in ("system", "user", "assistant"):
            raise LLMConfigurationError(
                f"message[{i}] : role invalide {role!r} (system|user|assistant)"
            )
        if not isinstance(content, str) or not content.strip():
            raise LLMConfigurationError(f"message[{i}] : content doit être une chaîne non vide")


# ──────────────────────────────────────────────
# EXCEPTIONS TYPÉES
# ──────────────────────────────────────────────
class LLMError(Exception):
    """Erreur de base de la couche LLM — jamais raisée directement."""


class LLMConfigurationError(LLMError):
    """Configuration absente ou entrée appelant invalide (clé manquante,
    messages mal formés...). Action appelant : 503 / message d'aide."""


class LLMAuthError(LLMError):
    """Authentification rejetée par le provider (clé invalide/révoquée)."""


class LLMRateLimitError(LLMError):
    """Quota/rythme dépassé (429). NE JAMAIS être rejouée à l'identique :
    respecter retry_after_s côté appelant."""

    def __init__(self, message: str, retry_after_s: Optional[float] = None):
        super().__init__(message)
        self.retry_after_s = retry_after_s


class LLMTimeoutError(LLMError):
    """Délai d'attente dépassé avant la réponse complète."""


class LLMProviderError(LLMError):
    """Panne ou réponse ininterprétable du provider (5xx, JSON cassé...)."""
