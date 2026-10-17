# -*- coding: utf-8 -*-
"""
src/llm/groq_service.py
=======================
Adaptateur Groq du contrat LLMService (Phase 4, PR 3).

Choix de conception, vérifiés sur le SDK groq 1.7.0 :
  - `max_retries=0` côté SDK : la politique de retry est LA NÔTRE, explicite
    et bornée — le SDK ne rejoue jamais rien à notre insu, et les 429 ne
    sont JAMAIS rejoués à l'identique : on lit Retry-After, on plafonne
    l'attente, on dort (fonction de sleep injectable → les tests ne dorment
    pas), et chaque tentative consomme le budget `groq_max_retries`.
  - mapping typé complet : aucune exception SDK ne franchit la frontière.

La clé API vient EXCLUSIVEMENT de la configuration environnementale
(src.config.Settings). Elle n'est jamais loguée ni incluse dans une réponse.
"""

from __future__ import annotations

import time as _time
from typing import Callable, Optional

from src.config import Settings
from src.llm.base import (
    LLMAuthError,
    LLMConfigurationError,
    LLMProviderError,
    LLMRateLimitError,
    LLMTimeoutError,
    validate_messages,
)

# plafond d'attente Retry-After (s) : au-delà, on rend la main à l'appelant
MAX_RETRY_AFTER_S = 30.0
# erreurs transitoires dignes d'un retry (hors 429, géré à part)
_TRANSIENT_FACTORY = None  # résolu paresseusement (import SDK)


class GroqService:
    """LLMService concret via l'API Groq (clé obligatoire côté serveur).

    Paramètres :
      settings : Settings existants (groq_api_key, groq_model,
                 groq_timeout_s, groq_max_retries).
      client   : client Groq INJECTABLE pour les tests (aucun réseau en test).
      sleep_fn : fonction de sommeil injectable (les tests ne dorment pas).

    Sans clé ET sans client injecté → LLMConfigurationError dès la
    construction (mode dégradé détectable au démarrage).
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        client=None,
        sleep_fn: Callable[[float], None] = _time.sleep,
    ):
        self.settings = settings or _default_settings()
        self._sleep = sleep_fn
        self._injected_client = client
        self._client = None  # créé paresseusement
        if client is None and not self.settings.groq_configured:
            raise LLMConfigurationError(
                "GROQ_API_KEY absente : service Groq non configuré "
                "(mode dégradé — définir la variable d'environnement)"
            )

    @property
    def name(self) -> str:
        return f"groq:{self.settings.groq_model}"

    # ── client ──
    def _get_client(self):
        if self._injected_client is not None:
            return self._injected_client
        if self._client is None:
            import groq  # import paresseux : le module reste importable sans SDK

            # max_retries=0 : TOUTE la politique de retry est locale (voir module)
            self._client = groq.Groq(
                api_key=self.settings.groq_api_key,
                timeout=self.settings.groq_timeout_s,
                max_retries=0,
            )
        return self._client

    # ── disponibilité du modèle (health checks) ──
    def is_available(self) -> bool:
        """True si le provider est joignable ET le modèle configuré présent.

        ATTENTION : effectue une VRAIE requête provider (models.list) —
        consomme du quota, peut elle-même être rate-limitée. Ne lève jamais
        (retourne False) et n'expose ni identifiants ni exception brute ;
        l'appelant doit mettre le résultat en cache (health checks).
        """
        try:
            models = self._get_client().models.list()
            model_ids = {m.id for m in models.data}
            return self.settings.groq_model in model_ids
        except Exception:
            return False

    # ── génération ──
    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        validate_messages(messages)
        client = self._get_client()

        attempts_left = 1 + max(0, self.settings.groq_max_retries)
        last_error: Optional[Exception] = None

        while attempts_left > 0:
            attempts_left -= 1
            try:
                kwargs = {
                    "model": self.settings.groq_model,
                    "messages": messages,
                    "timeout": self.settings.groq_timeout_s,
                }
                if max_tokens is not None:
                    kwargs["max_tokens"] = max_tokens
                if temperature is not None:
                    kwargs["temperature"] = temperature
                response = client.chat.completions.create(**kwargs)
                content = response.choices[0].message.content
                if not isinstance(content, str) or not content.strip():
                    raise LLMProviderError("réponse Groq vide ou ininterprétable")
                return content.strip()

            except Exception as e:  # mapping typé ci-dessous
                mapped = self._map_error(e)
                last_error = mapped

                # 429 : on NE rejouE PAS à l'identique — Retry-After plafonné,
                # cette attente consomme une tentative du budget.
                if isinstance(mapped, LLMRateLimitError) and attempts_left > 0:
                    wait = mapped.retry_after_s if mapped.retry_after_s is not None else 1.0
                    self._sleep(min(wait, MAX_RETRY_AFTER_S))
                    continue

                # transitoire (timeout/réseau/5xx) : retry borné
                if isinstance(mapped, (LLMTimeoutError, LLMProviderError)) and attempts_left > 0:
                    self._sleep(0.5)  # backoff fixe minimal, borné par le budget
                    continue

                raise mapped from e

        raise last_error or LLMProviderError("échec de génération inexpliqué")

    # ── mapping SDK → exceptions typées du contrat ──
    def _map_error(self, e: Exception):
        import groq

        if isinstance(e, groq.AuthenticationError):
            return LLMAuthError(f"clé Groq rejetée : {e}")
        if isinstance(e, groq.RateLimitError):
            retry_after = _read_retry_after(e)
            return LLMRateLimitError(
                f"quota Groq dépassé (retry-after={retry_after}s)", retry_after_s=retry_after
            )
        if isinstance(e, groq.APITimeoutError):
            return LLMTimeoutError(f"délai Groq dépassé : {e}")
        if isinstance(e, groq.APIConnectionError):
            return LLMTimeoutError(f"Groq injoignable : {e}")
        if isinstance(e, groq.APIStatusError):
            return LLMProviderError(f"erreur provider Groq {e.status_code} : {e}")
        if isinstance(e, (LLMAuthError, LLMRateLimitError, LLMTimeoutError,
                          LLMProviderError, LLMConfigurationError)):
            return e  # déjà typée (ex. réponse vide)
        return LLMProviderError(f"erreur inattendue : {e.__class__.__name__}: {e}")


def _read_retry_after(rate_limit_error: Exception) -> Optional[float]:
    """Lit Retry-After depuis la réponse HTTP de l'erreur 429, plafonné."""
    try:
        headers = rate_limit_error.response.headers  # type: ignore[attr-defined]
        raw = headers.get("retry-after")
        return min(float(raw), MAX_RETRY_AFTER_S) if raw is not None else None
    except (AttributeError, TypeError, ValueError):
        return None


def _default_settings() -> Settings:
    from src.config import load_settings

    return load_settings()
