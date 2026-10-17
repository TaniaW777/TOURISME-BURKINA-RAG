# -*- coding: utf-8 -*-
"""
src/llm/mock.py
===============
Implémentation DÉTERMINISTE du contrat LLMService — tests et CI.

- Aucun réseau, aucun modèle, aucune dépendance provider.
- Même entrée → même sortie, à des fins d'assertions stables.
- Configurable : réponse fixe, séquence de réponses, ou erreur à simuler
  (pour tester les contrats d'erreur des appelants).

N'est JAMAIS instanciée par le code de production : réserved aux tests
et au mode démonstration hors-ligne explicite.
"""

from __future__ import annotations

from typing import Optional

from src.llm.base import (
    LLMError,
    LLMConfigurationError,
    validate_messages,
)

DEFAULT_MOCK_RESPONSE = "[MockLLM] Réponse déterministe pour les tests."


class MockLLMService:
    """LLMService déterministe.

    Paramètres :
      response   : réponse unique constante (défaut : chaîne standard).
      responses  : séquence consommée dans l'ordre ; la dernière valeur est
                   répétée ensuite. Prioritaire sur `response`.
      fail_with  : exception LLMError (ou sous-classe) à lever à chaque appel
                   — pour tester le mapping d'erreurs de l'appelant.

    Comportement déterministe : la réponse ne dépend QUE de la configuration
    et de l'ordre des appels — jamais du temps, du réseau ou d'un modèle.
    """

    def __init__(
        self,
        response: str = DEFAULT_MOCK_RESPONSE,
        responses: Optional[list[str]] = None,
        fail_with: Optional[LLMError] = None,
    ):
        if responses is not None and not responses:
            raise LLMConfigurationError("responses ne peut pas être une liste vide")
        self._responses = list(responses) if responses is not None else [response]
        self._fail_with = fail_with
        self.calls: list[list[dict[str, str]]] = []  # historique pour assertions

    @property
    def name(self) -> str:
        return "mock"

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        validate_messages(messages)
        if self._fail_with is not None:
            raise self._fail_with
        self.calls.append([dict(m) for m in messages])
        # Déterminisme : pop du devant, dernière réponse répétée
        if len(self._responses) > 1:
            return self._responses.pop(0)
        return self._responses[0]
