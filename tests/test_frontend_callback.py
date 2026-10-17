# -*- coding: utf-8 -*-
"""Tests Phase 5 — callback Gradio `answer_question`.

Régression du défaut livré par #11 : le callback utilisait le global `pipeline`
(None à la première question → AttributeError) au lieu de `_get_pipeline()`,
et n'injectait pas le top_k du slider. Léger et hors-ligne : pipeline factice
+ monkeypatch ; aucune interaction Gradio, aucun modèle, aucun réseau.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# importer le module SANS déclencher demo.launch (gradio se construit,
# rien ne tourne) — identique à un import de bibliothèque.
import frontend.app as ui  # noqa: E402

GOOD_RESULT = {
    "response": "• [Source 1] Réponse factice citée.",
    "sources": ["https://exemple.bf/1"],
    "retrieved_docs": [
        {"text": "Passage sur le tourisme burkinabè." * 3,
         "url": "https://exemple.bf/1", "title": "Titre 1",
         "source_type": "web", "language": "fr", "trust": "encyclopedia",
         "similarity": 0.82},
    ],
    "response_time": 1.25,
    "safe": True,
    "citations": [1],
    "abstained": False,
    "model": "fake",
}


class FakePipeline:
    def __init__(self, result=None, error=None):
        self.result = result or GOOD_RESULT
        self.error = error
        self.calls: list[tuple[str, dict]] = []

    def query(self, question, top_k=None):
        self.calls.append({"question": question, "top_k": top_k})
        if self.error:
            raise self.error
        return self.result


@pytest.fixture(autouse=True)
def reset_module_state(monkeypatch):
    """Chaque test part d'un pipeline module None (état première question)."""
    monkeypatch.setattr(ui, "pipeline", None)
    monkeypatch.setattr(ui, "_pipeline_error", None)
    yield


class TestSuccessfulFirstQuestion:
    def test_uses_lazy_pipeline_when_global_is_none(self, monkeypatch):
        fake = FakePipeline()
        monkeypatch.setattr(ui, "_get_pipeline", lambda: fake)
        out = ui.answer_question("Que visiter à Banfora ?")
        # 4 sorties (contrat Gradio : réponse, sources, documents, métriques)
        assert len(out) == 4
        assert out[0] == GOOD_RESULT["response"]
        assert "https://exemple.bf/1" in out[1]
        assert "Titre 1" in out[2] and "82" in out[2]  # docs formatés
        assert "1.25s" in out[3]
        assert fake.calls  # le pipeline factice a bien été interrogé

    def test_does_not_touch_the_raw_global(self, monkeypatch):
        """Régression #11 : si le callback utilisait le global nu (None),
        une AttributeError serait convertie en message générique — on
        prouve que c'est bien _get_pipeline() qui est consulté."""
        def boom():
            raise AssertionError("_get_pipeline() doit être appelé")

        monkeypatch.setattr(ui, "_get_pipeline", boom)
        with pytest.raises(AssertionError):
            ui.answer_question("question valide ?")


class TestTopKPropagation:
    def test_slider_value_reaches_the_pipeline(self, monkeypatch):
        fake = FakePipeline()
        monkeypatch.setattr(ui, "_get_pipeline", lambda: fake)
        ui.answer_question("question", top_k=7)
        assert fake.calls[0]["top_k"] == 7

    def test_default_top_k_is_five(self, monkeypatch):
        fake = FakePipeline()
        monkeypatch.setattr(ui, "_get_pipeline", lambda: fake)
        ui.answer_question("question")
        assert fake.calls[0]["top_k"] == 5


class TestInitializationFailure:
    def test_safe_message_not_attribute_error(self, monkeypatch):
        monkeypatch.setattr(ui, "_get_pipeline", lambda: None)
        monkeypatch.setattr(
            ui, "_pipeline_error",
            "PipelineError: GROQ_API_KEY absente : pipeline v2 non configurable")
        out = ui.answer_question("question valide ?")
        assert len(out) == 4
        assert out[0].startswith("⚙️ Assistant non configuré")
        assert "GROQ_API_KEY" in out[0]
        assert out[1] == out[2] == out[3] == ""

    def test_degraded_message_displayed_verbatim(self, monkeypatch):
        """Le callback affiche le message d'initialisation typé TEL QUEL —
        il n'ajoute rien par lui-même. La garantie « aucune clé dans les
        erreurs » vit en amont (erreurs typées, testées dans
        test_groq_service.py::test_key_never_in_exceptions_or_name)."""
        monkeypatch.setattr(ui, "_get_pipeline", lambda: None)
        msg = "PipelineError: GROQ_API_KEY absente (mode dégradé)"
        monkeypatch.setattr(ui, "_pipeline_error", msg)
        out = ui.answer_question("question")
        assert msg in out[0]


class TestCallbackContract:
    def test_empty_question_returns_four_values(self):
        out = ui.answer_question("   ")
        assert out[0].startswith("⚠️") and out[1:] == ("", "", "")

    def test_unsafe_question_returns_four_values(self, monkeypatch):
        fake = FakePipeline(result={**GOOD_RESULT, "safe": False})
        monkeypatch.setattr(ui, "_get_pipeline", lambda: fake)
        out = ui.answer_question("question")
        assert out[0].startswith("🚫") and out[1:] == ("", "", "")

    def test_query_failure_handled_with_generic_message(self, monkeypatch):
        fake = FakePipeline(error=RuntimeError("panne simulée du pipeline"))
        monkeypatch.setattr(ui, "_get_pipeline", lambda: fake)
        out = ui.answer_question("question valide ?")
        assert len(out) == 4
        assert out[0].startswith("❌ Erreur")
        assert out[1] == out[2] == out[3] == ""
