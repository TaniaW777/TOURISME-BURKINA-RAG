# -*- coding: utf-8 -*-
"""Tests Phase 2 — ingestion/normalize.py. Aucun réseau requis."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ingestion.normalize import (
    detect_language,
    doc_id_for,
    fix_encoding,
    normalize_text,
    sanitize_text,
)


class TestFixEncoding:
    def test_mojibake_sequence_repaired(self):
        assert fix_encoding("caractÃ©ristiques") == "caractéristiques"

    def test_clean_text_unchanged(self):
        clean = "Les cascades de Banfora"
        assert fix_encoding(clean) == clean

    def test_apostrophe_and_quotes(self):
        text = "l\u00e2\u20ac\u2122histoire"  # l'histoire en mojibake
        assert fix_encoding(text) == "l’histoire"

    def test_empty(self):
        assert fix_encoding("") == ""


class TestSanitizeText:
    def test_html_stripped(self):
        assert "<script>alert(1)</script>" not in sanitize_text("<p>Texte</p><script>alert(1)</script>")

    def test_control_chars_removed(self):
        assert "\x00" not in sanitize_text("a\x00b\x1fc")

    def test_sentence_punctuation_preserved(self):
        out = sanitize_text("Première phrase. Deuxième phrase ! Troisième ?")
        assert out.count(".") == 1 and out.count("!") == 1 and out.count("?") == 1


class TestNormalizeText:
    def test_full_pipeline(self):
        raw = "Site touristique du Burkina.\n\n   Les ruines de LoropÃ©ni."
        out = normalize_text(raw)
        assert "Loropéni" in out
        assert "\n\n" not in out


class TestDetectLanguage:
    FR = ("Le Burkina Faso est un pays d'Afrique de l'Ouest. La capitale est "
          "Ouagadougou et les habitants sont très accueillants pour les visiteurs.")
    EN = ("Burkina Faso is a country in West Africa. The capital city is "
          "Ouagadougou and the people are very welcoming to foreign visitors.")

    def test_french_detected(self):
        assert detect_language(self.FR) == "fr"

    def test_english_detected(self):
        assert detect_language(self.EN) == "en"

    def test_gibberish_is_unknown(self):
        assert detect_language("xyzzy qwerty zzz abcdef ghijkl mno pqrstu vwx") == "unknown"

    def test_short_text_is_unknown(self):
        assert detect_language("trop court") == "unknown"


class TestDocId:
    def test_deterministic(self):
        assert doc_id_for("https://example.fr") == doc_id_for("https://example.fr")

    def test_distinct_urls_distinct_ids(self):
        assert doc_id_for("https://a.fr") != doc_id_for("https://b.fr")

    def test_length_16(self):
        assert len(doc_id_for("x")) == 16
