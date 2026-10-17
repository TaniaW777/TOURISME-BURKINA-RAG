# -*- coding: utf-8 -*-
"""Tests Phase 2 — ingestion/chunker.py.

Le tokenizer réel n'est PAS requis : un FakeTokenCounter déterministe
(≈ mots + ponctuation) teste la LOGIQUE ; la conformité à la limite du
vrai modèle est vérifiée dans TestRealTokenizerLimits, sauté sans cache
HF (marqué skip, pas d'échec) — et exécutée réellement dans le run CI
après premier téléchargement des fichiers tokenizer (~1 Mo).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ingestion.chunker import (  # noqa: E402
    MAX_TOKENS,
    OVERLAP_TOKENS,
    TARGET_TOKENS,
    Chunk,
    TransformersTokenCounter,
    chunk_document,
    chunk_text,
    dedup_exact,
    split_sentences,
)


class FakeTokenCounter:
    """Compteur déterministe : 1 token par mot/signe de ponctuation.
    Suffisant pour borner la logique de packing (le vrai tokenizer
    produit plus de tokens par texte → le vrai test de limite utilise
    des marges via TARGET < MAX)."""

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return list(range(max(1, len(text.split()))))


# Phrases courtes en français (8-12 mots chacune)
S1 = "Les ruines de Loropéni se trouvent dans le sud-ouest du Burkina Faso."
S2 = "Le site est classé au patrimoine mondial de l'UNESCO depuis 2009."
S3 = "Les remparts de pierre sèche atteignent six mètres de hauteur."
S4 = "La période de construction reste débattue par les chercheurs."
S5 = "Le site est accessible depuis la ville de Gaoua en voiture."
S6 = "Des guides locaux proposent des visites commentées du lieu."
TEXT = " ".join([S1, S2, S3, S4, S5, S6])


class TestSplitSentences:
    def test_split_count(self):
        assert len(split_sentences(TEXT)) == 6

    def test_no_mid_sentence_cut(self):
        for s in split_sentences(TEXT):
            assert s.endswith((".", "!", "?"))


class TestChunkTextLogic:
    def test_all_chunks_within_token_limit(self):
        chunks = chunk_text(TEXT, FakeTokenCounter(), target_tokens=12, max_tokens=15)
        for c in chunks:
            assert c["token_count"] <= 15
            assert len(c["text"].split()) <= 15

    def test_chunks_cover_all_sentences(self):
        chunks = chunk_text(TEXT, FakeTokenCounter(), target_tokens=12, max_tokens=15)
        # chaque phrase apparaît dans au moins un chunk
        rejoined = " ".join(c["text"] for c in chunks)
        for s in [S1, S2, S3, S4, S5, S6]:
            assert s in rejoined

    def test_overlap_exists_between_consecutive_chunks(self):
        # target assez grand pour packer plusieurs phrases par chunk,
        # sinon des chunks d'une seule phrase ne peuvent pas se recouvrir
        chunks = chunk_text(TEXT, FakeTokenCounter(), target_tokens=24, max_tokens=30,
                            overlap_tokens=12)
        assert len(chunks) >= 2
        # le recouvrement reprend les DERNIÈRES phrases du chunk précédent :
        # la dernière phrase du chunk 0 doit figurer dans le chunk 1
        last_sentence_prev = chunks[0]["text"].split(". ")[-1].rstrip(".")
        assert last_sentence_prev in chunks[1]["text"], (
            f"aucun recouvrement détecté :\n0: {chunks[0]['text']}\n1: {chunks[1]['text']}"
        )

    def test_single_short_sentence_single_chunk(self):
        chunks = chunk_text(S1, FakeTokenCounter())
        assert len(chunks) == 1
        assert chunks[0]["text"] == S1

    def test_empty_text(self):
        assert chunk_text("", FakeTokenCounter()) == []

    def test_meaningful_fragment_repack_not_dropped(self):
        """Régression du cas réel fasoeducation (Phase 3 review) :
        le chunk précédent est presque plein (126) et le fragment court
        meaningful 'conférences).' doit être PRÉSERVÉ par repack en deux
        chunks, sans jamais dépasser la limite."""
        # ~9 phrases longues (11 mots) pour remplir, puis une queue courte
        sentences = [
            "Les grandes rencontres internationales disposent d'infrastructures modernes et adaptées.",
            "Le Burkina Faso investit dans les équipements hôteliers et les espaces de congrès.",
            "Les visiteurs étrangers apprécient particulièrement l'accueil et l'organisation locale.",
            "Le tourisme d'affaires constitue un secteur en pleine expansion dans la capitale.",
            "Les pouvoirs publics encouragent les partenariats public-privé pour les infrastructures.",
            "Les manifestations culturelles rythment le calendrier touristique toute l'année.",
            "Les opérateurs économiques participent activement au financement des grands événements.",
        ]
        text = " ".join(sentences) + " (colloques, symposiums, conférences)."
        chunks = chunk_text(text, FakeTokenCounter(), target_tokens=13,
                            max_tokens=14, overlap_tokens=0)
        # 1. aucun texte meaningful perdu
        rejoined = " ".join(c["text"] for c in chunks)
        assert "conférences)" in rejoined
        # 2. limite dure respectée partout
        for c in chunks:
            assert c["token_count"] <= 14
        # 3. pas de micro-chunk orphan (le fragment est repacké avec du texte)
        assert all(len(c["text"]) >= 15 for c in chunks)

    def test_boilerplate_fragment_may_remain_alone(self):
        """Le boilerplate classé peut rester seul (l'ingestion l'auditera) :
        le chunker ne décide PAS du drop, il préserve le texte."""
        text = S1 + " (consulté le )"
        chunks = chunk_text(text, FakeTokenCounter())
        rejoined = " ".join(c["text"] for c in chunks)
        assert "consulté" in rejoined  # le chunker ne perd rien non plus
        # 5 phrases : 4 longues + une queue très courte
        text = " ".join([S1, S2, S3, S4, "Peu de texte."])
        chunks = chunk_text(text, FakeTokenCounter(), target_tokens=12,
                            max_tokens=15, overlap_tokens=0)
        # pas de micro-chunk final : la queue est fusionnée ou absorbée
        assert all(len(c["text"]) >= 15 for c in chunks)
        rejoined = " ".join(c["text"] for c in chunks)
        assert "Peu de texte." in rejoined  # aucune perte

    def test_oversized_sentence_is_fragmented_not_dropped(self):
        giant = ("Le site " * 50).strip() + "."  # 50 mots : dépasse max_tokens=15
        chunks = chunk_text(giant, FakeTokenCounter(), target_tokens=12, max_tokens=15)
        assert chunks  # rien n'est perdu
        rejoined_words = set(giant.replace(".", "").split())
        out_words = set(" ".join(c["text"] for c in chunks).replace(".", "").split())
        assert rejoined_words <= out_words


class TestChunkDocument:
    DOC = {
        "id": "abc123def4567890",
        "url": "https://fr.wikipedia.org/wiki/Test",
        "title": "Page de test",
        "source_type": "web",
        "language": "fr",
        "trust": "encyclopedia",
        "text": TEXT,
    }

    def test_deterministic_ids(self):
        c1 = chunk_document(self.DOC, FakeTokenCounter())
        c2 = chunk_document(self.DOC, FakeTokenCounter())
        assert [c.chunk_id for c in c1] == [c.chunk_id for c in c2]

    def test_id_format_and_sequence(self):
        chunks = chunk_document(self.DOC, FakeTokenCounter())
        for i, c in enumerate(chunks):
            assert c.chunk_id == f"abc123def4567890_chunk_{i:04d}"
            assert c.parent_id == "abc123def4567890"

    def test_metadata_propagated(self):
        chunks = chunk_document(self.DOC, FakeTokenCounter())
        for c in chunks:
            assert c.language == "fr"
            assert c.trust == "encyclopedia"
            assert c.url == self.DOC["url"]
            rec = c.to_record()
            assert rec["token_count"] == c.token_count


class TestDedupExact:
    def _mk(self, text: str, i: int) -> Chunk:
        return Chunk(chunk_id=f"doc{i}_chunk_0000", parent_id=f"doc{i}", url=f"u{i}",
                     title="t", source_type="web", language="fr", trust="blog",
                     text=text, chunk_idx=0, token_count=5, char_start=0, char_end=len(text))

    def test_identical_text_removed(self):
        chunks = [self._mk("Texte identique.", 0), self._mk("Texte identique.", 1)]
        unique, removed = dedup_exact(chunks)
        assert len(unique) == 1 and removed == 1

    def test_different_text_kept(self):
        chunks = [self._mk("Texte A.", 0), self._mk("Texte B.", 1)]
        unique, removed = dedup_exact(chunks)
        assert len(unique) == 2 and removed == 0

    def test_first_occurrence_wins(self):
        chunks = [self._mk("X", 0), self._mk("X", 1)]
        unique, _ = dedup_exact(chunks)
        assert unique[0].chunk_id.startswith("doc0")


class TestRealTokenizerLimits:
    """Conformité au tokenizer MiniLM réel (skip si fichiers non cachés)."""

    @pytest.fixture(scope="class")
    def real_counter(self, request):
        try:
            counter = TransformersTokenCounter()
        except Exception as e:  # cache HF absent et réseau indisponible
            pytest.skip(f"tokenizer MiniLM non disponible localement : {e}")
        request.cls.real_counter = counter
        return counter

    FRENCH_PARAGRAPH = (
        "Le parc national du W est un parc national du Burkina Faso, classé "
        "réserve de biosphère par l'UNESCO. Il abrite une faune remarquable : "
        "éléphants, buffles, lions, antilopes et de nombreuses espèces d'oiseaux. "
        "La meilleure période de visite s'étend de novembre à février, pendant "
        "la saison sèche, lorsque les animaux se concentrent autour des points "
        "d'eau. Des safaris accompagnés en véhicule 4x4 sont organisés depuis "
        "la ville d'Arly et les campements environnants proposent des hébergements."
    ) * 3  # assez long pour produire plusieurs chunks

    def test_every_chunk_under_hard_limit(self, real_counter):
        chunks = chunk_text(self.FRENCH_PARAGRAPH, real_counter)
        assert chunks, "au moins un chunk attendu"
        for c in chunks:
            # +2 tokens spéciaux <s></s> : la contrainte modèle est 128
            assert c["token_count"] + 2 <= 128, (
                f"chunk de {c['token_count'] + 2} tokens dépasse max_seq_length=128"
            )

    def test_token_counts_are_honest(self, real_counter):
        # le compte annoncé doit être celui du tokenizer réel, pas une approximation
        chunks = chunk_text(self.FRENCH_PARAGRAPH, real_counter)
        for c in chunks:
            assert c["token_count"] == len(real_counter.encode(c["text"]))

    def test_constant_values_match_verified_model_config(self):
        assert MAX_TOKENS == 126
        assert TARGET_TOKENS == 110
        assert OVERLAP_TOKENS == 15
