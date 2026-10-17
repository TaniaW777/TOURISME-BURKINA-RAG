# -*- coding: utf-8 -*-
"""Tests Phase 3 — scripts/build_index.py.

Aucun réseau, aucun vrai modèle : FakeEmbedder déterministe + FakeCounter.
Le vrai index (data/chroma_db_v2) n'est PAS construit ici — les tests
travaillent dans des répertoires temporaires.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.build_index import (  # noqa: E402
    SafetyError,
    ValidationError,
    build_index,
    load_chunks,
    validate_chunks,
    verify_manifest,
)

LEGACY = REPO_ROOT / "data" / "chroma_db"


# ──────────────────────────────────────────────
# Fixtures / doublures
# ──────────────────────────────────────────────
class FakeCounter:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return list(range(max(1, len(text.split()))))


class FakeEmbedder:
    """Vecteurs déterministes de dim 8, normalisés (norme 1)."""

    dim = 8

    def __init__(self):
        self.calls = 0

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        out = []
        for t in texts:
            vals = [(ord(ch) % 97) / 97 + 0.01 for ch in (t + "padding")[: self.dim]]
            norm = sum(v * v for v in vals) ** 0.5
            out.append([v / norm for v in vals])
        return out


class FailingEmbedder:
    def embed(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError("panne simulée du modèle d'embeddings")


def make_chunk(i: int, text: str = None, **overrides) -> dict:
    base = {
        "id": f"doc0000000000000_chunk_{i:04d}",
        "parent_id": "doc0000000000000",
        "url": f"https://example.bf/page{i}",
        "title": f"Page {i}",
        "source_type": "web",
        "language": "fr",
        "trust": "encyclopedia",
        "text": text or (
            "Les cascades de Banfora attirent de nombreux visiteurs chaque "
            "année pendant la saison sèche au Burkina Faso."
        ),
        "chunk_idx": i,
        "token_count": 17,  # nb de mots du texte par défaut (FakeCounter)
    }
    base.update(overrides)
    return base


def write_corpus(tmp_path: Path, chunks: list[dict]) -> Path:
    p = tmp_path / "chunks.jsonl"
    p.write_text("\n".join(json.dumps(c, ensure_ascii=False) for c in chunks),
                 encoding="utf-8")
    return p


GOOD = [make_chunk(i) for i in range(3)]


# ──────────────────────────────────────────────
# Validation du corpus
# ──────────────────────────────────────────────
class TestValidateChunks:
    def test_valid_corpus_passes(self):
        assert validate_chunks(GOOD, FakeCounter()) == []

    def test_missing_field_rejected(self):
        bad = make_chunk(0)
        del bad["trust"]
        with pytest.raises(ValidationError, match="champs manquants"):
            validate_chunks([*GOOD, bad], FakeCounter())

    def test_empty_text_rejected(self):
        with pytest.raises(ValidationError, match="texte vide"):
            validate_chunks([make_chunk(0, text="  ")], FakeCounter())

    def test_duplicate_ids_rejected(self):
        dup = [make_chunk(0), make_chunk(0)]
        with pytest.raises(ValidationError, match="dupliqué"):
            validate_chunks(dup, FakeCounter())

    def test_bad_source_type_and_language_rejected(self):
        with pytest.raises(ValidationError, match="source_type"):
            validate_chunks([make_chunk(0, source_type="video")], FakeCounter())
        with pytest.raises(ValidationError, match="language"):
            validate_chunks([make_chunk(0, language="de")], FakeCounter())

    def test_token_limit_violation_rejected(self):
        long_text = "mot " * 200  # 200 tokens > 128 - 2
        with pytest.raises(ValidationError, match="limite modèle"):
            validate_chunks([make_chunk(0, text=long_text, token_count=200)], FakeCounter())

    def test_dishonest_token_count_rejected(self):
        with pytest.raises(ValidationError, match="token_count stocké"):
            validate_chunks([make_chunk(0, token_count=3)], FakeCounter())


class TestLoadChunks:
    def test_malformed_json_rejected(self, tmp_path):
        p = tmp_path / "bad.jsonl"
        p.write_text('{"id": "x"}\n{pas du json', encoding="utf-8")
        with pytest.raises(ValidationError, match="JSON invalide"):
            load_chunks(p)

    def test_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_chunks(tmp_path / "absent.jsonl")

    def test_empty_file_rejected(self, tmp_path):
        p = tmp_path / "empty.jsonl"
        p.write_text("", encoding="utf-8")
        with pytest.raises(ValidationError, match="corpus vide"):
            load_chunks(p)


# ──────────────────────────────────────────────
# Protection de l'index legacy
# ──────────────────────────────────────────────
class TestLegacyProtection:
    def test_builder_refuses_legacy_path(self):
        with pytest.raises(SafetyError, match="REFUS"):
            build_index(chunks_path=REPO_ROOT / "data" / "chunks.jsonl",
                        index_path=LEGACY, embedder=FakeEmbedder(), counter=FakeCounter())

    def test_builder_refuses_subdir_of_legacy(self):
        with pytest.raises(SafetyError):
            build_index(index_path=LEGACY / "nested", embedder=FakeEmbedder(),
                        counter=FakeCounter())

    def test_legacy_index_untouched_by_guard(self):
        # le simple fait que les tests ci-dessus lèvent SafetyError AVANT
        # toute écriture protège le legacy ; on le vérifie factuellement —
        # mais seulement là où l'index legacy local existe (gitignoré :
        # absent d'un clone propre / CI).
        import chromadb

        if not LEGACY.is_dir():
            pytest.skip("index legacy local absent (clone propre/CI - "
                        "chroma_db n'est volontairement pas versionne)")
        count = chromadb.PersistentClient(path=str(LEGACY))             .get_or_create_collection("tourisme_burkina").count()
        assert count == 785


# ──────────────────────────────────────────────
# Construction (répertoires temporaires)
# ──────────────────────────────────────────────
class TestBuildIndex:
    def test_successful_build_and_manifest(self, tmp_path):
        corpus = write_corpus(tmp_path, GOOD)
        idx = tmp_path / "chroma_v2"
        result = build_index(chunks_path=corpus, index_path=idx,
                             embedder=FakeEmbedder(), counter=FakeCounter())
        assert result.indexed_count == 3
        assert result.embedding_dim == 8
        m = verify_manifest(result.manifest_path)
        assert m["source_chunk_count"] == m["indexed_count"] == 3

        import chromadb

        col = chromadb.PersistentClient(path=str(idx)).get_collection("tourisme_burkina_v2")
        assert col.count() == 3
        got = col.get(include=["metadatas"])
        for meta in got["metadatas"]:
            for field in ("url", "title", "source_type", "language", "trust",
                          "parent_id", "chunk_idx", "token_count"):
                assert field in meta

    def test_rebuild_is_reproducible_and_replaces(self, tmp_path):
        corpus = write_corpus(tmp_path, GOOD)
        idx = tmp_path / "chroma_v2"
        r1 = build_index(chunks_path=corpus, index_path=idx,
                         embedder=FakeEmbedder(), counter=FakeCounter())
        # corpus modifié : un chunk de moins → recréation complète
        write_corpus(tmp_path, GOOD[:2])
        r2 = build_index(chunks_path=corpus, index_path=idx,
                         embedder=FakeEmbedder(), counter=FakeCounter())
        assert r2.indexed_count == 2  # pas d'accumulation : 3 puis 2, pas 5
        assert verify_manifest(r2.manifest_path)["indexed_count"] == 2

    def test_validation_failure_leaves_no_index_no_manifest(self, tmp_path):
        corpus = write_corpus(tmp_path, [make_chunk(0), make_chunk(0)])  # dupliqués
        idx = tmp_path / "chroma_v2"
        with pytest.raises(ValidationError):
            build_index(chunks_path=corpus, index_path=idx,
                        embedder=FakeEmbedder(), counter=FakeCounter())
        assert not idx.exists()
        assert not (tmp_path / "index_manifest.json").exists()

    def test_embedding_failure_no_manifest(self, tmp_path):
        corpus = write_corpus(tmp_path, GOOD)
        idx = tmp_path / "chroma_v2"
        with pytest.raises(Exception, match="panne simulée|échec"):
            build_index(chunks_path=corpus, index_path=idx,
                        embedder=FailingEmbedder(), counter=FakeCounter())
        assert not (idx.parent / "index_manifest.json").exists()

    def test_partial_batch_failure_no_manifest(self, tmp_path):
        corpus = write_corpus(tmp_path, [make_chunk(i) for i in range(10)])
        idx = tmp_path / "chroma_v2"

        class HalfFailing:
            def __init__(self):
                self.calls = 0

            def embed(self, texts):
                self.calls += 1
                if self.calls > 1:  # premier lot OK, second en panne
                    raise RuntimeError("panne au lot 2")
                return FakeEmbedder().embed(texts)

        with pytest.raises(Exception):
            build_index(chunks_path=corpus, index_path=idx,
                        embedder=HalfFailing(), counter=FakeCounter(), batch_size=4)
        assert not (idx.parent / "index_manifest.json").exists()
        # un index partiel peut exister mais sans manifest il est invalide :
        with pytest.raises(ValidationError, match="manifest absent"):
            verify_manifest(idx.parent / "index_manifest.json")


# ──────────────────────────────────────────────
# Manifest ↔ index
# ──────────────────────────────────────────────
class TestManifest:
    def test_verify_detects_count_mismatch(self, tmp_path):
        corpus = write_corpus(tmp_path, GOOD)
        idx = tmp_path / "chroma_v2"
        result = build_index(chunks_path=corpus, index_path=idx,
                             embedder=FakeEmbedder(), counter=FakeCounter())
        m = json.loads(result.manifest_path.read_text(encoding="utf-8"))
        m["indexed_count"] = 99  # manifest falsifié
        result.manifest_path.write_text(json.dumps(m), encoding="utf-8")
        with pytest.raises(ValidationError, match="incohérent"):
            verify_manifest(result.manifest_path)

    def test_verify_detects_changed_corpus(self, tmp_path):
        corpus = write_corpus(tmp_path, GOOD)
        idx = tmp_path / "chroma_v2"
        result = build_index(chunks_path=corpus, index_path=idx,
                             embedder=FakeEmbedder(), counter=FakeCounter())
        write_corpus(tmp_path, GOOD + [make_chunk(99)])  # corpus modifié
        with pytest.raises(ValidationError, match="corpus source a changé"):
            verify_manifest(result.manifest_path)

    def test_missing_manifest_is_failure(self, tmp_path):
        with pytest.raises(ValidationError, match="manifest absent"):
            verify_manifest(tmp_path / "index_manifest.json")

class TestManifestRepoRelativePaths:
    """Régression : le manifeste ne doit dépendre d'aucun chemin
    développeur (chemins absolus type Windows) et verify_manifest doit résoudre les
    chemins relatifs contre REPO_ROOT — tout en acceptant les manifestes
    legacy à chemins absolus."""

    def test_in_repo_paths_are_relative_posix(self, tmp_path, monkeypatch):
        import scripts.build_index as bi

        # simule un repo racine dans tmp : REPO_ROOT patché des deux côtés
        monkeypatch.setattr(bi, "REPO_ROOT", tmp_path)
        corpus = write_corpus(tmp_path, GOOD)
        idx = tmp_path / "data" / "chroma_db_v2"
        result = bi.build_index(chunks_path=corpus, index_path=idx,
                                embedder=FakeEmbedder(), counter=FakeCounter())
        m = json.loads(result.manifest_path.read_text(encoding="utf-8"))
        assert m["index_path"] == "data/chroma_db_v2"
        assert m["corpus_path"] == "chunks.jsonl"  # corpus écrit à la racine simulée
        # aucun path machine : pas de lettre de lecteur, pas d'anti-slash
        for key in ("index_path", "corpus_path"):
            assert ":" not in m[key] and "\\" not in m[key] and "Users" not in m[key]
        # la vérification résout le chemin relatif contre REPO_ROOT
        verified = bi.verify_manifest(result.manifest_path)
        assert verified["indexed_count"] == len(GOOD)

    def test_out_of_repo_paths_stay_absolute_but_verify(self, tmp_path, monkeypatch):
        import scripts.build_index as bi

        monkeypatch.setattr(bi, "REPO_ROOT", tmp_path / "repo_simule")
        corpus = write_corpus(tmp_path, GOOD)          # hors du "repo" simulé
        idx = tmp_path / "ailleurs" / "chroma_v2"      # hors du repo aussi
        result = bi.build_index(chunks_path=corpus, index_path=idx,
                                embedder=FakeEmbedder(), counter=FakeCounter())
        m = json.loads(result.manifest_path.read_text(encoding="utf-8"))
        assert Path(m["index_path"]).is_absolute()
        assert bi.verify_manifest(result.manifest_path)["indexed_count"] == len(GOOD)

    def test_legacy_absolute_manifest_still_verifies(self, tmp_path):
        # manifeste à l'ancienne (chemins absolus) : compatibilité arrière
        corpus = write_corpus(tmp_path, GOOD)
        idx = tmp_path / "chroma_v2"
        result = build_index(chunks_path=corpus, index_path=idx,
                             embedder=FakeEmbedder(), counter=FakeCounter())
        m = json.loads(result.manifest_path.read_text(encoding="utf-8"))
        m["index_path"] = str(idx.resolve())       # forme legacy absolue
        m["corpus_path"] = str(corpus.resolve())
        result.manifest_path.write_text(json.dumps(m), encoding="utf-8")
        assert verify_manifest(result.manifest_path)["indexed_count"] == len(GOOD)

    def test_relative_path_resolution_failure(self, tmp_path, monkeypatch):
        import scripts.build_index as bi

        monkeypatch.setattr(bi, "REPO_ROOT", tmp_path / "autre_racine")
        corpus = write_corpus(tmp_path, GOOD)
        idx = tmp_path / "chroma_v2"
        result = bi.build_index(chunks_path=corpus, index_path=idx,
                                embedder=FakeEmbedder(), counter=FakeCounter())
        m = json.loads(result.manifest_path.read_text(encoding="utf-8"))
        m["index_path"] = "data/inexistant"          # relatif qui ne résout pas
        result.manifest_path.write_text(json.dumps(m), encoding="utf-8")
        with pytest.raises(ValidationError):
            bi.verify_manifest(result.manifest_path)

