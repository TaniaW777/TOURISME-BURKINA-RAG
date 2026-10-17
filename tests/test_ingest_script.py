# -*- coding: utf-8 -*-
"""Tests Phase 2 — scripts/ingest.py (fonctions pures, sans réseau)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.ingest import (  # noqa: E402
    MIN_PUBLISHED_CHUNK_CHARS,
    build_document,
    filter_fragments,
    load_sources,
)
from ingestion.chunker import Chunk, is_boilerplate_fragment  # noqa: E402
from ingestion.fetch import FetchResult  # noqa: E402


class TestLoadSources:
    def test_sources_yaml_parses_with_required_fields(self):
        cfg = load_sources()
        assert "sources" in cfg and len(cfg["sources"]) > 10
        for s in cfg["sources"]:
            assert "url" in s and s["type"] in ("web", "pdf")
            assert s["trust"] in ("institutional", "encyclopedia", "press", "blog")

    def test_urls_unique(self):
        cfg = load_sources()
        urls = [s["url"] for s in cfg["sources"]]
        assert len(urls) == len(set(urls)), "doublons dans sources.yaml"

    def test_local_pdfs_config(self):
        cfg = load_sources()
        lp = cfg.get("local_pdfs")
        assert lp and "dir" in lp


GOOD_TEXT = (
    "Les ruines de Loropéni sont un site archéologique du sud-ouest du Burkina Faso. "
    "Inscrites au patrimoine mondial de l'UNESCO en 2009, elles comprennent des "
    "remparts de pierre sèche. La région de Gaoua abrite également des peuples Lobi "
    "et leurs traditions artisanales remarquables."
)


class TestBuildDocument:
    def _ok_result(self, text: str) -> FetchResult:
        return FetchResult(url="https://fr.wikipedia.org/wiki/X",
                           status="ok",
                           canonical_url="https://fr.wikipedia.org/wiki/X",
                           content_type="text/html",
                           raw_text=text, title="X — Wikipédia")

    def test_ok_produces_canonical_document(self):
        doc = build_document(self._ok_result(GOOD_TEXT), "web", "encyclopedia")
        assert doc is not None
        assert doc["language"] == "fr"
        assert doc["trust"] == "encyclopedia"
        assert doc["source_type"] == "web"
        assert len(doc["id"]) == 16
        assert "fetched_at" in doc and "requested_url" in doc  # provenance

    def test_short_content_rejected_explicitly(self):
        assert build_document(self._ok_result("trop court."), "web", "blog") is None

    def test_failed_fetch_rejected(self):
        r = FetchResult(url="u", status="http_error", error="404")
        assert build_document(r, "web", "blog") is None

    def test_encoding_fixed_at_ingestion(self):
        mojibake = GOOD_TEXT.replace("Loropéni", "LoropÃ©ni").replace("UNESCO", "UNESCO")
        doc = build_document(self._ok_result(mojibake), "web", "encyclopedia")
        assert doc is not None
        assert "Loropéni" in doc["text"] and "Ã©" not in doc["text"]


def _chunk(text: str) -> Chunk:
    return Chunk(chunk_id="x_chunk_0000", parent_id="x", url="u", title="t",
                 source_type="web", language="fr", trust="blog", text=text,
                 chunk_idx=0, token_count=len(text.split()), char_start=0,
                 char_end=len(text))


class TestBoilerplateClassification:
    """Règle de classification explicite — chaque motif est testé."""

    @pytest.mark.parametrize("frag,expected", [
        ("(consulté le )", True),        # citation Wikipedia, date absente
        ("(consulté le 12 mars 2024)", True),  # même modèle, date remplie
        ("B.", True),                    # initiale isolée
        ("p. 42", True),                 # pagination
        ("...", True),                   # ponctuation pure
        ("", True),                      # vide
        ("conférences).", False),        # MOT meaningful — jamais du boilerplate
        ("Ouagadougou", False),
        ("le parc du W", False),
    ])
    def test_classification(self, frag, expected):
        assert is_boilerplate_fragment(frag) is expected


class TestFilterFragments:
    DOC = {"id": "abc", "url": "https://exemple.bf/page", "title": "Page"}

    def test_boilerplate_dropped_with_full_audit_record(self):
        kept, dropped = filter_fragments(
            [_chunk("(consulté le )"), _chunk("Un vrai chunk substantiel de contenu.")],
            self.DOC,
        )
        assert len(kept) == 1 and len(dropped) == 1
        rec = dropped[0]
        assert rec["fragment"] == "(consulté le )"
        assert rec["url"] == "u"  # url portée par le chunk (helper _chunk)
        assert rec["document_id"]
        assert rec["chunk_id"]
        assert rec["reason"] and rec["rule"]

    def test_meaningful_short_fragment_raises_instead_of_dropping(self):
        with pytest.raises(ValueError, match="meaningful"):
            filter_fragments([_chunk("conférences).")], self.DOC)

    def test_normal_chunks_all_kept(self):
        chunks = [_chunk("Phrase assez longue pour être publiée sans problème.")]
        kept, dropped = filter_fragments(chunks, self.DOC)
        assert kept == chunks and dropped == []


class TestOfflineRechunk:
    """Mode --from-documents : reconstruction SANS réseau, prouvée par test."""

    DOC = {
        "id": "offline0000000000",
        "url": "https://exemple.bf/offline",
        "requested_url": "https://exemple.bf/offline",
        "title": "Page hors ligne",
        "text": GOOD_TEXT + " " + GOOD_TEXT,
        "source_type": "web",
        "language": "fr",
        "trust": "encyclopedia",
        "fetched_at": "2026-01-01T00:00:00Z",
    }

    def test_offline_rechunk_never_touches_network(self, tmp_path, monkeypatch):
        """run(documents_path=...) doit réussir même si TOUT accès réseau lève."""
        import scripts.ingest as ingest_mod

        docs_file = tmp_path / "documents.jsonl"
        docs_file.write_text(json.dumps(self.DOC, ensure_ascii=False) + "\n",
                             encoding="utf-8")

        def _no_network(*a, **k):
            raise AssertionError("ACCÈS RÉSEAU INTERDIT en mode --from-documents")

        monkeypatch.setattr("requests.Session.get", _no_network)
        monkeypatch.setattr("requests.Session.post", _no_network)
        for fn in ("fetch_web", "fetch_pdf", "fetch_local_pdf"):
            monkeypatch.setattr(ingest_mod, fn, _no_network)
        monkeypatch.setattr(ingest_mod, "RobotsCache", _no_network)

        monkeypatch.setattr(ingest_mod, "DOCUMENTS_OUT", tmp_path / "documents.out.jsonl")
        monkeypatch.setattr(ingest_mod, "CHUNKS_OUT", tmp_path / "chunks.out.jsonl")
        monkeypatch.setattr(ingest_mod, "REPORT_OUT", tmp_path / "report.out.json")

        report = ingest_mod.run(documents_path=docs_file)

        assert (tmp_path / "chunks.out.jsonl").exists()
        chunks = [json.loads(l) for l in
                  (tmp_path / "chunks.out.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
        assert chunks and all(c["parent_id"] == "offline0000000000" for c in chunks)
        assert report["documents"] == 1
        assert report["chunks"] == len(chunks)
        assert report["source_status"] == {"ok": 1}
        assert all(e["status"] == "ok" and e.get("offline_rechunk") for e in report["entries"])
