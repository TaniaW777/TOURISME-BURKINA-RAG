# -*- coding: utf-8 -*-
"""Tests Phase 2 — ingestion/fetch.py. HTTP entièrement simulé (requests-mock
via monkeypatch) : AUCUN réseau requis pour ces tests."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ingestion.fetch import (  # noqa: E402
    FetchResult,
    MAX_HTML_BYTES,
    RobotsCache,
    USER_AGENT,
    extract_pdf_bytes,
    fetch_local_pdf,
    fetch_web,
    resp_encoding_fallback,
)


# ──────────────────────────────────────────────
# Robots.txt — respect STRICT
# ──────────────────────────────────────────────
class TestRobotsCache:
    def test_disallow_is_blocked(self, monkeypatch):
        cache = RobotsCache()
        rp = MagicMock()
        rp.can_fetch.return_value = False
        monkeypatch.setattr(cache, "_cache", {"https://x.com": rp})
        allowed, reason = cache.allowed("https://x.com/page")
        assert not allowed and "robots" in reason

    def test_unreadable_robots_blocks_prudently(self, monkeypatch):
        cache = RobotsCache()
        monkeypatch.setattr(cache, "_fetch_robots", lambda origin: None)
        allowed, reason = cache.allowed("https://down.example/page")
        assert not allowed  # prudent : pas de crawl si robots injoignable

    def test_missing_robots_404_allows(self, monkeypatch):
        # sur 404, _fetch_robots retourne un parser avec allow_all=True
        # (un parser jamais parsé renvoie False — voir fix Phase 2)
        from urllib.robotparser import RobotFileParser

        cache = RobotsCache()

        def _robots_404(origin):
            rp = RobotFileParser()
            rp.allow_all = True
            return rp

        monkeypatch.setattr(cache, "_fetch_robots", _robots_404)
        allowed, _ = cache.allowed("https://ok.example/page")
        assert allowed


# ──────────────────────────────────────────────
# Fetch web — succès, échecs explicites, limites
# ──────────────────────────────────────────────
def _fake_response(url="", status=200, content=b"", ctype="text/html; charset=utf-8",
                   size=None):
    resp = MagicMock()
    resp.url = url
    resp.status_code = status
    resp.headers = {"Content-Type": ctype}
    if size is not None:
        resp.headers["Content-Length"] = str(size)
    resp.raise_for_status.side_effect = (
        None if status < 400 else _raise_http_error(status)
    )
    resp.iter_content.return_value = [content]
    return resp


def _raise_http_error(status):
    import requests

    return requests.HTTPError(f"{status} error")


def _allow_all_robots(monkeypatch):
    monkeypatch.setattr(RobotsCache, "allowed", lambda self, url: (True, ""))


LONG_HTML = "<html><head><title>Titre</title></head><body>" + \
    ("<p>Le Burkina Faso offre de nombreux sites touristiques remarquables. "
     "Les cascades de Karfiguéla près de Banfora attirent chaque année des "
     "visiteurs. Le Festival FESPACO célèbre le cinéma africain à Ouagadougou.</p>" * 10) + \
    "</body></html>"


class TestFetchWeb:
    def test_ok_page_extracted_with_title(self, monkeypatch):
        import requests

        _allow_all_robots(monkeypatch)
        monkeypatch.setattr(requests.Session, "get",
                            lambda self, *a, **k: _fake_response(
                                url="https://site.bf/page", content=LONG_HTML.encode()))
        monkeypatch.setattr("ingestion.fetch.time.sleep", lambda s: None)
        session = requests.Session()
        result = fetch_web("https://site.bf/page", session, RobotsCache())
        assert result.ok
        assert result.title == "Titre"
        assert "Karfiguéla" in result.raw_text
        assert result.canonical_url == "https://site.bf/page"

    def test_http_error_is_explicit_failure(self, monkeypatch):
        import requests

        _allow_all_robots(monkeypatch)
        monkeypatch.setattr(requests.Session, "get",
                            lambda self, *a, **k: _fake_response(status=404))
        monkeypatch.setattr("ingestion.fetch.time.sleep", lambda s: None)
        result = fetch_web("https://site.bf/absent", requests.Session(), RobotsCache())
        assert not result.ok
        assert result.status == "http_error"
        assert result.error  # jamais d'échec silencieux sans diagnostic

    def test_blocked_robots_never_fetches(self, monkeypatch):
        import requests

        called = {"n": 0}

        def _spy(self, *a, **k):
            called["n"] += 1
            return _fake_response(content=LONG_HTML.encode())

        monkeypatch.setattr(requests.Session, "get", _spy)
        monkeypatch.setattr("ingestion.fetch.time.sleep", lambda s: None)
        result = fetch_web("https://site.bf/interdit", requests.Session(), RobotsCache())
        assert result.status == "blocked_robots"
        assert called["n"] == 0  # aucune requête émise

    def test_oversized_content_rejected_before_extraction(self, monkeypatch):
        import requests

        _allow_all_robots(monkeypatch)
        monkeypatch.setattr(requests.Session, "get",
                            lambda self, *a, **k: _fake_response(
                                size=MAX_HTML_BYTES + 1, content=LONG_HTML.encode()))
        monkeypatch.setattr("ingestion.fetch.time.sleep", lambda s: None)
        result = fetch_web("https://gros.bf/x", requests.Session(), RobotsCache())
        assert result.status == "http_error"
        assert "volumineux" in result.error


class TestEncodingFallback:
    def test_charset_from_header(self):
        assert resp_encoding_fallback(b"", "text/html; charset=iso-8859-1") == "iso-8859-1"

    def test_utf8_detected(self):
        assert resp_encoding_fallback("é".encode("utf-8"), "") == "utf-8"

    def test_latin1_fallback(self):
        assert resp_encoding_fallback("é".encode("latin-1"), "") == "latin-1"


# ──────────────────────────────────────────────
# PDF — validation du contenu et échecs propres
# ──────────────────────────────────────────────
@pytest.fixture(scope="module")
def minimal_pdf() -> bytes:
    """PDF minimal mais valide, généré avec pdfplumber s'il est disponible,
    sinon fixture binaire construite à la main."""
    try:
        import io

        from reportlab.pdfgen import canvas  # type: ignore

        buf = io.BytesIO()
        c = canvas.Canvas(buf)
        c.drawString(100, 750, "Les ruines de Lorpéni au Burkina Faso sont un site majeur.")
        c.drawString(100, 730, "Les cascades de Banfora et le festival FESPACO attirent les visiteurs.")
        c.drawString(100, 710, "Le parc du W et la mare aux hippopotames de Bala sont protégés.")
        c.drawString(100, 690, "Ouagadougou accueille le SIAO, salon international de l'artisanat.")
        c.drawString(100, 670, "Bobo-Dioulasso est réputée pour sa grande mosquée et sa musique.")
        c.save()
        return buf.getvalue()
    except ImportError:
        pytest.skip("reportlab absent : fixture PDF non générable")


class TestPdf:
    def test_valid_pdf_extracted(self, minimal_pdf):
        result = extract_pdf_bytes(minimal_pdf, "https://x.bf/doc.pdf", "Doc")
        assert result.ok
        assert "Burkina" in result.raw_text or len(result.raw_text) > 20

    def test_non_pdf_bytes_rejected(self):
        result = extract_pdf_bytes(b"ceci n'est pas un pdf", "https://x.bf/fake.pdf")
        assert not result.ok  # extract_failed : échec explicite, pas de contenu vide

    def test_local_pdf_nonexistent(self):
        result = fetch_local_pdf(Path("Z:/inexistant.pdf"))
        assert not result.ok and result.error

    def test_local_pdf_wrong_type(self, tmp_path):
        p = tmp_path / "note.pdf"
        p.write_text("pas un pdf", encoding="utf-8")
        result = fetch_local_pdf(p)
        assert result.status == "invalid_content"


# ──────────────────────────────────────────────
# Invariants de politique
# ──────────────────────────────────────────────
class TestPolicy:
    def test_user_agent_is_honest_not_fake_chrome(self):
        assert "Mozilla" not in USER_AGENT and "Chrome" not in USER_AGENT
        assert "TourismeBurkinaRAG" in USER_AGENT
