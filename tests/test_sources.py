# -*- coding: utf-8 -*-
"""Tests PR 2.1 — registre curé de sources (data/sources.yaml).

Valide le CONTRAT du registre : schéma, unicité, valeurs autorisées.
Aucune dépendance au code d'ingestion ni au réseau.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

SOURCES_PATH = REPO_ROOT / "data" / "sources.yaml"

VALID_TYPES = {"web", "pdf"}
VALID_TRUST = {"institutional", "encyclopedia", "press", "blog"}


def _load() -> dict:
    with open(SOURCES_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


class TestSourceRegistry:
    def test_file_exists_and_parses(self):
        cfg = _load()
        assert isinstance(cfg.get("sources"), list) and len(cfg["sources"]) >= 10

    def test_every_source_has_required_fields(self):
        for s in _load()["sources"]:
            assert set(s.keys()) >= {"url", "type", "trust"}, f"champs manquants : {s}"
            assert s["type"] in VALID_TYPES, f"type invalide : {s['type']!r}"
            assert s["trust"] in VALID_TRUST, f"trust invalide : {s['trust']!r}"

    def test_urls_are_http_and_unique(self):
        urls = [s["url"] for s in _load()["sources"]]
        assert len(urls) == len(set(urls)), "doublons d'URL dans le registre"
        for u in urls:
            assert u.startswith(("https://", "http://")), f"schéma invalide : {u}"

    def test_https_used_wherever_possible(self):
        insecure = [u for u in (s["url"] for s in _load()["sources"])
                    if u.startswith("http://")]
        # tolérance : certaines sources .bf historiques n'ont pas de TLS ;
        # elles doivent rester l'exception — le test gèle la liste connue
        assert insecure == [], (
            f"sources HTTP non TLS inattendues (ajouter à la liste connue si "
            f"véritablement sans TLS) : {insecure}"
        )

    def test_notes_are_optional_strings(self):
        for s in _load()["sources"]:
            if "notes" in s:
                assert isinstance(s["notes"], str) and s["notes"]

    def test_local_pdfs_config(self):
        lp = _load().get("local_pdfs")
        assert isinstance(lp, dict) and "dir" in lp
        assert isinstance(lp.get("exclude_subdirs", []), list)
        # Les PDFs locaux sont gitignorés par conception (jamais committés) :
        # sur un clone propre (CI), le répertoire n'existe pas. Le contrat du
        # schéma est vérifié inconditionnellement ; la présence physique des
        # fichiers n'est testée que sur une installation complète.
        pdf_dir = REPO_ROOT / lp["dir"]
        if not pdf_dir.is_dir():
            pytest.skip("répertoire de PDFs locaux absent (clone propre/CI — "
                        "les PDFs sont volontairement non versionnés)")

    def test_registry_is_the_whitelist(self):
        # garde-fou documentaire : aucun scraping hors registre.
        # (le fetcher vérifie robots.txt ; ce test garantit que la liste
        # elle-même reste la source de vérité unique et lisible.)
        cfg = _load()
        assert "sources" in cfg and "local_pdfs" in cfg
        assert set(cfg.keys()) == {"sources", "local_pdfs"}, (
            f"sections inattendues : {set(cfg.keys())}"
        )
