# -*- coding: utf-8 -*-
"""
tests/test_config.py
====================
Tests Phase 1 — configuration centralisée.

Objectifs :
  1. Résolution des chemins identique quel que soit le CWD.
  2. Comportement déterminé quand GROQ_API_KEY est absente
     (mode dégradé, pas de crash) et présente.
  3. Valeurs par défaut et validation.
  4. Le vrai environnement du développeur ne fait pas fuiter
     de configuration dans les tests (env injecté explicite).

Ces tests n'ont aucune dépendance réseau ni modèle lourd.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.config import (  # noqa: E402
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_GROQ_MODEL,
    REPO_ROOT,
    Settings,
    load_settings,
    resolve_repo_relative,
)


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────
def clean_env() -> dict[str, str]:
    """Environnement minimal sans aucune variable du projet."""
    excluded = {
        "GROQ_API_KEY", "GROQ_MODEL", "GROQ_TIMEOUT_S", "GROQ_MAX_RETRIES",
        "EMBEDDING_MODEL", "EMBEDDING_DEVICE", "CHROMA_PATH", "CORPUS_PATH",
        "COLLECTION_NAME", "TOP_K", "MIN_SIMILARITY", "MAX_QUERY_LENGTH",
        "RATE_LIMIT_PER_MINUTE", "LOG_RETRIEVED_CHUNKS",
    }
    return {k: v for k, v in os.environ.items() if k not in excluded}


# ──────────────────────────────────────────────
# 1. Ancrage des chemins — indépendant du CWD
# ──────────────────────────────────────────────
class TestPathAnchoring:
    def test_repo_root_is_this_repository(self):
        # La racine détectée contient bien le repo (marqueurs fiables)
        assert (REPO_ROOT / "src" / "config.py").exists()
        assert (REPO_ROOT / ".env.example").exists()

    def test_default_paths_are_repo_anchored(self):
        s = load_settings(env=clean_env())
        assert s.chroma_path == REPO_ROOT / "data" / "chroma_db"
        assert s.corpus_path == REPO_ROOT / "data" / "corpus.json"
        assert s.audit_log_path == REPO_ROOT / "logs" / "audit.log"

    def test_relative_env_paths_resolve_against_repo_root(self):
        # Un chemin relatif passé par l'environnement doit désigner le
        # même répertoire, même si le CWD est ailleurs.
        env = clean_env() | {"CHROMA_PATH": "data/chroma_db", "CORPUS_PATH": "data/corpus.json"}
        s = load_settings(env=env)
        assert s.chroma_path == Path("data/chroma_db")  # conservé tel quel...
        assert resolve_repo_relative(s.chroma_path) == REPO_ROOT / "data" / "chroma_db"

    def test_absolute_env_path_is_preserved(self):
        env = clean_env() | {"CHROMA_PATH": "/tmp/autre_chroma"}
        s = load_settings(env=env)
        assert resolve_repo_relative(s.chroma_path).is_absolute()

    def test_paths_identical_from_different_cwd(self):
        """Résolution en sous-processus depuis deux CWD différents."""
        code = (
            "import sys; sys.path.insert(0, r'%s'); "
            "from src.config import load_settings; "
            "s = load_settings(env={}); "
            "print(s.chroma_path); print(s.corpus_path)" % str(REPO_ROOT)
        )
        results = []
        for cwd in [str(REPO_ROOT), os.path.join(os.path.expanduser("~"))]:
            out = subprocess.run(
                [sys.executable, "-c", code],
                capture_output=True, text=True, cwd=cwd, check=True,
            )
            results.append(out.stdout.strip())
        assert results[0] == results[1], (
            f"chemins divergents selon le CWD :\n{results[0]}\n{results[1]}"
        )


# ──────────────────────────────────────────────
# 2. GROQ_API_KEY absente → mode dégradé, pas de crash
# ──────────────────────────────────────────────
class TestMissingGroqKey:
    def test_load_settings_succeeds_without_key(self):
        s = load_settings(env=clean_env())  # ne lève pas
        assert isinstance(s, Settings)

    def test_degraded_mode_flag(self):
        s = load_settings(env=clean_env())
        assert s.groq_api_key == ""
        assert s.groq_configured is False

    def test_validate_reports_missing_key_as_warning_not_error(self):
        s = load_settings(env=clean_env())
        problems = s.validate()
        # exactement un problème : la clé manquante (aucun autre par défaut)
        assert len(problems) == 1
        assert "GROQ_API_KEY" in problems[0]
        assert s.groq_configured is False  # l'objet reste utilisable

    def test_key_with_whitespace_only_is_not_configured(self):
        s = load_settings(env=clean_env() | {"GROQ_API_KEY": "   "})
        assert s.groq_configured is False


# ──────────────────────────────────────────────
# 3. Clé présente / valeurs par défaut / validation
# ──────────────────────────────────────────────
class TestConfiguredState:
    def test_key_present_is_configured(self):
        s = load_settings(env=clean_env() | {"GROQ_API_KEY": "gsk_test_123"})
        assert s.groq_configured is True
        assert s.validate() == []

    def test_defaults(self):
        s = load_settings(env=clean_env())
        assert s.groq_model == DEFAULT_GROQ_MODEL == "openai/gpt-oss-20b"
        assert s.embedding_model == DEFAULT_EMBEDDING_MODEL
        assert s.top_k == 5
        assert 0.0 <= s.min_similarity <= 1.0
        assert s.groq_timeout_s == 20.0

    def test_env_overrides(self):
        env = clean_env() | {
            "GROQ_API_KEY": "gsk_test_123",
            "GROQ_MODEL": "openai/gpt-oss-120b",
            "TOP_K": "8",
            "GROQ_TIMEOUT_S": "30",
        }
        s = load_settings(env=env)
        assert s.groq_model == "openai/gpt-oss-120b"
        assert s.top_k == 8
        assert s.groq_timeout_s == 30.0

    def test_invalid_numeric_values_fall_back_to_defaults(self):
        env = clean_env() | {"TOP_K": "pas_un_nombre", "GROQ_TIMEOUT_S": "abc"}
        s = load_settings(env=env)
        assert s.top_k == 5
        assert s.groq_timeout_s == 20.0

    def test_out_of_range_values_flagged_by_validate(self):
        env = clean_env() | {
            "GROQ_API_KEY": "gsk_test_123",
            "GROQ_TIMEOUT_S": "999",
            "TOP_K": "42",
        }
        s = load_settings(env=env)
        problems = s.validate()
        assert any("GROQ_TIMEOUT_S" in p for p in problems)
        assert any("TOP_K" in p for p in problems)

    def test_settings_are_frozen(self):
        s = load_settings(env=clean_env())
        try:
            s.top_k = 99  # type: ignore[misc]
        except Exception:
            return  # attendu : dataclass frozen
        raise AssertionError("Settings doit être immuable (frozen=True)")
