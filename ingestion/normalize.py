# -*- coding: utf-8 -*-
"""
ingestion/normalize.py
======================
Normalisation du texte brut et détection de langue, À L'INGESTION
(plus tard ne jamais corriger l'encodage au moment de la requête).

Reprend et déplace la logique historique de rag_pipeline.fix_encoding
et scraping_script.sanitize_text — les fichiers legacy restent intacts.
"""

from __future__ import annotations

import re

import bleach

# ──────────────────────────────────────────────
# CORRECTION D'ENCODAGE (héritée de [FIX-11], exécutée désormais à l'ingestion)
# ──────────────────────────────────────────────
_ENCODING_REPLACEMENTS = {
    "Ã©": "é", "Ã¨": "è", "Ãª": "ê", "Ã«": "ë", "Ã ": "à",
    "Ã¢": "â", "Ã¤": "ä", "Ã®": "î", "Ã¯": "ï", "Ã´": "ô",
    "Ã¶": "ö", "Ã²": "ò", "Ã»": "û", "Ã¹": "ù", "Ã¼": "ü",
    "Ã§": "ç", "Ã†": "Æ", "Ã\u2030": "É", "Ã\u02c6": "È", "Ã\u009c": "Ü",
    "\u00e2\u20ac\u2122": "’",
    "\u00e2\u20ac\u0153": "“",
    "\u00e2\u20ac\x9d": "”",
    "\u00e2\u20ac\x93": "–",
    "\u00e2\u20ac\x94": "—",
    "\u00c2\u00a0": " ",
    "\u00c2\u00ab": "«",
    "\u00c2\u00bb": "»",
}


def fix_encoding(text: str) -> str:
    """Corrige les artéfacts Latin-1/UTF-8 mal interprété.

    D'abord un round-trip encode/decode (récupère les vrais caractères
    quand le texte a été mal décodé à la source), puis table de
    remplacement pour les cas résiduels.
    """
    if not text:
        return text
    try:
        return text.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass
    for bad, good in _ENCODING_REPLACEMENTS.items():
        text = text.replace(bad, good)
    return text


# ──────────────────────────────────────────────
# ASSAINISSEMENT
# ──────────────────────────────────────────────
_CTRL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def sanitize_text(text: str) -> str:
    """Supprime HTML résiduel, caractères de contrôle ; normalise les espaces.

    Les fins de phrase (. ! ? …) sont préservées — indispensables au
    chunker qui découpe par phrases.
    """
    text = bleach.clean(text, tags=[], strip=True)
    text = _CTRL_CHARS_RE.sub("", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s*\n\s*", "\n", text)
    return text.strip()


def normalize_text(text: str) -> str:
    """Pipeline complet de normalisation d'un texte extrait."""
    if not text:
        return ""
    text = fix_encoding(text)
    text = sanitize_text(text)
    return text


# ──────────────────────────────────────────────
# DÉTECTION DE LANGUE — heuristique déterministe FR/EN
# ──────────────────────────────────────────────
# Pas de dépendance externe : ratio de mots-outils caractéristiques.
# Suffisant pour notre corpus (sources FR majoritaires, wikivoyage EN connu).
_FR_MARKERS = {
    "le", "la", "les", "des", "du", "une", "un", "et", "est", "dans",
    "pour", "que", "qui", "au", "aux", "avec", "sur", "plus", "sont",
    "cette", "par", "pas", "vous", "nous", "il", "elle", "mais", "ou",
    "où", "leur", "c'est", "d'un", "d'une", "l'on", "était", "être",
}
_EN_MARKERS = {
    "the", "of", "and", "to", "in", "is", "you", "that", "it", "he",
    "was", "for", "on", "are", "as", "with", "his", "they", "at", "be",
    "this", "have", "from", "or", "had", "by", "but", "not", "what",
    "all", "were", "we", "your", "can", "their", "said", "there",
}


def detect_language(text: str) -> str:
    """Retourne 'fr', 'en' ou 'unknown' à partir d'un échantillon de texte.

    Seuil : la langue détectée doit avoir un ratio de marqueurs au moins
    double de l'autre ; sinon 'unknown' (conservateur — on ne devine pas).
    """
    if not text or len(text) < 40:
        return "unknown"
    words = re.findall(r"[a-zà-ÿ']+", text.lower())
    # Élimine les mots accentués du côté anglais (ils ne comptent pas comme
    # mots-outils EN mais les marqueurs FR accentés existent : c'est, où...)
    sample = words[:500]
    if len(sample) < 10:
        return "unknown"
    fr = sum(1 for w in sample if w in _FR_MARKERS)
    en = sum(1 for w in sample if w in _EN_MARKERS)
    total = len(sample)
    fr_ratio, en_ratio = fr / total, en / total
    if fr_ratio >= 2 * en_ratio and fr_ratio >= 0.04:
        return "fr"
    if en_ratio >= 2 * fr_ratio and en_ratio >= 0.04:
        return "en"
    return "unknown"


# ──────────────────────────────────────────────
# SCHÉMA CANONIQUE DE DOCUMENT
# ──────────────────────────────────────────────
def doc_id_for(url: str) -> str:
    """ID déterministe de document : sha256(url)[:16]."""
    import hashlib

    return hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
