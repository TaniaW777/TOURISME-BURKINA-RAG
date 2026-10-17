# -*- coding: utf-8 -*-
"""
ingestion/chunker.py
====================
Découpage conscient des phrases, borné par le tokenizer RÉEL du modèle
d'embeddings — pas par une approximation en mots.

Fait vérifié (Phase 2) : sentence-transformers/paraphrase-multilingual-
MiniLM-L12-v2 a max_seq_length = 128 (sentence_bert_config.json officiel),
tokenizer XLM-R avec 2 tokens spéciaux (<s>, </s>).
L'ancien corpus (~190 tokens/chunk en moyenne) dépassait cette limite :
les chunks étaient silencieusement tronqués à l'encodage.

Garanties testées :
  - chaque chunk fait <= MAX_TOKENS tokens (tokenizer réel ou injecté)
  - découpage aux frontières de phrases, jamais au milieu
  - recouvrement (overlap) en phrases entières
  - IDs déterministes : sha256(url)[:16]_chunk_N
  - déduplication exacte par hash de texte normalisé
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Protocol

# ──────────────────────────────────────────────
# LIMITES — alignées sur le modèle d'embeddings
# ──────────────────────────────────────────────
MODEL_MAX_SEQ_LENGTH = 128          # vérifié : sentence_bert_config.json
SPECIAL_TOKENS = 2                  # <s> ... </s> (XLM-R)
MAX_TOKENS = MODEL_MAX_SEQ_LENGTH - SPECIAL_TOKENS   # 126 tokens de texte
TARGET_TOKENS = 110                 # marge avant la limite dure
OVERLAP_TOKENS = 15                 # recouvrement entre chunks consécutifs
MIN_CHUNK_CHARS = 60                # en dessous : bruit, on fusionne/abandonne

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?…])\s+")


class TokenCounter(Protocol):
    """Interface minimale d'un tokenizer (testable sans réseau)."""

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]: ...


class TransformersTokenCounter:
    """Adaptateur tokenizer HuggingFace, chargé une seule fois (cached)."""

    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        from transformers import AutoTokenizer

        self._tok = AutoTokenizer.from_pretrained(model_name)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return self._tok.encode(text, add_special_tokens=add_special_tokens)


@dataclass
class Chunk:
    chunk_id: str
    parent_id: str
    url: str
    title: str
    source_type: str
    language: str
    trust: str
    text: str
    chunk_idx: int
    token_count: int
    char_start: int
    char_end: int

    def to_record(self) -> dict:
        return {
            "id": self.chunk_id,
            "parent_id": self.parent_id,
            "url": self.url,
            "title": self.title,
            "source_type": self.source_type,
            "language": self.language,
            "trust": self.trust,
            "text": self.text,
            "chunk_idx": self.chunk_idx,
            "token_count": self.token_count,
        }


# ──────────────────────────────────────────────
# CLASSIFICATION DES FRAGMENTS COURTS (règle testable)
# ──────────────────────────────────────────────
# Un fragment court ne peut être ABANDONNÉ que s'il est du boilerplate sans
# valeur informationnelle, selon ces motifs EXPLICITES. Tout fragment court
# qui ne matche AUCUN motif est du texte meaningful : il doit être préservé
# (repack) — s'il ne peut l'être, l'ingestion échoue bruyamment, jamais de
# drop silencieux.
_BOILERPLATE_PATTERNS = [
    r"^\(\s*consulté le[^)]*\)?\.?$",   # modèle de citation Wikipedia, date absente
    r"^\(\s*consulté le\b",             # variante tronquée du même modèle
    r"^[A-Z]\.?$",                      # initiale isolée (fin de bibliographie)
    r"^(p|pp)\.\s*\d+",                 # référence de pagination
    r"^[\W\d_]+$",                      # aucun caractère alphabétique (ponctuation pure)
]


def is_boilerplate_fragment(text: str) -> bool:
    """True ssi le fragment est du boilerplate classé, sans contenu informationnel."""
    stripped = text.strip()
    if not stripped:
        return True
    return any(re.match(p, stripped, re.IGNORECASE) for p in _BOILERPLATE_PATTERNS)


# ──────────────────────────────────────────────
# DÉCOUPAGE PAR PHRASES
# ──────────────────────────────────────────────
def split_sentences(text: str) -> list[str]:
    """Découpe en phrases ; les séparateurs de fin de phrase sont conservés."""
    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]
    return sentences


def _split_long_sentence(sentence: str, counter: TokenCounter, max_tokens: int) -> list[str]:
    """Une phrase seule dépasse la limite (rare, ex. énumérations) :
    coupe aux virgules/points-virgules, sinon par mots."""
    parts: list[str] = []
    current = ""
    for piece in re.split(r"(?<=[,;:])\s+", sentence):
        candidate = f"{current} {piece}".strip()
        if len(counter.encode(candidate)) <= max_tokens:
            current = candidate
        else:
            if current:
                parts.append(current)
            if len(counter.encode(piece)) <= max_tokens:
                current = piece
            else:
                # coupe par mots — dernier recours
                words = piece.split()
                current = ""
                for w in words:
                    c2 = f"{current} {w}".strip()
                    if len(counter.encode(c2)) <= max_tokens:
                        current = c2
                    else:
                        if current:
                            parts.append(current)
                        current = w
    if current:
        parts.append(current)
    return parts


def chunk_text(
    text: str,
    counter: TokenCounter,
    target_tokens: int = TARGET_TOKENS,
    max_tokens: int = MAX_TOKENS,
    overlap_tokens: int = OVERLAP_TOKENS,
    min_chars: int = MIN_CHUNK_CHARS,
) -> list[dict]:
    """Découpe un texte normalisé en chunks conformes à la limite de tokens.

    Retourne une liste de dicts {text, token_count, char_start, char_end}.
    Propriétés :
      - chaque chunk <= max_tokens (testé, pas supposé)
      - recouvrement : le chunk suivant redémarre sur les dernières phrases
        du précédent tant que le budget overlap_tokens n'est pas épuisé
    """
    sentences = split_sentences(text)
    # fragmente les phrases trop longues AVANT le packing
    prepared: list[str] = []
    for s in sentences:
        if len(counter.encode(s)) <= max_tokens:
            prepared.append(s)
        else:
            prepared.extend(_split_long_sentence(s, counter, max_tokens))

    # positions de caractères pour la provenance (passage citable)
    char_positions: list[tuple[int, int]] = []
    cursor = 0
    for s in prepared:
        start = text.find(s[:40], cursor)
        if start == -1:
            start = cursor
        char_positions.append((start, start + len(s)))
        cursor = start + len(s)

    chunks: list[dict] = []
    i = 0
    carry: list[int] = []  # indices de phrases trop courtes, reportées au chunk suivant
    while i < len(prepared) or carry:
        # 1. packing glouton : le carry d'abord, puis de nouvelles phrases.
        #    En reprise de carry, on packing jusqu'à max_tokens (pas target) :
        #    sinon un reste court + une phrase longue dépasseraient target et
        #    le reste court serait émis seul au milieu du document.
        limit = max_tokens if carry else target_tokens
        packed_idx = list(carry)
        token_count = sum(len(counter.encode(prepared[k])) for k in packed_idx)
        j = i
        while j < len(prepared):
            s = prepared[j]
            s_tokens = len(counter.encode(s))
            if packed_idx and token_count + s_tokens > limit:
                break
            packed_idx.append(j)
            token_count += s_tokens
            j += 1

        chunk_text_str = " ".join(prepared[k] for k in packed_idx)
        actual_tokens = len(counter.encode(chunk_text_str))

        # 2. contrainte dure : dépassement (rare, phrases déjà fragmentées) → on retire
        while packed_idx and actual_tokens > max_tokens:
            packed_idx.pop()
            chunk_text_str = " ".join(prepared[k] for k in packed_idx)
            actual_tokens = len(counter.encode(chunk_text_str))

        if not packed_idx:
            break  # plus rien à packer ni à porter

        progressed = j > i or len(packed_idx) > len(carry)

        # 3. chunk trop court ET du texte reste après : report au chunk suivant
        #    (seulement si on a progressé, sinon boucle infinie → on fusionne en 3b)
        if len(chunk_text_str) < min_chars and j < len(prepared) and progressed:
            carry = packed_idx
            i = j
            continue

        # 3b. reste trop court : fusion arrière, sinon REPACK du chunk précédent
        #     avec le fragment en deux chunks équilibrés — sans jamais dépasser
        #     max_tokens. Le drop n'existe pas ici : c'est l'ingestion qui
        #     décide (et seulement pour du boilerplate classé).
        if len(chunk_text_str) < min_chars and chunks:
            merged = chunks[-1]["text"] + " " + chunk_text_str
            merged_tokens = len(counter.encode(merged))
            if merged_tokens <= max_tokens:
                # fusion simple : le précédent a du budget
                chunks[-1]["text"] = merged
                chunks[-1]["token_count"] = merged_tokens
                chunks[-1]["char_end"] = char_positions[packed_idx[-1]][1]
                chunks[-1]["idx_list"] = chunks[-1]["idx_list"] + packed_idx
                carry = []
                if j >= len(prepared):
                    break  # fin de document
                i = j
                continue

            # fusion impossible (précédent presque plein) : repack du précédent
            # + fragment en chunks équilibrés. Le repack travaille en UNITÉS DE
            # TEXTE : si une pièce du chunk précédent est elle-même énorme (ex.
            # phrase géante fragmentée en 126+3 tokens), on la re-découpe à une
            # frontière plus fine pour équilibrer — sans jamais dépasser max.
            combined_texts = [prepared[k] for k in chunks[-1]["idx_list"]] + \
                [prepared[k] for k in packed_idx]
            total_tokens = sum(len(counter.encode(t)) for t in combined_texts)
            half = max(1, total_tokens // 2)

            units: list[str] = []
            for t in combined_texts:
                if len(counter.encode(t)) > half:
                    units.extend(_split_long_sentence(t, counter, half))
                else:
                    units.append(t)

            groups: list[list[str]] = []
            current: list[str] = []
            cur_tok = 0
            for u in units:
                ut = len(counter.encode(u))
                if current and cur_tok + ut > half:
                    groups.append(current)
                    current, cur_tok = [], 0
                current.append(u)
                cur_tok += ut
            if current:
                groups.append(current)

            if len(groups) >= 2:
                group_texts = [" ".join(g) for g in groups]
                group_toks = [len(counter.encode(t)) for t in group_texts]
                if (all(t <= max_tokens for t in group_toks)
                        and all(len(t) >= min_chars for t in group_texts)):
                    # positions recalculées depuis le début du chunk précédent
                    cursor = chunks[-1]["char_start"]
                    new_entries = []
                    for gt, gtok in zip(group_texts, group_toks):
                        start = text.find(gt[:40], cursor)
                        if start == -1:
                            start = cursor
                        new_entries.append({
                            "text": gt, "token_count": gtok,
                            "char_start": start, "char_end": start + len(gt),
                            "idx_list": [],
                        })
                        cursor = start + len(gt)
                    chunks[-1:] = new_entries
                    carry = []
                    if j >= len(prepared):
                        break
                    i = j
                    continue
            # repack impossible aussi : on émet seul — l'ingestion classera
            # (boilerplate → drop audité ; meaningful → échec bruyant)

        chunks.append({
            "text": chunk_text_str,
            "token_count": actual_tokens,
            "char_start": char_positions[packed_idx[0]][0],
            "char_end": char_positions[packed_idx[-1]][1],
            "idx_list": list(packed_idx),
        })
        carry = []

        # 4. avance avec recouvrement : reprend les dernières phrases du chunk
        #    émis dans la limite du budget overlap_tokens
        if j >= len(prepared):
            break  # dernier chunk : pas de recouvrement après la fin
        overlap_budget = overlap_tokens
        back = 0
        k = len(packed_idx) - 1
        while k > 0:
            s_tokens = len(counter.encode(prepared[packed_idx[k]]))
            if overlap_budget - s_tokens < 0:
                break
            overlap_budget -= s_tokens
            back += 1
            k -= 1
        i = j - back if back else j

    return chunks


# ──────────────────────────────────────────────
# CHUNKS D'UN DOCUMENT + DÉDUPLICATION EXACTE
# ──────────────────────────────────────────────
def chunk_document(
    doc: dict,
    counter: TokenCounter,
    **chunk_kwargs,
) -> list[Chunk]:
    """Produit les chunks d'un document normalisé avec IDs déterministes."""
    pieces = chunk_text(doc["text"], counter, **chunk_kwargs)
    chunks: list[Chunk] = []
    for idx, piece in enumerate(pieces):
        chunks.append(Chunk(
            chunk_id=f"{doc['id']}_chunk_{idx:04d}",
            parent_id=doc["id"],
            url=doc["url"],
            title=doc["title"],
            source_type=doc["source_type"],
            language=doc["language"],
            trust=doc.get("trust", ""),
            text=piece["text"],
            chunk_idx=idx,
            token_count=piece["token_count"],
            char_start=piece["char_start"],
            char_end=piece["char_end"],
        ))
    return chunks


def dedup_exact(chunks: list[Chunk]) -> tuple[list[Chunk], int]:
    """Supprime les chunks au texte strictement identique (toutes sources confondues).

    Retourne (chunks_uniques, nb_doublons_supprimés). Le premier encountered
    est conservé ; les doublons typiques viennent des recouvrements
    Wikipedia FR / Wikivoyage FR sur les mêmes sujets.
    """
    seen: dict[str, str] = {}  # sha256(texte) -> chunk_id gardé
    unique: list[Chunk] = []
    removed = 0
    for c in chunks:
        h = hashlib.sha256(c.text.encode("utf-8")).hexdigest()
        if h in seen:
            removed += 1
            continue
        seen[h] = c.chunk_id
        unique.append(c)
    return unique, removed
