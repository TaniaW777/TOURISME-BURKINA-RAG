# -*- coding: utf-8 -*-
"""
scripts/build_index.py
======================
Construction de l'index vectoriel Phase 3 — ChromaDB v2.

  data/chunks.jsonl ─► validation stricte ─► embeddings MiniLM (normalisés)
        ─► data/chroma_db_v2 (collection recréée de zéro à chaque build)
        ─► vérification post-build ─► data/index_manifest.json (atomique,
            écrit SEULEMENT si tout a réussi)

Politique de reconstruction : RÉCRÉATION COMPLÈTE. La collection v2 est
supprimée puis reconstruite à chaque exécution — un build interrompu
laisse un index partiel SANS manifest : il ne peut pas être confondu
avec un build complet (l'absence de manifest invalide l'index).

PROTECTION DU LEGACY : le builder refuse catégoriquement d'écrire dans
data/chroma_db (l'index 785 documents qui sert encore à l'application
existante). Aucune migration, aucune modification.

Usage :
    python scripts/build_index.py [--chunks data/chunks.jsonl]
                                  [--index-path data/chroma_db_v2]

Aucun réseau au runtime (pas de scraping) : le modèle d'embeddings est
chargé depuis le cache HF local.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import chromadb
from loguru import logger

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.config import load_settings  # noqa: E402

# ──────────────────────────────────────────────
# CONSTANTES PHASE 3
# ──────────────────────────────────────────────
DEFAULT_CHUNKS_PATH = REPO_ROOT / "data" / "chunks.jsonl"
DEFAULT_INDEX_PATH = REPO_ROOT / "data" / "chroma_db_v2"


def _repo_relative(path: Path) -> str:
    """Chemin de manifeste : relatif au repo (POSIX) si dedans, sinon absolu.

    Un manifeste ne doit jamais dépendre du filesystem d'un développeur :
    à l'intérieur du repo on écrit 'data/chroma_db_v2', pas
    'C:\\Users\\...\\chroma_db_v2'.
    """
    resolved = path.resolve()
    root = REPO_ROOT.resolve()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError:
        return resolved.as_posix()  # hors repo : absolu, mais POSIX


def _resolve_manifest_path(raw: str) -> Path:
    """Résout un chemin de manifeste : relatif → contre REPO_ROOT ;
    absolu (manifestes legacy) → tel quel."""
    p = Path(raw)
    return p if p.is_absolute() else (REPO_ROOT / p).resolve()
LEGACY_INDEX_PATHS = {  # intouchables, en chemin absolu résolu
    (REPO_ROOT / "data" / "chroma_db").resolve(),
}
V2_COLLECTION_NAME = "tourisme_burkina_v2"
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"  # ne change JAMAIS silencieusement
MODEL_MAX_SEQ_LENGTH = 128  # vérifié : sentence_bert_config.json (Phase 2)
BATCH_SIZE = 32

REQUIRED_FIELDS = {"id", "parent_id", "url", "title", "source_type",
                   "language", "trust", "text", "chunk_idx", "token_count"}
VALID_SOURCE_TYPES = {"web", "pdf", "pdf_local"}
VALID_LANGUAGES = {"fr", "en", "unknown"}
MIN_TEXT_CHARS = 30  # plus permissif que l'ingestion (60) : on ne rejette
# pas un corpus déjà construit pour des queues de document un peu courtes,
# mais on refuse le vide / le non-sens.


class ValidationError(Exception):
    """Entrée de corpus invalide — le build échoue, rien n'est indexé."""


class SafetyError(Exception):
    """Opération qui toucherait l'index legacy — refusée avant toute écriture."""


class EmbeddingError(Exception):
    """Échec de génération d'embeddings — build interrompu, pas de manifest."""


class TokenCounter(Protocol):
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]: ...


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


# ──────────────────────────────────────────────
# EMBEDDER RÉEL — MiniLM, normalisé (compatible cosine)
# ──────────────────────────────────────────────
class MiniLMEmbedder:
    def __init__(self, model_name: str = EMBEDDING_MODEL, device: str = "cpu"):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self._model = SentenceTransformer(model_name, device=device)

    def embed(self, texts: list[str]) -> list[list[float]]:
        embs = self._model.encode(texts, normalize_embeddings=True,
                                  show_progress_bar=False)
        return embs.tolist()


def real_token_counter() -> TokenCounter:
    from ingestion.chunker import TransformersTokenCounter

    return TransformersTokenCounter(f"sentence-transformers/{EMBEDDING_MODEL}")


# ──────────────────────────────────────────────
# VALIDATION STRICTE DU CORPUS
# ──────────────────────────────────────────────
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def validate_chunks(chunks: list[dict], counter: TokenCounter) -> list[str]:
    """Valide chaque chunk ; lève ValidationError avec le détail complet.

    Un seul chunk invalide fait échouer tout le build : jamais de saut
    silencieux (contrainte Phase 3).
    """
    errors: list[str] = []
    seen_ids: dict[str, int] = {}

    for i, c in enumerate(chunks):
        where = f"chunk[{i}] id={c.get('id', '?')}"
        missing = REQUIRED_FIELDS - set(c.keys())
        if missing:
            errors.append(f"{where}: champs manquants {sorted(missing)}")
            continue
        if not isinstance(c["text"], str) or len(c["text"].strip()) < MIN_TEXT_CHARS:
            errors.append(f"{where}: texte vide ou < {MIN_TEXT_CHARS} caractères")
        if not isinstance(c["id"], str) or not c["id"]:
            errors.append(f"{where}: id invalide")
        elif c["id"] in seen_ids:
            errors.append(f"{where}: id dupliqué (déjà vu en position {seen_ids[c['id']]})")
        else:
            seen_ids[c["id"]] = i
        if c["source_type"] not in VALID_SOURCE_TYPES:
            errors.append(f"{where}: source_type invalide {c['source_type']!r}")
        if c["language"] not in VALID_LANGUAGES:
            errors.append(f"{where}: language invalide {c['language']!r}")
        if not isinstance(c["url"], str) or not c["url"]:
            errors.append(f"{where}: url manquante")
        # limite de tokens réelle — le compte stocké doit être honnête
        if counter is not None:
            actual = len(counter.encode(c["text"]))
            if actual + 2 > MODEL_MAX_SEQ_LENGTH:
                errors.append(
                    f"{where}: {actual + 2} tokens > limite modèle {MODEL_MAX_SEQ_LENGTH}"
                )
            elif isinstance(c["token_count"], int) and c["token_count"] != actual:
                errors.append(f"{where}: token_count stocké {c['token_count']} != réel {actual}")

    if errors:
        # on rapporte tout (max 20 pour rester lisible), on lève
        shown = "\n  ".join(errors[:20])
        more = f"\n  ... et {len(errors) - 20} autres erreurs" if len(errors) > 20 else ""
        raise ValidationError(f"{len(errors)} chunks invalides :\n  {shown}{more}")
    return []


def load_chunks(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"corpus introuvable : {path}")
    chunks: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                chunks.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValidationError(f"JSON invalide ligne {line_no} de {path} : {e}")
    if not chunks:
        raise ValidationError(f"corpus vide : {path}")
    return chunks


# ──────────────────────────────────────────────
# CONSTRUCTION
# ──────────────────────────────────────────────
@dataclass
class BuildResult:
    indexed_count: int
    embedding_dim: int
    index_path: Path
    manifest_path: Path


def assert_safe_index_path(index_path: Path) -> None:
    resolved = index_path.resolve()
    for legacy in LEGACY_INDEX_PATHS:
        if resolved == legacy or legacy in resolved.parents or resolved in legacy.parents:
            raise SafetyError(
                f"REFUS : {index_path} chevauche l'index legacy {legacy} "
                "(785 documents, application existante). L'index v2 doit être "
                "un chemin distinct (ex. data/chroma_db_v2)."
            )


def build_index(
    chunks_path: Path = DEFAULT_CHUNKS_PATH,
    index_path: Path = DEFAULT_INDEX_PATH,
    embedder: Embedder | None = None,
    counter: TokenCounter | None = None,
    batch_size: int = BATCH_SIZE,
) -> BuildResult:
    """Construit l'index v2 complet. Levée d'exception ⇒ pas de manifest."""
    assert_safe_index_path(index_path)

    embedder = embedder or MiniLMEmbedder()
    counter = counter or real_token_counter()

    # 1. charger + valider (échec ici ⇒ aucune écriture)
    chunks = load_chunks(chunks_path)
    logger.info(f"Validation de {len(chunks)} chunks (tokenizer réel)...")
    validate_chunks(chunks, counter)
    corpus_sha = sha256_file(chunks_path)

    # 2. recréation from scratch de la collection v2.
    # NB Windows : chromadb garde des verrous sur ses fichiers — on supprime
    # la collection via le client (libère les verrous) plutôt que rmtree du
    # répertoire, qui échouerait avec PermissionError WinError 32.
    client = chromadb.PersistentClient(path=str(index_path))
    try:
        client.delete_collection(V2_COLLECTION_NAME)
        logger.warning(f"Collection v2 existante supprimée (reconstruction complète) : {V2_COLLECTION_NAME}")
    except Exception:
        pass  # collection absente : premier build
    collection = client.create_collection(
        name=V2_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine", "embedding_model": EMBEDDING_MODEL},
    )

    # 3. embeddings par lots + écriture ; toute erreur interrompt tout
    try:
        dim: int | None = None
        indexed = 0
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            embeddings = embedder.embed([c["text"] for c in batch])
            if len(embeddings) != len(batch):
                raise EmbeddingError(
                    f"batch {i // batch_size}: {len(embeddings)} embeddings "
                    f"pour {len(batch)} chunks"
                )
            batch_dim = len(embeddings[0])
            if dim is None:
                dim = batch_dim
            elif any(len(e) != dim for e in embeddings):
                raise EmbeddingError(f"batch {i // batch_size}: dimensions incohérentes")
            collection.add(
                ids=[c["id"] for c in batch],
                embeddings=embeddings,
                documents=[c["text"] for c in batch],
                metadatas=[{
                    "url": c["url"],
                    "title": c["title"],
                    "source_type": c["source_type"],
                    "language": c["language"],
                    "trust": c["trust"],
                    "parent_id": c["parent_id"],
                    "chunk_idx": c["chunk_idx"],
                    "token_count": c["token_count"],
                } for c in batch],
            )
            indexed += len(batch)
            logger.info(f"indexés {indexed}/{len(chunks)}")
    except EmbeddingError:
        raise
    except Exception as e:
        raise EmbeddingError(f"échec d'écriture/embedding au batch {indexed}: {e}") from e

    # 4. vérification post-build : le compte doit être exact
    actual = collection.count()
    if actual != len(chunks):
        raise EmbeddingError(f"écriture incomplète : {actual}/{len(chunks)} chunks dans l'index")

    # 5. manifest — écriture atomique (tmp + rename), seulement maintenant
    manifest = {
        "embedding_model": EMBEDDING_MODEL,
        "collection_name": V2_COLLECTION_NAME,
        "index_path": _repo_relative(index_path),
        "corpus_path": _repo_relative(chunks_path),
        "corpus_sha256": corpus_sha,
        "source_chunk_count": len(chunks),
        "indexed_count": indexed,
        "embedding_dim": dim,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model_max_seq_length": MODEL_MAX_SEQ_LENGTH,
        "distance_metric": "cosine",
        "normalized_embeddings": True,
        "batch_size": batch_size,
    }
    manifest_path = index_path.parent / "index_manifest.json"
    tmp = manifest_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(manifest_path)

    logger.success(f"Index v2 prêt : {indexed} chunks, dim={dim}, manifest={manifest_path}")
    return BuildResult(indexed, dim, index_path, manifest_path)


# ──────────────────────────────────────────────
# VALIDATION MANIFEST ↔ INDEX
# ──────────────────────────────────────────────
def verify_manifest(manifest_path: Path, index_path: Path | None = None) -> dict:
    """Vérifie que le manifest correspond à l'index réellement présent."""
    if not manifest_path.exists():
        raise ValidationError(f"manifest absent : {manifest_path} (build incomplet ou échoué)")
    m = json.loads(manifest_path.read_text(encoding="utf-8"))
    idx = _resolve_manifest_path(m["index_path"]) if index_path is None else index_path
    client = chromadb.PersistentClient(path=str(idx))
    try:
        col = client.get_collection(m["collection_name"])
    except Exception as e:
        raise ValidationError(f"collection {m['collection_name']!r} introuvable dans {idx} : {e}")
    count = col.count()
    if count != m["indexed_count"] or count != m["source_chunk_count"]:
        raise ValidationError(
            f"manifest incohérent : annonce {m['indexed_count']}, index contient {count}"
        )
    # dimension réelle d'un échantillon
    sample = col.get(limit=3, include=["embeddings"])
    dims = {len(e) for e in sample["embeddings"]}
    if dims != {m["embedding_dim"]}:
        raise ValidationError(f"dimensions réelles {dims} != manifest {m['embedding_dim']}")
    if sha256_file(_resolve_manifest_path(m["corpus_path"])) != m["corpus_sha256"]:
        raise ValidationError("le corpus source a changé depuis le build (sha256 ≠ manifest)")
    return m


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build index v2 (Phase 3)")
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS_PATH)
    parser.add_argument("--index-path", type=Path, default=DEFAULT_INDEX_PATH)
    args = parser.parse_args()

    result = build_index(chunks_path=args.chunks, index_path=args.index_path)
    verified = verify_manifest(result.manifest_path)
    print(f"\nVérification manifest OK : {verified['indexed_count']} chunks, "
          f"dim={verified['embedding_dim']}, modèle={verified['embedding_model']}")
