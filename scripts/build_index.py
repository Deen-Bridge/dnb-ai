"""Backfill the retrieval vector index from the bundled corpora.

Rebuilds the full index deterministically from ``data/`` (surah reference table,
Quran ayat, hadith grading records) by streaming every chunk through
:class:`retrieval.index.RetrievalIndex`. The job is **idempotent**: because
chunking and content hashing are pure, a second run re-embeds nothing and leaves
the index byte-for-byte identical — the invariant issue #88 requires.

Run from the repository root:

    # Durable SQLite index, embeddings via the real text-embedding-004 seam:
    RETRIEVAL_INDEX_PATH=data/retrieval_index.db python scripts/build_index.py

    # Offline demo / CI — deterministic fake embeddings, no network, no key:
    python scripts/build_index.py --index-path /tmp/idx.db --fake-embeddings

Follows the conventions of ``scripts/build_surah_index.py``: deterministic,
no hidden state, and safe to re-run. The index database is a build artifact and
is not checked in (it is large and binary); its provenance is this script plus
the checked-in corpora.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

import numpy as np

# Allow ``python scripts/build_index.py`` from the repo root without installing.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from retrieval.chunking import DATA_DIR, iter_corpus_chunks  # noqa: E402
from retrieval.index import (  # noqa: E402
    ReindexResult,
    RetrievalIndex,
    SQLiteVectorStore,
    VectorStore,
    create_vector_store,
)

# Dimension of the deterministic offline embedding. Only used with
# ``--fake-embeddings``; the real seam returns text-embedding-004's own size.
_FAKE_DIM = 64


def deterministic_embedding(text: str, dim: int = _FAKE_DIM) -> np.ndarray:
    """A pure, offline embedding: same text always yields the same vector.

    Seeded from a SHA-256 of the text so the whole backfill runs without a
    network call or API key — which is what lets CI exercise the pipeline and
    prove idempotency without touching Gemini.
    """
    seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")
    rng = np.random.default_rng(seed)
    return rng.standard_normal(dim).astype(np.float32)


def build_index(
    store: VectorStore,
    *,
    data_dir: Path = DATA_DIR,
    fake_embeddings: bool = False,
    reset: bool = False,
) -> ReindexResult:
    """Sync *store* to the full corpus and return the reindex result."""
    embed = deterministic_embedding if fake_embeddings else None
    index = RetrievalIndex(store, embed=embed)
    if reset:
        store.clear()
    return index.sync(iter_corpus_chunks(data_dir))


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill the retrieval vector index.")
    parser.add_argument(
        "--index-path",
        default=None,
        help="SQLite index path. Defaults to $RETRIEVAL_INDEX_PATH; if neither is set, an "
        "in-memory store is used (build only, nothing persisted).",
    )
    parser.add_argument(
        "--data-dir",
        default=str(DATA_DIR),
        help="Corpus directory to index (default: the bundled data/).",
    )
    parser.add_argument(
        "--fake-embeddings",
        action="store_true",
        help="Use deterministic offline embeddings (no network / API key).",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Clear the index before building (a clean full rebuild).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    store: VectorStore
    if args.index_path:
        store = SQLiteVectorStore(args.index_path)
        where = args.index_path
    else:
        store = create_vector_store()
        env_path = os.environ.get("RETRIEVAL_INDEX_PATH")
        where = env_path if env_path else "in-memory (not persisted)"

    result = build_index(
        store,
        data_dir=Path(args.data_dir),
        fake_embeddings=args.fake_embeddings,
        reset=args.reset,
    )

    print(f"Index target: {where}")
    print(
        "Backfill complete: "
        f"{result.added} added, {result.updated} updated, "
        f"{result.unchanged} unchanged, {result.removed} removed, "
        f"{result.embedded} embedded."
    )
    print(f"Index now holds {store.count()} chunks across {len(store.source_ids())} sources.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
