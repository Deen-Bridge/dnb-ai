"""Persistent vector store + the incremental embedding/index pipeline.

Layout mirrors the store patterns already in the repo (``store.py``,
``memory/store.py``): an abstract :class:`VectorStore` with a real, persistent
backend and an in-memory fallback for local/CI, chosen by
:func:`create_vector_store` the way ``create_session_store`` picks its backend.

* :class:`InMemoryVectorStore` — process-local, lost on restart. Keeps CI offline
  and fast; also the backing store the semantic cache migrates onto.
* :class:`SQLiteVectorStore` — durable on disk via stdlib ``sqlite3`` (the same
  dependency-free persistence ``feedback.py`` already uses). Survives process
  restart, needs no external service, and stays deterministic in CI.

Both do brute-force cosine search: at the corpus sizes here that is exact and
cheap, and it keeps the abstraction free of an ANN-index dependency. A
similarity backend (Redis-vector / sqlite-vss) can slot in behind the same
interface later without touching callers.

:class:`RetrievalIndex` is the pipeline on top: it embeds chunks through the
existing ``semantic_cache`` seam, **dedupes by ``content_hash`` so unchanged
content is never re-embedded**, and keeps the index in sync — upserting changed
chunks and deleting removed ``source_id``s — which is what makes both incremental
reindex and the full backfill idempotent.
"""

from __future__ import annotations

import abc
import json
import os
import sqlite3
import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from retrieval.chunking import Chunk

# Type of the embedding seam: text -> vector. Defaults to the existing
# ``semantic_cache.embed_text`` (resolved lazily to avoid an import cycle: the
# semantic cache migrates onto this module).
EmbedFn = Callable[[str], np.ndarray]

# When set, ``create_vector_store`` builds a durable SQLite-backed store at this
# path; otherwise it returns the in-memory fallback (the CI/local default).
RETRIEVAL_INDEX_PATH = os.getenv("RETRIEVAL_INDEX_PATH", "")


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity, defined as 0.0 when either vector has zero norm.

    This is the single implementation shared by every vector search in the
    codebase — the corpus stores here and the semantic cache, which re-exports it
    (retiring its former bespoke copy)."""
    dot = float(np.dot(a, b))
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


@dataclass(frozen=True)
class ScoredChunk:
    """A retrieved chunk and its cosine similarity to the query."""

    chunk: Chunk
    score: float


@dataclass(frozen=True)
class ReindexResult:
    """Outcome of a :meth:`RetrievalIndex.sync` — what changed and what it cost.

    ``embedded`` is the number of embedding calls the sync actually made; it is
    ``added + updated`` and, crucially, does **not** count ``unchanged`` chunks —
    that is the observable proof unchanged content is never re-embedded.
    """

    added: int = 0
    updated: int = 0
    unchanged: int = 0
    removed: int = 0
    embedded: int = 0

    @property
    def changed(self) -> bool:
        return bool(self.added or self.updated or self.removed)


def _as_float32(vector: np.ndarray) -> np.ndarray:
    """Return *vector* as a contiguous float32 array (the stored representation)."""
    arr = np.asarray(vector, dtype=np.float32)
    return np.ascontiguousarray(arr)


# ---------------------------------------------------------------------------
# Abstract store
# ---------------------------------------------------------------------------


class VectorStore(abc.ABC):
    """A persisted set of ``(Chunk, embedding)`` records with cosine search.

    Records are keyed by ``chunk_id``; ``source_id`` groups the chunks of one
    logical record so incremental reindex can delete a source wholesale.
    """

    @abc.abstractmethod
    def upsert(self, chunk: Chunk, embedding: np.ndarray) -> None: ...

    @abc.abstractmethod
    def delete_chunk(self, chunk_id: str) -> bool: ...

    @abc.abstractmethod
    def delete_source(self, source_id: str) -> int: ...

    @abc.abstractmethod
    def get(self, chunk_id: str) -> tuple[Chunk, np.ndarray] | None: ...

    @abc.abstractmethod
    def hashes(self) -> dict[str, str]:
        """Return ``{chunk_id: content_hash}`` for every stored chunk (dedup key)."""

    @abc.abstractmethod
    def query(
        self,
        embedding: np.ndarray,
        *,
        top_k: int = 5,
        min_score: float = 0.0,
        scope: str | None = None,
        published_only: bool = False,
    ) -> list[ScoredChunk]: ...

    @abc.abstractmethod
    def all_chunks(self) -> list[Chunk]: ...

    @abc.abstractmethod
    def source_ids(self) -> set[str]: ...

    @abc.abstractmethod
    def count(self) -> int: ...

    @abc.abstractmethod
    def clear(self) -> None: ...

    def upsert_many(self, items: Iterable[tuple[Chunk, np.ndarray]]) -> None:
        for chunk, embedding in items:
            self.upsert(chunk, embedding)


def _passes_filters(chunk: Chunk, scope: str | None, published_only: bool) -> bool:
    """Shared #3 access filter: scope match and/or published-only."""
    if scope is not None and chunk.scope != scope:
        return False
    if published_only and not chunk.published:
        return False
    return True


def _rank(
    scored: list[ScoredChunk],
    top_k: int,
    min_score: float,
) -> list[ScoredChunk]:
    """Filter by ``min_score``, sort by score desc (chunk_id tiebreak), take top_k."""
    filtered = [s for s in scored if s.score >= min_score]
    filtered.sort(key=lambda s: (-s.score, s.chunk.chunk_id))
    return filtered[:top_k]


# ---------------------------------------------------------------------------
# In-memory backend (fallback; also backs the semantic cache)
# ---------------------------------------------------------------------------


class InMemoryVectorStore(VectorStore):
    """Process-local vector store. Fast, offline, lost on restart."""

    def __init__(self) -> None:
        self._chunks: dict[str, Chunk] = {}
        self._vectors: dict[str, np.ndarray] = {}

    def upsert(self, chunk: Chunk, embedding: np.ndarray) -> None:
        self._chunks[chunk.chunk_id] = chunk
        self._vectors[chunk.chunk_id] = _as_float32(embedding)

    def delete_chunk(self, chunk_id: str) -> bool:
        existed = self._chunks.pop(chunk_id, None) is not None
        self._vectors.pop(chunk_id, None)
        return existed

    def delete_source(self, source_id: str) -> int:
        victims = [cid for cid, chunk in self._chunks.items() if chunk.source_id == source_id]
        for cid in victims:
            self.delete_chunk(cid)
        return len(victims)

    def get(self, chunk_id: str) -> tuple[Chunk, np.ndarray] | None:
        chunk = self._chunks.get(chunk_id)
        if chunk is None:
            return None
        return chunk, self._vectors[chunk_id]

    def hashes(self) -> dict[str, str]:
        return {cid: chunk.content_hash for cid, chunk in self._chunks.items()}

    def query(
        self,
        embedding: np.ndarray,
        *,
        top_k: int = 5,
        min_score: float = 0.0,
        scope: str | None = None,
        published_only: bool = False,
    ) -> list[ScoredChunk]:
        query_vec = _as_float32(embedding)
        scored: list[ScoredChunk] = []
        for cid, chunk in self._chunks.items():
            if not _passes_filters(chunk, scope, published_only):
                continue
            scored.append(ScoredChunk(chunk=chunk, score=cosine_similarity(query_vec, self._vectors[cid])))
        return _rank(scored, top_k, min_score)

    def all_chunks(self) -> list[Chunk]:
        return list(self._chunks.values())

    def source_ids(self) -> set[str]:
        return {chunk.source_id for chunk in self._chunks.values()}

    def count(self) -> int:
        return len(self._chunks)

    def clear(self) -> None:
        self._chunks.clear()
        self._vectors.clear()


# ---------------------------------------------------------------------------
# SQLite backend (durable; survives restart)
# ---------------------------------------------------------------------------

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id      TEXT PRIMARY KEY,
    source        TEXT NOT NULL,
    source_id     TEXT NOT NULL,
    content_hash  TEXT NOT NULL,
    scope         TEXT NOT NULL DEFAULT 'public',
    published     INTEGER NOT NULL DEFAULT 1,
    chunk_json    TEXT NOT NULL,
    dim           INTEGER NOT NULL,
    vector        BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_source_id ON chunks(source_id);
CREATE INDEX IF NOT EXISTS idx_chunks_scope     ON chunks(scope);
"""


class SQLiteVectorStore(VectorStore):
    """Durable vector store backed by stdlib ``sqlite3``.

    Vectors are stored as float32 blobs and matched by brute-force cosine loaded
    in Python — exact and dependency-free. The on-disk file is what lets the
    index survive a process restart; ``chunk_id`` is the primary key, so a
    re-upsert of the same chunk replaces its row rather than duplicating it,
    which is what keeps the backfill idempotent.
    """

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = str(db_path)
        self._local = threading.local()
        self._init_db()

    def _init_db(self) -> None:
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        try:
            conn.executescript(_CREATE_TABLE)
            conn.commit()
        finally:
            conn.close()

    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self._db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            self._local.conn = conn
        return conn

    _UPSERT_SQL = """
        INSERT INTO chunks
            (chunk_id, source, source_id, content_hash, scope, published, chunk_json, dim, vector)
        VALUES (?,?,?,?,?,?,?,?,?)
        ON CONFLICT(chunk_id) DO UPDATE SET
            source       = excluded.source,
            source_id    = excluded.source_id,
            content_hash = excluded.content_hash,
            scope        = excluded.scope,
            published    = excluded.published,
            chunk_json   = excluded.chunk_json,
            dim          = excluded.dim,
            vector       = excluded.vector
    """

    @staticmethod
    def _row(chunk: Chunk, embedding: np.ndarray) -> tuple[object, ...]:
        vector = _as_float32(embedding)
        return (
            chunk.chunk_id,
            chunk.source,
            chunk.source_id,
            chunk.content_hash,
            chunk.scope,
            1 if chunk.published else 0,
            json.dumps(chunk.to_dict(), ensure_ascii=False),
            int(vector.shape[0]),
            vector.tobytes(),
        )

    def upsert(self, chunk: Chunk, embedding: np.ndarray) -> None:
        conn = self._conn()
        conn.execute(self._UPSERT_SQL, self._row(chunk, embedding))
        conn.commit()

    def upsert_many(self, items: Iterable[tuple[Chunk, np.ndarray]]) -> None:
        """Batch upsert in a single transaction — the fast path the backfill uses."""
        rows = [self._row(chunk, embedding) for chunk, embedding in items]
        if not rows:
            return
        conn = self._conn()
        conn.executemany(self._UPSERT_SQL, rows)
        conn.commit()

    def delete_chunk(self, chunk_id: str) -> bool:
        conn = self._conn()
        cur = conn.execute("DELETE FROM chunks WHERE chunk_id = ?", (chunk_id,))
        conn.commit()
        return cur.rowcount > 0

    def delete_source(self, source_id: str) -> int:
        conn = self._conn()
        cur = conn.execute("DELETE FROM chunks WHERE source_id = ?", (source_id,))
        conn.commit()
        return int(cur.rowcount)

    def get(self, chunk_id: str) -> tuple[Chunk, np.ndarray] | None:
        row = (
            self._conn()
            .execute(
                "SELECT chunk_json, vector FROM chunks WHERE chunk_id = ?",
                (chunk_id,),
            )
            .fetchone()
        )
        if row is None:
            return None
        return self._row_to_chunk(row), np.frombuffer(row["vector"], dtype=np.float32)

    def hashes(self) -> dict[str, str]:
        rows = self._conn().execute("SELECT chunk_id, content_hash FROM chunks").fetchall()
        return {row["chunk_id"]: row["content_hash"] for row in rows}

    def query(
        self,
        embedding: np.ndarray,
        *,
        top_k: int = 5,
        min_score: float = 0.0,
        scope: str | None = None,
        published_only: bool = False,
    ) -> list[ScoredChunk]:
        query_vec = _as_float32(embedding)
        sql = "SELECT chunk_json, vector FROM chunks"
        clauses: list[str] = []
        params: list[object] = []
        if scope is not None:
            clauses.append("scope = ?")
            params.append(scope)
        if published_only:
            clauses.append("published = 1")
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        rows = self._conn().execute(sql, params).fetchall()
        scored = [
            ScoredChunk(
                chunk=self._row_to_chunk(row),
                score=cosine_similarity(query_vec, np.frombuffer(row["vector"], dtype=np.float32)),
            )
            for row in rows
        ]
        return _rank(scored, top_k, min_score)

    def all_chunks(self) -> list[Chunk]:
        rows = self._conn().execute("SELECT chunk_json FROM chunks ORDER BY chunk_id").fetchall()
        return [self._row_to_chunk(row) for row in rows]

    def source_ids(self) -> set[str]:
        rows = self._conn().execute("SELECT DISTINCT source_id FROM chunks").fetchall()
        return {row["source_id"] for row in rows}

    def count(self) -> int:
        return int(self._conn().execute("SELECT COUNT(*) FROM chunks").fetchone()[0])

    def clear(self) -> None:
        conn = self._conn()
        conn.execute("DELETE FROM chunks")
        conn.commit()

    @staticmethod
    def _row_to_chunk(row: sqlite3.Row) -> Chunk:
        return Chunk.from_dict(json.loads(row["chunk_json"]))


# ---------------------------------------------------------------------------
# Store factory
# ---------------------------------------------------------------------------


def create_vector_store(index_path: str | None = None) -> VectorStore:
    """Build the vector store, mirroring ``create_session_store``'s tiering.

    ``RETRIEVAL_INDEX_PATH`` (or an explicit *index_path*) selects the durable
    SQLite backend; with neither set the in-memory fallback is returned, which is
    what keeps local development and CI offline and restart-free.
    """
    path = index_path if index_path is not None else RETRIEVAL_INDEX_PATH
    if path:
        return SQLiteVectorStore(path)
    return InMemoryVectorStore()


# ---------------------------------------------------------------------------
# Embedding + incremental index pipeline
# ---------------------------------------------------------------------------


def _default_embed(text: str) -> np.ndarray:
    """Resolve the embedding seam lazily to avoid an import cycle.

    ``semantic_cache`` imports this module (it migrated its store here), so a
    top-level ``from semantic_cache import embed_text`` would be circular. The
    import is deferred to first use instead."""
    from semantic_cache import embed_text

    return embed_text(text)


class RetrievalIndex:
    """Embed-and-sync pipeline over a :class:`VectorStore`.

    :meth:`sync` reconciles the store with a set of desired chunks: it embeds
    only chunks whose ``content_hash`` is new or changed, upserts them, and
    deletes chunks whose ``chunk_id`` is no longer desired. Because chunking and
    hashing are deterministic, calling :meth:`sync` again with the same corpus is
    a no-op (0 embeddings) — the property both incremental reindex and the
    backfill rely on.
    """

    def __init__(self, store: VectorStore, *, embed: EmbedFn | None = None) -> None:
        self._store = store
        self._embed = embed if embed is not None else _default_embed
        # Cumulative embedding-call counter — the observable dedup seam. Tests
        # assert it does not advance when content is unchanged.
        self.embed_calls = 0

    @property
    def store(self) -> VectorStore:
        return self._store

    def _embed_text(self, text: str) -> np.ndarray:
        self.embed_calls += 1
        return self._embed(text)

    def sync(self, chunks: Iterable[Chunk]) -> ReindexResult:
        """Reconcile the store to *chunks*; embed only new/changed content."""
        desired: dict[str, Chunk] = {chunk.chunk_id: chunk for chunk in chunks}
        existing = self._store.hashes()

        added = updated = unchanged = 0
        to_write: list[Chunk] = []
        for chunk_id, chunk in desired.items():
            prior = existing.get(chunk_id)
            if prior is None:
                added += 1
                to_write.append(chunk)
            elif prior != chunk.content_hash:
                updated += 1
                to_write.append(chunk)
            else:
                unchanged += 1

        removed_ids = [chunk_id for chunk_id in existing if chunk_id not in desired]

        # Embed only new/changed chunks, then write them in one batch so a full
        # backfill is a single transaction rather than tens of thousands.
        writes = [(chunk, self._embed_text(chunk.text)) for chunk in to_write]
        self._store.upsert_many(writes)
        for chunk_id in removed_ids:
            self._store.delete_chunk(chunk_id)

        return ReindexResult(
            added=added,
            updated=updated,
            unchanged=unchanged,
            removed=len(removed_ids),
            embedded=len(to_write),
        )

    def upsert_chunk(self, chunk: Chunk) -> ReindexResult:
        """Incrementally index a single chunk (new or edited)."""
        prior = self._store.hashes().get(chunk.chunk_id)
        if prior == chunk.content_hash:
            return ReindexResult(unchanged=1)
        self._store.upsert(chunk, self._embed_text(chunk.text))
        if prior is None:
            return ReindexResult(added=1, embedded=1)
        return ReindexResult(updated=1, embedded=1)

    def delete_source(self, source_id: str) -> ReindexResult:
        """Incrementally drop every chunk of a deleted ``source_id``."""
        removed = self._store.delete_source(source_id)
        return ReindexResult(removed=removed)

    def query(
        self,
        text: str,
        *,
        top_k: int = 5,
        min_score: float = 0.0,
        scope: str | None = None,
        published_only: bool = False,
    ) -> list[ScoredChunk]:
        """Embed *text* and return the most similar chunks."""
        return self._store.query(
            self._embed_text(text),
            top_k=top_k,
            min_score=min_score,
            scope=scope,
            published_only=published_only,
        )
