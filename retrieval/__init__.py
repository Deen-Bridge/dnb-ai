"""Shared retrieval infrastructure — chunking, embedding, and a vector store.

This package is the foundation the RAG epic builds on (#1 personal-context, #2
public-knowledge, #3 access-scoped retrieval, #5 hybrid + reranking). It owns:

* :mod:`retrieval.chunking` — deterministic, content-type-aware splitting of the
  bundled corpora into retrievable :class:`~retrieval.chunking.Chunk` spans, each
  carrying stable ``source`` / ``source_id`` / ``content_hash`` metadata plus the
  ``scope`` / ``published`` fields access-scoped retrieval (#3) filters on.
* :mod:`retrieval.index` — a persistent :class:`~retrieval.index.VectorStore`
  abstraction (in-memory fallback + SQLite backend) and
  :class:`~retrieval.index.RetrievalIndex`, which embeds chunks through the
  existing ``semantic_cache`` seam, dedupes unchanged content by hash, and keeps
  the index incrementally in sync as content changes.

Everything runs fully offline in CI through the ``set_fake_embedding`` seam that
already lives in :mod:`semantic_cache`.
"""

from retrieval.chunking import (
    Chunk,
    chunk_prose,
    content_hash,
    iter_corpus_chunks,
    iter_hadith_chunks,
    iter_quran_chunks,
    iter_surah_index_chunks,
)
from retrieval.index import (
    InMemoryVectorStore,
    ReindexResult,
    RetrievalIndex,
    ScoredChunk,
    SQLiteVectorStore,
    VectorStore,
    create_vector_store,
)

__all__ = [
    "Chunk",
    "InMemoryVectorStore",
    "ReindexResult",
    "RetrievalIndex",
    "SQLiteVectorStore",
    "ScoredChunk",
    "VectorStore",
    "chunk_prose",
    "content_hash",
    "create_vector_store",
    "iter_corpus_chunks",
    "iter_hadith_chunks",
    "iter_quran_chunks",
    "iter_surah_index_chunks",
]
