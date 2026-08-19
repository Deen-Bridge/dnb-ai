# Retrieval infrastructure

This document describes the shared retrieval pipeline in [`retrieval/`](../retrieval):
the chunk schema, the vector-store abstraction, how the index stays in sync as
content changes, and the idempotent backfill job. It is the foundation the RAG
epic depends on (#1 personal-context, #2 public-knowledge, #3 access-scoped
retrieval, #5 hybrid + reranking) — it delivers the pipeline and nothing
product-facing on top of it.

## Overview

```
corpora (data/)  ─chunking─▶  Chunk[]  ─embed (content-hash dedup)─▶  VectorStore
                                                                         │
                                            query(text) ──cosine top-k──┘
```

- **Chunking** ([`retrieval/chunking.py`](../retrieval/chunking.py)) turns the
  bundled corpora into retrievable `Chunk` spans, deterministically.
- **Embedding** reuses the existing `text-embedding-004` seam
  (`semantic_cache.embed_text`), and **dedupes by `content_hash`** so unchanged
  content is never re-embedded.
- **Vector store** ([`retrieval/index.py`](../retrieval/index.py)) persists
  `(Chunk, embedding)` records and answers cosine top-k queries. A durable SQLite
  backend and an in-memory fallback sit behind one `VectorStore` interface.
- **`RetrievalIndex`** is the pipeline that embeds, dedupes, and keeps the store
  in sync — the same machinery both incremental reindex and the full backfill
  use.

## Chunk schema

A **chunk** is the smallest retrievable unit the index stores. Produced by
`retrieval.chunking`, every chunk is a frozen dataclass with these fields:

| Field          | Type              | Meaning |
|----------------|-------------------|---------|
| `chunk_id`     | `str`             | Unique id within the index. Equals `source_id` for an atomic record, or `"{source_id}#{part}"` for a split prose part. |
| `source`       | `str`             | Corpus family: `"quran"`, `"hadith"`, `"surah_index"`, … |
| `source_id`    | `str`             | Stable id of the **logical record** the chunk came from (e.g. `"quran:2:255"`, `"hadith:bukhari:1"`). Incremental reindex upserts/deletes by this id. |
| `text`         | `str`             | The embeddable text. |
| `content_hash` | `str`             | SHA-256 over the normalized `text`. The dedup key: an unchanged hash means no re-embed; a changed hash replaces the old vector. |
| `scope`        | `str`             | Access scope. `"public"` for the bundled corpora; #3 emits private/per-user scopes through this same field. |
| `published`    | `bool`            | Publish flag. Retrieval can filter to `published_only`. |
| `part`         | `int`             | Part index within `source_id` (0 for atomic records). |
| `part_count`   | `int`             | Number of parts `source_id` was split into (1 for atomic records). |
| `metadata`     | `dict[str, Any]`  | Source-specific fields (surah/ayah numbers, hadith grade/chain, surah aliases, …). |

`scope` and `published` are set at chunk-creation time so the pipeline is ready
for **access-scoped retrieval (#3)** without a later schema change. `VectorStore.query`
already honors both:

```python
store.query(embedding, top_k=5, scope="public", published_only=True)
```

### Chunking rules

1. **Reference records stay atomic.** A Quran ayah (`chunk_ayah`), a hadith
   grading record (`chunk_hadith`), and a surah reference-table entry
   (`chunk_surah`) each become exactly one chunk — splitting `2:255` across
   chunks would let a search return half a verse.
2. **Long prose is split by a token budget with overlap** (`chunk_prose`). Text
   longer than `max_tokens` (default 256) is windowed with `overlap` tokens
   (default 32) of carry-over, so content on a boundary is retrievable from
   either side. Tokens are estimated by whitespace splitting — deterministic and
   dependency-free.

Chunking is **pure and deterministic**: `iter_corpus_chunks()` walks every
corpus in sorted order and hashing has no hidden state, so a backfill produces
byte-identical chunks run to run. That determinism is what makes the index
idempotent.

## Vector store

`VectorStore` (in [`retrieval/index.py`](../retrieval/index.py)) is the
abstraction; `create_vector_store()` selects the backend the way
`create_session_store` does:

| Backend                | Selected when                   | Persistence | Use |
|------------------------|---------------------------------|-------------|-----|
| `SQLiteVectorStore`    | `RETRIEVAL_INDEX_PATH` is set   | Durable on disk (stdlib `sqlite3`) | Production / any run that must survive a restart |
| `InMemoryVectorStore`  | otherwise (the default)         | Process-local, lost on restart | Local dev and CI — offline, fast |

Both do **brute-force cosine** search (loaded in Python). At the corpus sizes
here that is exact and cheap, and it keeps the abstraction free of an ANN-index
dependency; a similarity backend (Redis-vector / sqlite-vss) can slot in behind
the same interface later without touching callers. `cosine_similarity` lives once,
in `retrieval.index`, and is shared by every vector search in the codebase — the
semantic cache re-exports it and now stores its entries in an
`InMemoryVectorStore`, retiring its former hand-rolled linear scan.

Core interface:

```python
store.upsert(chunk, embedding)          # add or replace by chunk_id
store.delete_chunk(chunk_id)            # -> bool
store.delete_source(source_id)          # -> int (all parts of a record)
store.query(embedding, top_k=5, min_score=0.0, scope=None, published_only=False)
store.hashes()                          # {chunk_id: content_hash} — the dedup map
store.count(); store.source_ids(); store.all_chunks(); store.clear()
```

## Incremental reindex & dedup

`RetrievalIndex.sync(chunks)` reconciles the store to a desired set of chunks:

- **added** — a `chunk_id` the store has never seen → embed + upsert.
- **updated** — a known `chunk_id` whose `content_hash` changed → re-embed +
  replace the old vector.
- **unchanged** — a known `chunk_id` with the same `content_hash` → **skipped, no
  embedding call** (dedup, ties into backlog #13).
- **removed** — a stored `chunk_id` no longer desired → deleted.

The result is a `ReindexResult(added, updated, unchanged, removed, embedded)`.
`embedded` counts the actual embedding calls and never includes `unchanged`
chunks — the observable proof that unchanged content is not re-embedded.

Single-record helpers back the "content change reflected within a target window"
path: `index.upsert_chunk(chunk)` (new/edited) and `index.delete_source(source_id)`
(deleted). Because the store is queried live, an upsert or delete is reflected in
the very next `query()`.

## Backfill

[`scripts/build_index.py`](../scripts/build_index.py) rebuilds the full index
from `data/`, following `scripts/build_surah_index.py` conventions
(deterministic, no hidden state, safe to re-run):

```bash
# Durable SQLite index, real embeddings:
RETRIEVAL_INDEX_PATH=data/retrieval_index.db python scripts/build_index.py

# Offline demo / CI — deterministic fake embeddings, no network or API key:
python scripts/build_index.py --index-path /tmp/idx.db --fake-embeddings
```

**Idempotency:** running the backfill twice re-embeds nothing and leaves the
index byte-for-byte identical (same `content_hash` map, same counts). Two clean
builds of the same corpus into separate stores produce the same index. Both
properties are covered by `tests/test_build_index.py`.

The index database is a **build artifact** — large and binary — and is not
checked in (see [`.gitignore`](../.gitignore)); its provenance is this script
plus the checked-in corpora.

## Testing

Everything runs **fully offline** via the embedding seam:

- `tests/test_retrieval_chunking.py` — determinism, atomic vs. prose splitting,
  overlap, corpus iterators.
- `tests/test_retrieval_index.py` — both store backends, cosine ranking, scope /
  published / `min_score` filters, SQLite persistence across a restart, and the
  sync pipeline (dedup, content-change reflection, deletion, idempotency).
- `tests/test_build_index.py` — the backfill: indexes every corpus, idempotency,
  cross-process match, restart survival.
- `tests/test_semantic_cache.py` — unchanged; proves the cache migration onto the
  shared store did not regress its behavior.

Tests inject a deterministic hash embedder (same text → same vector) so no
network call is ever made.
