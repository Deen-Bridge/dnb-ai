"""Tests for retrieval.index — vector store backends and the sync pipeline.

Fully offline: a deterministic hash embedder stands in for text-embedding-004, so
the same text always maps to the same vector and no network call is made.
"""

import hashlib
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

from retrieval.chunking import make_chunk
from retrieval.index import (
    InMemoryVectorStore,
    RetrievalIndex,
    SQLiteVectorStore,
    VectorStore,
    cosine_similarity,
    create_vector_store,
)

EMBED_DIM = 24


def fake_embed(text: str) -> np.ndarray:
    """Deterministic offline embedding: identical text -> identical vector."""
    seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")
    rng = np.random.default_rng(seed)
    return rng.standard_normal(EMBED_DIM).astype(np.float32)


def chunk(source_id: str, text: str, *, scope: str = "public", published: bool = True):
    return make_chunk(source="doc", source_id=source_id, text=text, scope=scope, published=published)


# ---------------------------------------------------------------------------
# cosine_similarity
# ---------------------------------------------------------------------------


def test_cosine_similarity_basic():
    a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    b = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    c = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    assert cosine_similarity(a, b) == pytest.approx(1.0)
    assert cosine_similarity(a, c) == pytest.approx(0.0)


def test_cosine_similarity_zero_vector_is_zero():
    z = np.zeros(3, dtype=np.float32)
    assert cosine_similarity(z, np.array([1.0, 0.0, 0.0], dtype=np.float32)) == 0.0


# ---------------------------------------------------------------------------
# Store behavior — parametrized across both backends
# ---------------------------------------------------------------------------


@pytest.fixture(params=["memory", "sqlite"])
def store(request, tmp_path: Path) -> VectorStore:
    if request.param == "memory":
        return InMemoryVectorStore()
    return SQLiteVectorStore(tmp_path / "index.db")


def test_upsert_get_count(store: VectorStore):
    c = chunk("doc:1", "hello world")
    store.upsert(c, fake_embed(c.text))
    assert store.count() == 1
    got = store.get("doc:1")
    assert got is not None
    assert got[0].text == "hello world"


def test_upsert_replaces_same_chunk_id(store: VectorStore):
    store.upsert(chunk("doc:1", "v1"), fake_embed("v1"))
    store.upsert(chunk("doc:1", "v2"), fake_embed("v2"))
    assert store.count() == 1
    got = store.get("doc:1")
    assert got is not None and got[0].text == "v2"


def test_query_returns_most_similar_first(store: VectorStore):
    for sid, text in [("doc:1", "prayer times"), ("doc:2", "zakat nisab"), ("doc:3", "hajj rites")]:
        store.upsert(chunk(sid, text), fake_embed(text))
    results = store.query(fake_embed("zakat nisab"), top_k=3)
    assert results[0].chunk.source_id == "doc:2"
    assert results[0].score == pytest.approx(1.0, abs=1e-5)


def test_query_top_k_limits_results(store: VectorStore):
    for i in range(5):
        store.upsert(chunk(f"doc:{i}", f"text {i}"), fake_embed(f"text {i}"))
    assert len(store.query(fake_embed("text 0"), top_k=2)) == 2


def test_query_min_score_floor(store: VectorStore):
    store.upsert(chunk("doc:1", "alpha"), fake_embed("alpha"))
    # An orthogonal-ish query below the floor returns nothing.
    assert store.query(fake_embed("totally different"), top_k=5, min_score=0.999) == []


def test_query_scope_filter(store: VectorStore):
    store.upsert(chunk("doc:1", "public one", scope="public"), fake_embed("public one"))
    store.upsert(chunk("doc:2", "private one", scope="private"), fake_embed("private one"))
    results = store.query(fake_embed("private one"), top_k=5, scope="public")
    assert all(r.chunk.scope == "public" for r in results)
    assert "doc:2" not in {r.chunk.source_id for r in results}


def test_query_published_only_filter(store: VectorStore):
    store.upsert(chunk("doc:1", "draft", published=False), fake_embed("draft"))
    results = store.query(fake_embed("draft"), top_k=5, published_only=True)
    assert results == []


def test_delete_chunk(store: VectorStore):
    store.upsert(chunk("doc:1", "x"), fake_embed("x"))
    assert store.delete_chunk("doc:1") is True
    assert store.delete_chunk("doc:1") is False
    assert store.count() == 0


def test_delete_source_removes_all_parts(store: VectorStore):
    store.upsert(make_chunk(source="d", source_id="d:1", text="a", part=0, part_count=2), fake_embed("a"))
    store.upsert(make_chunk(source="d", source_id="d:1", text="b", part=1, part_count=2), fake_embed("b"))
    store.upsert(chunk("d:2", "c"), fake_embed("c"))
    assert store.delete_source("d:1") == 2
    assert store.source_ids() == {"d:2"}


def test_hashes_maps_chunk_id_to_content_hash(store: VectorStore):
    c = chunk("doc:1", "content")
    store.upsert(c, fake_embed(c.text))
    assert store.hashes() == {"doc:1": c.content_hash}


def test_clear(store: VectorStore):
    store.upsert(chunk("doc:1", "x"), fake_embed("x"))
    store.clear()
    assert store.count() == 0


# ---------------------------------------------------------------------------
# SQLite persistence — survives a "restart" (a fresh store on the same file)
# ---------------------------------------------------------------------------


def test_sqlite_survives_restart(tmp_path: Path):
    db = tmp_path / "persist.db"
    first = SQLiteVectorStore(db)
    c = chunk("doc:1", "durable content")
    first.upsert(c, fake_embed(c.text))
    assert first.count() == 1

    # Simulate a process restart: a brand-new store object on the same file.
    second = SQLiteVectorStore(db)
    assert second.count() == 1
    results = second.query(fake_embed("durable content"), top_k=1)
    assert results and results[0].chunk.source_id == "doc:1"
    assert results[0].score == pytest.approx(1.0, abs=1e-5)


def test_create_vector_store_factory(tmp_path: Path):
    assert isinstance(create_vector_store(), InMemoryVectorStore)
    assert isinstance(create_vector_store(str(tmp_path / "x.db")), SQLiteVectorStore)


# ---------------------------------------------------------------------------
# RetrievalIndex — incremental sync, dedup, content-change reflection
# ---------------------------------------------------------------------------


def make_index(store: VectorStore) -> tuple[RetrievalIndex, Mock]:
    embed = Mock(side_effect=fake_embed)
    return RetrievalIndex(store, embed=embed), embed


def test_sync_adds_new_chunks(store: VectorStore):
    index, embed = make_index(store)
    result = index.sync([chunk("d:1", "a"), chunk("d:2", "b")])
    assert (result.added, result.updated, result.unchanged, result.removed) == (2, 0, 0, 0)
    assert result.embedded == 2
    assert embed.call_count == 2
    assert store.count() == 2


def test_unchanged_content_is_not_reembedded(store: VectorStore):
    """AC: same content_hash triggers NO re-embedding call."""
    index, embed = make_index(store)
    chunks = [chunk("d:1", "stable text"), chunk("d:2", "other text")]
    index.sync(chunks)
    assert embed.call_count == 2

    result = index.sync(chunks)  # identical corpus
    assert (result.added, result.updated, result.unchanged) == (0, 0, 2)
    assert result.embedded == 0
    assert embed.call_count == 2  # seam NOT invoked again
    assert index.embed_calls == 2


def test_content_change_is_reflected_in_retrieval(store: VectorStore):
    """AC: an edited source_id is reflected in retrieval results."""
    index, embed = make_index(store)
    index.sync([chunk("d:1", "prayer at dawn")])

    before = index.query("prayer at dawn", top_k=1)
    assert before[0].score == pytest.approx(1.0, abs=1e-5)

    # Edit the same source_id (new content_hash) and re-sync.
    result = index.sync([chunk("d:1", "prayer at dusk")])
    assert result.updated == 1 and result.embedded == 1
    assert store.count() == 1  # replaced, not duplicated

    after = index.query("prayer at dusk", top_k=1)
    assert after[0].chunk.source_id == "d:1"
    assert after[0].score == pytest.approx(1.0, abs=1e-5)
    # The old text no longer matches perfectly.
    assert index.query("prayer at dawn", top_k=1)[0].score < 0.999


def test_deleted_source_is_removed_on_sync(store: VectorStore):
    """AC: a deleted source_id disappears from retrieval."""
    index, _ = make_index(store)
    index.sync([chunk("d:1", "keep"), chunk("d:2", "drop")])
    result = index.sync([chunk("d:1", "keep")])  # d:2 no longer desired
    assert result.removed == 1
    assert store.source_ids() == {"d:1"}


def test_incremental_upsert_and_delete_helpers(store: VectorStore):
    index, _ = make_index(store)
    added = index.upsert_chunk(chunk("d:1", "hello"))
    assert added.added == 1 and added.embedded == 1

    same = index.upsert_chunk(chunk("d:1", "hello"))  # unchanged
    assert same.unchanged == 1 and same.embedded == 0

    edited = index.upsert_chunk(chunk("d:1", "hello there"))
    assert edited.updated == 1 and edited.embedded == 1

    removed = index.delete_source("d:1")
    assert removed.removed == 1
    assert store.count() == 0


def test_sync_is_idempotent(store: VectorStore):
    """AC: re-running the sync yields identical contents and no work."""
    index, _ = make_index(store)
    chunks = [chunk(f"d:{i}", f"text number {i}") for i in range(6)]
    index.sync(chunks)
    snapshot = sorted(store.hashes().items())

    result = index.sync(chunks)
    assert not result.changed
    assert sorted(store.hashes().items()) == snapshot
    assert store.count() == 6
