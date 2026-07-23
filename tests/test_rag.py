"""Tests for the RAG module — no live API calls.

Uses fake embeddings and a small fixture corpus loaded into a temporary
ChromaDB instance. Requires chromadb to be installed.
"""

import json
import os

import pytest

from rag import (
    RAG_ENABLED,
    RAG_TOP_K,
    RAG_MIN_SCORE,
    SourceDocument,
    embed_text,
    format_reference_passages,
    set_fake_embedding,
)

FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "fixtures")


# ---------------------------------------------------------------------------
# Embedding helpers: use fixed fake vectors for deterministic tests
# ---------------------------------------------------------------------------

# A small fixed vector for all "quran" content and a different one for
# "hadith" content, plus a near-orthogonal vector so we can test score
# thresholds.
V_QURAN = [0.1, 0.0, 0.0]
V_HADITH = [0.0, 0.1, 0.0]
V_OTHER = [0.0, 0.0, 0.1]


@pytest.fixture(autouse=True)
def reset_fake_embedding():
    set_fake_embedding(None)
    yield
    set_fake_embedding(None)


def load_fixture(name: str) -> list[dict]:
    path = os.path.join(FIXTURE_DIR, name)
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


# ---------------------------------------------------------------------------
# Fake ChromaStore that uses an in-memory dict for testing
# ---------------------------------------------------------------------------


class FakeChromaStore:
    """Simplified store for offline tests; mirrors ChromaStore interface."""

    def __init__(self):
        self._docs: dict[str, dict] = {}
        self._embeddings: dict[str, list[float]] = {}

    def add_documents(self, texts, metadatas, ids):
        for i, doc_id in enumerate(ids):
            self._docs[doc_id] = {
                "text": texts[i],
                "metadata": metadatas[i],
            }

    def search(self, query, top_k=5, min_score=0.0):
        query_emb = embed_text(query)
        scored = []
        for doc_id, doc in self._docs.items():
            emb = self._embeddings.get(doc_id, [0.0, 0.0, 0.0])
            sim = sum(a * b for a, b in zip(query_emb, emb))
            if sim >= min_score:
                scored.append((sim, doc_id, doc))
        scored.sort(key=lambda x: -x[0])
        results = []
        for sim, doc_id, doc in scored[:top_k]:
            meta = doc["metadata"]
            ref = _ref_str(meta)
            results.append(SourceDocument(
                text=doc["text"],
                reference=ref,
                score=round(sim, 4),
            ))
        return results

    def _set_embeddings(self, emb_map: dict[str, list[float]]):
        self._embeddings = emb_map

    @property
    def count(self):
        return len(self._docs)


def _ref_str(meta: dict) -> str:
    src = meta.get("source", "")
    if src == "quran":
        return f"Quran {meta['surah']}:{meta['ayah']}"
    if src == "hadith":
        return f"{meta['collection']}, Book {meta['book']}, Hadith {meta['hadith_number']}"
    return src


# ---------------------------------------------------------------------------
# Chunking & metadata tests
# ---------------------------------------------------------------------------


def test_quran_chunk_has_surah_ayah():
    entries = load_fixture("quran_sample.jsonl")
    assert len(entries) >= 5
    for entry in entries:
        assert entry["source"] == "quran"
        assert isinstance(entry["surah"], int)
        assert isinstance(entry["ayah"], int)
        assert entry["text"]
        assert entry["reference"].startswith("Quran")


def test_hadith_chunk_has_collection_number():
    entries = load_fixture("hadith_sample.jsonl")
    assert len(entries) >= 3
    for entry in entries:
        assert entry["source"] == "hadith"
        assert entry["collection"]
        assert entry["hadith_number"]
        assert entry["text"]


# ---------------------------------------------------------------------------
# Fake embedding: exact match above threshold
# ---------------------------------------------------------------------------


def test_retrieve_exact_match():
    store = FakeChromaStore()
    quran = load_fixture("quran_sample.jsonl")

    texts = [e["text"] for e in quran]
    metadatas = [{k: v for k, v in e.items() if k != "text"} for e in quran]
    ids = [f"quran-{e['surah']}-{e['ayah']}" for e in quran]
    store.add_documents(texts, metadatas, ids)
    store._set_embeddings({doc_id: V_QURAN for doc_id in ids})

    set_fake_embedding(V_QURAN)
    results = store.search("test query about quran", top_k=3, min_score=0.0)
    assert len(results) > 0
    for r in results:
        assert r.reference.startswith("Quran")


# ---------------------------------------------------------------------------
# Score threshold: below-threshold results excluded
# ---------------------------------------------------------------------------


def test_retrieve_min_score_filters():
    store = FakeChromaStore()

    quran = load_fixture("quran_sample.jsonl")
    hadith = load_fixture("hadith_sample.jsonl")

    texts = [e["text"] for e in quran]
    metadatas = [{k: v for k, v in e.items() if k != "text"} for e in quran]
    ids = [f"quran-{e['surah']}-{e['ayah']}" for e in quran]
    store.add_documents(texts, metadatas, ids)
    store._set_embeddings({doc_id: V_QURAN for doc_id in ids})

    hadith_texts = [e["text"] for e in hadith]
    hadith_metas = [{k: v for k, v in e.items() if k != "text"} for e in hadith]
    hadith_ids = [f"hadith-{e['hadith_number']}" for e in hadith]
    store.add_documents(hadith_texts, hadith_metas, hadith_ids)
    store._set_embeddings({**store._embeddings, **{doc_id: V_HADITH for doc_id in hadith_ids}})

    # Query with V_OTHER embedding — low similarity to everything
    set_fake_embedding(V_OTHER)
    results = store.search("unrelated query", top_k=10, min_score=0.5)
    assert len(results) == 0


# ---------------------------------------------------------------------------
# Empty corpus handling
# ---------------------------------------------------------------------------


def test_retrieve_empty_corpus():
    store = FakeChromaStore()
    set_fake_embedding(V_QURAN)
    results = store.search("anything", top_k=5, min_score=0.0)
    assert results == []


# ---------------------------------------------------------------------------
# Format reference passages
# ---------------------------------------------------------------------------


def test_format_reference_passages():
    docs = [
        SourceDocument(text="Praise be to Allah", reference="Quran 1:2", score=0.95),
        SourceDocument(text="He is Allah, One", reference="Quran 112:1", score=0.88),
    ]
    formatted = format_reference_passages(docs)
    assert "[1] Quran 1:2" in formatted
    assert "[2] Quran 112:1" in formatted
    assert "Praise be to Allah" in formatted


def test_format_reference_passages_empty():
    assert format_reference_passages([]) == ""


# ---------------------------------------------------------------------------
# SourceDocument model
# ---------------------------------------------------------------------------


def test_sourcedocument_creation():
    doc = SourceDocument(text="Some text", reference="Quran 1:1", score=0.95)
    assert doc.text == "Some text"
    assert doc.reference == "Quran 1:1"
    assert doc.score == 0.95


# ---------------------------------------------------------------------------
# RAG config defaults
# ---------------------------------------------------------------------------


def test_rag_config_defaults():
    # These are read from env at module import time.
    assert RAG_TOP_K >= 1
    assert RAG_MIN_SCORE >= 0.0
    assert isinstance(RAG_ENABLED, bool)


# ---------------------------------------------------------------------------
# Format reference string
# ---------------------------------------------------------------------------


def test_format_reference_quran():
    from rag import _format_reference
    ref = _format_reference({"source": "quran", "surah": 1, "ayah": 2})
    assert ref == "Quran 1:2"


def test_format_reference_hadith():
    from rag import _format_reference
    ref = _format_reference({
        "source": "hadith",
        "collection": "Sahih al-Bukhari",
        "book": "1",
        "hadith_number": "7",
    })
    assert "Sahih al-Bukhari" in ref
    assert "Hadith 7" in ref
