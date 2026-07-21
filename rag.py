"""Retrieval-augmented generation over Quran and Hadith sources.

Store choice: ChromaDB in embedded (persistent local) mode.
-----------------------------------------------------------
The service is a single Render process with no database add-on.
Qdrant (separate server) and pgvector (needs Postgres) add operational
surface this repo does not have. Chroma runs in-process and persists to
disk. Behind a thin VectorStore interface so a later swap is a one-file
change.
"""

import logging
import os
from typing import Any, Optional

from pydantic import BaseModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Environment configuration
# ---------------------------------------------------------------------------

RAG_ENABLED = os.getenv("RAG_ENABLED", "0").lower() in ("1", "true", "yes")
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "5"))
RAG_MIN_SCORE = float(os.getenv("RAG_MIN_SCORE", "0.0"))
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "chroma_data")
CHROMA_COLLECTION = "islamic_sources"

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


class SourceDocument(BaseModel):
    text: str
    reference: str
    score: float


# ---------------------------------------------------------------------------
# Embedding seam
# ---------------------------------------------------------------------------

_FAKE_EMBEDDING: Optional[list[float]] = None


def set_fake_embedding(vec: Optional[list[float]]) -> None:
    global _FAKE_EMBEDDING
    _FAKE_EMBEDDING = vec


def embed_text(text: str) -> list[float]:
    """Embed a single text string using Gemini text-embedding-004.

    When _FAKE_EMBEDDING is set (tests), return that instead.
    """
    if _FAKE_EMBEDDING is not None:
        return _FAKE_EMBEDDING
    import google.generativeai as genai

    result = genai.embed_content(
        model="models/text-embedding-004",
        content=text,
    )
    return list(result["embedding"])


def embed_batch(texts: list[str]) -> list[list[float]]:
    return [embed_text(t) for t in texts]


# ---------------------------------------------------------------------------
# Vector store
# ---------------------------------------------------------------------------


class VectorStore:
    """Abstract interface for the RAG vector store."""

    def add_documents(
        self,
        texts: list[str],
        metadatas: list[dict[str, Any]],
        ids: list[str],
    ) -> None:
        raise NotImplementedError

    def search(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> list[SourceDocument]:
        raise NotImplementedError

    @property
    def count(self) -> int:
        raise NotImplementedError


class ChromaStore(VectorStore):
    """ChromaDB-backed vector store, embedded/persistent mode."""

    def __init__(self, persist_dir: str = CHROMA_PERSIST_DIR) -> None:
        self._persist_dir = persist_dir
        self._collection: Any = None
        try:
            import chromadb
            from chromadb.config import Settings

            self._client = chromadb.PersistentClient(
                path=persist_dir,
                settings=Settings(anonymized_telemetry=False),
            )
            self._collection = self._client.get_or_create_collection(
                name=CHROMA_COLLECTION,
            )
            logger.info(
                "ChromaDB ready: persist_dir=%s count=%d",
                persist_dir,
                self._collection.count(),
            )
        except Exception as exc:
            logger.warning("ChromaDB init failed: %s — RAG disabled", exc)
            self._collection = None

    @property
    def _ready(self) -> bool:
        return self._collection is not None

    def add_documents(
        self,
        texts: list[str],
        metadatas: list[dict[str, Any]],
        ids: list[str],
    ) -> None:
        if not self._ready:
            logger.warning("Chroma not available — skipping add")
            return
        embeddings = embed_batch(texts)
        self._collection.add(
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
            ids=ids,
        )
        logger.info("Added %d documents to Chroma collection", len(texts))

    def search(
        self,
        query: str,
        top_k: int = RAG_TOP_K,
        min_score: float = RAG_MIN_SCORE,
    ) -> list[SourceDocument]:
        if not self._ready:
            logger.warning("Chroma not available — returning empty results")
            return []

        query_emb = embed_text(query)
        results = self._collection.query(
            query_embeddings=[query_emb],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        if not results["ids"] or not results["ids"][0]:
            return []

        documents: list[SourceDocument] = []
        for i in range(len(results["ids"][0])):
            distance = results["distances"][0][i] if results.get("distances") else 0.0
            # Chroma returns L2-squared by default; convert to a similarity score.
            # Cosine distance = 1 - cosine_similarity, so score = 1 - distance.
            score = max(0.0, 1.0 - distance)
            if score < min_score:
                continue

            meta = results["metadatas"][0][i] if results.get("metadatas") else {}
            reference = _format_reference(meta)
            documents.append(SourceDocument(
                text=results["documents"][0][i],
                reference=reference,
                score=round(score, 4),
            ))

        return documents

    @property
    def count(self) -> int:
        if not self._ready:
            return 0
        return self._collection.count()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _format_reference(meta: dict[str, Any]) -> str:
    source = meta.get("source", "")
    if source == "quran":
        return f"Quran {meta.get('surah', '?')}:{meta.get('ayah', '?')}"
    if source == "hadith":
        parts = [meta.get("collection", "")]
        if meta.get("book"):
            parts.append(f"Book {meta['book']}")
        if meta.get("hadith_number"):
            parts.append(f"Hadith {meta['hadith_number']}")
        return ", ".join(parts)
    return source


def format_reference_passages(docs: list[SourceDocument]) -> str:
    """Format retrieved passages for injection into the model prompt."""
    if not docs:
        return ""
    lines = ["\n\nReference passages:"]
    for i, doc in enumerate(docs, 1):
        lines.append(f"[{i}] {doc.reference} — {doc.text}")
    lines.append(
        "\nWhere relevant, prefer these passages over your training data. "
        "Cite the reference in square brackets.\n"
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Singleton store
# ---------------------------------------------------------------------------

_store: Optional[ChromaStore] = None


def get_store() -> ChromaStore:
    global _store
    if _store is None:
        _store = ChromaStore()
    return _store


def retrieve(query: str) -> list[SourceDocument]:
    """Convenience: search the singleton store."""
    if not RAG_ENABLED:
        logger.info("RAG disabled — skipping retrieval")
        return []
    return get_store().search(query)
