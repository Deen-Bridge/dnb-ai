"""Semantic response cache with embedding-similarity matching.

Cache correctness invariant
---------------------------
A follow-up question depends on conversation history, so its answer must
never be served to someone else. Therefore we consult/populate the cache
*only* when the chat has no prior history (new chat_id / first turn) AND
request.context is None.

Store choice
------------
In-memory store with numpy for cosine similarity. Chosen over ChromaDB to
keep dependencies minimal — numpy alone is sufficient for this use case,
and avoids coupling to ChromaDB's full vector-store infrastructure. If the
RAG infrastructure lands with ChromaDB, the cache can be migrated to share
its collection.
"""

import logging
import os
import time
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Environment configuration
# ---------------------------------------------------------------------------

SEMANTIC_CACHE_ENABLED = os.getenv("SEMANTIC_CACHE_ENABLED", "0").lower() in (
    "1",
    "true",
    "yes",
)
SEMANTIC_CACHE_THRESHOLD = float(os.getenv("SEMANTIC_CACHE_THRESHOLD", "0.95"))
SEMANTIC_CACHE_TTL_SECONDS = int(os.getenv("SEMANTIC_CACHE_TTL_SECONDS", "86400"))
SEMANTIC_CACHE_MAX_ENTRIES = int(os.getenv("SEMANTIC_CACHE_MAX_ENTRIES", "1000"))

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    dot = float(np.dot(a, b))
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def normalize_text(text: str) -> str:
    return " ".join(text.casefold().split())


# ---------------------------------------------------------------------------
# Embedding seam
# ---------------------------------------------------------------------------

_FAKE_EMBEDDING: Optional[np.ndarray] = None


def set_fake_embedding(vec: Optional[np.ndarray]) -> None:
    global _FAKE_EMBEDDING
    _FAKE_EMBEDDING = vec


def embed_text(text: str) -> np.ndarray:
    if _FAKE_EMBEDDING is not None:
        return _FAKE_EMBEDDING
    import google.generativeai as genai

    result = genai.embed_content(
        model="models/text-embedding-004",
        content=text,
    )
    return np.array(result["embedding"], dtype=np.float32)


# ---------------------------------------------------------------------------
# Cache entry
# ---------------------------------------------------------------------------


class CacheEntry:
    __slots__ = ("embedding", "response", "chat_id", "history", "expires_at")

    def __init__(
        self,
        embedding: np.ndarray,
        response: str,
        chat_id: str,
        history: list[Any],
        expires_at: float,
    ) -> None:
        self.embedding = embedding
        self.response = response
        self.chat_id = chat_id
        self.history = history
        self.expires_at = expires_at

    @property
    def expired(self) -> bool:
        return time.time() > self.expires_at


# ---------------------------------------------------------------------------
# In-memory semantic cache
# ---------------------------------------------------------------------------


class SemanticCache:
    def __init__(self) -> None:
        self._entries: list[CacheEntry] = []
        self._access_times: list[float] = []

        self.hits = 0
        self.misses = 0
        self.bypasses = 0
        self.evictions = 0

    # -- public API ---------------------------------------------------------

    def get(self, embedding: np.ndarray) -> Optional[CacheEntry]:
        if not SEMANTIC_CACHE_ENABLED:
            return None
        match = self._find_best_match(embedding)
        if match is not None:
            entry, idx = match
            self._access_times[idx] = time.time()
            self.hits += 1
            return entry
        self.misses += 1
        return None

    def put(
        self,
        embedding: np.ndarray,
        response: str,
        chat_id: str,
        history: list[Any],
    ) -> None:
        if not SEMANTIC_CACHE_ENABLED:
            return
        self._evict_lru_if_full()
        entry = CacheEntry(
            embedding=embedding,
            response=response,
            chat_id=chat_id,
            history=history,
            expires_at=time.time() + SEMANTIC_CACHE_TTL_SECONDS,
        )
        self._entries.append(entry)
        self._access_times.append(time.time())

    def get_stats(self) -> dict[str, Any]:
        total = self.hits + self.misses + self.bypasses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "bypasses": self.bypasses,
            "evictions": self.evictions,
            "hit_rate": round(self.hits / total, 4) if total > 0 else 0.0,
            "size": len(self._entries),
            "max_entries": SEMANTIC_CACHE_MAX_ENTRIES,
            "threshold": SEMANTIC_CACHE_THRESHOLD,
            "ttl_seconds": SEMANTIC_CACHE_TTL_SECONDS,
            "enabled": SEMANTIC_CACHE_ENABLED,
        }

    def clear(self) -> None:
        self._entries.clear()
        self._access_times.clear()

    # -- internals ----------------------------------------------------------

    def _find_best_match(
        self, embedding: np.ndarray
    ) -> Optional[tuple[CacheEntry, int]]:
        best_score = SEMANTIC_CACHE_THRESHOLD
        best_idx: Optional[int] = None

        surviving_entries: list[CacheEntry] = []
        surviving_times: list[float] = []

        for i, entry in enumerate(self._entries):
            if entry.expired:
                self.evictions += 1
                continue
            surviving_entries.append(entry)
            surviving_times.append(self._access_times[i])

        self._entries = surviving_entries
        self._access_times = surviving_times

        for i, entry in enumerate(self._entries):
            score = cosine_similarity(embedding, entry.embedding)
            if score >= best_score:
                best_score = score
                best_idx = i

        if best_idx is not None:
            return self._entries[best_idx], best_idx
        return None

    def _evict_lru_if_full(self) -> None:
        if len(self._entries) < SEMANTIC_CACHE_MAX_ENTRIES:
            return
        lru_idx = int(np.argmin(self._access_times))
        self._entries.pop(lru_idx)
        self._access_times.pop(lru_idx)
        self.evictions += 1


_cache: SemanticCache = SemanticCache()


def get_cache() -> SemanticCache:
    return _cache
