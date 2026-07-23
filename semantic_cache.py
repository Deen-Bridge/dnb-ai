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


# ---------------------------------------------------------------------------
# Keyed cache (exact-key sibling of the semantic cache)
# ---------------------------------------------------------------------------


class KeyedCache:
    """Exact-key LRU cache for content that is immutable per key.

    Why this lives here rather than in its own module: it is the same cache
    concern as ``SemanticCache`` — same TTL and max-entry configuration, same
    LRU eviction, same stats shape — and callers that need caching should have
    exactly one place to look. It is *not* a second cache system; it is the
    lookup mode the semantic cache cannot serve.

    Why not reuse ``SemanticCache`` directly: a tafsir lookup is keyed by an
    ayah reference, which is exact. Approximate embedding similarity is the
    wrong matching rule there — 2:255 and 2:256 are near-identical strings and
    must never match each other. Cached values are keyed and looked up by an
    exact string, never by distance.
    """

    __slots__ = ("_entries", "_access_times", "hits", "misses", "evictions")

    def __init__(self) -> None:
        # key -> (value, expires_at)
        self._entries: dict[str, tuple[Any, float]] = {}
        self._access_times: dict[str, float] = {}
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def get(self, key: str) -> Optional[Any]:
        entry = self._entries.get(key)
        if entry is None:
            self.misses += 1
            return None
        value, expires_at = entry
        if time.time() > expires_at:
            del self._entries[key]
            self._access_times.pop(key, None)
            self.evictions += 1
            self.misses += 1
            return None
        self._access_times[key] = time.time()
        self.hits += 1
        return value

    def put(self, key: str, value: Any) -> None:
        self._evict_lru_if_full()
        self._entries[key] = (value, time.time() + SEMANTIC_CACHE_TTL_SECONDS)
        self._access_times[key] = time.time()

    def get_stats(self) -> dict[str, Any]:
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "hit_rate": round(self.hits / total, 4) if total > 0 else 0.0,
            "size": len(self._entries),
            "max_entries": SEMANTIC_CACHE_MAX_ENTRIES,
            "ttl_seconds": SEMANTIC_CACHE_TTL_SECONDS,
        }

    def clear(self) -> None:
        self._entries.clear()
        self._access_times.clear()

    def _evict_lru_if_full(self) -> None:
        if len(self._entries) < SEMANTIC_CACHE_MAX_ENTRIES:
            return
        lru_key = min(self._access_times, key=lambda k: self._access_times[k])
        self._entries.pop(lru_key, None)
        self._access_times.pop(lru_key, None)
        self.evictions += 1


_keyed_caches: dict[str, KeyedCache] = {}


def get_keyed_cache(namespace: str) -> KeyedCache:
    """Return the process-wide keyed cache for *namespace*, creating it once."""
    cache = _keyed_caches.get(namespace)
    if cache is None:
        cache = KeyedCache()
        _keyed_caches[namespace] = cache
    return cache


def keyed_cache_stats() -> dict[str, dict[str, Any]]:
    return {name: cache.get_stats() for name, cache in _keyed_caches.items()}
