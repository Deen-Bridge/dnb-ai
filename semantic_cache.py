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

import hashlib
import logging
import os
import time
from typing import Any

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

_FAKE_EMBEDDING: np.ndarray | None = None


def set_fake_embedding(vec: np.ndarray | None) -> None:
    global _FAKE_EMBEDDING
    _FAKE_EMBEDDING = vec


def embed_text(text: str) -> np.ndarray:
    if _FAKE_EMBEDDING is not None:
        return _FAKE_EMBEDDING
    import google.generativeai as genai

    try:
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=text,
        )
        return np.array(result["embedding"], dtype=np.float32)
    except Exception as exc:
        logger.debug("Failed to embed content via Gemini API: %s; using deterministic fallback", exc)
        h = hashlib.sha256(text.encode("utf-8")).digest()
        vec = np.frombuffer(h * 4, dtype=np.uint8)[:128].astype(np.float32)
        norm = float(np.linalg.norm(vec))
        return vec / (norm if norm > 0 else 1.0)


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

    def get(self, embedding: np.ndarray) -> CacheEntry | None:
        if not SEMANTIC_CACHE_ENABLED:
            return None
        self._sweep_expired()
        matches = self._store.query(embedding, top_k=1, min_score=SEMANTIC_CACHE_THRESHOLD)
        if matches:
            chunk_id = matches[0].chunk.chunk_id
            entry = self._entry_by_id.get(chunk_id)
            if entry is not None and not entry.expired:
                self._access_times[chunk_id] = time.time()
                self.hits += 1
                try:
                    import metrics

                    metrics.record_cache_hit(cache_type="semantic")
                except Exception:  # noqa: BLE001
                    pass
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

    def _remove(self, chunk_id: str) -> None:
        self._store.delete_chunk(chunk_id)
        self._entry_by_id.pop(chunk_id, None)
        self._access_times.pop(chunk_id, None)

    def _sweep_expired(self) -> None:
        """Lazily drop expired entries before a lookup (matches old semantics)."""
        expired = [chunk_id for chunk_id, entry in self._entry_by_id.items() if entry.expired]
        for chunk_id in expired:
            self._remove(chunk_id)
            self.evictions += 1

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

    def get(self, key: str) -> Any | None:
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
        try:
            import metrics

            metrics.record_cache_hit(cache_type="exact")
        except Exception:  # noqa: BLE001
            pass
        return value

    def put(self, key: str, value: Any, ttl_seconds: int | None = None) -> None:
        """Store *value*, expiring after *ttl_seconds* (default: the cache TTL).

        The override exists for content that is cacheable but not immutable —
        a market price is worth caching for hours, not for the day-long TTL
        that suits a fixed tafsir passage.
        """
        self._evict_lru_if_full()
        ttl = SEMANTIC_CACHE_TTL_SECONDS if ttl_seconds is None else ttl_seconds
        self._entries[key] = (value, time.time() + ttl)
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
