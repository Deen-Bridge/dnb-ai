"""Memory store abstraction — Redis-backed with in-memory fallback.

REDIS_URL absent  → InMemoryMemoryStore (process-local, lost on restart).
REDIS_URL present → RedisMemoryStore. Connection failures surface
                    (log + raise) — never silently switch to in-memory,
                    because two workers would diverge on user memory.
"""

from __future__ import annotations

import abc
import json
import logging
import os
import time

from memory.bookmarks import Bookmark, BookmarkCollection
from memory.models import ChatSummary, UserProfile

logger = logging.getLogger(__name__)

MEMORY_TTL_SECONDS = int(os.getenv("MEMORY_TTL_DAYS", "90")) * 86400
REDIS_URL = os.getenv("REDIS_URL", "")

try:
    import redis.asyncio as aioredis

    _redis_available = True
except ImportError:
    _redis_available = False


def _profile_key(user_id: str) -> str:
    return f"memory:profile:{user_id}"


def _summary_key(chat_id: str) -> str:
    return f"memory:summary:{chat_id}"


def _bookmark_key(user_id: str, bookmark_id: str) -> str:
    return f"memory:bookmark:{user_id}:{bookmark_id}"


def _collection_key(user_id: str, collection_id: str) -> str:
    return f"memory:collection:{user_id}:{collection_id}"


class MemoryStore(abc.ABC):
    @abc.abstractmethod
    async def get_profile(self, user_id: str) -> UserProfile | None: ...

    @abc.abstractmethod
    async def save_profile(self, user_id: str, profile: UserProfile) -> None: ...

    @abc.abstractmethod
    async def delete_profile(self, user_id: str) -> bool: ...

    @abc.abstractmethod
    async def get_chat_summary(self, chat_id: str) -> ChatSummary | None: ...

    @abc.abstractmethod
    async def save_chat_summary(self, chat_id: str, summary: ChatSummary) -> None: ...

    @abc.abstractmethod
    async def delete_chat_summary(self, chat_id: str) -> bool: ...

    @abc.abstractmethod
    async def save_bookmark(self, user_id: str, bookmark: Bookmark) -> None: ...

    @abc.abstractmethod
    async def get_bookmark(self, user_id: str, bookmark_id: str) -> Bookmark | None: ...

    @abc.abstractmethod
    async def list_bookmarks(self, user_id: str) -> list[Bookmark]: ...

    @abc.abstractmethod
    async def delete_bookmark(self, user_id: str, bookmark_id: str) -> bool: ...

    @abc.abstractmethod
    async def save_collection(self, user_id: str, collection: BookmarkCollection) -> None: ...

    @abc.abstractmethod
    async def get_collection(self, user_id: str, collection_id: str) -> BookmarkCollection | None: ...

    @abc.abstractmethod
    async def list_collections(self, user_id: str) -> list[BookmarkCollection]: ...

    @abc.abstractmethod
    async def delete_collection(self, user_id: str, collection_id: str) -> bool: ...


class InMemoryMemoryStore(MemoryStore):
    def __init__(self) -> None:
        self._profiles: dict[str, tuple[float, UserProfile]] = {}
        self._summaries: dict[str, tuple[float, ChatSummary]] = {}
        self._bookmarks: dict[str, dict[str, tuple[float, Bookmark]]] = {}
        self._collections: dict[str, dict[str, tuple[float, BookmarkCollection]]] = {}

    async def get_profile(self, user_id: str) -> UserProfile | None:
        entry = self._profiles.get(user_id)
        if entry is None:
            return None
        expires_at, profile = entry
        if time.monotonic() >= expires_at:
            del self._profiles[user_id]
            return None
        return profile

    async def save_profile(self, user_id: str, profile: UserProfile) -> None:
        self._profiles[user_id] = (time.monotonic() + MEMORY_TTL_SECONDS, profile)

    async def delete_profile(self, user_id: str) -> bool:
        return self._profiles.pop(user_id, None) is not None

    async def get_chat_summary(self, chat_id: str) -> ChatSummary | None:
        entry = self._summaries.get(chat_id)
        if entry is None:
            return None
        expires_at, summary = entry
        if time.monotonic() >= expires_at:
            del self._summaries[chat_id]
            return None
        return summary

    async def save_chat_summary(self, chat_id: str, summary: ChatSummary) -> None:
        self._summaries[chat_id] = (time.monotonic() + MEMORY_TTL_SECONDS, summary)

    async def delete_chat_summary(self, chat_id: str) -> bool:
        return self._summaries.pop(chat_id, None) is not None

    async def save_bookmark(self, user_id: str, bookmark: Bookmark) -> None:
        self._bookmarks.setdefault(user_id, {})[bookmark.id] = (
            time.monotonic() + MEMORY_TTL_SECONDS,
            bookmark,
        )

    async def get_bookmark(self, user_id: str, bookmark_id: str) -> Bookmark | None:
        entry = self._bookmarks.get(user_id, {}).get(bookmark_id)
        if entry is None:
            return None
        expires_at, bookmark = entry
        if time.monotonic() >= expires_at:
            del self._bookmarks[user_id][bookmark_id]
            return None
        return bookmark

    async def list_bookmarks(self, user_id: str) -> list[Bookmark]:
        now = time.monotonic()
        entries = self._bookmarks.get(user_id, {})
        expired = [bid for bid, (exp, _) in entries.items() if now >= exp]
        for bid in expired:
            del entries[bid]
        return [b for _, b in entries.values()]

    async def delete_bookmark(self, user_id: str, bookmark_id: str) -> bool:
        return self._bookmarks.get(user_id, {}).pop(bookmark_id, None) is not None

    async def save_collection(self, user_id: str, collection: BookmarkCollection) -> None:
        self._collections.setdefault(user_id, {})[collection.id] = (
            time.monotonic() + MEMORY_TTL_SECONDS,
            collection,
        )

    async def get_collection(self, user_id: str, collection_id: str) -> BookmarkCollection | None:
        entry = self._collections.get(user_id, {}).get(collection_id)
        if entry is None:
            return None
        expires_at, collection = entry
        if time.monotonic() >= expires_at:
            del self._collections[user_id][collection_id]
            return None
        return collection

    async def list_collections(self, user_id: str) -> list[BookmarkCollection]:
        now = time.monotonic()
        entries = self._collections.get(user_id, {})
        expired = [cid for cid, (exp, _) in entries.items() if now >= exp]
        for cid in expired:
            del entries[cid]
        return [c for _, c in entries.values()]

    async def delete_collection(self, user_id: str, collection_id: str) -> bool:
        return self._collections.get(user_id, {}).pop(collection_id, None) is not None


class RedisMemoryStore(MemoryStore):
    def __init__(self, redis_url: str) -> None:
        if not _redis_available:
            raise RuntimeError("redis package not installed")
        self._redis = aioredis.from_url(redis_url, decode_responses=True)

    async def get_profile(self, user_id: str) -> UserProfile | None:
        raw = await self._redis.get(_profile_key(user_id))
        if raw is None:
            return None
        try:
            return UserProfile.model_validate(json.loads(raw))
        except (json.JSONDecodeError, Exception):
            logger.warning("Corrupt profile for user %s", user_id)
            return None

    async def save_profile(self, user_id: str, profile: UserProfile) -> None:
        await self._redis.setex(
            _profile_key(user_id),
            MEMORY_TTL_SECONDS,
            profile.model_dump_json(),
        )

    async def delete_profile(self, user_id: str) -> bool:
        deleted = await self._redis.delete(_profile_key(user_id))
        return deleted > 0

    async def get_chat_summary(self, chat_id: str) -> ChatSummary | None:
        raw = await self._redis.get(_summary_key(chat_id))
        if raw is None:
            return None
        try:
            return ChatSummary.model_validate(json.loads(raw))
        except (json.JSONDecodeError, Exception):
            logger.warning("Corrupt chat summary for chat %s", chat_id)
            return None

    async def save_chat_summary(self, chat_id: str, summary: ChatSummary) -> None:
        await self._redis.setex(
            _summary_key(chat_id),
            MEMORY_TTL_SECONDS,
            summary.model_dump_json(),
        )

    async def delete_chat_summary(self, chat_id: str) -> bool:
        deleted = await self._redis.delete(_summary_key(chat_id))
        return deleted > 0

    async def save_bookmark(self, user_id: str, bookmark: Bookmark) -> None:
        await self._redis.setex(
            _bookmark_key(user_id, bookmark.id),
            MEMORY_TTL_SECONDS,
            bookmark.model_dump_json(),
        )
        await self._redis.sadd(f"memory:bookmark_ids:{user_id}", bookmark.id)
        await self._redis.expire(f"memory:bookmark_ids:{user_id}", MEMORY_TTL_SECONDS)

    async def get_bookmark(self, user_id: str, bookmark_id: str) -> Bookmark | None:
        raw = await self._redis.get(_bookmark_key(user_id, bookmark_id))
        if raw is None:
            return None
        try:
            return Bookmark.model_validate(json.loads(raw))
        except (json.JSONDecodeError, Exception):
            logger.warning("Corrupt bookmark %s for user %s", bookmark_id, user_id)
            return None

    async def list_bookmarks(self, user_id: str) -> list[Bookmark]:
        ids = await self._redis.smembers(f"memory:bookmark_ids:{user_id}")
        bookmarks: list[Bookmark] = []
        for bid in ids:
            bookmark = await self.get_bookmark(user_id, bid)
            if bookmark is not None:
                bookmarks.append(bookmark)
            else:
                await self._redis.srem(f"memory:bookmark_ids:{user_id}", bid)
        return bookmarks

    async def delete_bookmark(self, user_id: str, bookmark_id: str) -> bool:
        deleted = await self._redis.delete(_bookmark_key(user_id, bookmark_id))
        await self._redis.srem(f"memory:bookmark_ids:{user_id}", bookmark_id)
        return deleted > 0

    async def save_collection(self, user_id: str, collection: BookmarkCollection) -> None:
        await self._redis.setex(
            _collection_key(user_id, collection.id),
            MEMORY_TTL_SECONDS,
            collection.model_dump_json(),
        )
        await self._redis.sadd(f"memory:collection_ids:{user_id}", collection.id)
        await self._redis.expire(f"memory:collection_ids:{user_id}", MEMORY_TTL_SECONDS)

    async def get_collection(self, user_id: str, collection_id: str) -> BookmarkCollection | None:
        raw = await self._redis.get(_collection_key(user_id, collection_id))
        if raw is None:
            return None
        try:
            return BookmarkCollection.model_validate(json.loads(raw))
        except (json.JSONDecodeError, Exception):
            logger.warning("Corrupt collection %s for user %s", collection_id, user_id)
            return None

    async def list_collections(self, user_id: str) -> list[BookmarkCollection]:
        ids = await self._redis.smembers(f"memory:collection_ids:{user_id}")
        collections: list[BookmarkCollection] = []
        for cid in ids:
            collection = await self.get_collection(user_id, cid)
            if collection is not None:
                collections.append(collection)
            else:
                await self._redis.srem(f"memory:collection_ids:{user_id}", cid)
        return collections

    async def delete_collection(self, user_id: str, collection_id: str) -> bool:
        deleted = await self._redis.delete(_collection_key(user_id, collection_id))
        await self._redis.srem(f"memory:collection_ids:{user_id}", collection_id)
        return deleted > 0
