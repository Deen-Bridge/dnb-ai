"""Persistence for book moderation records (issue #314).

REDIS_URL present → RedisBookModerationStore. The state then survives a
restart and is shared by every worker, which matters: two workers that disagree
about whether a book is taken down would show it to different learners.

REDIS_URL absent → InMemoryBookModerationStore, for local development only.

The key namespace is deliberately *not* ``books:*``: that prefix belongs to the
catalog cache of the REST API, and a cache flush must never take the moderation
state with it.
"""

from __future__ import annotations

import abc
import json
import logging
import os

from book_moderation.models import BookModeration

logger = logging.getLogger(__name__)

MODERATION_KEY_PREFIX = "moderation:book:"

try:
    import redis.asyncio as aioredis

    _redis_available = True
except ImportError:  # pragma: no cover - the redis package is a hard dependency
    _redis_available = False


class BookModerationStore(abc.ABC):
    """Where the moderation state of the catalog lives."""

    @abc.abstractmethod
    async def get(self, book_id: str) -> BookModeration | None: ...

    @abc.abstractmethod
    async def save(self, record: BookModeration) -> None: ...

    @abc.abstractmethod
    async def delete(self, book_id: str) -> bool: ...

    @abc.abstractmethod
    async def list_all(self) -> list[BookModeration]: ...


class InMemoryBookModerationStore(BookModerationStore):
    """Process-local store — development and tests only."""

    def __init__(self) -> None:
        self._records: dict[str, BookModeration] = {}

    async def get(self, book_id: str) -> BookModeration | None:
        return self._records.get(book_id)

    async def save(self, record: BookModeration) -> None:
        self._records[record.book_id] = record

    async def delete(self, book_id: str) -> bool:
        return self._records.pop(book_id, None) is not None

    async def list_all(self) -> list[BookModeration]:
        return list(self._records.values())


class RedisBookModerationStore(BookModerationStore):
    """Redis-backed store, shared with the rest of the platform."""

    def __init__(self, redis_url: str) -> None:
        if not _redis_available:
            raise RuntimeError("redis package not installed")
        self._redis = aioredis.from_url(redis_url, decode_responses=True)

    async def get(self, book_id: str) -> BookModeration | None:
        raw = await self._redis.get(f"{MODERATION_KEY_PREFIX}{book_id}")
        if raw is None:
            return None
        try:
            return BookModeration.model_validate(json.loads(raw))
        except (json.JSONDecodeError, ValueError):
            logger.warning("Corrupt moderation record for book %s", book_id)
            return None

    async def save(self, record: BookModeration) -> None:
        await self._redis.set(f"{MODERATION_KEY_PREFIX}{record.book_id}", record.model_dump_json())

    async def delete(self, book_id: str) -> bool:
        deleted = await self._redis.delete(f"{MODERATION_KEY_PREFIX}{book_id}")
        return deleted > 0

    async def list_all(self) -> list[BookModeration]:
        records: list[BookModeration] = []
        async for key in self._redis.scan_iter(match=f"{MODERATION_KEY_PREFIX}*"):
            raw = await self._redis.get(key)
            if raw is None:
                continue
            try:
                records.append(BookModeration.model_validate(json.loads(raw)))
            except (json.JSONDecodeError, ValueError):
                logger.warning("Corrupt moderation record at %s", key)
        return records


def build_store(redis_url: str | None = None) -> BookModerationStore:
    """Return the configured store: Redis when available, in-memory otherwise."""
    url = redis_url if redis_url is not None else os.getenv("REDIS_URL", "")
    if url:
        try:
            return RedisBookModerationStore(url)
        except Exception as exc:  # noqa: BLE001 - any setup failure degrades to memory
            logger.warning("Redis moderation store unavailable (%s); using in-memory store.", exc)
    return InMemoryBookModerationStore()
