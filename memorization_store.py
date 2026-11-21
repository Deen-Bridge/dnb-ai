"""Per-user storage for memorization plans and Muraja'ah progress (#843, AI-067).

Each user's planner data is one JSON record under one key, so isolation and
reset are a property of the key, not of filtering:

    memorization:user:{user_id} -> {"schema_version": 1, "plans": [...], "progress": [...]}

Backends (mirrors ``memory/store.py``):

* ``REDIS_URL`` unset -> :class:`InMemoryPlannerStore`. Process-local, lost on
  restart, capped at ``MEMORIZATION_MAX_IN_MEMORY_USERS`` records (least
  recently used evicted first).
* ``REDIS_URL`` set   -> :class:`RedisPlannerStore`. Records expire
  ``MEMORIZATION_TTL_DAYS`` (default 365) after the user's last write. Errors
  surface as :class:`PlannerStoreUnavailable`; the store never silently falls
  back to memory, because two workers would then disagree about a user's data.

Raw user identifiers are never logged; log lines carry a short hash instead.
"""

from __future__ import annotations

import abc
import hashlib
import json
import logging
import os
from collections import OrderedDict
from typing import Any

logger = logging.getLogger(__name__)

REDIS_URL = os.getenv("REDIS_URL", "")
MEMORIZATION_TTL_SECONDS = int(os.getenv("MEMORIZATION_TTL_DAYS", "365")) * 86400
MAX_IN_MEMORY_USERS = int(os.getenv("MEMORIZATION_MAX_IN_MEMORY_USERS", "10000"))
KEY_PREFIX = "memorization:user:"

try:
    import redis.asyncio as aioredis

    _redis_available = True
except ImportError:  # pragma: no cover - redis is a runtime dependency
    _redis_available = False


class PlannerStoreError(RuntimeError):
    """Base class for planner storage failures. Messages never contain user data."""


class PlannerStoreUnavailable(PlannerStoreError):
    """The backend could not be reached or rejected the operation."""


class PlannerRecordCorrupt(PlannerStoreError):
    """A stored record exists but cannot be decoded."""


def user_ref(user_id: str) -> str:
    """Short, non-reversible reference for logs."""
    return hashlib.sha256(user_id.encode("utf-8")).hexdigest()[:12]


def _key(user_id: str) -> str:
    return f"{KEY_PREFIX}{user_id}"


def _decode(raw: str, user_id: str) -> dict[str, Any]:
    try:
        record = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Corrupt memorization record for user_ref=%s", user_ref(user_id))
        raise PlannerRecordCorrupt("stored planner record is not valid JSON") from None
    if not isinstance(record, dict):
        logger.warning("Corrupt memorization record for user_ref=%s", user_ref(user_id))
        raise PlannerRecordCorrupt("stored planner record is not an object")
    return record


class PlannerStore(abc.ABC):
    """Load, save and delete one user's planner record."""

    backend: str = "abstract"

    @abc.abstractmethod
    async def load(self, user_id: str) -> dict[str, Any] | None: ...

    @abc.abstractmethod
    async def save(self, user_id: str, record: dict[str, Any]) -> None: ...

    @abc.abstractmethod
    async def delete(self, user_id: str) -> bool: ...


class InMemoryPlannerStore(PlannerStore):
    """Process-local store. Records are kept as JSON strings so callers never share objects."""

    backend = "memory"

    def __init__(self, max_users: int = MAX_IN_MEMORY_USERS) -> None:
        self._records: OrderedDict[str, str] = OrderedDict()
        self._max_users = max(1, max_users)

    async def load(self, user_id: str) -> dict[str, Any] | None:
        raw = self._records.get(_key(user_id))
        if raw is None:
            return None
        self._records.move_to_end(_key(user_id))
        return _decode(raw, user_id)

    async def save(self, user_id: str, record: dict[str, Any]) -> None:
        key = _key(user_id)
        self._records[key] = json.dumps(record, ensure_ascii=False)
        self._records.move_to_end(key)
        while len(self._records) > self._max_users:
            self._records.popitem(last=False)
            logger.info("Evicted least recently used memorization record (%d user cap)", self._max_users)

    async def delete(self, user_id: str) -> bool:
        return self._records.pop(_key(user_id), None) is not None

    def __len__(self) -> int:
        return len(self._records)


class RedisPlannerStore(PlannerStore):
    """Redis-backed store. Every write refreshes the record's TTL."""

    backend = "redis"

    def __init__(self, redis_url: str = "", client: Any = None, ttl_seconds: int = MEMORIZATION_TTL_SECONDS) -> None:
        if client is None:
            if not _redis_available:
                raise RuntimeError("redis package not installed")
            client = aioredis.from_url(redis_url, decode_responses=True)
        self._redis = client
        self._ttl = ttl_seconds

    async def load(self, user_id: str) -> dict[str, Any] | None:
        try:
            raw = await self._redis.get(_key(user_id))
        except Exception:
            logger.warning("Memorization store read failed for user_ref=%s", user_ref(user_id), exc_info=True)
            raise PlannerStoreUnavailable("planner storage read failed") from None
        return None if raw is None else _decode(raw, user_id)

    async def save(self, user_id: str, record: dict[str, Any]) -> None:
        try:
            await self._redis.setex(_key(user_id), self._ttl, json.dumps(record, ensure_ascii=False))
        except Exception:
            logger.warning("Memorization store write failed for user_ref=%s", user_ref(user_id), exc_info=True)
            raise PlannerStoreUnavailable("planner storage write failed") from None

    async def delete(self, user_id: str) -> bool:
        try:
            deleted = await self._redis.delete(_key(user_id))
        except Exception:
            logger.warning("Memorization store delete failed for user_ref=%s", user_ref(user_id), exc_info=True)
            raise PlannerStoreUnavailable("planner storage delete failed") from None
        return bool(deleted)


def create_planner_store() -> PlannerStore:
    if REDIS_URL:
        logger.info("Memorization planner store: Redis (TTL %d days)", MEMORIZATION_TTL_SECONDS // 86400)
        return RedisPlannerStore(REDIS_URL)
    logger.info("Memorization planner store: in-memory (process-local, cap %d users)", MAX_IN_MEMORY_USERS)
    return InMemoryPlannerStore()


_store: PlannerStore | None = None


def get_planner_store() -> PlannerStore:
    global _store
    if _store is None:
        _store = create_planner_store()
    return _store


def set_planner_store(store: PlannerStore | None) -> None:
    """Replace the process-wide store (tests and operational tooling only)."""
    global _store
    _store = store
