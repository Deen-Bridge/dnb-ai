"""Persistence for book moderation records (issue #314, #881).

REDIS_URL present -> RedisBookModerationStore wrapped in ResilientBookModerationStore.
The state then survives a restart and is shared by every worker.

During a Redis outage, ResilientBookModerationStore writes to a durable fallback
store and queues reconciliation rather than silently losing state in process memory.

The key namespace is deliberately *not* books:* - that prefix belongs to the
catalog cache of the REST API, and a cache flush must never take the moderation
state with it.
"""

from __future__ import annotations

import abc
import json
import logging
import os
from pathlib import Path

from book_moderation.models import BookModeration

logger = logging.getLogger(__name__)

MODERATION_KEY_PREFIX = "moderation:book:"

try:
    import redis.asyncio as aioredis

    _redis_available = True
except ImportError:  # pragma: no cover - the redis package is a hard dependency
    _redis_available = False


class ModerationStoreOutageError(RuntimeError):
    """Raised when moderation persistence cannot guarantee durability during an outage."""


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
    """Process-local store - development and tests only."""

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


class DurableFileBookModerationStore(BookModerationStore):
    """Durable file-backed store surviving process restarts and pod crashes."""

    def __init__(self, file_path: str | Path) -> None:
        self._path = Path(file_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if not self._path.exists():
            self._write_records({})

    def _read_records(self) -> dict[str, BookModeration]:
        if not self._path.exists():
            return {}
        try:
            with open(self._path, "r", encoding="utf-8") as handle:
                raw_data = json.load(handle)
            return {
                book_id: BookModeration.model_validate(raw)
                for book_id, raw in raw_data.items()
            }
        except (json.JSONDecodeError, OSError, ValueError) as exc:
            logger.warning("Could not read durable moderation file (%s)", exc)
            return {}

    def _write_records(self, records: dict[str, BookModeration]) -> None:
        data = {k: v.model_dump(mode="json") for k, v in records.items()}
        temp_file = self._path.with_suffix(".tmp")
        with open(temp_file, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, default=str)
        temp_file.replace(self._path)

    async def get(self, book_id: str) -> BookModeration | None:
        records = self._read_records()
        return records.get(book_id)

    async def save(self, record: BookModeration) -> None:
        records = self._read_records()
        records[record.book_id] = record
        self._write_records(records)

    async def delete(self, book_id: str) -> bool:
        records = self._read_records()
        if book_id in records:
            del records[book_id]
            self._write_records(records)
            return True
        return False

    async def list_all(self) -> list[BookModeration]:
        return list(self._read_records().values())


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


class ResilientBookModerationStore(BookModerationStore):
    """Resilient store pairing primary Redis with durable fallback during outage."""

    def __init__(
        self,
        primary: BookModerationStore,
        fallback: BookModerationStore | None = None,
        *,
        strict_durability: bool = True,
    ) -> None:
        self._primary = primary
        self._fallback = fallback
        self._strict_durability = strict_durability
        self._pending_sync: set[str] = set()

    @property
    def pending_sync_ids(self) -> list[str]:
        return sorted(self._pending_sync)

    async def get(self, book_id: str) -> BookModeration | None:
        try:
            return await self._primary.get(book_id)
        except Exception as exc:
            logger.warning("Primary moderation store read failed (%s); checking fallback.", exc)
            if self._fallback is not None:
                return await self._fallback.get(book_id)
            if self._strict_durability:
                raise ModerationStoreOutageError(f"Moderation store unavailable during read for book {book_id}") from exc
            return None

    async def save(self, record: BookModeration) -> None:
        try:
            await self._primary.save(record)
            if self._fallback is not None:
                try:
                    await self._fallback.save(record)
                    self._pending_sync.discard(record.book_id)
                except Exception as fb_exc:
                    logger.warning("Failed to sync primary write to fallback store (%s)", fb_exc)
        except Exception as exc:
            # Emit operator alert with affected book ID and count, without leaking content
            logger.error(
                "MODERATION_ALERT: Redis store outage during save. affected_count=1 book_ids=['%s']. Error: %s",
                record.book_id,
                type(exc).__name__,
            )
            if self._fallback is not None:
                try:
                    await self._fallback.save(record)
                    self._pending_sync.add(record.book_id)
                    logger.info(
                        "Durable fallback preserved moderation state for book %s during Redis outage.",
                        record.book_id,
                    )
                    return
                except Exception as fb_exc:
                    logger.error("Durable fallback also failed during Redis outage: %s", fb_exc)

            if self._strict_durability:
                raise ModerationStoreOutageError(
                    f"Durable write failed for book {record.book_id}: primary store unavailable."
                ) from exc
            raise

    async def delete(self, book_id: str) -> bool:
        try:
            deleted = await self._primary.delete(book_id)
            if self._fallback is not None:
                await self._fallback.delete(book_id)
            self._pending_sync.discard(book_id)
            return deleted
        except Exception as exc:
            logger.error(
                "MODERATION_ALERT: Redis store outage during delete. affected_count=1 book_ids=['%s']. Error: %s",
                book_id,
                type(exc).__name__,
            )
            if self._fallback is not None:
                self._pending_sync.discard(book_id)
                return await self._fallback.delete(book_id)
            if self._strict_durability:
                raise ModerationStoreOutageError(f"Delete failed for book {book_id} due to store outage") from exc
            raise

    async def list_all(self) -> list[BookModeration]:
        try:
            return await self._primary.list_all()
        except Exception as exc:
            logger.warning("Primary moderation store list failed (%s); reading from fallback.", exc)
            if self._fallback is not None:
                return await self._fallback.list_all()
            if self._strict_durability:
                raise ModerationStoreOutageError("Moderation store unavailable during list_all") from exc
            return []

    async def reconcile_pending(self) -> int:
        """Sync records saved to fallback during an outage back into primary."""
        if not self._pending_sync or self._fallback is None:
            return 0
        synced = 0
        to_sync = list(self._pending_sync)
        for book_id in to_sync:
            record = await self._fallback.get(book_id)
            if record is not None:
                try:
                    await self._primary.save(record)
                    self._pending_sync.discard(book_id)
                    synced += 1
                except Exception as exc:
                    logger.warning("Reconciliation to primary failed for book %s (%s)", book_id, exc)
                    break
        if synced:
            logger.info("MODERATION_RECONCILIATION: Synced %d pending records to primary store.", synced)
        return synced


def build_store(
    redis_url: str | None = None,
    *,
    fallback_path: str | Path | None = None,
    strict_durability: bool | None = None,
) -> BookModerationStore:
    """Return the configured store: resilient Redis with durable fallback in production, in-memory for testing."""
    url = redis_url if redis_url is not None else os.getenv("REDIS_URL", "")
    is_prod = (
        os.getenv("ENVIRONMENT", "").lower() == "production"
        or os.getenv("ENV", "").lower() == "production"
        or os.getenv("MODERATION_STRICT_DURABILITY", "").lower() in ("true", "1")
    )
    if strict_durability is None:
        strict_durability = is_prod

    default_fallback_path = fallback_path or os.getenv("MODERATION_FALLBACK_FILE", "data/moderation_store.json")
    durable_fallback = DurableFileBookModerationStore(default_fallback_path) if (is_prod or fallback_path) else None

    if url:
        try:
            primary = RedisBookModerationStore(url)
            return ResilientBookModerationStore(
                primary=primary,
                fallback=durable_fallback,
                strict_durability=strict_durability,
            )
        except Exception as exc:
            if strict_durability:
                logger.error(
                    "MODERATION_ALERT: Redis connection failed during store initialization (%s). Strict durability enforced.",
                    exc,
                )
                if durable_fallback is not None:
                    logger.info("Operating in degraded mode with durable fallback store.")
                    return durable_fallback
                raise ModerationStoreOutageError(f"Cannot initialize moderation store: Redis unavailable ({exc})") from exc
            logger.warning("Redis moderation store unavailable (%s); using in-memory store.", exc)
            return InMemoryBookModerationStore()

    if is_prod:
        logger.warning(
            "MODERATION_ALERT: REDIS_URL not configured in production. Using durable file store at %s",
            default_fallback_path,
        )
        return durable_fallback or DurableFileBookModerationStore(default_fallback_path)

    return InMemoryBookModerationStore()
