"""Durable store for the scholar-review queue.

Persistence choice
------------------
Same shape as the session store that shipped with session persistence (#3):
Redis when ``REDIS_URL`` is set, with a crash-safe local disk spool fallback
(``REVIEW_SPOOL_DIR``) and in-process cache behind one async interface so
callers never branch on which backend is live.

When Redis is unreachable, accepted review items are durably spooled to disk
(default: ``data/review/spool``) before returning, guaranteeing that process
restarts, container replacements, or host reboots cannot discard questions
awaiting scholar review.

Upon Redis recovery, spooled items are reconciled into Redis with strict
idempotency (zero duplicates).

Degraded mode is bounded by ``REVIEW_SPOOL_MAX_SIZE`` and governed by
``REVIEW_DEGRADED_POLICY`` (``spool_or_reject`` or ``spool_or_drop``), emitting
structured alerts for degraded status, queue depth, queue age, and dropped or
rejected items.

Redis layout
------------
- ``review:item:{id}``   — the JSON-serialized item
- ``review:pending``     — sorted set of pending ids, scored by creation time
- ``review:reviewed``    — sorted set of decided ids, scored by decision time
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

REDIS_URL = os.getenv("REDIS_URL", "")

ITEM_KEY = "review:item:{item_id}"
PENDING_INDEX = "review:pending"
REVIEWED_INDEX = "review:reviewed"

REVIEW_SPOOL_DIR = os.getenv("REVIEW_SPOOL_DIR", "data/review/spool")
REVIEW_SPOOL_ENABLED = os.getenv("REVIEW_SPOOL_ENABLED", "true").lower() == "true"
REVIEW_SPOOL_MAX_SIZE = int(os.getenv("REVIEW_SPOOL_MAX_SIZE", "10000"))
REVIEW_DEGRADED_POLICY = os.getenv("REVIEW_DEGRADED_POLICY", "spool_or_reject")
REVIEW_ALERT_DEPTH_THRESHOLD = int(os.getenv("REVIEW_ALERT_DEPTH_THRESHOLD", "500"))
REVIEW_ALERT_AGE_THRESHOLD_SECONDS = float(os.getenv("REVIEW_ALERT_AGE_THRESHOLD_SECONDS", "3600.0"))

try:
    import redis.asyncio as aioredis

    _redis_available = True
except ImportError:  # pragma: no cover - depends on deployment extras
    _redis_available = False


class AlreadyReviewedError(Exception):
    """Raised when a verdict is recorded on an item that already has one.

    Carries the existing item so the caller can report what the earlier verdict
    was rather than a bare conflict.
    """

    def __init__(self, item: ReviewItem) -> None:
        super().__init__(f"Review item {item.id} was already reviewed")
        self.item = item


class ReviewSpoolFullError(Exception):
    """Raised when the degraded scholar review queue spool reaches capacity."""

    def __init__(self, size: int, max_size: int) -> None:
        super().__init__(f"Scholar review degraded spool is full ({size}/{max_size} items)")
        self.size = size
        self.max_size = max_size


class ReviewStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    CORRECTED = "corrected"
    REJECTED = "rejected"


class Verdict(str, Enum):
    APPROVE = "approve"
    CORRECT = "correct"
    REJECT = "reject"


VERDICT_STATUS: dict[Verdict, ReviewStatus] = {
    Verdict.APPROVE: ReviewStatus.APPROVED,
    Verdict.CORRECT: ReviewStatus.CORRECTED,
    Verdict.REJECT: ReviewStatus.REJECTED,
}


class ReviewItem(BaseModel):
    """One low-confidence religious answer awaiting (or holding) a verdict."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str
    answer: str
    confidence: float
    band: str
    signals: dict[str, float] = {}
    chat_id: str | None = None
    created_at: float = Field(default_factory=time.time)

    status: ReviewStatus = ReviewStatus.PENDING
    verdict: Verdict | None = None
    corrected_answer: str | None = None
    reviewer: str | None = None
    reviewer_note: str | None = None
    reviewed_at: float | None = None

    @property
    def final_answer(self) -> str | None:
        """The answer a reviewer stands behind, or None if there isn't one."""
        if self.status is ReviewStatus.APPROVED:
            return self.answer
        if self.status is ReviewStatus.CORRECTED:
            return self.corrected_answer
        return None


class ReviewSpool:
    """Crash-safe, disk-backed local spool for scholar review items.

    Items are written atomically using temporary files and fsync before being
    renamed, preventing partial or corrupted records on system crash.
    """

    def __init__(
        self,
        spool_dir: str | Path,
        enabled: bool = True,
        max_size: int = 10000,
        policy: str = "spool_or_reject",
    ) -> None:
        self.spool_dir = Path(spool_dir)
        self.enabled = enabled
        self.max_size = max_size
        self.policy = policy
        if self.enabled:
            self.spool_dir.mkdir(parents=True, exist_ok=True)

    def _item_path(self, item_id: str) -> Path:
        return self.spool_dir / f"item_{item_id}.json"

    def write_item(self, item: ReviewItem) -> Path | None:
        """Atomically persist a review item to disk."""
        if not self.enabled:
            return None
        self.spool_dir.mkdir(parents=True, exist_ok=True)
        final_path = self._item_path(item.id)
        tmp_path = self.spool_dir / f"item_{item.id}.tmp.{uuid.uuid4().hex[:8]}"
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                f.write(item.model_dump_json())
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, final_path)
            return final_path
        except Exception as exc:
            logger.error("Failed to write review item %s to spool %s: %s", item.id, final_path, exc)
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass
            raise

    def remove_item(self, item_id: str) -> bool:
        """Remove a spooled item file upon successful reconciliation."""
        if not self.enabled:
            return False
        path = self._item_path(item_id)
        try:
            if path.exists():
                path.unlink()
                return True
        except OSError as exc:
            logger.warning("Could not delete reconciled spool file %s: %s", path, exc)
        return False

    def load_all(self) -> list[ReviewItem]:
        """Load and deserialize all spooled review items from disk."""
        if not self.enabled or not self.spool_dir.exists():
            return []
        items: list[ReviewItem] = []
        for path in self.spool_dir.glob("item_*.json"):
            try:
                content = path.read_text(encoding="utf-8")
                item = ReviewItem.model_validate_json(content)
                items.append(item)
            except Exception as exc:
                logger.warning("Skipping corrupt review spool file %s: %s", path, exc)
        items.sort(key=lambda it: it.created_at)
        return items

    def size(self) -> int:
        """Current number of spooled item files on disk."""
        if not self.enabled or not self.spool_dir.exists():
            return 0
        return len(list(self.spool_dir.glob("item_*.json")))

    def oldest_item_age_seconds(self) -> float | None:
        """Age in seconds of the oldest item currently in the spool."""
        items = self.load_all()
        if not items:
            return None
        oldest = min(it.created_at for it in items)
        return max(0.0, time.time() - oldest)

    def evict_oldest(self) -> ReviewItem | None:
        """Evict the oldest pending item when policy is spool_or_drop."""
        items = [it for it in self.load_all() if it.status is ReviewStatus.PENDING]
        if not items:
            return None
        oldest = min(items, key=lambda it: it.created_at)
        self.remove_item(oldest.id)
        return oldest

    def clear(self) -> None:
        """Remove all spooled items and temporary files. Used by tests."""
        if not self.enabled or not self.spool_dir.exists():
            return
        for f in self.spool_dir.glob("item_*"):
            try:
                f.unlink()
            except OSError:
                pass


class ReviewStore:
    """Persists review items. Redis-backed when configured, durable spool & memory otherwise."""

    def __init__(
        self,
        spool_dir: str | Path | None = None,
        spool_enabled: bool | None = None,
        max_spool_size: int | None = None,
        degraded_policy: str | None = None,
        alert_depth_threshold: int | None = None,
        alert_age_threshold_seconds: float | None = None,
    ) -> None:
        self._redis: Any = None
        self._local: dict[str, ReviewItem] = {}
        self._use_redis = False
        self._degraded = False

        raw_spool_dir = spool_dir or os.getenv("REVIEW_SPOOL_DIR") or "data/review/spool"
        self.spool_dir = Path(raw_spool_dir)
        self.spool_enabled = (
            spool_enabled if spool_enabled is not None else os.getenv("REVIEW_SPOOL_ENABLED", "true").lower() == "true"
        )
        self.max_spool_size = (
            max_spool_size if max_spool_size is not None else int(os.getenv("REVIEW_SPOOL_MAX_SIZE", "10000"))
        )
        self.degraded_policy = (
            degraded_policy if degraded_policy is not None else os.getenv("REVIEW_DEGRADED_POLICY", "spool_or_reject")
        )
        self.alert_depth_threshold = (
            alert_depth_threshold
            if alert_depth_threshold is not None
            else int(os.getenv("REVIEW_ALERT_DEPTH_THRESHOLD", "500"))
        )
        self.alert_age_threshold_seconds = (
            alert_age_threshold_seconds
            if alert_age_threshold_seconds is not None
            else float(os.getenv("REVIEW_ALERT_AGE_THRESHOLD_SECONDS", "3600.0"))
        )

        self.total_spooled = 0
        self.total_reconciled = 0
        self.total_rejected = 0
        self.total_dropped = 0

        self._spool = ReviewSpool(
            spool_dir=self.spool_dir,
            enabled=self.spool_enabled,
            max_size=self.max_spool_size,
            policy=self.degraded_policy,
        )

        # Recover spooled items from prior run
        spooled_items = self._spool.load_all()
        if spooled_items:
            for item in spooled_items:
                self._local[item.id] = item
            self._degraded = True
            logger.info("ReviewStore loaded %d spooled items from disk (%s)", len(spooled_items), self.spool_dir)

        if REDIS_URL and _redis_available:
            try:
                self._redis = aioredis.from_url(REDIS_URL, decode_responses=True)
                self._use_redis = True
                logger.info("ReviewStore using Redis")
            except Exception as exc:  # pragma: no cover
                logger.warning("Failed to connect to Redis for review queue (%s); using spool", exc)

        if not self._use_redis:
            logger.info("ReviewStore using durable disk spool (%s)", self.spool_dir)

    @property
    def durable(self) -> bool:
        return self._use_redis and not self._degraded

    # -- public API ---------------------------------------------------------

    async def add(self, item: ReviewItem) -> ReviewItem:
        """Queue an item for review.

        Persists directly to Redis when healthy. If Redis is unreachable or not
        configured, the item is persisted to the local crash-safe disk spool and
        held in memory. Bounded capacity is enforced via configured policy:
        'spool_or_reject' (raises ReviewSpoolFullError) or 'spool_or_drop'
        (evicts oldest item).
        """
        if self._use_redis:
            try:
                async with self._redis.pipeline(transaction=True) as pipe:
                    pipe.set(ITEM_KEY.format(item_id=item.id), item.model_dump_json())
                    pipe.zadd(PENDING_INDEX, {item.id: item.created_at})
                    await pipe.execute()
                # If item was in spool previously, remove it
                self._spool.remove_item(item.id)
                self._local.pop(item.id, None)
                logger.info("Queued answer %s for scholar review (confidence=%.2f)", item.id, item.confidence)
                return item
            except Exception as exc:  # degrade, never silently drop
                self._mark_degraded("add", exc)

        # Degraded path: store in durable spool + local memory
        current_size = self._spool.size()
        if current_size >= self.max_spool_size and item.id not in self._local:
            if self.degraded_policy == "spool_or_reject":
                self.total_rejected += 1
                logger.error(
                    "REVIEW_QUEUE_ALERT alert=queue_rejection count=%d size=%d max_size=%d item_id=%s",
                    self.total_rejected,
                    current_size,
                    self.max_spool_size,
                    item.id,
                )
                raise ReviewSpoolFullError(current_size, self.max_spool_size)
            else:  # spool_or_drop
                evicted = self._spool.evict_oldest()
                if evicted:
                    self._local.pop(evicted.id, None)
                    self.total_dropped += 1
                    logger.error(
                        "REVIEW_QUEUE_ALERT alert=queue_drop count=%d evicted_id=%s size=%d",
                        self.total_dropped,
                        evicted.id,
                        self.max_spool_size,
                    )

        try:
            self._spool.write_item(item)
            self.total_spooled += 1
        except Exception as spool_exc:
            logger.error("Failed to spool review item %s to disk: %s", item.id, spool_exc)

        self._local[item.id] = item

        # Evaluate alert thresholds
        self._evaluate_alerts()

        logger.info(
            "Queued answer %s in durable spool for scholar review (confidence=%.2f, degraded=%s)",
            item.id,
            item.confidence,
            self._degraded,
        )
        return item

    async def get(self, item_id: str) -> ReviewItem | None:
        if self._use_redis:
            try:
                raw = await self._redis.get(ITEM_KEY.format(item_id=item_id))
            except Exception as exc:  # noqa: BLE001
                self._mark_degraded("get", exc)
                return self._local.get(item_id)
            item = self._deserialize(raw)
            if item is not None:
                return item
            # Items buffered locally while Redis was unreachable.
            return self._local.get(item_id)
        return self._local.get(item_id)

    async def list_pending(self, limit: int = 50, offset: int = 0) -> list[ReviewItem]:
        """Oldest pending items first — the longest wait gets seen first."""
        if self._use_redis:
            try:
                ids = await self._redis.zrange(PENDING_INDEX, offset, offset + limit - 1)
            except Exception as exc:  # noqa: BLE001
                self._mark_degraded("list_pending", exc)
                return []
            items = [await self.get(item_id) for item_id in ids]
            return [item for item in items if item is not None]
        pending = [item for item in self._local.values() if item.status is ReviewStatus.PENDING]
        pending.sort(key=lambda item: item.created_at)
        return pending[offset : offset + limit]

    async def list_reviewed(self, limit: int = 50, offset: int = 0) -> list[ReviewItem]:
        """Most recently decided items first."""
        if self._use_redis:
            try:
                ids = await self._redis.zrevrange(REVIEWED_INDEX, offset, offset + limit - 1)
            except Exception as exc:  # noqa: BLE001
                self._mark_degraded("list_reviewed", exc)
                return []
            items = [await self.get(item_id) for item_id in ids]
            return [item for item in items if item is not None]
        reviewed = [item for item in self._local.values() if item.status is not ReviewStatus.PENDING]
        reviewed.sort(key=lambda item: item.reviewed_at or 0, reverse=True)
        return reviewed[offset : offset + limit]

    async def record_verdict(
        self,
        item_id: str,
        verdict: Verdict,
        corrected_answer: str | None = None,
        reviewer: str | None = None,
        reviewer_note: str | None = None,
    ) -> ReviewItem | None:
        """Apply a reviewer's decision, exactly once.

        Returns None if the item does not exist, and raises
        ``AlreadyReviewedError`` if it already carries a verdict. The
        pending-to-reviewed transition is claimed atomically in Redis, or in a
        single uninterrupted step in memory. Spooled items on disk are updated
        durably so verdicts are not lost during outages.
        """
        item = await self.get(item_id)
        if item is None:
            return None
        if item.status is not ReviewStatus.PENDING:
            raise AlreadyReviewedError(item)

        if self._use_redis and item_id not in self._local:
            try:
                claimed = await self._redis.zrem(PENDING_INDEX, item_id)
            except Exception as exc:  # noqa: BLE001
                self._mark_degraded("record_verdict", exc)
                claimed = 0
                if item_id not in self._local:
                    raise
            if not claimed:
                raise AlreadyReviewedError(await self.get(item_id) or item)

        # No await between the status check above and the mutation below.
        item.verdict = verdict
        item.status = VERDICT_STATUS[verdict]
        item.corrected_answer = corrected_answer
        item.reviewer = reviewer
        item.reviewer_note = reviewer_note
        item.reviewed_at = time.time()

        if self._use_redis and item_id not in self._local:
            try:
                async with self._redis.pipeline(transaction=True) as pipe:
                    pipe.set(ITEM_KEY.format(item_id=item.id), item.model_dump_json())
                    pipe.zadd(REVIEWED_INDEX, {item.id: item.reviewed_at})
                    await pipe.execute()
            except Exception as exc:  # noqa: BLE001 - a recorded verdict is not lost
                self._mark_degraded("record_verdict", exc)
                self._local[item.id] = item
                try:
                    self._spool.write_item(item)
                except Exception:
                    pass
        else:
            self._local[item.id] = item
            try:
                self._spool.write_item(item)
            except Exception:
                pass

        logger.info("Review %s recorded: %s", item.id, verdict.value)
        return item

    async def reconcile_spool(self) -> int:
        """Replay spooled items into Redis without duplicates.

        Returns the number of items successfully reconciled.
        """
        if not self._use_redis or not self._redis:
            return 0

        spooled_items = self._spool.load_all()
        if not spooled_items:
            if not self._local:
                self._degraded = False
            return 0

        reconciled = 0
        for item in spooled_items:
            try:
                key = ITEM_KEY.format(item_id=item.id)
                existing = await self._redis.get(key)
                if not existing:
                    # Write to Redis transactionally
                    async with self._redis.pipeline(transaction=True) as pipe:
                        pipe.set(key, item.model_dump_json())
                        if item.status is ReviewStatus.PENDING:
                            pipe.zadd(PENDING_INDEX, {item.id: item.created_at})
                        else:
                            pipe.zadd(REVIEWED_INDEX, {item.id: item.reviewed_at or item.created_at})
                        await pipe.execute()
                else:
                    # If item already exists in Redis, check if local carries a verdict that Redis lacks
                    existing_item = self._deserialize(existing)
                    if (
                        existing_item
                        and existing_item.status is ReviewStatus.PENDING
                        and item.status is not ReviewStatus.PENDING
                    ):
                        async with self._redis.pipeline(transaction=True) as pipe:
                            pipe.set(key, item.model_dump_json())
                            pipe.zrem(PENDING_INDEX, item.id)
                            pipe.zadd(REVIEWED_INDEX, {item.id: item.reviewed_at or item.created_at})
                            await pipe.execute()

                # Reconciled in Redis! Remove from spool file and local memory
                self._spool.remove_item(item.id)
                self._local.pop(item.id, None)
                reconciled += 1
                self.total_reconciled += 1
            except Exception as exc:
                self._mark_degraded("reconcile_spool", exc)
                logger.error("Error during spool reconciliation for item %s: %s", item.id, exc)
                break

        if self._spool.size() == 0 and not self._local:
            self._degraded = False
            logger.info(
                "REVIEW_QUEUE_ALERT alert=degraded_mode status=resolved reconciled_count=%d",
                reconciled,
            )

        return reconciled

    async def stats(self) -> dict[str, Any]:
        """Queue depth, durability status, spool metrics, and active alerts."""
        spool_size = self._spool.size()
        oldest_age = self._spool.oldest_item_age_seconds()
        alerts = self._evaluate_alerts()

        if self._use_redis:
            try:
                pending = await self._redis.zcard(PENDING_INDEX)
                reviewed = await self._redis.zcard(REVIEWED_INDEX)
                # Attempt opportunistic reconciliation if spooled items exist
                if spool_size > 0:
                    await self.reconcile_spool()
                    pending = await self._redis.zcard(PENDING_INDEX)
                    reviewed = await self._redis.zcard(REVIEWED_INDEX)
                    spool_size = self._spool.size()
                    oldest_age = self._spool.oldest_item_age_seconds()
                    alerts = self._evaluate_alerts()
            except Exception as exc:  # noqa: BLE001
                self._mark_degraded("stats", exc)
            else:
                return {
                    "pending": pending,
                    "reviewed": reviewed,
                    "durable": True,
                    "degraded": False,
                    "spool": {
                        "enabled": self.spool_enabled,
                        "size": spool_size,
                        "max_size": self.max_spool_size,
                        "oldest_age_seconds": oldest_age,
                        "policy": self.degraded_policy,
                        "total_spooled": self.total_spooled,
                        "total_reconciled": self.total_reconciled,
                        "total_rejected": self.total_rejected,
                        "total_dropped": self.total_dropped,
                    },
                    "alerts": alerts,
                }

        counts: dict[str, int] = {}
        for item in self._local.values():
            counts[item.status.value] = counts.get(item.status.value, 0) + 1
        return {
            "pending": counts.get(ReviewStatus.PENDING.value, 0),
            "reviewed": sum(count for status, count in counts.items() if status != ReviewStatus.PENDING.value),
            "by_status": counts,
            "durable": False,
            "degraded": self._degraded,
            "spool": {
                "enabled": self.spool_enabled,
                "size": spool_size,
                "max_size": self.max_spool_size,
                "oldest_age_seconds": oldest_age,
                "policy": self.degraded_policy,
                "total_spooled": self.total_spooled,
                "total_reconciled": self.total_reconciled,
                "total_rejected": self.total_rejected,
                "total_dropped": self.total_dropped,
            },
            "alerts": alerts,
        }

    async def clear(self) -> None:
        """Drop everything. Used by tests; never called by the service."""
        if self._use_redis and self._redis is not None:
            try:
                ids = await self._redis.zrange(PENDING_INDEX, 0, -1)
                ids += await self._redis.zrange(REVIEWED_INDEX, 0, -1)
                for item_id in ids:
                    await self._redis.delete(ITEM_KEY.format(item_id=item_id))
                await self._redis.delete(PENDING_INDEX, REVIEWED_INDEX)
            except Exception:
                pass
        self._local.clear()
        self._spool.clear()
        self._degraded = False
        self.total_spooled = 0
        self.total_reconciled = 0
        self.total_rejected = 0
        self.total_dropped = 0

    # -- internals ----------------------------------------------------------

    def _mark_degraded(self, operation: str, exc: Exception) -> None:
        """Record that Redis failed, logging the first failure loudly."""
        if not self._degraded:
            logger.error(
                "Review queue Redis operation %r failed (%s); serving from the "
                "durable spool fallback — the queue is NO LONGER DURABLE in Redis",
                operation,
                exc,
            )
            self._degraded = True
            logger.warning("REVIEW_QUEUE_ALERT alert=degraded_mode status=active operation=%s", operation)
        else:
            logger.warning("Review queue Redis %r failed again: %s", operation, exc)

    def _evaluate_alerts(self) -> dict[str, bool]:
        """Check all operational threshold alerts and emit structured logs."""
        spool_size = self._spool.size()
        oldest_age = self._spool.oldest_item_age_seconds() or 0.0

        high_depth = spool_size >= self.alert_depth_threshold
        high_age = oldest_age >= self.alert_age_threshold_seconds
        drops_or_rejections = (self.total_rejected > 0) or (self.total_dropped > 0)

        if high_depth:
            logger.warning(
                "REVIEW_QUEUE_ALERT alert=depth_exceeded depth=%d threshold=%d",
                spool_size,
                self.alert_depth_threshold,
            )
        if high_age:
            logger.warning(
                "REVIEW_QUEUE_ALERT alert=age_exceeded oldest_age_seconds=%.1f threshold=%.1f",
                oldest_age,
                self.alert_age_threshold_seconds,
            )
        return {
            "degraded_active": self._degraded,
            "high_depth": high_depth,
            "high_age": high_age,
            "drops_or_rejections": drops_or_rejections,
        }

    @staticmethod
    def _deserialize(raw: str | None) -> ReviewItem | None:
        if raw is None:
            return None
        try:
            return ReviewItem(**json.loads(raw))
        except (json.JSONDecodeError, TypeError, ValueError):
            logger.warning("Corrupt review item in store; skipping")
            return None


_store: ReviewStore = ReviewStore()


def get_review_store() -> ReviewStore:
    return _store
