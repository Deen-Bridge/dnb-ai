"""Durable store for the scholar-review queue.

Persistence choice
------------------
Same shape as the session store that shipped with session persistence (#3):
Redis when ``REDIS_URL`` is set, an in-process dict otherwise, behind one
async interface so callers never branch on which backend is live. This is
deliberately the *same* store, not a parallel one — a review queue and a chat
session have the same durability requirement, and a service with two unrelated
persistence layers is a service where one of them quietly stops being backed up.

Unlike sessions, review items have **no TTL**. A question waiting on a scholar
must not evaporate because nobody got to it within a day.

Redis layout
------------
- ``review:item:{id}``   — the JSON-serialized item
- ``review:pending``     — sorted set of pending ids, scored by creation time
- ``review:reviewed``    — sorted set of decided ids, scored by decision time

Items are moved between the two indexes on verdict, so listing the pending queue
never scans every item ever recorded.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

REDIS_URL = os.getenv("REDIS_URL", "")

ITEM_KEY = "review:item:{item_id}"
PENDING_INDEX = "review:pending"
REVIEWED_INDEX = "review:reviewed"

try:
    import redis.asyncio as aioredis

    _redis_available = True
except ImportError:  # pragma: no cover - depends on deployment extras
    _redis_available = False


class ReviewStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    CORRECTED = "corrected"
    REJECTED = "rejected"


class Verdict(str, Enum):
    APPROVE = "approve"
    CORRECT = "correct"
    REJECT = "reject"


VERDICT_STATUS: Dict[Verdict, ReviewStatus] = {
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
    signals: Dict[str, float] = {}
    chat_id: Optional[str] = None
    created_at: float = Field(default_factory=time.time)

    status: ReviewStatus = ReviewStatus.PENDING
    verdict: Optional[Verdict] = None
    corrected_answer: Optional[str] = None
    reviewer: Optional[str] = None
    reviewer_note: Optional[str] = None
    reviewed_at: Optional[float] = None

    @property
    def final_answer(self) -> Optional[str]:
        """The answer a reviewer stands behind, or None if there isn't one."""
        if self.status is ReviewStatus.APPROVED:
            return self.answer
        if self.status is ReviewStatus.CORRECTED:
            return self.corrected_answer
        return None


class ReviewStore:
    """Persists review items. Redis-backed when configured, in-memory otherwise."""

    def __init__(self) -> None:
        self._redis: Optional[Any] = None
        self._local: Dict[str, ReviewItem] = {}
        self._use_redis = False

        if REDIS_URL and _redis_available:
            try:
                self._redis = aioredis.from_url(REDIS_URL, decode_responses=True)
                self._use_redis = True
                logger.info("ReviewStore using Redis")
            except Exception as exc:  # pragma: no cover - connection-time only
                logger.warning(
                    "Failed to connect to Redis for the review queue (%s); "
                    "falling back to in-memory",
                    exc,
                )

        if not self._use_redis:
            logger.info(
                "ReviewStore using in-memory dict — set REDIS_URL to persist "
                "the scholar-review queue across restarts"
            )

    @property
    def durable(self) -> bool:
        return self._use_redis

    # -- public API ---------------------------------------------------------

    async def add(self, item: ReviewItem) -> ReviewItem:
        if self._use_redis:
            await self._redis.set(
                ITEM_KEY.format(item_id=item.id), item.model_dump_json()
            )
            await self._redis.zadd(PENDING_INDEX, {item.id: item.created_at})
        else:
            self._local[item.id] = item
        logger.info(
            "Queued answer %s for scholar review (confidence=%.2f)",
            item.id,
            item.confidence,
        )
        return item

    async def get(self, item_id: str) -> Optional[ReviewItem]:
        if self._use_redis:
            raw = await self._redis.get(ITEM_KEY.format(item_id=item_id))
            return self._deserialize(raw)
        return self._local.get(item_id)

    async def list_pending(self, limit: int = 50, offset: int = 0) -> List[ReviewItem]:
        """Oldest pending items first — the longest wait gets seen first."""
        if self._use_redis:
            ids = await self._redis.zrange(PENDING_INDEX, offset, offset + limit - 1)
            items = [await self.get(item_id) for item_id in ids]
            return [item for item in items if item is not None]
        pending = [
            item
            for item in self._local.values()
            if item.status is ReviewStatus.PENDING
        ]
        pending.sort(key=lambda item: item.created_at)
        return pending[offset:offset + limit]

    async def list_reviewed(self, limit: int = 50, offset: int = 0) -> List[ReviewItem]:
        """Most recently decided items first."""
        if self._use_redis:
            ids = await self._redis.zrevrange(
                REVIEWED_INDEX, offset, offset + limit - 1
            )
            items = [await self.get(item_id) for item_id in ids]
            return [item for item in items if item is not None]
        reviewed = [
            item
            for item in self._local.values()
            if item.status is not ReviewStatus.PENDING
        ]
        reviewed.sort(key=lambda item: item.reviewed_at or 0, reverse=True)
        return reviewed[offset:offset + limit]

    async def record_verdict(
        self,
        item_id: str,
        verdict: Verdict,
        corrected_answer: Optional[str] = None,
        reviewer: Optional[str] = None,
        reviewer_note: Optional[str] = None,
    ) -> Optional[ReviewItem]:
        """Apply a reviewer's decision. Returns None if the item doesn't exist."""
        item = await self.get(item_id)
        if item is None:
            return None

        item.verdict = verdict
        item.status = VERDICT_STATUS[verdict]
        item.corrected_answer = corrected_answer
        item.reviewer = reviewer
        item.reviewer_note = reviewer_note
        item.reviewed_at = time.time()

        if self._use_redis:
            await self._redis.set(
                ITEM_KEY.format(item_id=item.id), item.model_dump_json()
            )
            await self._redis.zrem(PENDING_INDEX, item.id)
            await self._redis.zadd(REVIEWED_INDEX, {item.id: item.reviewed_at})
        else:
            self._local[item.id] = item

        logger.info("Review %s recorded: %s", item.id, verdict.value)
        return item

    async def stats(self) -> Dict[str, Any]:
        if self._use_redis:
            pending = await self._redis.zcard(PENDING_INDEX)
            reviewed = await self._redis.zcard(REVIEWED_INDEX)
            return {
                "pending": pending,
                "reviewed": reviewed,
                "durable": True,
            }
        counts: Dict[str, int] = {}
        for item in self._local.values():
            counts[item.status.value] = counts.get(item.status.value, 0) + 1
        return {
            "pending": counts.get(ReviewStatus.PENDING.value, 0),
            "reviewed": sum(
                count
                for status, count in counts.items()
                if status != ReviewStatus.PENDING.value
            ),
            "by_status": counts,
            "durable": False,
        }

    async def clear(self) -> None:
        """Drop everything. Used by tests; never called by the service."""
        if self._use_redis:  # pragma: no cover - not exercised offline
            ids = await self._redis.zrange(PENDING_INDEX, 0, -1)
            ids += await self._redis.zrange(REVIEWED_INDEX, 0, -1)
            for item_id in ids:
                await self._redis.delete(ITEM_KEY.format(item_id=item_id))
            await self._redis.delete(PENDING_INDEX, REVIEWED_INDEX)
        else:
            self._local.clear()

    # -- internals ----------------------------------------------------------

    @staticmethod
    def _deserialize(raw: Optional[str]) -> Optional[ReviewItem]:
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
