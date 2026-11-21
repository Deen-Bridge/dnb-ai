"""User-scoped study progress and personalization store (#846 / #847).

Provides isolated, opt-in, and resettable storage for learner study progress,
mastery tracking, and adaptive quiz results. Supports in-memory LRU caching
for local development and Redis for production persistence.
"""

from __future__ import annotations

import json
import logging
import os
from collections import OrderedDict
from datetime import UTC, datetime

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

MAX_IN_MEMORY_USERS = 5000
STUDY_PROGRESS_TTL_SECONDS = 365 * 86400  # 1 year


class UserStudyRecord(BaseModel):
    """User-scoped study profile and mastery history."""

    user_id: str
    opt_in: bool = True
    quizzes_completed: int = Field(default=0, ge=0)
    total_items_answered: int = Field(default=0, ge=0)
    cumulative_score: float = Field(default=0.0, ge=0.0)
    mastery_by_discipline: dict[str, float] = Field(default_factory=dict)
    topics_studied: list[str] = Field(default_factory=list)
    last_updated_iso: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


class StudyStore:
    """Storage interface for user study progress."""

    def __init__(self, redis_url: str | None = None):
        self.redis_url = redis_url or os.getenv("REDIS_URL")
        self._memory_cache: OrderedDict[str, UserStudyRecord] = OrderedDict()
        self._redis_client = None

        if self.redis_url:
            try:
                import redis.asyncio as aioredis

                self._redis_client = aioredis.from_url(self.redis_url, decode_responses=True)
                logger.info("StudyStore initialized with Redis backend")
            except Exception as exc:
                logger.warning(
                    "StudyStore failed to connect to Redis (%s); falling back to in-memory store",
                    exc,
                )
                self._redis_client = None
        else:
            logger.info("StudyStore initialized with in-memory backend")

    def _format_key(self, user_id: str) -> str:
        clean_id = user_id.strip()
        return f"study:user:{clean_id}"

    async def get_progress(self, user_id: str) -> UserStudyRecord | None:
        """Retrieve progress for a specific user. Returns None if absent or not opted in."""
        key = self._format_key(user_id)

        if self._redis_client:
            try:
                raw = await self._redis_client.get(key)
                if raw:
                    data = json.loads(raw)
                    record = UserStudyRecord.model_validate(data)
                    return record if record.opt_in else None
                return None
            except Exception as exc:
                logger.error("Redis read error in StudyStore.get_progress: %s", exc)

        # In-memory fallback
        record = self._memory_cache.get(key)
        if record:
            self._memory_cache.move_to_end(key)
            return record if record.opt_in else None
        return None

    async def record_progress(
        self,
        user_id: str,
        discipline: str,
        score: float,
        items_count: int,
        topic: str | None = None,
    ) -> UserStudyRecord:
        """Update study progress for an opted-in user."""
        key = self._format_key(user_id)
        current = await self.get_progress(user_id)

        if current is None:
            current = UserStudyRecord(user_id=user_id, opt_in=True)

        current.quizzes_completed += 1
        current.total_items_answered += items_count
        current.cumulative_score += score

        # Update discipline mastery with rolling exponential moving average
        existing_mastery = current.mastery_by_discipline.get(discipline, 0.5)
        new_mastery = round(0.7 * existing_mastery + 0.3 * (score / 100.0), 3)
        current.mastery_by_discipline[discipline] = new_mastery

        if topic and topic not in current.topics_studied:
            current.topics_studied.append(topic)
            if len(current.topics_studied) > 50:
                current.topics_studied = current.topics_studied[-50:]

        current.last_updated_iso = datetime.now(UTC).isoformat()

        # Persist to Redis if available
        if self._redis_client:
            try:
                await self._redis_client.set(
                    key,
                    current.model_dump_json(),
                    ex=STUDY_PROGRESS_TTL_SECONDS,
                )
            except Exception as exc:
                logger.error("Redis write error in StudyStore.record_progress: %s", exc)

        # In-memory store
        self._memory_cache[key] = current
        self._memory_cache.move_to_end(key)
        if len(self._memory_cache) > MAX_IN_MEMORY_USERS:
            self._memory_cache.popitem(last=False)

        return current

    async def reset_user(self, user_id: str) -> bool:
        """Permanently erase all stored study progress for a user (opt-out / reset)."""
        key = self._format_key(user_id)
        existed = False

        if self._redis_client:
            try:
                deleted = await self._redis_client.delete(key)
                if deleted > 0:
                    existed = True
            except Exception as exc:
                logger.error("Redis delete error in StudyStore.reset_user: %s", exc)

        if key in self._memory_cache:
            del self._memory_cache[key]
            existed = True

        logger.info("Study progress reset for user hash %s", user_id[:8] if len(user_id) >= 8 else user_id)
        return existed


# Global store instance
study_store = StudyStore()
