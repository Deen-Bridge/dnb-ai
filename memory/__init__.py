"""Per-user long-term memory and conversation summarization.

See README.md for usage; see tests/ for offline-verifiable contracts.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from memory.models import ChatSummary, UserProfile
from memory.store import (
    InMemoryMemoryStore,
    MemoryStore,
    RedisMemoryStore,
)

logger = logging.getLogger(__name__)


def create_memory_store() -> MemoryStore:
    """Factory: ``REDIS_URL`` set → ``RedisMemoryStore``, else in-memory.

    When Redis is configured but fails at startup the error is surfaced
    (logged + raised) — workers must not silently diverge on user memory.
    """
    url = os.getenv("REDIS_URL", "")
    if url:
        logger.info("MemoryStore using Redis at %s", url)
        return RedisMemoryStore(url)
    logger.info("MemoryStore using in-memory dict (local development)")
    return InMemoryMemoryStore()


def render_user_context(
    profile: Optional[UserProfile],
    summary: Optional[ChatSummary],
) -> str:
    """Render profile and chat summary as a delimited DATA block.

    Returns an empty string when neither has content so anonymous traffic
    is completely unaffected.
    """
    parts: list[str] = []

    if profile is not None and (
        profile.knowledge_level
        or profile.madhhab
        or profile.preferred_language
        or profile.topics_studied
        or profile.remembered_facts
    ):
        lines = ["--- Known about this student ---"]
        if profile.knowledge_level:
            lines.append(f"Knowledge level: {profile.knowledge_level}")
        if profile.madhhab:
            lines.append(f"Madhhab: {profile.madhhab}")
        if profile.preferred_language:
            lines.append(f"Preferred language: {profile.preferred_language}")
        if profile.topics_studied:
            topics_str = ", ".join(
                f"{t.topic}" for t in profile.topics_studied[-10:]
            )
            lines.append(f"Topics studied: {topics_str}")
        if profile.remembered_facts:
            for fact in profile.remembered_facts[-5:]:
                lines.append(f"- {fact.fact}")
        parts.append("\n".join(lines))

    if summary is not None and summary.content:
        parts.append(f"--- Conversation summary ---\n{summary.content}")

    if not parts:
        return ""

    return "\n\n".join(parts) + "\n---------------------------------\n"


__all__ = [
    "ChatSummary",
    "InMemoryMemoryStore",
    "MemoryStore",
    "RedisMemoryStore",
    "UserProfile",
    "create_memory_store",
    "render_user_context",
]
