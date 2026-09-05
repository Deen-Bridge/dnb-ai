"""Per-user long-term memory and conversation summarization.

See README.md for usage; see tests/ for offline-verifiable contracts.
"""

from __future__ import annotations

import logging
import os

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
    profile: UserProfile | None,
    summary: ChatSummary | None,
    max_chars: int | None = None,
) -> str:
    """Render profile and chat summary as a delimited DATA block.

    Returns an empty string when neither has content so anonymous traffic
    is completely unaffected.  When *max_chars* is given, the returned
    block is truncated to that many characters to fit context windows.
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
            topics_str = ", ".join(f"{t.topic}" for t in profile.topics_studied[-10:])
            lines.append(f"Topics studied: {topics_str}")
        if profile.remembered_facts:
            for fact in profile.remembered_facts[-5:]:
                lines.append(f"- {fact.fact}")
        parts.append("\n".join(lines))

    if summary is not None and summary.content:
        parts.append(f"--- Conversation summary ---\n{summary.content}")

    if not parts:
        return ""

    result = "\n\n".join(parts) + "\n---------------------------------\n"
    if max_chars is not None:
        result = result[:max_chars]
    return result


def retrieve_context(
    store: MemoryStore,
    user_id: str,
    query: str = "",
    *,
    max_chars: int | None = None,
) -> str:
    """Retrieve a user's memory context for an agent query.

    Loads profile and chat summary from *store*, optionally ranks remembered
    facts by token overlap with *query*, and renders the result.  This lets
    agents share a single backing store while keeping context bounded.
    """
    load_profile = getattr(store, "load_user_profile", None)
    if load_profile is None:
        load_profile = getattr(store, "get_user_profile")
    profile = load_profile(user_id)

    load_summary = getattr(store, "load_chat_summary", None)
    if load_summary is None:
        load_summary = getattr(store, "get_chat_summary")
    summary = load_summary(user_id)

    if query and profile is not None and profile.remembered_facts:
        q_tokens = set(query.lower().split())
        profile.remembered_facts = sorted(
            profile.remembered_facts,
            key=lambda f: len(q_tokens & set(f.fact.lower().split())),
            reverse=True,
        )[:5]

    return render_user_context(profile, summary, max_chars=max_chars)


__all__ = [
    "ChatSummary",
    "InMemoryMemoryStore",
    "MemoryStore",
    "RedisMemoryStore",
    "UserProfile",
    "create_memory_store",
    "render_user_context",
    "retrieve_context",
]
