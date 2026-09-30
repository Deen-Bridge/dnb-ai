"""Pydantic models for per-user memory and per-chat summaries."""

from __future__ import annotations

import time

from pydantic import BaseModel, Field

MAX_FACTS = 20
MAX_TOPICS = 30
MAX_FACT_LENGTH = 500
MAX_TOPIC_LENGTH = 100
MAX_SUMMARY_LENGTH = 2000
VALID_KNOWLEDGE_LEVELS = frozenset({"beginner", "intermediate", "advanced"})


class TopicEntry(BaseModel):
    topic: str = Field(max_length=MAX_TOPIC_LENGTH)
    last_asked: float


class FactEntry(BaseModel):
    fact: str = Field(max_length=MAX_FACT_LENGTH)
    created_at: float


class UserProfile(BaseModel):
    user_id: str
    knowledge_level: str | None = None
    madhhab: str | None = None
    preferred_language: str | None = None
    topics_studied: list[TopicEntry] = Field(default_factory=list)
    remembered_facts: list[FactEntry] = Field(default_factory=list)
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)

    model_config = {"extra": "forbid"}


class ChatSummary(BaseModel):
    chat_id: str
    content: str = Field(max_length=MAX_SUMMARY_LENGTH)
    turn_count: int = 0
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)

    model_config = {"extra": "forbid"}


# Personal-context (per-user RAG) records — enrollments, course progress,
# purchases, pledges/donations, saved items, past Q&A. Each record is owned
# by exactly one user; ``user_id`` is the isolation boundary and is re-checked
# after retrieval so one user's records can never surface for another.
MAX_RECORD_TEXT_LENGTH = 1000

VALID_RECORD_TYPES = frozenset(
    {
        "enrollment",
        "course_progress",
        "purchase",
        "pledge",
        "saved_item",
        "past_qa",
    }
)


class PersonalRecord(BaseModel):
    """One retrievable per-user platform signal.

    ``text`` is the human-readable, embeddable content; ``source`` is the
    citable origin the model must attribute; ``metadata`` holds a few extra
    display fields. Everything here is one user's own data — never another's.
    """

    user_id: str
    record_type: str = Field(..., description="One of VALID_RECORD_TYPES")
    record_id: str = Field(..., description="Stable id within its signal type")
    text: str = Field(max_length=MAX_RECORD_TEXT_LENGTH)
    source: str = Field(..., description="Citable origin, e.g. 'enrollments'")
    metadata: dict[str, str] = Field(default_factory=dict)
    updated_at: float = Field(default_factory=time.time)

    model_config = {"extra": "forbid"}


class PersonalContextBundle(BaseModel):
    """A user's ingested records plus the freshness timestamp for staleness.

    ``fetched_at`` drives the on-read TTL check so a new enrollment or pledge
    is picked up without a full rebuild.
    """

    user_id: str
    records: list[PersonalRecord] = Field(default_factory=list)
    fetched_at: float = Field(default_factory=time.time)

    model_config = {"extra": "forbid"}
