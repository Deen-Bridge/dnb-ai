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
    embedding: list[float] = Field(default_factory=list)
    shared: bool = False


class FactEntry(BaseModel):
    fact: str = Field(max_length=MAX_FACT_LENGTH)
    created_at: float
    embedding: list[float] = Field(default_factory=list)
    shared: bool = False
    version: int = 1
    last_accessed: float = Field(default_factory=time.time)
    archived: bool = False
    metadata: dict = Field(default_factory=dict)


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
    embedding: list[float] = Field(default_factory=list)
    shared: bool = False
    version: int = 1

    model_config = {"extra": "forbid"}
