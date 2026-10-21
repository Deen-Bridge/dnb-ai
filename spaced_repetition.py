"""Durable, per-learner Islamic flashcards and SM-2 review scheduling.

The backend authenticates the learner and supplies their opaque user id; these
routes are protected by the service API key when mounted by main.py. Media URLs
are stored as references and are never fetched by this service.
"""

from __future__ import annotations

import math
import os
import sqlite3
import threading
import uuid
from datetime import UTC, datetime, timedelta
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Path as PathParam, Query
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, HttpUrl, computed_field, field_validator

router = APIRouter(prefix="/study/repetition", tags=["spaced repetition"])


class ContentType(str, Enum):
    hadith = "hadith"
    vocabulary = "vocabulary"
    ruling = "ruling"
    date = "date"
    terminology = "terminology"


class CardInput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    front: str = Field(min_length=1, max_length=10000)
    back: str = Field(min_length=1, max_length=20000)
    content_type: ContentType
    source: str | None = Field(default=None, max_length=2000)
    tags: list[str] = Field(default_factory=list, max_length=30)
    audio_url: HttpUrl | None = None
    image_url: HttpUrl | None = None

    @field_validator("tags")
    @classmethod
    def validate_tags(cls, tags: list[str]) -> list[str]:
        if any(not tag.strip() or len(tag) > 100 for tag in tags):
            raise ValueError("Tags must contain 1 to 100 characters")
        return list(dict.fromkeys(tag.strip() for tag in tags))


class Schedule(BaseModel):
    repetitions: int = Field(default=0, ge=0)
    interval_days: int = Field(default=0, ge=0, le=36500)
    ease: float = Field(default=2.5, ge=1.3, allow_inf_nan=False)
    due_at: AwareDatetime
    last_reviewed_at: AwareDatetime | None = None
    reviews: int = Field(default=0, ge=0)
    successes: int = Field(default=0, ge=0)
    lapses: int = Field(default=0, ge=0)

    # Mypy does not support Pydantic computed fields layered over properties.
    @computed_field  # type: ignore[prop-decorator]
    @property
    def retention_rate(self) -> float | None:
        return self.successes / self.reviews if self.reviews else None


class Card(BaseModel):
    id: str
    content: CardInput
    schedule: Schedule
    revision: int = 0


class ImportCards(BaseModel):
    model_config = ConfigDict(extra="forbid")
    cards: list[CardInput] = Field(min_length=1, max_length=500)


class ReviewInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    grade: int = Field(ge=0, le=5, strict=True)
    revision: int = Field(ge=0, strict=True, description="Revision from the card; prevents duplicate or stale reviews")


class Statistics(BaseModel):
    total_cards: int
    due_cards: int
    reviewed_cards: int
    reviews: int
    successful_reviews: int
    lapses: int
    retention_rate: float | None = Field(
        description="Observed successful recalls / reviews; null before the first review"
    )
    by_content_type: dict[str, int]


def utc_now() -> datetime:
    return datetime.now(UTC)


def schedule_review(state: Schedule, grade: int, now: datetime) -> Schedule:
    """Apply SM-2 interval/ease updates, with a 100 year interval safety cap.

    Grades 0..2 reset the repetition sequence; 3..5 count as recall. A failed
    card is due again in ten minutes for relearning rather than disappearing
    from today's queue. Successes use 1 day, 6 days, then ceil(interval * ease).
    """
    if isinstance(grade, bool) or not isinstance(grade, int) or not 0 <= grade <= 5:
        raise ValueError("grade must be an integer between 0 and 5")
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("Review time must have a timezone")
    now = now.astimezone(UTC)
    if state.last_reviewed_at is not None and now < state.last_reviewed_at:
        raise ValueError("Review time cannot precede the previous review")
    success = grade >= 3
    repetitions = state.repetitions + 1 if success else 0
    interval = (
        (1 if repetitions == 1 else 6 if repetitions == 2 else min(36500, math.ceil(state.interval_days * state.ease)))
        if success
        else 1
    )
    ease = max(1.3, state.ease + 0.1 - (5 - grade) * (0.08 + (5 - grade) * 0.02))
    return Schedule(
        repetitions=repetitions,
        interval_days=interval,
        ease=round(ease, 4),
        due_at=now + (timedelta(days=interval) if success else timedelta(minutes=10)),
        last_reviewed_at=now,
        reviews=state.reviews + 1,
        successes=state.successes + int(success),
        lapses=state.lapses + int(not success),
    )


class RepetitionStore:
    """SQLite transactions keep reviews atomic across workers and restarts."""

    def __init__(self, path: str) -> None:
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path, check_same_thread=False, timeout=15)
        self._lock = threading.Lock()
        with self._db:
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS repetition_cards "
                "(user_id TEXT NOT NULL, card_id TEXT NOT NULL, due REAL NOT NULL, "
                "card TEXT NOT NULL, PRIMARY KEY(user_id, card_id))"
            )
            self._db.execute("CREATE INDEX IF NOT EXISTS repetition_due ON repetition_cards(user_id, due, card_id)")

    def close(self) -> None:
        self._db.close()

    def create(self, user_id: str, inputs: list[CardInput], now: datetime) -> list[Card]:
        cards = [Card(id=str(uuid.uuid4()), content=item, schedule=Schedule(due_at=now)) for item in inputs]
        with self._lock, self._db:
            self._db.executemany(
                "INSERT INTO repetition_cards VALUES (?, ?, ?, ?)",
                [(user_id, card.id, card.schedule.due_at.timestamp(), card.model_dump_json()) for card in cards],
            )
        return cards

    def due(self, user_id: str, now: datetime, limit: int) -> list[Card]:
        with self._lock:
            rows = self._db.execute(
                "SELECT card FROM repetition_cards WHERE user_id=? AND due<=? ORDER BY due, card_id LIMIT ?",
                (user_id, now.timestamp(), limit),
            ).fetchall()
        return [Card.model_validate_json(row[0]) for row in rows]

    def get(self, user_id: str, card_id: str) -> Card:
        with self._lock:
            row = self._db.execute(
                "SELECT card FROM repetition_cards WHERE user_id=? AND card_id=?", (user_id, card_id)
            ).fetchone()
        if row is None:
            raise HTTPException(404, "Card not found")
        return Card.model_validate_json(row[0])

    def review(self, user_id: str, card_id: str, review: ReviewInput, now: datetime) -> Card:
        with self._lock, self._db:
            # Acquire the SQLite write reservation before reading; workers must
            # never both grade the same revision or overwrite each other's work.
            self._db.execute("BEGIN IMMEDIATE")
            row = self._db.execute(
                "SELECT card FROM repetition_cards WHERE user_id=? AND card_id=?", (user_id, card_id)
            ).fetchone()
            if row is None:
                raise HTTPException(404, "Card not found")
            card = Card.model_validate_json(row[0])
            if card.revision != review.revision:
                raise HTTPException(409, "Card changed; reload it before submitting another review")
            if now < card.schedule.due_at:
                raise HTTPException(409, "Card is not due yet")
            card.schedule = schedule_review(card.schedule, review.grade, now)
            card.revision += 1
            self._db.execute(
                "UPDATE repetition_cards SET due=?, card=? WHERE user_id=? AND card_id=?",
                (card.schedule.due_at.timestamp(), card.model_dump_json(), user_id, card_id),
            )
        return card

    def statistics(self, user_id: str, now: datetime) -> Statistics:
        with self._lock:
            rows = self._db.execute("SELECT card FROM repetition_cards WHERE user_id=?", (user_id,)).fetchall()
        cards = [Card.model_validate_json(row[0]) for row in rows]
        reviews = sum(card.schedule.reviews for card in cards)
        successes = sum(card.schedule.successes for card in cards)
        return Statistics(
            total_cards=len(cards),
            due_cards=sum(card.schedule.due_at <= now for card in cards),
            reviewed_cards=sum(card.schedule.reviews > 0 for card in cards),
            reviews=reviews,
            successful_reviews=successes,
            lapses=sum(card.schedule.lapses for card in cards),
            retention_rate=successes / reviews if reviews else None,
            by_content_type={
                kind.value: sum(card.content.content_type == kind for card in cards) for kind in ContentType
            },
        )


@lru_cache
def get_store() -> RepetitionStore:
    return RepetitionStore(os.getenv("REPETITION_DB_PATH", "data/repetition.db"))


UserId = Annotated[str, PathParam(min_length=1, max_length=128)]


@router.post("/{user_id}/cards", response_model=Card, status_code=201)
def create_card(user_id: UserId, body: CardInput, store: RepetitionStore = Depends(get_store)) -> Card:
    return store.create(user_id, [body], utc_now())[0]


@router.post("/{user_id}/import", response_model=list[Card], status_code=201)
def import_cards(user_id: UserId, body: ImportCards, store: RepetitionStore = Depends(get_store)) -> list[Card]:
    return store.create(user_id, body.cards, utc_now())


@router.get("/{user_id}/session", response_model=list[Card])
def daily_session(
    user_id: UserId,
    limit: int = Query(default=20, ge=1, le=200),
    store: RepetitionStore = Depends(get_store),
) -> list[Card]:
    return store.due(user_id, utc_now(), limit)


@router.get("/{user_id}/cards/{card_id}", response_model=Card)
def get_card(user_id: UserId, card_id: str, store: RepetitionStore = Depends(get_store)) -> Card:
    return store.get(user_id, card_id)


@router.post("/{user_id}/cards/{card_id}/review", response_model=Card)
def review_card(user_id: UserId, card_id: str, body: ReviewInput, store: RepetitionStore = Depends(get_store)) -> Card:
    return store.review(user_id, card_id, body, utc_now())


@router.get("/{user_id}/statistics", response_model=Statistics)
def statistics(user_id: UserId, store: RepetitionStore = Depends(get_store)) -> Statistics:
    return store.statistics(user_id, utc_now())
