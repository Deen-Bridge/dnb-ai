"""Domain model for the book take-down / restore flow (issue #314).

A book is either ``active`` — listed in the catalog and readable — or
``taken-down``, which hides it from the catalog and from reading. The state
lives in this service so that every consumer (the catalog, the learner
library, the reading views) asks the same authority, and so that a restore can
put a book back exactly where it was before.

The vocabulary deliberately mirrors the product contract: the web app renders
``"taken-down"`` on the library card and offers exactly the four reason
categories below in its take-down dialog.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, field_validator


class BookStatus(str, Enum):
    """Catalog state of a book."""

    ACTIVE = "active"
    TAKEN_DOWN = "taken-down"


class TakeDownReason(str, Enum):
    """Why an admin took a book down."""

    CONTENT_VIOLATION = "content-violation"
    DMCA = "dmca"
    QUALITY_ISSUES = "quality-issues"
    OTHER = "other"


class BookModeration(BaseModel):
    """Moderation state of a single book."""

    book_id: str
    status: BookStatus = BookStatus.ACTIVE
    # The state a restore returns the book to.
    previous_status: BookStatus | None = None
    reason: TakeDownReason | None = None
    note: str | None = None
    taken_down_at: datetime | None = None
    taken_down_by: str | None = None
    restored_at: datetime | None = None
    # The book's author keeps access to their own work while it is taken down.
    author_id: str | None = None


class TakeDownRequest(BaseModel):
    """Body of a take-down request."""

    reason: TakeDownReason
    note: str | None = Field(default=None, max_length=2000)

    @field_validator("note")
    @classmethod
    def _blank_note_is_no_note(cls, value: str | None) -> str | None:
        if value is None:
            return None
        trimmed = value.strip()
        return trimmed or None


class VisibilityRequest(BaseModel):
    """Body of a batch visibility query."""

    book_ids: list[str] = Field(default_factory=list, max_length=500)


class BookVisibility(BaseModel):
    """What a viewer is allowed to see of one book."""

    book_id: str
    visible: bool
    # True when the viewer keeps access because of what they already own
    # (a purchase, or authorship of the book), not because the book is active.
    retained: bool
    # Why the decision was made: "active", "purchaser", "author" or
    # "taken-down".
    basis: str
