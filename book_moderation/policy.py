"""Visibility rules for books under moderation (issue #314).

Pure functions, no I/O: the same rules decide what the catalog shows, what a
learner may still read, and what the library card is allowed to claim.

The rule set mirrors the access policy the REST API enforces for reading a
book: the author of the book and the learners who bought it keep their access
when it is taken down, everybody else loses the catalog entry and the reading
view. The API owns the purchase list and it does not compare the purchase
date with the take-down timestamp, so neither does this service — a stricter
rule here would disagree with the file access the API still grants.
"""

from __future__ import annotations

from dataclasses import dataclass

from book_moderation.models import BookModeration, BookStatus


@dataclass(frozen=True)
class Viewer:
    """Who is asking."""

    user_id: str | None = None


@dataclass(frozen=True)
class Purchase:
    """A purchase the platform holds for the viewer."""

    book_id: str


@dataclass(frozen=True)
class AccessDecision:
    """Outcome of the visibility rules."""

    visible: bool
    retained: bool
    basis: str


def decide_access(
    record: BookModeration | None,
    viewer: Viewer,
    purchase: Purchase | None = None,
    author_id: str | None = None,
) -> AccessDecision:
    """Decide whether ``viewer`` may still see and read the book.

    ``author_id`` is the author the platform holds for the book; the id stored
    with the moderation record is the fallback when the catalog is not at hand.
    """
    if record is None or record.status is BookStatus.ACTIVE:
        return AccessDecision(visible=True, retained=False, basis="active")

    owner = author_id or record.author_id
    if owner and viewer.user_id and owner == viewer.user_id:
        return AccessDecision(visible=True, retained=True, basis="author")

    if purchase and purchase.book_id == record.book_id:
        return AccessDecision(visible=True, retained=True, basis="purchaser")

    return AccessDecision(visible=False, retained=False, basis="taken-down")
