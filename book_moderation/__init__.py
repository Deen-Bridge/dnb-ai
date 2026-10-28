"""Book take-down and restore controls (issue #314).

The moderation state of the catalog lives here: a book is ``active`` or
``taken-down``, a take-down records why, and a restore puts the book back
exactly where it was. The rules that decide who keeps access to a taken-down
book are in :mod:`book_moderation.policy`; the routes are in
:mod:`book_moderation.api`.
"""

from __future__ import annotations

from book_moderation.models import (
    BookModeration,
    BookStatus,
    BookVisibility,
    TakeDownReason,
    TakeDownRequest,
    VisibilityRequest,
)
from book_moderation.policy import AccessDecision, Purchase, Viewer, decide_access
from book_moderation.service import BookModerationService, get_service
from book_moderation.store import (
    BookModerationStore,
    InMemoryBookModerationStore,
    RedisBookModerationStore,
    build_store,
)

__all__ = [
    "AccessDecision",
    "BookModeration",
    "BookModerationService",
    "BookModerationStore",
    "BookStatus",
    "BookVisibility",
    "InMemoryBookModerationStore",
    "Purchase",
    "RedisBookModerationStore",
    "TakeDownReason",
    "TakeDownRequest",
    "Viewer",
    "VisibilityRequest",
    "build_store",
    "decide_access",
    "get_service",
]
