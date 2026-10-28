"""Book take-down and restore controls (issue #314).

The service owns the moderation state of the catalog: an admin can take a book
down with a reason category, which hides it from the catalog and from reading
while the learners who bought it keep the access they paid for, and a restore
puts the book back exactly where it was.

Two routers are exposed:

* ``admin_router`` — the management routes, behind the admin gate.
* ``public_router`` — what a learner may see: the moderated catalog, the
  moderated book detail and the visibility of a book.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Header, Query
from fastapi.security import APIKeyHeader

from book_moderation.models import (
    BookStatus,
    BookVisibility,
    TakeDownRequest,
    VisibilityRequest,
)
from book_moderation.service import BookModerationService, get_service
from errors import APIException

_admin_header = APIKeyHeader(name="X-Admin-Token", auto_error=False)

admin_router = APIRouter(prefix="/books/moderation", tags=["book-moderation"])
public_router = APIRouter(prefix="/books", tags=["books"])


def get_book_moderation_service() -> BookModerationService:
    """Dependency seam so a test can run against an isolated store."""
    return get_service()


async def require_admin(token: str | None = Depends(_admin_header)) -> None:
    """The admin gate of this service.

    The gate itself lives in ``main`` — this module is mounted by the
    application, so importing it at module level would be circular. The import
    happens per request instead, which keeps one implementation of the gate.
    """
    from main import require_admin as app_require_admin

    await app_require_admin(token)


def _bearer_token(authorization: str | None) -> str | None:
    """The caller's platform token, when one was sent."""
    if not authorization:
        return None
    header = authorization.strip()
    if not header.lower().startswith("bearer "):
        return None
    token = header[7:].strip()
    return token or None


# -- admin ------------------------------------------------------------------


@admin_router.post(
    "/{book_id}/takedown",
    dependencies=[Depends(require_admin)],
    summary="Take a book down (admin)",
    description=(
        "Hides the book from the catalog and from the reading views and records why. "
        "Its author and the learners who bought it keep their access. "
        "Requires the `X-Admin-Token` header."
    ),
)
async def take_down_book(
    book_id: str,
    payload: TakeDownRequest,
    service: BookModerationService = Depends(get_book_moderation_service),
) -> dict:
    """Take a book down: it leaves the catalog and the reading views."""
    record = await service.take_down(
        book_id,
        payload.reason,
        note=payload.note,
        actor="admin",
    )
    return {"success": True, "moderation": record.model_dump(mode="json")}


@admin_router.post(
    "/{book_id}/restore",
    dependencies=[Depends(require_admin)],
    summary="Restore a taken-down book (admin)",
    description=(
        "Returns the book to the state it had before the take-down and clears the "
        "moderation metadata. Requires the `X-Admin-Token` header."
    ),
)
async def restore_book(
    book_id: str,
    service: BookModerationService = Depends(get_book_moderation_service),
) -> dict:
    """Restore a taken-down book to the state it had before."""
    record = await service.restore(book_id, actor="admin")
    return {"success": True, "moderation": record.model_dump(mode="json")}


@admin_router.get(
    "",
    dependencies=[Depends(require_admin)],
    summary="List the moderation state of the catalog (admin)",
    description=(
        "Every book the moderation state is known for, optionally filtered by "
        "`status` (`active` or `taken-down`). Requires the `X-Admin-Token` header."
    ),
)
async def list_moderation(
    status: str | None = Query(default=None),
    service: BookModerationService = Depends(get_book_moderation_service),
) -> dict:
    """List the moderation state of the catalog."""
    parsed: BookStatus | None = None
    if status is not None:
        try:
            parsed = BookStatus(status)
        except ValueError as exc:
            raise APIException(
                400,
                f"Invalid status filter '{status}'.",
                hint="Expected 'active' or 'taken-down'.",
            ) from exc
    records = await service.list_states(parsed)
    return {"success": True, "moderation": [record.model_dump(mode="json") for record in records]}


@admin_router.get(
    "/{book_id}",
    dependencies=[Depends(require_admin)],
    summary="The moderation state of one book (admin)",
    description="Requires the `X-Admin-Token` header.",
)
async def moderation_state(
    book_id: str,
    service: BookModerationService = Depends(get_book_moderation_service),
) -> dict:
    """The moderation state of one book."""
    record = await service.get_state(book_id)
    return {"success": True, "moderation": record.model_dump(mode="json") if record else None}


# -- learners ---------------------------------------------------------------


@public_router.get(
    "/catalog",
    summary="The catalog, with the taken-down books removed",
    description=(
        "The books the platform publishes. A book that was taken down is left out "
        "unless the caller keeps the access they already have — its author or a "
        "learner who bought it. The books that stay carry their `status`, so the "
        "library can badge them."
    ),
)
async def moderated_catalog(
    authorization: str | None = Header(default=None),
    category: str | None = Query(default=None),
    page: int | None = Query(default=None, ge=1),
    limit: int | None = Query(default=None, ge=1, le=100),
    service: BookModerationService = Depends(get_book_moderation_service),
) -> dict:
    """The catalog the learner may see."""
    params = {
        key: value for key, value in (("category", category), ("page", page), ("limit", limit)) if value is not None
    }
    books = await service.moderated_catalog(_bearer_token(authorization), params=params)
    return {"success": True, "books": books}


@public_router.get(
    "/{book_id}",
    summary="One book, if the caller may see it",
    description=(
        "The book as the platform publishes it. A taken-down book the caller may not see is reported as missing."
    ),
)
async def moderated_book(
    book_id: str,
    authorization: str | None = Header(default=None),
    service: BookModerationService = Depends(get_book_moderation_service),
) -> dict:
    """The book, unless it was taken down and the caller may not see it."""
    book = await service.moderated_book(book_id, _bearer_token(authorization))
    if book is None:
        raise APIException(
            404,
            f"Book {book_id} was not found.",
            hint="The book does not exist or it is not available to you.",
        )
    return {"success": True, "book": book}


@public_router.get(
    "/moderation/{book_id}/visibility",
    summary="What the caller may see of one book",
    description=(
        "Whether the caller may still see the book, whether that access is retained "
        "from a purchase made before the take-down, and on what basis the decision "
        "was made (`active`, `admin`, `author`, `purchaser` or `taken-down`)."
    ),
)
async def book_visibility(
    book_id: str,
    authorization: str | None = Header(default=None),
    service: BookModerationService = Depends(get_book_moderation_service),
) -> dict:
    """What the caller may still see of one book."""
    account = await service.account_for(_bearer_token(authorization))
    decision = await service.visibility(book_id, account)
    visibility = BookVisibility(
        book_id=book_id,
        visible=decision.visible,
        retained=decision.retained,
        basis=decision.basis,
    )
    return {"success": True, "visibility": visibility.model_dump(mode="json")}


@public_router.post(
    "/moderation/visibility",
    summary="What the caller may see of a list of books",
    description=(
        "The same decision as the single-book visibility route, for the books of a "
        "listing — the learner library asks for all of them in one pass."
    ),
)
async def books_visibility(
    payload: VisibilityRequest,
    authorization: str | None = Header(default=None),
    service: BookModerationService = Depends(get_book_moderation_service),
) -> dict:
    """What the caller may still see of a list of books — the library uses this."""
    account = await service.account_for(_bearer_token(authorization))
    decisions = await service.visibility_many(payload.book_ids, account)
    return {
        "success": True,
        "visibility": {
            book_id: {"visible": decision.visible, "retained": decision.retained, "basis": decision.basis}
            for book_id, decision in decisions.items()
        },
    }
