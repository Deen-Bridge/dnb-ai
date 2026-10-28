"""Book moderation service — take-down, restore and the visibility that follows.

This service is the authority on whether a book is allowed to appear in the
catalog and in the reading views (issue #314). It never duplicates the
platform's data: purchases are read from the REST API, which owns them, and a
status change is pushed into the catalog caches of that same API so the change
is visible on the next request instead of after the cache TTL.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

from book_moderation.models import BookModeration, BookStatus, TakeDownReason
from book_moderation.platform import (
    Account,
    AccountLookup,
    BackendAccountLookup,
    BookCacheInvalidator,
    PlatformCatalogClient,
    book_author_id,
    book_identifier,
)
from book_moderation.policy import AccessDecision, Viewer, decide_access
from book_moderation.store import BookModerationStore, build_store
from errors import APIException

logger = logging.getLogger(__name__)


class BookModerationService:
    """The moderation flow and the visibility rules that depend on it."""

    def __init__(
        self,
        store: BookModerationStore,
        *,
        cache_invalidator: BookCacheInvalidator | None = None,
        account_lookup: AccountLookup | None = None,
        catalog_client: PlatformCatalogClient | None = None,
        clock: Any | None = None,
    ) -> None:
        self._store = store
        self._cache_invalidator = cache_invalidator
        self._account_lookup = account_lookup
        self._catalog_client = catalog_client
        self._clock = clock or (lambda: datetime.now(UTC))

    # -- state ---------------------------------------------------------------

    async def get_state(self, book_id: str) -> BookModeration | None:
        return await self._store.get(book_id)

    async def list_states(self, status: BookStatus | None = None) -> list[BookModeration]:
        records = await self._store.list_all()
        if status is None:
            return records
        return [record for record in records if record.status is status]

    # -- flow ----------------------------------------------------------------

    async def take_down(
        self,
        book_id: str,
        reason: TakeDownReason,
        *,
        note: str | None = None,
        actor: str | None = None,
    ) -> BookModeration:
        """Hide a book from the catalog and from reading."""
        record = await self._store.get(book_id)
        if record is not None and record.status is BookStatus.TAKEN_DOWN:
            raise APIException(
                409,
                f"Book {book_id} is already taken down.",
                hint="Restore the book before taking it down again.",
            )

        author_id = await self._author_of(book_id)

        updated = BookModeration(
            book_id=book_id,
            status=BookStatus.TAKEN_DOWN,
            previous_status=record.status if record is not None else BookStatus.ACTIVE,
            reason=reason,
            note=note,
            taken_down_at=self._clock(),
            taken_down_by=actor,
            restored_at=None,
            author_id=author_id or (record.author_id if record is not None else None),
        )
        await self._store.save(updated)
        await self._invalidate_caches()
        logger.info("Book %s taken down by %s (%s).", book_id, actor or "unknown", reason.value)
        return updated

    async def restore(self, book_id: str, *, actor: str | None = None) -> BookModeration:
        """Return a taken-down book to the state it had before."""
        record = await self._store.get(book_id)
        if record is None:
            raise APIException(
                404,
                f"No moderation record for book {book_id}.",
                hint="Only a book that was taken down can be restored.",
            )
        if record.status is not BookStatus.TAKEN_DOWN:
            raise APIException(
                409,
                f"Book {book_id} is not taken down.",
                hint="There is nothing to restore.",
            )

        restored = record.model_copy(
            update={
                "status": record.previous_status or BookStatus.ACTIVE,
                "previous_status": None,
                "reason": None,
                "note": None,
                "taken_down_at": None,
                "taken_down_by": None,
                "restored_at": self._clock(),
            }
        )
        await self._store.save(restored)
        await self._invalidate_caches()
        logger.info("Book %s restored by %s.", book_id, actor or "unknown")
        return restored

    # -- visibility ----------------------------------------------------------

    async def visibility(self, book_id: str, account: Account | None) -> AccessDecision:
        """Decide what a viewer may see of one book."""
        record = await self._store.get(book_id)
        return self._decide(record, account)

    async def visibility_many(self, book_ids: Sequence[str], account: Account | None) -> dict[str, AccessDecision]:
        """Decide what a viewer may see of many books in one pass."""
        records = {record.book_id: record for record in await self._store.list_all()}
        viewer = _viewer_of(account)
        decisions: dict[str, AccessDecision] = {}
        for book_id in book_ids:
            purchase = account.purchase_for(book_id) if account else None
            decisions[book_id] = decide_access(records.get(book_id), viewer, purchase)
        return decisions

    async def account_for(self, auth_token: str | None) -> Account | None:
        """Resolve the viewer through the platform, when a token was given."""
        if not auth_token or self._account_lookup is None:
            return None
        try:
            return await self._account_lookup.account(auth_token)
        except Exception as exc:  # noqa: BLE001 - the platform being down must not leak a taken-down book
            logger.warning("Could not resolve the viewer account (%s); treating the viewer as unentitled.", exc)
            return None

    async def moderated_catalog(
        self,
        auth_token: str | None = None,
        *,
        params: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """The platform's catalog with the hidden books removed.

        A book that was taken down leaves the listing for everybody except the
        people who keep the access they already have: its author and the
        learners who bought it. The books that stay carry their moderation
        state, so the library can badge them.
        """
        if self._catalog_client is None:
            raise APIException(
                503,
                "The catalog is not available on this server.",
                hint="DNB_BACKEND_URL is not configured.",
            )
        try:
            books = await self._catalog_client.books(auth_token, params=params)
        except Exception as exc:  # noqa: BLE001 - the platform is the source of the catalog
            logger.warning("Could not read the catalog from the platform (%s).", exc)
            raise APIException(
                502,
                "Could not read the catalog from the platform.",
                hint="The book catalog is temporarily unavailable. Please retry.",
            ) from exc
        if not books:
            return []

        records = {record.book_id: record for record in await self._store.list_all()}
        account = await self.account_for(auth_token)
        viewer = _viewer_of(account)

        visible: list[dict[str, Any]] = []
        for book in books:
            book_id = book_identifier(book)
            record = records.get(book_id) if book_id else None
            purchase = account.purchase_for(book_id) if account and book_id else None
            decision = decide_access(record, viewer, purchase, book_author_id(book))
            if not decision.visible:
                continue
            book["status"] = record.status.value if record else BookStatus.ACTIVE.value
            if record is not None and record.reason is not None:
                book["moderationReason"] = record.reason.value
            visible.append(book)
        return visible

    async def moderated_book(
        self,
        book_id: str,
        auth_token: str | None = None,
    ) -> dict[str, Any] | None:
        """One book, or None when the viewer may not see it."""
        if self._catalog_client is None:
            raise APIException(
                503,
                "The catalog is not available on this server.",
                hint="DNB_BACKEND_URL is not configured.",
            )
        try:
            book = await self._catalog_client.book(book_id, auth_token)
        except Exception as exc:  # noqa: BLE001 - the platform is the source of the book
            logger.warning("Could not read book %s from the platform (%s).", book_id, exc)
            raise APIException(
                502,
                f"Could not read book {book_id} from the platform.",
                hint="The book catalog is temporarily unavailable. Please retry.",
            ) from exc
        if book is None:
            return None

        record = await self._store.get(book_id)
        account = await self.account_for(auth_token)
        decision = self._decide(record, account, book_author_id(book))
        if not decision.visible:
            return None
        book["status"] = record.status.value if record else BookStatus.ACTIVE.value
        if record is not None and record.reason is not None:
            book["moderationReason"] = record.reason.value
        return book

    # -- internals -----------------------------------------------------------

    @staticmethod
    def _decide(
        record: BookModeration | None,
        account: Account | None,
        author_id: str | None = None,
    ) -> AccessDecision:
        purchase = account.purchase_for(record.book_id) if account and record else None
        return decide_access(record, _viewer_of(account), purchase, author_id)

    async def _author_of(self, book_id: str) -> str | None:
        """The author of the book, according to the platform.

        The author keeps access to their own book while it is taken down, so
        the id has to come from the platform — a caller could otherwise claim
        the authorship of any book. A book the platform does not publish cannot
        be taken down; a platform that cannot be reached only costs the author
        their badge, so the take-down goes ahead.
        """
        if self._catalog_client is None:
            return None
        try:
            book = await self._catalog_client.book(book_id)
        except Exception as exc:  # noqa: BLE001 - the platform holds the catalog
            logger.warning("Could not read book %s from the platform (%s).", book_id, exc)
            return None
        if book is None:
            raise APIException(
                404,
                f"Book {book_id} does not exist.",
                hint="Only a book the platform publishes can be taken down.",
            )
        return book_author_id(book)

    async def _invalidate_caches(self) -> None:
        if self._cache_invalidator is None:
            return
        try:
            await self._cache_invalidator.invalidate()
        except Exception as exc:  # noqa: BLE001 - a cache failure must not undo a moderation change
            logger.warning("Catalog cache invalidation failed (%s).", exc)


def _viewer_of(account: Account | None) -> Viewer:
    return Viewer(user_id=account.user_id if account else None)


_service: BookModerationService | None = None


def get_service() -> BookModerationService:
    """The service used by the API routes."""
    global _service
    if _service is None:
        _service = BookModerationService(
            build_store(),
            cache_invalidator=BookCacheInvalidator(),
            account_lookup=BackendAccountLookup(),
            catalog_client=PlatformCatalogClient(),
        )
    return _service
