"""Everything that talks to the platform (issue #314).

The moderation service does not own the catalog, the purchases or the admin
identity — the REST API does. This module is the only place where the service
reads from that API, so the rest of the package stays free of I/O details and
can be tested against a mocked transport.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx

from book_moderation.policy import Purchase

logger = logging.getLogger(__name__)

DNB_BACKEND_URL = os.getenv("DNB_BACKEND_URL", "https://dnb-backend-api.onrender.com").rstrip("/")
ACCOUNT_LOOKUP_TIMEOUT = float(os.getenv("PURCHASE_FETCH_TIMEOUT", "10"))
CATALOG_LOOKUP_TIMEOUT = float(os.getenv("CATALOG_FETCH_TIMEOUT", "10"))

# The cache keys the REST API keeps for the catalog, the book detail and the
# educator listings that embed book titles. Dropping them is the invalidation
# pattern the API itself uses after a write.
BOOK_CACHE_PATTERNS = ("books:*", "book:*", "educators:*")


@dataclass(frozen=True)
class Account:
    """What the platform knows about the viewer."""

    user_id: str | None = None
    purchases: tuple[Purchase, ...] = field(default_factory=tuple)

    def purchase_for(self, book_id: str) -> Purchase | None:
        for purchase in self.purchases:
            if purchase.book_id == book_id:
                return purchase
        return None


class AccountLookup(Protocol):
    """Resolves the viewer and what they own."""

    async def account(self, auth_token: str) -> Account | None: ...


class BackendAccountLookup:
    """Reads the viewer and their purchases from the REST API.

    The API owns the purchase records, so the moderation service asks it
    instead of keeping a second copy of what people own. The call also tells
    us who the viewer is — the API validates the token, so the moderation
    service never has to trust an unverified identity.
    """

    def __init__(self, base_url: str | None = None, timeout: float | None = None) -> None:
        self._base_url = (base_url or DNB_BACKEND_URL).rstrip("/")
        self._timeout = timeout if timeout is not None else ACCOUNT_LOOKUP_TIMEOUT

    async def account(self, auth_token: str, *, client: httpx.AsyncClient | None = None) -> Account | None:
        url = f"{self._base_url}/api/users/me/export"
        headers = {"Authorization": f"Bearer {auth_token.strip()}"}
        owns_client = client is None
        client = client or httpx.AsyncClient(timeout=self._timeout)
        try:
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            payload = response.json()
        finally:
            if owns_client:
                await client.aclose()

        if not isinstance(payload, dict):
            return None
        data = payload.get("data")
        if not isinstance(data, dict):
            return None

        purchases: list[Purchase] = []
        for entry in data.get("purchasedBooks") or []:
            if not isinstance(entry, dict):
                continue
            book_id = entry.get("bookId") or entry.get("book_id")
            if not book_id:
                continue
            purchases.append(Purchase(book_id=str(book_id)))

        user_id = data.get("_id") or data.get("id")
        return Account(
            user_id=str(user_id) if user_id else None,
            purchases=tuple(purchases),
        )


class InMemoryAccountLookup:
    """Static account source — tests and local development."""

    def __init__(self, accounts: dict[str, Account] | None = None) -> None:
        self._accounts = accounts or {}

    async def account(self, auth_token: str) -> Account | None:
        return self._accounts.get(auth_token.strip())


class PlatformCatalogClient:
    """Reads the catalog and the book detail from the REST API.

    The catalog itself stays where it is — this service does not copy it. It
    reads the books the platform publishes and applies the moderation rules on
    top, so a taken-down book disappears from the listing without the platform
    having to know about the moderation state.
    """

    def __init__(self, base_url: str | None = None, timeout: float | None = None) -> None:
        self._base_url = (base_url or DNB_BACKEND_URL).rstrip("/")
        self._timeout = timeout if timeout is not None else CATALOG_LOOKUP_TIMEOUT

    async def books(
        self,
        auth_token: str | None = None,
        *,
        params: dict[str, Any] | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> list[dict[str, Any]]:
        """The books the platform publishes.

        The API answers with ``{"success": true, "data": [...]}`` — the
        paginated variant carries the page counters next to the books — so the
        list is read from the ``data`` key.
        """
        payload = await self._get("/api/books", auth_token, params=params, client=client)
        books: Any = None
        if isinstance(payload, list):
            books = payload
        elif isinstance(payload, dict):
            for key in ("data", "books"):
                candidate = payload.get(key)
                if isinstance(candidate, list):
                    books = candidate
                    break
        if not isinstance(books, list):
            return []
        return [book for book in books if isinstance(book, dict)]

    async def book(
        self,
        book_id: str,
        auth_token: str | None = None,
        *,
        client: httpx.AsyncClient | None = None,
    ) -> dict[str, Any] | None:
        """One book, or None when the platform does not publish it."""
        try:
            payload = await self._get(f"/api/books/{book_id}", auth_token, client=client)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                return None
            raise
        if not isinstance(payload, dict):
            return None
        for key in ("data", "book"):
            candidate = payload.get(key)
            if isinstance(candidate, dict):
                return candidate
        return payload if payload.get("_id") or payload.get("id") else None

    async def _get(
        self,
        path: str,
        auth_token: str | None,
        *,
        params: dict[str, Any] | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> Any:
        url = f"{self._base_url}{path}"
        headers = {"Authorization": f"Bearer {auth_token.strip()}"} if auth_token else {}
        owns_client = client is None
        client = client or httpx.AsyncClient(timeout=self._timeout)
        try:
            response = await client.get(url, headers=headers, params=params)
            response.raise_for_status()
            return response.json()
        finally:
            if owns_client:
                await client.aclose()


class BookCacheInvalidator:
    """Drops the catalog caches of the REST API after a status change."""

    def __init__(self, redis_url: str | None = None, *, client: Any | None = None) -> None:
        self._redis_url = redis_url if redis_url is not None else os.getenv("REDIS_URL", "")
        self._client = client

    async def invalidate(self) -> int:
        """Delete the cached catalog entries; returns the number of keys removed."""
        client = self._client
        if client is None:
            if not self._redis_url:
                logger.info("REDIS_URL is not configured; the catalog caches are left untouched.")
                return 0
            try:
                import redis.asyncio as aioredis
            except ImportError:  # pragma: no cover - redis is a hard dependency
                logger.warning("redis package is not installed; the catalog caches are left untouched.")
                return 0
            client = aioredis.from_url(self._redis_url, decode_responses=True)
            self._client = client

        deleted = 0
        for pattern in BOOK_CACHE_PATTERNS:
            keys = [key async for key in client.scan_iter(match=pattern)]
            if keys:
                await client.delete(*keys)
                deleted += len(keys)
        logger.info("Invalidated %d catalog cache keys after a moderation change.", deleted)
        return deleted


def book_identifier(book: dict[str, Any]) -> str | None:
    """The platform's id of a book."""
    for key in ("_id", "id", "bookId"):
        value = book.get(key)
        if value:
            return str(value)
    return None


def book_author_id(book: dict[str, Any]) -> str | None:
    """The platform's id of the book's author.

    The catalog populates the author with the profile fields, so the id sits
    next to the name; a book that was not populated carries the raw id.
    """
    author = book.get("author")
    if isinstance(author, dict):
        value = author.get("_id") or author.get("id")
        return str(value) if value else None
    if isinstance(author, str) and author:
        return author
    return None
