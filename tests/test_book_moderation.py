"""Tests for the book take-down and restore controls (issue #314).

Everything runs offline: the platform calls are mocked with httpx's transport,
the moderation state lives in an in-memory store, and the catalog cache is
faked. No Redis, no network, no model calls.
"""

from __future__ import annotations

import fnmatch
from datetime import UTC, datetime
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

import main
from book_moderation.api import get_book_moderation_service
from book_moderation.models import (
    BookModeration,
    BookStatus,
    TakeDownReason,
    TakeDownRequest,
    VisibilityRequest,
)
from book_moderation.platform import (
    Account,
    BackendAccountLookup,
    BookCacheInvalidator,
    InMemoryAccountLookup,
    PlatformCatalogClient,
    book_author_id,
    book_identifier,
)
from book_moderation.policy import Purchase, Viewer, decide_access
from book_moderation.service import BookModerationService
from book_moderation.store import (
    InMemoryBookModerationStore,
    RedisBookModerationStore,
    build_store,
)

ADMIN_TOKEN = "test-admin-token"

NOW = datetime(2026, 3, 1, 12, 0, tzinfo=UTC)


def fixed_clock() -> datetime:
    return NOW


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeRedis:
    """The slice of the Redis API the stores use."""

    def __init__(self, data: dict[str, str] | None = None) -> None:
        self.data: dict[str, str] = dict(data or {})
        self.deleted: list[str] = []
        self.scans: list[str] = []

    async def get(self, key: str) -> str | None:
        return self.data.get(key)

    async def set(self, key: str, value: str) -> None:
        self.data[key] = value

    async def delete(self, *keys: str) -> int:
        removed = 0
        for key in keys:
            if key in self.data:
                del self.data[key]
                removed += 1
        self.deleted.extend(keys)
        return removed

    async def scan_iter(self, match: str | None = None):
        self.scans.append(match or "*")
        for key in list(self.data):
            if match is None or fnmatch.fnmatch(key, match):
                yield key


class _FailingInvalidator(BookCacheInvalidator):
    """A cache invalidator whose Redis is down."""

    def __init__(self) -> None:
        super().__init__(client=_FakeRedis())

    async def invalidate(self) -> int:
        raise RuntimeError("redis is down")


class _FakeCatalogClient(PlatformCatalogClient):
    """A platform catalog that lives in memory."""

    def __init__(self, books: list[dict[str, Any]] | None = None) -> None:
        super().__init__(base_url="https://platform.test")
        self._books = books or []
        self.book_calls: list[str] = []

    async def books(
        self,
        auth_token: str | None = None,
        *,
        params: dict[str, Any] | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> list[dict[str, Any]]:
        return [dict(book) for book in self._books]

    async def book(
        self,
        book_id: str,
        auth_token: str | None = None,
        *,
        client: httpx.AsyncClient | None = None,
    ) -> dict[str, Any] | None:
        self.book_calls.append(book_id)
        for book in self._books:
            if book_identifier(book) == book_id:
                return dict(book)
        return None


def make_service(
    *,
    books: list[dict[str, Any]] | None = None,
    accounts: dict[str, Account] | None = None,
    cache: _FakeRedis | None = None,
) -> tuple[BookModerationService, InMemoryBookModerationStore, _FakeCatalogClient]:
    """A service wired to in-memory doubles."""
    store = InMemoryBookModerationStore()
    catalog = _FakeCatalogClient(books if books is not None else [a_book("b1"), a_book("b2")])
    service = BookModerationService(
        store,
        cache_invalidator=BookCacheInvalidator(client=cache or _FakeRedis()),
        account_lookup=InMemoryAccountLookup(accounts or {}),
        catalog_client=catalog,
        clock=fixed_clock,
    )
    return service, store, catalog


def a_book(book_id: str, **extra: Any) -> dict[str, Any]:
    """A book as the platform publishes it: the author is populated."""
    book = {"_id": book_id, "title": f"Book {book_id}", "author": {"_id": f"author-of-{book_id}", "name": "An Author"}}
    book.update(extra)
    return book


# ---------------------------------------------------------------------------
# The visibility rules
# ---------------------------------------------------------------------------


class TestPolicy:
    def test_active_book_is_visible_to_everybody(self):
        record = BookModeration(book_id="b1", status=BookStatus.ACTIVE)
        decision = decide_access(record, Viewer())
        assert decision.visible is True
        assert decision.retained is False
        assert decision.basis == "active"

    def test_unknown_book_is_visible(self):
        decision = decide_access(None, Viewer())
        assert decision.visible is True
        assert decision.basis == "active"

    def test_taken_down_book_is_hidden(self):
        record = BookModeration(book_id="b1", status=BookStatus.TAKEN_DOWN, taken_down_at=NOW)
        decision = decide_access(record, Viewer())
        assert decision.visible is False
        assert decision.retained is False
        assert decision.basis == "taken-down"

    def test_the_author_keeps_access(self):
        record = BookModeration(book_id="b1", status=BookStatus.TAKEN_DOWN, author_id="author-1", taken_down_at=NOW)
        decision = decide_access(record, Viewer(user_id="author-1"))
        assert decision.visible is True
        assert decision.retained is True
        assert decision.basis == "author"

    def test_the_author_of_the_platform_wins_over_the_stored_one(self):
        record = BookModeration(book_id="b1", status=BookStatus.TAKEN_DOWN, author_id="author-1", taken_down_at=NOW)
        decision = decide_access(record, Viewer(user_id="author-2"), None, "author-2")
        assert decision.visible is True
        assert decision.basis == "author"

    def test_another_user_does_not_get_the_author_access(self):
        record = BookModeration(book_id="b1", status=BookStatus.TAKEN_DOWN, author_id="author-1", taken_down_at=NOW)
        assert decide_access(record, Viewer(user_id="someone-else")).visible is False

    def test_purchaser_keeps_access(self):
        record = BookModeration(book_id="b1", status=BookStatus.TAKEN_DOWN, taken_down_at=NOW)
        purchase = Purchase(book_id="b1")
        decision = decide_access(record, Viewer(user_id="buyer-1"), purchase)
        assert decision.visible is True
        assert decision.retained is True
        assert decision.basis == "purchaser"

    def test_a_purchase_keeps_the_access_the_platform_grants(self):
        # The platform owns the purchase list and does not compare the purchase
        # date with the take-down timestamp, so neither does this rule.
        record = BookModeration(book_id="b1", status=BookStatus.TAKEN_DOWN, taken_down_at=NOW)
        purchase = Purchase(book_id="b1")
        assert decide_access(record, Viewer(user_id="buyer-1"), purchase).visible is True

    def test_purchase_of_another_book_does_not_grant_access(self):
        record = BookModeration(book_id="b1", status=BookStatus.TAKEN_DOWN, taken_down_at=NOW)
        purchase = Purchase(book_id="b2")
        assert decide_access(record, Viewer(user_id="buyer-1"), purchase).visible is False

    def test_an_admin_without_an_entitlement_does_not_keep_the_book(self):
        # The platform grants reading access to the author and to the buyers;
        # an admin manages the taken-down books through the admin routes.
        record = BookModeration(book_id="b1", status=BookStatus.TAKEN_DOWN, taken_down_at=NOW)
        assert decide_access(record, Viewer(user_id="admin-1")).visible is False


# ---------------------------------------------------------------------------
# The request models
# ---------------------------------------------------------------------------


class TestModels:
    @pytest.mark.parametrize("reason", ["content-violation", "dmca", "quality-issues", "other"])
    def test_the_four_reason_categories_are_accepted(self, reason):
        assert TakeDownRequest(reason=reason).reason == reason

    def test_an_unknown_reason_is_rejected(self):
        with pytest.raises(ValidationError):
            TakeDownRequest(reason="because-i-said-so")

    def test_a_blank_note_is_no_note(self):
        assert TakeDownRequest(reason="dmca", note="   ").note is None
        assert TakeDownRequest(reason="dmca", note="\n\t ").note is None

    def test_a_note_is_trimmed(self):
        assert TakeDownRequest(reason="dmca", note="  spaced  ").note == "spaced"

    def test_a_note_is_bounded(self):
        with pytest.raises(ValidationError):
            TakeDownRequest(reason="dmca", note="x" * 2001)

    def test_the_visibility_list_is_bounded(self):
        with pytest.raises(ValidationError):
            VisibilityRequest(book_ids=[f"b{i}" for i in range(501)])

    def test_the_author_is_not_taken_from_the_request(self):
        # The author of the book is read from the platform, so a caller cannot
        # claim the authorship of a book it does not own.
        assert "author_id" not in TakeDownRequest.model_fields


# ---------------------------------------------------------------------------
# The stores
# ---------------------------------------------------------------------------


class TestInMemoryStore:
    async def test_roundtrip(self):
        store = InMemoryBookModerationStore()
        record = BookModeration(book_id="b1", status=BookStatus.TAKEN_DOWN, reason=TakeDownReason.DMCA)
        await store.save(record)
        assert await store.get("b1") == record
        assert await store.get("missing") is None

    async def test_delete_reports_whether_it_removed_anything(self):
        store = InMemoryBookModerationStore()
        await store.save(BookModeration(book_id="b1"))
        assert await store.delete("b1") is True
        assert await store.delete("b1") is False
        assert await store.get("b1") is None

    async def test_list_all(self):
        store = InMemoryBookModerationStore()
        await store.save(BookModeration(book_id="b1"))
        await store.save(BookModeration(book_id="b2"))
        assert {record.book_id for record in await store.list_all()} == {"b1", "b2"}


class TestRedisStore:
    def _store(self, data: dict[str, str] | None = None) -> tuple[RedisBookModerationStore, _FakeRedis]:
        fake = _FakeRedis(data)
        store = RedisBookModerationStore.__new__(RedisBookModerationStore)
        store._redis = fake  # noqa: SLF001 - the point is to avoid a live Redis
        return store, fake

    async def test_roundtrip(self):
        store, fake = self._store()
        record = BookModeration(book_id="b1", status=BookStatus.TAKEN_DOWN, reason=TakeDownReason.OTHER)
        await store.save(record)
        assert fake.data["moderation:book:b1"]
        assert await store.get("b1") == record

    async def test_missing_key(self):
        store, _ = self._store()
        assert await store.get("b1") is None

    async def test_a_corrupt_record_is_ignored(self):
        store, _ = self._store({"moderation:book:b1": "{not json"})
        assert await store.get("b1") is None

    async def test_delete_reports_whether_it_removed_anything(self):
        store, _ = self._store({"moderation:book:b1": "{}"})
        assert await store.delete("b1") is True
        assert await store.delete("b1") is False

    async def test_list_all_reads_the_moderation_namespace(self):
        store, fake = self._store(
            {
                "moderation:book:b1": '{"book_id": "b1", "status": "taken-down"}',
                "moderation:book:b2": '{"book_id": "b2", "status": "active"}',
                "books:list": "[]",
            }
        )
        records = await store.list_all()
        assert {record.book_id for record in records} == {"b1", "b2"}
        assert fake.scans == ["moderation:book:*"]


class TestBuildStore:
    def test_without_redis_url_the_store_is_in_memory(self, monkeypatch):
        monkeypatch.delenv("REDIS_URL", raising=False)
        assert isinstance(build_store(), InMemoryBookModerationStore)

    def test_an_unusable_redis_url_degrades_to_memory(self, monkeypatch):
        monkeypatch.setenv("REDIS_URL", "not-a-redis-url")
        assert isinstance(build_store(), InMemoryBookModerationStore)


# ---------------------------------------------------------------------------
# The service
# ---------------------------------------------------------------------------


class TestTakeDown:
    async def test_a_takedown_records_the_state(self):
        service, store, _ = make_service()
        record = await service.take_down("b1", TakeDownReason.DMCA, note="copyright", actor="admin-1")

        assert record.status is BookStatus.TAKEN_DOWN
        assert record.previous_status is BookStatus.ACTIVE
        assert record.reason is TakeDownReason.DMCA
        assert record.note == "copyright"
        assert record.taken_down_by == "admin-1"
        assert record.taken_down_at == NOW
        assert await store.get("b1") == record

    async def test_the_author_is_read_from_the_platform(self):
        service, _, _ = make_service(books=[a_book("b1", author={"_id": "author-1", "name": "An Author"})])
        record = await service.take_down("b1", TakeDownReason.OTHER)
        assert record.author_id == "author-1"

    async def test_a_book_the_platform_does_not_publish_cannot_be_taken_down(self):
        service, _, _ = make_service(books=[])
        with pytest.raises(main.APIException) as exc:
            await service.take_down("b1", TakeDownReason.DMCA)
        assert exc.value.status_code == 404

    async def test_taking_down_twice_is_a_conflict(self):
        service, _, _ = make_service()
        await service.take_down("b1", TakeDownReason.DMCA)
        with pytest.raises(main.APIException) as exc:
            await service.take_down("b1", TakeDownReason.OTHER)
        assert exc.value.status_code == 409
        assert "already taken down" in str(exc.value.detail)

    async def test_a_takedown_flushes_the_catalog_cache(self):
        cache = _FakeRedis({"books:list": "[]", "book:b1": "{}", "educators:list": "[]"})
        service, _, _ = make_service(cache=cache)
        await service.take_down("b1", TakeDownReason.DMCA)
        assert set(cache.deleted) == {"books:list", "book:b1", "educators:list"}

    async def test_a_broken_cache_does_not_undo_the_takedown(self):
        service, store, _ = make_service()
        service._cache_invalidator = _FailingInvalidator()  # noqa: SLF001
        record = await service.take_down("b1", TakeDownReason.DMCA)
        assert record.status is BookStatus.TAKEN_DOWN
        assert (await store.get("b1")).status is BookStatus.TAKEN_DOWN

    async def test_an_unreachable_platform_does_not_block_the_takedown(self):
        service, store, _ = make_service()

        class _BrokenCatalog(PlatformCatalogClient):
            def __init__(self) -> None:
                super().__init__(base_url="https://platform.test")

            async def book(self, book_id, auth_token=None, *, client=None):
                raise RuntimeError("the platform is down")

        service._catalog_client = _BrokenCatalog()  # noqa: SLF001
        record = await service.take_down("b1", TakeDownReason.DMCA)
        assert record.status is BookStatus.TAKEN_DOWN
        assert record.author_id is None
        assert (await store.get("b1")).status is BookStatus.TAKEN_DOWN


class TestRestore:
    async def test_a_restore_returns_the_book_to_its_previous_state(self):
        service, store, _ = make_service()
        await service.take_down("b1", TakeDownReason.DMCA, note="copyright", actor="admin-1")

        restored = await service.restore("b1", actor="admin-2")

        assert restored.status is BookStatus.ACTIVE
        assert restored.previous_status is None
        assert restored.reason is None
        assert restored.note is None
        assert restored.taken_down_at is None
        assert restored.taken_down_by is None
        assert restored.restored_at == NOW
        assert (await store.get("b1")).status is BookStatus.ACTIVE

    async def test_the_author_survives_a_restore(self):
        service, _, _ = make_service(books=[a_book("b1", author={"_id": "author-1", "name": "An Author"})])
        await service.take_down("b1", TakeDownReason.OTHER)
        restored = await service.restore("b1")
        assert restored.author_id == "author-1"

    async def test_restoring_an_unknown_book_is_a_not_found(self):
        service, _, _ = make_service()
        with pytest.raises(main.APIException) as exc:
            await service.restore("b1")
        assert exc.value.status_code == 404

    async def test_restoring_a_book_that_is_not_taked_down_is_a_conflict(self):
        service, _, _ = make_service()
        await service.take_down("b1", TakeDownReason.DMCA)
        await service.restore("b1")
        with pytest.raises(main.APIException) as exc:
            await service.restore("b1")
        assert exc.value.status_code == 409
        assert "not taken down" in str(exc.value.detail)

    async def test_a_restore_flushes_the_catalog_cache(self):
        cache = _FakeRedis({"books:list": "[]"})
        service, _, _ = make_service(cache=cache)
        await service.take_down("b1", TakeDownReason.DMCA)
        # The take-down flushed the cache; the book is published again, so the
        # restore has to flush it a second time.
        cache.data["books:list"] = "[]"
        cache.deleted.clear()
        await service.restore("b1")
        assert cache.deleted == ["books:list"]

    async def test_a_broken_cache_does_not_undo_the_restore(self):
        service, store, _ = make_service()
        await service.take_down("b1", TakeDownReason.DMCA)
        service._cache_invalidator = _FailingInvalidator()  # noqa: SLF001
        restored = await service.restore("b1")
        assert restored.status is BookStatus.ACTIVE
        assert (await store.get("b1")).status is BookStatus.ACTIVE


class TestVisibility:
    async def test_an_active_book_is_visible(self):
        service, _, _ = make_service()
        decision = await service.visibility("b1", None)
        assert decision.visible is True
        assert decision.basis == "active"

    async def test_a_taken_down_book_is_hidden_from_a_learner(self):
        service, _, _ = make_service()
        await service.take_down("b1", TakeDownReason.DMCA)
        decision = await service.visibility("b1", Account(user_id="reader-1"))
        assert decision.visible is False
        assert decision.basis == "taken-down"

    async def test_a_purchaser_keeps_the_access(self):
        service, _, _ = make_service()
        await service.take_down("b1", TakeDownReason.DMCA)
        account = Account(user_id="buyer-1", purchases=(Purchase(book_id="b1"),))
        decision = await service.visibility("b1", account)
        assert decision.visible is True
        assert decision.retained is True
        assert decision.basis == "purchaser"

    async def test_the_author_keeps_the_access(self):
        service, _, _ = make_service(books=[a_book("b1", author={"_id": "author-1", "name": "An Author"})])
        await service.take_down("b1", TakeDownReason.DMCA)
        decision = await service.visibility("b1", Account(user_id="author-1"))
        assert decision.visible is True
        assert decision.basis == "author"

    async def test_many_books_are_decided_in_one_pass(self):
        service, _, _ = make_service()
        await service.take_down("b1", TakeDownReason.DMCA)
        account = Account(user_id="buyer-1", purchases=(Purchase(book_id="b1"),))
        decisions = await service.visibility_many(["b1", "b2"], account)
        assert decisions["b1"].visible is True
        assert decisions["b1"].retained is True
        assert decisions["b2"].visible is True
        assert decisions["b2"].retained is False

    async def test_an_unreachable_platform_leaves_the_viewer_without_access(self):
        service, _, _ = make_service()
        await service.take_down("b1", TakeDownReason.DMCA)

        class _BrokenLookup:
            async def account(self, auth_token: str) -> Account | None:
                raise RuntimeError("the platform is down")

        service._account_lookup = _BrokenLookup()  # noqa: SLF001
        assert await service.account_for("token") is None


class TestModeratedCatalog:
    async def test_an_active_catalog_is_passed_through(self):
        service, _, _ = make_service(books=[a_book("b1"), a_book("b2")])
        books = await service.moderated_catalog()
        assert [book["_id"] for book in books] == ["b1", "b2"]
        assert all(book["status"] == "active" for book in books)

    async def test_a_taken_down_book_leaves_the_catalog(self):
        service, _, _ = make_service(books=[a_book("b1"), a_book("b2")])
        await service.take_down("b1", TakeDownReason.DMCA)
        books = await service.moderated_catalog()
        assert [book["_id"] for book in books] == ["b2"]

    async def test_a_purchaser_still_sees_the_book_with_its_status(self):
        account = Account(user_id="buyer-1", purchases=(Purchase(book_id="b1"),))
        service, _, _ = make_service(books=[a_book("b1")])
        await service.take_down("b1", TakeDownReason.DMCA)
        service._account_lookup = InMemoryAccountLookup({"token": account})  # noqa: SLF001
        books = await service.moderated_catalog("token")
        assert [book["_id"] for book in books] == ["b1"]
        assert books[0]["status"] == "taken-down"
        assert books[0]["moderationReason"] == "dmca"

    async def test_an_admin_without_an_entitlement_does_not_see_it_in_the_catalog(self):
        service, _, _ = make_service(books=[a_book("b1")])
        await service.take_down("b1", TakeDownReason.DMCA)
        service._account_lookup = InMemoryAccountLookup({"token": Account(user_id="a1")})  # noqa: SLF001
        books = await service.moderated_catalog("token")
        assert books == []

    async def test_the_author_sees_the_taken_down_book(self):
        service, _, _ = make_service(books=[a_book("b1", author={"_id": "author-1", "name": "An Author"})])
        await service.take_down("b1", TakeDownReason.DMCA)
        service._account_lookup = InMemoryAccountLookup({"token": Account(user_id="author-1")})  # noqa: SLF001
        books = await service.moderated_catalog("token")
        assert [book["_id"] for book in books] == ["b1"]

    async def test_one_book_is_hidden_or_shown_for_the_viewer(self):
        service, _, _ = make_service(books=[a_book("b1")])
        await service.take_down("b1", TakeDownReason.DMCA)

        assert await service.moderated_book("b1") is None

        service, _, _ = make_service(books=[a_book("b1")])
        await service.take_down("b1", TakeDownReason.DMCA)
        service._account_lookup = InMemoryAccountLookup(
            {"token": Account(user_id="buyer-1", purchases=(Purchase(book_id="b1"),))}  # noqa: SLF001
        )
        book = await service.moderated_book("b1", "token")
        assert book is not None
        assert book["status"] == "taken-down"

    async def test_an_unknown_book_is_none(self):
        service, _, _ = make_service(books=[])
        assert await service.moderated_book("missing") is None

    async def test_a_broken_platform_is_reported_as_unavailable(self):
        service, _, _ = make_service()

        class _BrokenCatalog(PlatformCatalogClient):
            def __init__(self) -> None:
                super().__init__(base_url="https://platform.test")

            async def books(self, auth_token=None, *, params=None, client=None):
                raise RuntimeError("the platform is down")

        service._catalog_client = _BrokenCatalog()  # noqa: SLF001
        with pytest.raises(main.APIException) as exc:
            await service.moderated_catalog()
        assert exc.value.status_code == 502


# ---------------------------------------------------------------------------
# The platform client
# ---------------------------------------------------------------------------


class TestBackendAccountLookup:
    async def test_the_account_is_read_from_the_export_route(self):
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.headers["authorization"] == "Bearer token-1"
            return httpx.Response(
                200,
                json={
                    "success": True,
                    "exportedAt": "2026-03-01T12:00:00.000Z",
                    "data": {
                        "_id": "user-1",
                        "role": "admin",
                        "purchasedBooks": [
                            {"bookId": "b1", "purchaseDate": "2026-01-01T00:00:00Z"},
                            {"bookId": "b2"},
                        ],
                    },
                },
            )

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            lookup = BackendAccountLookup(base_url="https://platform.test")
            account = await lookup.account("token-1", client=client)

        assert account is not None
        assert account.user_id == "user-1"
        assert account.purchase_for("b1") == Purchase(book_id="b1")
        assert account.purchase_for("b2") == Purchase(book_id="b2")
        assert account.purchase_for("b3") is None

    async def test_an_unexpected_payload_gives_no_account(self):
        transport = httpx.MockTransport(lambda request: httpx.Response(200, json={"nope": True}))
        async with httpx.AsyncClient(transport=transport) as client:
            lookup = BackendAccountLookup(base_url="https://platform.test")
            assert await lookup.account("token-1", client=client) is None

    async def test_a_rejected_token_raises(self):
        transport = httpx.MockTransport(lambda request: httpx.Response(401, json={"error": "nope"}))
        async with httpx.AsyncClient(transport=transport) as client:
            lookup = BackendAccountLookup(base_url="https://platform.test")
            with pytest.raises(httpx.HTTPStatusError):
                await lookup.account("token-1", client=client)


class TestPlatformCatalogClient:
    async def test_the_books_are_read_from_the_data_key(self):
        # This is the shape the platform answers with.
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/api/books"
            return httpx.Response(200, json={"success": True, "data": [{"_id": "b1"}, {"_id": "b2"}]})

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            catalog = PlatformCatalogClient(base_url="https://platform.test")
            books = await catalog.books(client=client)
        assert [book["_id"] for book in books] == ["b1", "b2"]

    async def test_a_paginated_catalog_is_read(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "success": True,
                    "page": 1,
                    "limit": 20,
                    "total": 100,
                    "hasMore": True,
                    "data": [{"_id": "b1"}],
                },
            )

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            catalog = PlatformCatalogClient(base_url="https://platform.test")
            books = await catalog.books(client=client)
        assert [book["_id"] for book in books] == ["b1"]

    async def test_a_bare_list_is_read(self):
        transport = httpx.MockTransport(lambda request: httpx.Response(200, json=[{"_id": "b1"}]))
        async with httpx.AsyncClient(transport=transport) as client:
            catalog = PlatformCatalogClient(base_url="https://platform.test")
            assert await catalog.books(client=client) == [{"_id": "b1"}]

    async def test_an_unexpected_payload_gives_no_books(self):
        transport = httpx.MockTransport(lambda request: httpx.Response(200, json={"success": True}))
        async with httpx.AsyncClient(transport=transport) as client:
            catalog = PlatformCatalogClient(base_url="https://platform.test")
            assert await catalog.books(client=client) == []

    async def test_one_book_is_read(self):
        transport = httpx.MockTransport(
            lambda request: httpx.Response(200, json={"success": True, "data": {"_id": "b1"}})
        )
        async with httpx.AsyncClient(transport=transport) as client:
            catalog = PlatformCatalogClient(base_url="https://platform.test")
            assert await catalog.book("b1", client=client) == {"_id": "b1"}

    async def test_a_book_the_platform_does_not_hold_is_none(self):
        transport = httpx.MockTransport(lambda request: httpx.Response(404, json={"success": False}))
        async with httpx.AsyncClient(transport=transport) as client:
            catalog = PlatformCatalogClient(base_url="https://platform.test")
            assert await catalog.book("missing", client=client) is None

    async def test_the_viewer_token_is_forwarded(self):
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.headers["authorization"] == "Bearer token-1"
            return httpx.Response(200, json={"success": True, "data": []})

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            catalog = PlatformCatalogClient(base_url="https://platform.test")
            await catalog.books("token-1", client=client)


class TestBookCacheInvalidator:
    async def test_the_catalog_caches_are_dropped(self):
        fake = _FakeRedis(
            {
                "books:list": "[]",
                "books:page:2": "[]",
                "book:b1": "{}",
                "educators:list": "[]",
                "moderation:book:b1": "{}",
                "users:list": "[]",
            }
        )
        invalidator = BookCacheInvalidator(client=fake)
        deleted = await invalidator.invalidate()

        assert deleted == 4
        assert set(fake.deleted) == {"books:list", "books:page:2", "book:b1", "educators:list"}
        assert "moderation:book:b1" in fake.data

    async def test_without_redis_nothing_is_dropped(self, monkeypatch):
        monkeypatch.delenv("REDIS_URL", raising=False)
        assert await BookCacheInvalidator().invalidate() == 0


class TestBookIdentifier:
    def test_the_platform_id_is_read_from_any_of_its_fields(self):
        assert book_identifier({"_id": "a"}) == "a"
        assert book_identifier({"id": "b"}) == "b"
        assert book_identifier({"bookId": "c"}) == "c"

    def test_a_book_without_an_id_has_none(self):
        assert book_identifier({"title": "x"}) is None


class TestBookAuthorId:
    def test_the_populated_author_carries_its_id(self):
        assert book_author_id({"author": {"_id": "author-1", "name": "An Author"}}) == "author-1"

    def test_a_book_without_a_populated_author_carries_the_raw_id(self):
        assert book_author_id({"author": "author-1"}) == "author-1"

    def test_a_book_without_an_author_has_none(self):
        assert book_author_id({"title": "x"}) is None
        assert book_author_id({"author": {}}) is None


# ---------------------------------------------------------------------------
# The HTTP surface
# ---------------------------------------------------------------------------


@pytest.fixture()
def client(monkeypatch):
    """The application with the moderation service wired to in-memory doubles."""
    service, store, catalog = make_service(
        books=[a_book("b1"), a_book("b2")],
        accounts={"token-1": Account(user_id="user-1", purchases=(Purchase(book_id="b1"),))},
    )
    monkeypatch.setattr(main, "ADMIN_TOKEN", ADMIN_TOKEN)
    main.app.dependency_overrides[get_book_moderation_service] = lambda: service
    with TestClient(main.app) as test_client:
        yield test_client
    main.app.dependency_overrides.pop(get_book_moderation_service, None)


class TestAdminRoutes:
    def test_the_admin_route_needs_the_admin_token(self, client):
        response = client.post("/books/moderation/b1/takedown", json={"reason": "dmca"})
        assert response.status_code == 403
        assert "hint" in response.json()

    def test_a_wrong_admin_token_is_rejected(self, client):
        response = client.post(
            "/books/moderation/b1/takedown",
            json={"reason": "dmca"},
            headers={"X-Admin-Token": "wrong"},
        )
        assert response.status_code == 403

    def test_the_admin_route_is_closed_without_a_configured_token(self, client, monkeypatch):
        monkeypatch.setattr(main, "ADMIN_TOKEN", "")
        response = client.post("/books/moderation/b1/takedown", json={"reason": "dmca"})
        assert response.status_code == 503

    def test_a_book_is_taken_down(self, client):
        response = client.post(
            "/books/moderation/b1/takedown",
            json={"reason": "content-violation", "note": "reported"},
            headers={"X-Admin-Token": ADMIN_TOKEN},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["success"] is True
        assert body["moderation"]["status"] == "taken-down"
        assert body["moderation"]["reason"] == "content-violation"
        assert body["moderation"]["note"] == "reported"

    def test_the_author_of_the_book_is_recorded(self, client):
        response = client.post(
            "/books/moderation/b1/takedown",
            json={"reason": "dmca"},
            headers={"X-Admin-Token": ADMIN_TOKEN},
        )
        assert response.json()["moderation"]["author_id"] == "author-of-b1"

    def test_an_unknown_reason_is_rejected(self, client):
        response = client.post(
            "/books/moderation/b1/takedown",
            json={"reason": "spam"},
            headers={"X-Admin-Token": ADMIN_TOKEN},
        )
        assert response.status_code == 422

    def test_taking_down_twice_is_a_conflict(self, client):
        headers = {"X-Admin-Token": ADMIN_TOKEN}
        assert client.post("/books/moderation/b1/takedown", json={"reason": "dmca"}, headers=headers).status_code == 200
        response = client.post("/books/moderation/b1/takedown", json={"reason": "dmca"}, headers=headers)
        assert response.status_code == 409
        assert "hint" in response.json()

    def test_a_book_is_restored(self, client):
        headers = {"X-Admin-Token": ADMIN_TOKEN}
        client.post("/books/moderation/b1/takedown", json={"reason": "dmca"}, headers=headers)
        response = client.post("/books/moderation/b1/restore", headers=headers)
        assert response.status_code == 200
        assert response.json()["moderation"]["status"] == "active"

    def test_restoring_an_unknown_book_is_a_not_found(self, client):
        response = client.post("/books/moderation/b1/restore", headers={"X-Admin-Token": ADMIN_TOKEN})
        assert response.status_code == 404

    def test_the_moderation_state_is_listed(self, client):
        headers = {"X-Admin-Token": ADMIN_TOKEN}
        client.post("/books/moderation/b1/takedown", json={"reason": "dmca"}, headers=headers)

        response = client.get("/books/moderation", headers=headers)
        assert response.status_code == 200
        assert [record["book_id"] for record in response.json()["moderation"]] == ["b1"]

        response = client.get("/books/moderation?status=taken-down", headers=headers)
        assert [record["book_id"] for record in response.json()["moderation"]] == ["b1"]

        response = client.get("/books/moderation?status=active", headers=headers)
        assert response.json()["moderation"] == []

    def test_an_invalid_status_filter_is_rejected(self, client):
        response = client.get("/books/moderation?status=paused", headers={"X-Admin-Token": ADMIN_TOKEN})
        assert response.status_code == 400
        assert "hint" in response.json()

    def test_the_state_of_one_book_is_read(self, client):
        headers = {"X-Admin-Token": ADMIN_TOKEN}
        client.post("/books/moderation/b1/takedown", json={"reason": "dmca"}, headers=headers)
        response = client.get("/books/moderation/b1", headers=headers)
        assert response.status_code == 200
        assert response.json()["moderation"]["status"] == "taken-down"

    def test_the_admin_state_of_an_untouched_book_is_none(self, client):
        response = client.get("/books/moderation/b1", headers={"X-Admin-Token": ADMIN_TOKEN})
        assert response.status_code == 200
        assert response.json()["moderation"] is None


class TestLearnerRoutes:
    def test_the_catalog_hides_a_taken_down_book(self, client):
        headers = {"X-Admin-Token": ADMIN_TOKEN}
        client.post("/books/moderation/b1/takedown", json={"reason": "dmca"}, headers=headers)

        response = client.get("/books/catalog")
        assert response.status_code == 200
        assert [book["_id"] for book in response.json()["books"]] == ["b2"]

    def test_the_catalog_keeps_the_books_a_purchaser_bought(self, client):
        headers = {"X-Admin-Token": ADMIN_TOKEN}
        client.post("/books/moderation/b1/takedown", json={"reason": "dmca"}, headers=headers)

        response = client.get("/books/catalog", headers={"Authorization": "Bearer token-1"})
        assert response.status_code == 200
        books = response.json()["books"]
        assert [book["_id"] for book in books] == ["b1", "b2"]
        assert books[0]["status"] == "taken-down"

    def test_a_hidden_book_is_not_found(self, client):
        headers = {"X-Admin-Token": ADMIN_TOKEN}
        client.post("/books/moderation/b1/takedown", json={"reason": "dmca"}, headers=headers)

        response = client.get("/books/b1")
        assert response.status_code == 404
        assert "hint" in response.json()

    def test_a_purchaser_still_reads_the_book(self, client):
        headers = {"X-Admin-Token": ADMIN_TOKEN}
        client.post("/books/moderation/b1/takedown", json={"reason": "dmca"}, headers=headers)

        response = client.get("/books/b1", headers={"Authorization": "Bearer token-1"})
        assert response.status_code == 200
        assert response.json()["book"]["status"] == "taken-down"

    def test_a_restored_book_is_published_again(self, client):
        headers = {"X-Admin-Token": ADMIN_TOKEN}
        client.post("/books/moderation/b1/takedown", json={"reason": "dmca"}, headers=headers)
        client.post("/books/moderation/b1/restore", headers=headers)

        response = client.get("/books/catalog")
        assert [book["_id"] for book in response.json()["books"]] == ["b1", "b2"]

    def test_the_visibility_of_one_book_is_reported(self, client):
        headers = {"X-Admin-Token": ADMIN_TOKEN}
        client.post("/books/moderation/b1/takedown", json={"reason": "dmca"}, headers=headers)

        response = client.get("/books/moderation/b1/visibility")
        assert response.status_code == 200
        visibility = response.json()["visibility"]
        assert visibility["visible"] is False
        assert visibility["retained"] is False
        assert visibility["basis"] == "taken-down"

        response = client.get("/books/moderation/b1/visibility", headers={"Authorization": "Bearer token-1"})
        visibility = response.json()["visibility"]
        assert visibility["visible"] is True
        assert visibility["retained"] is True
        assert visibility["basis"] == "purchaser"

    def test_the_visibility_of_many_books_is_reported(self, client):
        headers = {"X-Admin-Token": ADMIN_TOKEN}
        client.post("/books/moderation/b1/takedown", json={"reason": "dmca"}, headers=headers)

        response = client.post(
            "/books/moderation/visibility",
            json={"book_ids": ["b1", "b2"]},
            headers={"Authorization": "Bearer token-1"},
        )
        assert response.status_code == 200
        visibility = response.json()["visibility"]
        assert visibility["b1"]["visible"] is True
        assert visibility["b1"]["retained"] is True
        assert visibility["b2"]["visible"] is True
        assert visibility["b2"]["retained"] is False

    def test_a_bearer_token_that_is_not_one_is_ignored(self, client):
        headers = {"X-Admin-Token": ADMIN_TOKEN}
        client.post("/books/moderation/b1/takedown", json={"reason": "dmca"}, headers=headers)

        response = client.get("/books/catalog", headers={"Authorization": "Basic abc"})
        assert [book["_id"] for book in response.json()["books"]] == ["b2"]
