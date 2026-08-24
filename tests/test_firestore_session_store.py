"""Tests for FirestoreSessionStore.

These run against an in-memory fake of the google-cloud-firestore surface the
store touches, so no live credentials or network are required.
"""

import pytest

from store import FirestoreSessionStore, SessionStore, create_session_store


def _match(value, op, expected):
    if op == "==":
        return value == expected
    if op in ("<", "<=", ">", ">="):
        return {"<": value < expected, "<=": value <= expected, ">": value > expected, ">=": value >= expected}[op]
    return True


class _FakeSnapshot:
    def __init__(self, ref, data):
        self._ref = ref
        self._data = data
        self.id = ref.id

    @property
    def exists(self):
        return self._data is not None

    def to_dict(self):
        return self._data

    def get(self, key, default=None):
        if self._data is None:
            return default
        return self._data.get(key, default)

    @property
    def reference(self):
        return self._ref


class _FakeRef:
    def __init__(self, db, path, is_doc):
        self._db = db
        self.path = path
        self.id = path[-1]
        self._is_doc = is_doc

    def collection(self, name):
        return _FakeRef(self._db, self.path + (name,), False)

    def document(self, doc_id=None):
        if doc_id is None:
            doc_id = f"auto-{len(self._db._data)}"
        return _FakeRef(self._db, self.path + (doc_id,), True)

    def get(self):
        coll = self._db._data.get("/".join(self.path[:-1]), {})
        return _FakeSnapshot(self, coll.get(self.id))

    def set(self, data, merge=False):
        key = "/".join(self.path[:-1])
        coll = self._db._data.setdefault(key, {})
        if merge:
            coll.setdefault(self.id, {}).update(data)
        else:
            coll[self.id] = dict(data)

    def update(self, data):
        coll = self._db._data.get("/".join(self.path[:-1]), {})
        if self.id not in coll:
            raise KeyError("Document does not exist")
        coll[self.id].update(data)

    def delete(self):
        coll = self._db._data.get("/".join(self.path[:-1]))
        if coll:
            coll.pop(self.id, None)

    def stream(self):
        coll = self._db._data.get("/".join(self.path), {}) or {}
        return [_FakeSnapshot(_FakeRef(self._db, self.path + (k,), True), v) for k, v in sorted(coll.items())]

    def order_by(self, field, direction="ASCENDING"):
        return _FakeQuery(self._db, self.path, field, direction)

    def where(self, field_path=None, op_string=None, value=None, filter=None):
        if filter is not None:
            field_path, op_string, value = filter.field_path, filter.op_string, filter.value
        return _FakeQuery(self._db, self.path, None, None)._add(field_path, op_string, value)


class _FakeQuery:
    def __init__(self, db, coll_path, field, direction):
        self._db = db
        self._coll_path = coll_path
        self._order_field = field
        self._order_dir = direction
        self._filters = []

    def _add(self, field, op, value):
        self._filters.append((field, op, value))
        return self

    def order_by(self, field, direction="ASCENDING"):
        self._order_field = field
        self._order_dir = direction
        return self

    def where(self, field_path=None, op_string=None, value=None, filter=None):
        if filter is not None:
            field_path, op_string, value = filter.field_path, filter.op_string, filter.value
        return self._add(field_path, op_string, value)

    def stream(self):
        coll = self._db._data.get("/".join(self._coll_path), {}) or {}
        docs = [_FakeSnapshot(_FakeRef(self._db, self._coll_path + (k,), True), v) for k, v in coll.items()]
        if self._order_field:
            docs.sort(
                key=lambda d: d.get(self._order_field) or 0,
                reverse=self._order_dir == "DESCENDING",
            )
        for field, op, value in self._filters:
            docs = [d for d in docs if _match(d.get(field), op, value)]
        return docs


class _FakeBatch:
    def __init__(self, db):
        self._db = db
        self._ops = []

    def set(self, ref, data, merge=False):
        self._ops.append(("set", ref, data, merge))
        return self

    def update(self, ref, data):
        self._ops.append(("update", ref, data))
        return self

    def commit(self):
        for op in self._ops:
            if op[0] == "set":
                op[1].set(op[2], merge=op[3])
            else:
                op[1].update(op[2])


class FakeFirestore:
    def __init__(self):
        self._data = {}

    def collection(self, name):
        return _FakeRef(self, (name,), False)

    def batch(self):
        return _FakeBatch(self)


class TestFirestoreSessionStore:
    pytestmark = pytest.mark.asyncio(loop_scope="function")

    @pytest.fixture(autouse=True)
    def store(self):
        self.store = FirestoreSessionStore(client=FakeFirestore())
        yield

    async def test_load_empty_returns_empty_list(self):
        assert await self.store.load_history("nonexistent") == []

    async def test_save_and_load_roundtrip(self):
        history = [{"role": "user", "text": "hello"}, {"role": "model", "text": "hi"}]
        await self.store.save_history("chat-1", history)
        assert await self.store.load_history("chat-1") == history

    async def test_save_appends_only_new_messages(self):
        await self.store.save_history("c", [{"role": "user", "text": "q1"}])
        await self.store.save_history(
            "c",
            [
                {"role": "user", "text": "q1"},
                {"role": "model", "text": "a1"},
                {"role": "user", "text": "q2"},
            ],
        )
        assert await self.store.load_history("c") == [
            {"role": "user", "text": "q1"},
            {"role": "model", "text": "a1"},
            {"role": "user", "text": "q2"},
        ]

    async def test_messages_keep_seq_order_after_delete_and_resave(self):
        await self.store.save_history("c", [{"role": "user", "text": "q1"}])
        await self.store.delete_session("c")
        await self.store.save_history("c", [{"role": "user", "text": "new"}])
        assert await self.store.load_history("c") == [{"role": "user", "text": "new"}]

    async def test_delete_removes_session(self):
        await self.store.save_history("chat-1", [{"role": "user", "text": "hello"}])
        deleted = await self.store.delete_session("chat-1")
        assert deleted is True
        assert await self.store.load_history("chat-1") == []

    async def test_delete_nonexistent_returns_false(self):
        assert await self.store.delete_session("no-such-chat") is False

    async def test_add_user_chat_and_list(self):
        await self.store.add_user_chat("user-1", "chat-a")
        await self.store.add_user_chat("user-1", "chat-b")
        await self.store.add_user_chat("user-2", "chat-c")
        assert sorted(await self.store.get_user_chats("user-1")) == ["chat-a", "chat-b"]
        assert await self.store.get_user_chats("user-2") == ["chat-c"]

    async def test_get_user_chats_unknown_user(self):
        assert await self.store.get_user_chats("ghost") == []

    async def test_created_at_is_set_and_stable(self):
        await self.store.add_user_chat("u1", "c1")
        first = await self.store.get_chat_created_at("c1")
        assert first is not None
        await self.store.add_user_chat("u1", "c1")
        assert await self.store.get_chat_created_at("c1") == first

    async def test_created_at_missing_returns_none(self):
        assert await self.store.get_chat_created_at("nope") is None

    async def test_save_then_add_user_backfills_created_at(self):
        await self.store.save_history("c1", [{"role": "user", "text": "q"}])
        await self.store.add_user_chat("u1", "c1")
        assert await self.store.get_chat_created_at("c1") is not None

    async def test_remove_user_chat_unlinks(self):
        await self.store.add_user_chat("u1", "c1")
        await self.store.remove_user_chat("u1", "c1")
        assert await self.store.get_user_chats("u1") == []

    async def test_remove_user_chat_ignores_other_users(self):
        await self.store.add_user_chat("u1", "c1")
        await self.store.remove_user_chat("u2", "c1")
        assert await self.store.get_user_chats("u1") == ["c1"]


class TestCreateSessionStore:
    def test_falls_back_to_in_memory_without_credentials(self, monkeypatch):
        monkeypatch.setattr("store.FIREBASE_SERVICE_ACCOUNT", "")
        monkeypatch.setattr("store.FIREBASE_SERVICE_ACCOUNT_JSON", "")
        monkeypatch.setattr("store.GOOGLE_APPLICATION_CREDENTIALS", "")
        store = create_session_store()
        assert isinstance(store, SessionStore)

    def test_warns_when_credentials_set_but_sdk_missing(self, monkeypatch, caplog):
        monkeypatch.setattr("store.FIREBASE_SERVICE_ACCOUNT", "/tmp/does-not-exist.json")
        monkeypatch.setattr("store._firebase_available", False)
        store = create_session_store()
        assert isinstance(store, SessionStore)
        assert "firebase-admin is not installed" in caplog.text
