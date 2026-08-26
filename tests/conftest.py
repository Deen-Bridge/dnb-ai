"""Test configuration for importing repository modules without packaging the app."""

import os
import sys
from pathlib import Path

# Settings (config.py) requires GEMINI_API_KEY and is built when `main` is
# imported. Provide a dummy so tests that import the app collect in the hermetic
# CI env — the safety pipeline is disabled and Gemini is mocked, so no real key
# or network is ever used.
os.environ.setdefault("GEMINI_API_KEY", "test-key")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402


class _FakeSessionStore:
    """Fast in-memory stand-in for the real session store.

    Endpoint tests exercise the chat flow, not Firestore I/O: swapping the real
    store (which would make live network calls) for this keeps them hermetic and
    fast. The Firestore-backed store itself is tested directly in
    test_firestore_session_store.py against a fake client.
    """

    def __init__(self):
        self._histories: dict[str, list[dict[str, str]]] = {}
        self._user_chats: dict[str, set[str]] = {}

    async def load_history(self, chat_id: str) -> list[dict[str, str]]:
        return list(self._histories.get(chat_id, []))

    async def save_history(self, chat_id: str, history: list[dict[str, str]]) -> None:
        self._histories[chat_id] = [dict(m) for m in history]

    async def delete_session(self, chat_id: str) -> bool:
        return self._histories.pop(chat_id, None) is not None

    async def add_user_chat(self, user_id: str, chat_id: str) -> None:
        self._user_chats.setdefault(user_id, set()).add(chat_id)

    async def get_user_chats(self, user_id: str) -> list[str]:
        return list(self._user_chats.get(user_id, set()))

    async def get_chat_created_at(self, chat_id: str) -> float | None:
        return None

    async def remove_user_chat(self, user_id: str, chat_id: str) -> None:
        self._user_chats.get(user_id, set()).discard(chat_id)


@pytest.fixture(autouse=True)
def _fast_session_store(monkeypatch):
    """Replace main.session_store with an in-memory fake wherever main is loaded."""
    main = sys.modules.get("main")
    if main is None:
        return
    monkeypatch.setattr(main, "session_store", _FakeSessionStore())
