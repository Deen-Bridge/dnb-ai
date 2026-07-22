"""Tests for conversation history truncation.

All tests run offline — no GEMINI_API_KEY needed.
"""

from unittest.mock import MagicMock, PropertyMock

import pytest

from history import (
    MAX_HISTORY_TOKENS,
    MAX_HISTORY_TURNS,
    estimate_tokens,
    trim_history,
)


# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------


class TestEstimateTokens:
    def test_empty_string(self):
        assert estimate_tokens("") == 1

    def test_short_text(self):
        assert estimate_tokens("hello") == 1

    def test_typical_text(self):
        text = "What is the significance of Salah in Islam?"
        expected = max(1, len(text) // 4)
        assert estimate_tokens(text) == expected

    def test_long_text(self):
        text = "A" * 1000
        assert estimate_tokens(text) == 250


# ---------------------------------------------------------------------------
# Helpers for building fake history entries
# ---------------------------------------------------------------------------


def _make_msg(role: str, text: str):
    part = MagicMock(spec=[])
    part.text = text
    msg = MagicMock(spec=["role", "parts"])
    msg.role = role
    msg.parts = [part]
    return msg


def _make_turn_pair(user_text: str, model_text: str):
    return _make_msg("user", user_text), _make_msg("model", model_text)


def _make_chat_session(pairs: list[tuple[str, str]]):
    """Build a fake chat session with a mutable history list."""
    history = []
    for user_text, model_text in pairs:
        history.append(_make_msg("user", user_text))
        history.append(_make_msg("model", model_text))
    chat = MagicMock(spec=["history"])
    type(chat).history = PropertyMock(return_value=history)
    return chat


# ---------------------------------------------------------------------------
# Truncation behaviour
# ---------------------------------------------------------------------------


class TestTrimHistory:
    def test_no_history_is_noop(self):
        chat = _make_chat_session([])
        assert trim_history(chat) is False

    def test_single_turn_under_budget_is_noop(self):
        chat = _make_chat_session([("hello", "hi there")])
        assert trim_history(chat) is False

    def test_truncates_oldest_pairs_when_over_token_budget(self, monkeypatch):
        monkeypatch.setattr("history.MAX_HISTORY_TOKENS", 10)
        monkeypatch.setattr("history.MAX_HISTORY_TURNS", 50)
        pairs = [
            ("AAAAA AAAAA AA", "BBBBB BBBBB BB"),   # ~5+5 = 10 tokens — will be dropped
            ("CCCCC CCCCC CC", "DDDDD DDDDD DD"),   # ~5+5 = 10 tokens — will be dropped
            ("EEEEE EEEEE EE", "FFFFF FFFFF FF"),   # ~5+5 = 10 tokens — keeps this
        ]
        chat = _make_chat_session(pairs)
        assert trim_history(chat) is True
        assert len(chat.history) == 2  # one turn pair remains
        assert chat.history[0].parts[0].text == "EEEEE EEEEE EE"
        assert chat.history[1].parts[0].text == "FFFFF FFFFF FF"

    def test_truncates_to_turn_cap_first(self, monkeypatch):
        monkeypatch.setattr("history.MAX_HISTORY_TOKENS", 999999)
        monkeypatch.setattr("history.MAX_HISTORY_TURNS", 2)
        pairs = [
            ("A", "B"),
            ("C", "D"),
            ("E", "F"),
            ("G", "H"),
        ]
        chat = _make_chat_session(pairs)
        assert trim_history(chat) is True
        # MAX_HISTORY_TURNS=2 → at most 4 entries (2 pairs)
        assert len(chat.history) == 4
        assert chat.history[0].parts[0].text == "E"
        assert chat.history[1].parts[0].text == "F"
        assert chat.history[2].parts[0].text == "G"
        assert chat.history[3].parts[0].text == "H"

    def test_no_orphaned_turns_after_truncation(self, monkeypatch):
        """History must always have pairs — never a lone user or model turn."""
        monkeypatch.setattr("history.MAX_HISTORY_TOKENS", 10)
        monkeypatch.setattr("history.MAX_HISTORY_TURNS", 50)
        pairs = [
            ("AAAAA AAAAA AA", "BBBBB BBBBB BB"),
            ("CCCCC CCCCC CC", "DDDDD DDDDD DD"),
            ("EEEEE EEEEE EE", "FFFFF FFFFF FF"),
        ]
        chat = _make_chat_session(pairs)
        trim_history(chat)
        assert len(chat.history) % 2 == 0

    def test_returns_true_when_truncation_happens(self, monkeypatch):
        monkeypatch.setattr("history.MAX_HISTORY_TOKENS", 3)
        monkeypatch.setattr("history.MAX_HISTORY_TURNS", 50)
        chat = _make_chat_session([("AAAA AAAA", "BBBB BBBB")])
        assert trim_history(chat) is True

    def test_preserves_recent_context_after_truncation(self, monkeypatch):
        monkeypatch.setattr("history.MAX_HISTORY_TOKENS", 20)
        monkeypatch.setattr("history.MAX_HISTORY_TURNS", 50)
        pairs = [
            ("A" * 40 + " very old", "B" * 40 + " very old response"),
            ("C" * 40 + " old", "D" * 40 + " old response"),
            ("Recent question", "Recent answer"),
        ]
        chat = _make_chat_session(pairs)
        trim_history(chat)
        remaining = " ".join(
            m.parts[0].text for m in chat.history
        )
        assert "Recent" in remaining
        assert "Old question 1" not in remaining


# ---------------------------------------------------------------------------
# Integration: end-to-end scenario
# ---------------------------------------------------------------------------


class TestTrimHistoryIntegration:
    def test_long_conversation_triggers_truncation(self, monkeypatch):
        monkeypatch.setattr("history.MAX_HISTORY_TOKENS", 50)
        monkeypatch.setattr("history.MAX_HISTORY_TURNS", 50)
        pairs = [
            (f"User message {i} " + "X" * 60, f"Model response {i} " + "Y" * 60)
            for i in range(20)
        ]
        chat = _make_chat_session(pairs)
        truncated = trim_history(chat)
        assert truncated is True
        assert len(chat.history) < len(pairs) * 2

    def test_budget_configurable_via_env(self, monkeypatch):
        monkeypatch.setattr("history.MAX_HISTORY_TOKENS", 1000)
        monkeypatch.setattr("history.MAX_HISTORY_TURNS", 3)
        pairs = [
            ("A", "B"),
            ("C", "D"),
            ("E", "F"),
            ("G", "H"),
            ("I", "J"),
        ]
        chat = _make_chat_session(pairs)
        trim_history(chat)
        # 3 turn pairs max → 6 entries
        assert len(chat.history) <= 6
        assert len(chat.history) % 2 == 0
