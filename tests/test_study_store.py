"""Isolation and reset of user-scoped study data (#848).

Exercises StudyStore's in-memory backend directly: no Redis, no network, and
the FastAPI app is never imported.
"""

from __future__ import annotations

import asyncio
import logging

import pytest

from study_store import StudyStore


@pytest.fixture
def store(monkeypatch: pytest.MonkeyPatch) -> StudyStore:
    monkeypatch.delenv("REDIS_URL", raising=False)
    return StudyStore()


def _record(store: StudyStore, user_id: str, discipline: str = "fiqh") -> None:
    asyncio.run(store.record_progress(user_id=user_id, discipline=discipline, score=80.0, items_count=2))


def test_learners_do_not_see_each_others_progress(store: StudyStore) -> None:
    _record(store, "learner-a", "fiqh")
    _record(store, "learner-b", "tafsir")
    a = asyncio.run(store.get_progress("learner-a"))
    b = asyncio.run(store.get_progress("learner-b"))
    assert a is not None and b is not None
    assert a.user_id == "learner-a" and set(a.mastery_by_discipline) == {"fiqh"}
    assert b.user_id == "learner-b" and set(b.mastery_by_discipline) == {"tafsir"}


def test_reset_erases_one_learner_only_and_is_idempotent(store: StudyStore) -> None:
    _record(store, "learner-a")
    _record(store, "learner-b")
    assert asyncio.run(store.reset_user("learner-a")) is True
    assert asyncio.run(store.get_progress("learner-a")) is None
    assert asyncio.run(store.get_progress("learner-b")) is not None
    assert asyncio.run(store.reset_user("learner-a")) is False


def test_reset_then_new_progress_starts_from_zero(store: StudyStore) -> None:
    _record(store, "learner-a")
    _record(store, "learner-a")
    asyncio.run(store.reset_user("learner-a"))
    _record(store, "learner-a")
    progress = asyncio.run(store.get_progress("learner-a"))
    assert progress is not None and progress.quizzes_completed == 1


@pytest.mark.parametrize("blank", ["", " ", "\t\n"])
def test_blank_user_ids_are_refused_instead_of_sharing_a_key(store: StudyStore, blank: str) -> None:
    with pytest.raises(ValueError, match="blank"):
        _record(store, blank)
    with pytest.raises(ValueError):
        asyncio.run(store.get_progress(blank))
    with pytest.raises(ValueError):
        asyncio.run(store.reset_user(blank))


def test_arabic_user_ids_are_isolated(store: StudyStore) -> None:
    _record(store, "طالب-١")
    _record(store, "طالب-٢")
    assert asyncio.run(store.reset_user("طالب-١")) is True
    assert asyncio.run(store.get_progress("طالب-٢")) is not None


def test_reset_log_carries_a_hash_not_the_user_id(store: StudyStore, caplog: pytest.LogCaptureFixture) -> None:
    _record(store, "learner-private-123")
    with caplog.at_level(logging.INFO, logger="study_store"):
        asyncio.run(store.reset_user("learner-private-123"))
    text = caplog.text
    assert "learner" not in text
    assert "user hash" in text
