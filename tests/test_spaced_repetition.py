"""Offline scheduling, durable storage, concurrency and API tests."""

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from pydantic import ValidationError

from spaced_repetition import (
    CardInput,
    ContentType,
    ImportCards,
    RepetitionStore,
    ReviewInput,
    Schedule,
    get_store,
    router,
    schedule_review,
)

NOW = datetime(2026, 1, 1, tzinfo=UTC)


def content(**kwargs) -> CardInput:
    return CardInput(front="What does sabr mean?", back="Patience", content_type=ContentType.vocabulary, **kwargs)


@pytest.fixture
def store(tmp_path):
    instance = RepetitionStore(str(tmp_path / "cards.db"))
    yield instance
    instance.close()


def test_sm2_intervals_use_previous_ease_and_round_up():
    state = Schedule(due_at=NOW)
    for interval, ease in [(1, 2.6), (6, 2.7), (17, 2.8), (48, 2.9)]:
        at = state.due_at
        old = state.model_copy(deep=True)
        new = schedule_review(state, 5, at)
        assert new.interval_days == interval
        assert new.ease == ease
        assert new.due_at == at + timedelta(days=interval)
        assert state == old
        state = new
    assert state.repetitions == state.reviews == state.successes == 4


def test_grade_four_keeps_ease_and_failure_resets_sequence():
    state = Schedule(due_at=NOW)
    for _ in range(3):
        state = schedule_review(state, 4, state.due_at)
    assert state.interval_days == 15
    assert state.ease == 2.5
    at = state.due_at
    state = schedule_review(state, 1, at)
    assert state.repetitions == 0
    assert state.lapses == 1
    assert state.due_at == at + timedelta(minutes=10)
    state = schedule_review(state, 4, state.due_at)
    assert state.repetitions == 1
    assert state.interval_days == 1


@pytest.mark.parametrize("grade", range(6))
def test_grade_outcomes(grade):
    state = schedule_review(Schedule(due_at=NOW), grade, NOW)
    assert state.successes == int(grade >= 3)
    assert state.lapses == int(grade < 3)
    assert state.reviews == 1


def test_ease_floor_and_interval_cap():
    state = Schedule(due_at=NOW)
    for _ in range(30):
        state = schedule_review(state, 0, state.due_at)
    assert state.ease == 1.3
    state = Schedule(due_at=NOW, repetitions=50, interval_days=30000)
    assert schedule_review(state, 5, NOW).interval_days == 36500


@pytest.mark.parametrize("grade", [-1, 6, True, 2.5, "4"])
def test_invalid_grades(grade):
    with pytest.raises(ValueError):
        schedule_review(Schedule(due_at=NOW), grade, NOW)
    with pytest.raises(ValidationError):
        ReviewInput(grade=grade, revision=0)


def test_timezone_validation_and_clock_regression():
    with pytest.raises(ValueError, match="timezone"):
        schedule_review(Schedule(due_at=NOW), 4, NOW.replace(tzinfo=None))
    state = schedule_review(Schedule(due_at=NOW), 4, NOW)
    with pytest.raises(ValueError, match="previous review"):
        schedule_review(state, 4, NOW - timedelta(seconds=1))


def test_due_queue_boundaries_and_user_isolation(store):
    first = store.create("alice", [content()], NOW)[0]
    second = store.create("alice", [content()], NOW + timedelta(seconds=1))[0]
    store.create("bob", [content()], NOW)
    assert store.due("alice", NOW - timedelta(microseconds=1), 20) == []
    assert [c.id for c in store.due("alice", NOW, 20)] == [first.id]
    assert [c.id for c in store.due("alice", NOW + timedelta(days=1), 1)] == [first.id]
    store.review("alice", first.id, ReviewInput(grade=4, revision=0), NOW)
    assert [c.id for c in store.due("alice", NOW + timedelta(seconds=1), 20)] == [second.id]
    with pytest.raises(HTTPException) as error:
        store.get("bob", first.id)
    assert error.value.status_code == 404
    with pytest.raises(HTTPException) as error:
        store.review("bob", first.id, ReviewInput(grade=4, revision=0), NOW)
    assert error.value.status_code == 404


def test_stale_and_early_reviews_do_not_change_statistics(store):
    card = store.create("alice", [content()], NOW)[0]
    first = store.review("alice", card.id, ReviewInput(grade=4, revision=0), NOW)
    for request in [ReviewInput(grade=4, revision=0), ReviewInput(grade=4, revision=1)]:
        with pytest.raises(HTTPException) as error:
            store.review("alice", card.id, request, NOW)
        assert error.value.status_code == 409
    assert store.get("alice", card.id) == first
    assert store.statistics("alice", NOW).reviews == 1


def test_retention_is_measured_and_persists_after_restart(tmp_path):
    path = str(tmp_path / "durable.db")
    store = RepetitionStore(path)
    assert store.statistics("alice", NOW).retention_rate is None
    card = store.create("alice", [content()], NOW)[0]
    state = store.review("alice", card.id, ReviewInput(grade=5, revision=0), NOW)
    store.review("alice", card.id, ReviewInput(grade=1, revision=1), state.schedule.due_at)
    store.close()
    reopened = RepetitionStore(path)
    try:
        stats = reopened.statistics("alice", NOW + timedelta(days=2))
        assert stats.total_cards == stats.reviewed_cards == stats.due_cards == 1
        assert stats.reviews == 2
        assert stats.successful_reviews == stats.lapses == 1
        assert stats.retention_rate == 0.5
        assert stats.by_content_type["vocabulary"] == 1
        assert reopened.statistics("bob", NOW).total_cards == 0
    finally:
        reopened.close()


def test_concurrent_workers_cannot_grade_same_revision_twice(tmp_path):
    path = str(tmp_path / "concurrent.db")
    stores = [RepetitionStore(path), RepetitionStore(path)]
    card = stores[0].create("alice", [content()], NOW)[0]

    def grade(store):
        try:
            store.review("alice", card.id, ReviewInput(grade=4, revision=0), NOW)
            return 200
        except HTTPException as error:
            return error.status_code

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            assert sorted(pool.map(grade, stores)) == [200, 409]
        assert stores[0].statistics("alice", NOW).reviews == 1
    finally:
        for store in stores:
            store.close()


@pytest.mark.parametrize("kind", ContentType)
def test_all_content_types_and_multimedia(kind, store):
    card = CardInput(
        front="Recall",
        back="Answer",
        content_type=kind,
        source="Teacher supplied reference",
        audio_url="https://example.com/recitation.mp3",
        image_url="https://example.com/card.png",
        tags=[" Arabic ", "Arabic"],
    )
    saved = store.create("alice", [card], NOW)[0]
    assert store.get("alice", saved.id).content == card
    assert card.tags == ["Arabic"]


@pytest.mark.parametrize(
    "change",
    [
        {"front": "  "},
        {"content_type": "unknown"},
        {"audio_url": "file:///etc/passwd"},
        {"image_url": "javascript:alert(1)"},
        {"tags": [""]},
    ],
)
def test_invalid_content(change):
    data = content().model_dump(mode="json")
    data.update(change)
    with pytest.raises(ValidationError):
        CardInput.model_validate(data)


def test_import_caps_and_atomic_validation():
    with pytest.raises(ValidationError):
        ImportCards(cards=[])
    with pytest.raises(ValidationError):
        ImportCards(cards=[content()] * 501)


def test_api_create_import_review_and_statistics(store, monkeypatch):
    import spaced_repetition

    monkeypatch.setattr(spaced_repetition, "utc_now", lambda: NOW)
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_store] = lambda: store
    with TestClient(app) as client:
        prefix = "/study/repetition/alice"
        response = client.post(prefix + "/cards", json=content().model_dump(mode="json"))
        assert response.status_code == 201
        card = response.json()
        response = client.post(
            prefix + "/import", json={"cards": [content().model_dump(mode="json"), {"front": "broken"}]}
        )
        assert response.status_code == 422
        assert len(client.get(prefix + "/session").json()) == 1
        assert client.post(prefix + "/import", json={"cards": [content().model_dump(mode="json")]}).status_code == 201
        assert client.get(prefix + "/session?limit=0").status_code == 422
        assert client.get(prefix + "/cards/" + card["id"]).status_code == 200
        response = client.post(prefix + "/cards/" + card["id"] + "/review", json={"grade": 5, "revision": 0})
        assert response.status_code == 200
        assert response.json()["schedule"]["interval_days"] == 1
        assert (
            client.post(prefix + "/cards/" + card["id"] + "/review", json={"grade": 5, "revision": 0}).status_code
            == 409
        )
        stats = client.get(prefix + "/statistics").json()
        assert stats["total_cards"] == 2
        assert stats["due_cards"] == 1
        assert stats["retention_rate"] == 1.0


def test_main_mount_protects_learner_data_with_service_key(store, monkeypatch):
    import main

    monkeypatch.setattr(main, "AUTH_DISABLED", False)
    monkeypatch.setattr(main, "SERVICE_API_KEY", "trusted-backend-key")
    main.app.dependency_overrides[get_store] = lambda: store
    try:
        with TestClient(main.app) as client:
            route = "/study/repetition/alice/statistics"
            assert client.get(route).status_code == 401
            assert client.get(route, headers={"X-API-Key": "wrong"}).status_code == 401
            assert client.get(route, headers={"X-API-Key": "trusted-backend-key"}).status_code == 200
    finally:
        main.app.dependency_overrides.pop(get_store, None)


def test_import_rolls_back_on_database_failure(store):
    import sqlite3

    with store._db:
        store._db.execute(
            "CREATE TRIGGER reject_second BEFORE INSERT ON repetition_cards "
            "WHEN (SELECT COUNT(*) FROM repetition_cards) > 0 "
            "BEGIN SELECT RAISE(ABORT, 'simulated storage failure'); END"
        )
    with pytest.raises(sqlite3.IntegrityError):
        store.create("alice", [content(), content()], NOW)
    assert store.statistics("alice", NOW).total_cards == 0
