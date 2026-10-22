"""Tests for Muraja'ah Revision Intelligence (#253)."""

from datetime import date, datetime, timedelta

import pytest

from memorization_planner import (
    AyahProgress,
    MurajaahEngine,
    RecallQuality,
    RevisionOutcome,
    RevisionRecordRequest,
    RevisionSessionRequest,
    _forgetting_curve,
    _priority_score,
    _sm2_update,
    router,
)


@pytest.fixture
def client():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def test_sm2_good_recall_grows_interval():
    p = AyahProgress(surah=2, ayah=255)
    p = _sm2_update(p, RecallQuality.GOOD)
    assert p.interval_days == 1
    p = _sm2_update(p, RecallQuality.GOOD)
    assert p.interval_days == 3
    p = _sm2_update(p, RecallQuality.GOOD)
    assert p.interval_days > 3


def test_sm2_failure_resets_and_counts_lapse():
    p = AyahProgress(surah=2, ayah=255, repetitions=5, interval_days=30, review_count=5)
    p = _sm2_update(p, RecallQuality.WRONG)
    assert p.repetitions == 0
    assert p.lapses == 1
    assert p.error_count == 1
    assert p.interval_days == 1


def test_sm2_ease_floor():
    p = AyahProgress(surah=1, ayah=1)
    for _ in range(20):
        p = _sm2_update(p, RecallQuality.BLACKOUT)
    assert p.ease >= 1.3


def test_forgetting_curve_decays():
    assert _forgetting_curve(0) == 1.0
    assert _forgetting_curve(1) < 1.0
    assert _forgetting_curve(10, stability=1) < _forgetting_curve(1, stability=1)


def test_priority_never_reviewed_is_top():
    fresh = AyahProgress(surah=1, ayah=1)
    reviewed = AyahProgress(
        surah=1,
        ayah=2,
        last_reviewed=datetime.now(),
        next_review=date.today(),
    )
    assert _priority_score(fresh) > _priority_score(reviewed)


def test_build_session_balances_tiers():
    now = datetime.now()
    today = date.today()
    ayahs: list[AyahProgress] = []
    for i in range(1, 21):
        p = AyahProgress(surah=1, ayah=i)
        if i <= 10:
            p.next_review = today - timedelta(days=10)
            p.review_count, p.error_count = 10, 5
            p.last_reviewed = now - timedelta(days=10)
        elif i <= 15:
            p.next_review = today - timedelta(days=2)
            p.review_count, p.error_count = 10, 4
            p.last_reviewed = now - timedelta(days=2)
        else:
            p.next_review = today
        ayahs.append(p)

    session = MurajaahEngine.build_session(ayahs, RevisionSessionRequest(max_minutes=1, max_ayahs=20))
    assert len(session.ayahs) > 0
    assert session.breakdown.get("critical", 0) > 0


def test_build_session_respects_budget():
    ayahs = [AyahProgress(surah=2, ayah=i) for i in range(1, 11)]
    session = MurajaahEngine.build_session(ayahs, RevisionSessionRequest(max_minutes=5), seconds_per_ayah=60)
    assert len(session.ayahs) <= 5


def test_record_skipped_pushes_to_tomorrow():
    p = MurajaahEngine.record(
        [], RevisionRecordRequest(surah=1, ayah=1, quality=RecallQuality.GOOD, outcome=RevisionOutcome.SKIPPED)
    )
    assert p.next_review == date.today() + timedelta(days=1)


def test_record_partial_downgrades():
    p = MurajaahEngine.record(
        [], RevisionRecordRequest(surah=1, ayah=1, quality=RecallQuality.GOOD, outcome=RevisionOutcome.PARTIAL)
    )
    assert p.repetitions == 0


def test_reminder_empty_when_nothing_due():
    res = MurajaahEngine.reminder([])
    assert res.message is None


def test_reminder_mentions_count():
    ayahs = [AyahProgress(surah=1, ayah=1), AyahProgress(surah=1, ayah=2)]
    res = MurajaahEngine.reminder(ayahs)
    assert res.message is not None and "2" in res.message


def test_stats_shape():
    res = MurajaahEngine.stats([AyahProgress(surah=1, ayah=1)])
    assert res.total_ayahs == 1
    assert res.due_today == 1


def test_api_revision_session(client):
    resp = client.post("/memorization/revision/session", json={"max_minutes": 10})
    assert resp.status_code == 200
    assert "ayahs" in resp.json()


def test_api_revision_reminder(client):
    resp = client.get("/memorization/revision/reminder")
    assert resp.status_code == 200
    assert resp.json()["due_today"] == 0


def test_api_revision_stats(client):
    resp = client.get("/memorization/revision/stats")
    assert resp.status_code == 200
    assert resp.json()["total_ayahs"] == 0
