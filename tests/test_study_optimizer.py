"""Tests for Study Session Optimization (#258)."""

import pytest

from memorization_planner import AyahRange, SessionLog, SessionPerformance, UserCapacity
from study import (
    EnergyLevel,
    SessionBlockType,
    StudySessionOptimizer,
    StudySessionRequest,
    WorkSchedule,
    _content_mix,
    _fatigue_from_history,
    _infer_energy_level,
    _pick_start_hour,
    router,
)


@pytest.fixture
def client():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def _range() -> AyahRange:
    return AyahRange(
        surah_number=67,
        surah_name="Al-Mulk",
        start_ayah=1,
        end_ayah=5,
        ayah_count=5,
    )


def test_energy_peaks_morning_and_afternoon():
    assert _infer_energy_level(10, fasting=False, schedule=WorkSchedule.NONE) == EnergyLevel.PEAK
    assert _infer_energy_level(16, fasting=False, schedule=WorkSchedule.NONE) == EnergyLevel.PEAK


def test_energy_low_at_night():
    assert _infer_energy_level(2, fasting=False, schedule=WorkSchedule.NONE) == EnergyLevel.LOW


def test_energy_fasting_downgrades_afternoon_peak():
    assert _infer_energy_level(14, fasting=True, schedule=WorkSchedule.NONE) != EnergyLevel.PEAK


def test_content_mix_sums_to_one():
    for e in EnergyLevel:
        m = _content_mix(e, fasting=False)
        assert abs(sum(m.values()) - 1.0) < 0.02


def test_content_mix_fasting_shifts_toward_review():
    normal = _content_mix(EnergyLevel.PEAK, fasting=False)
    fasting = _content_mix(EnergyLevel.PEAK, fasting=True)
    assert fasting["review"] >= normal["review"]


def test_pick_start_hour_ramadan_shifts():
    assert _pick_start_hour(12, day_of_week=0, is_ramadan=True) in (5, 20)


def test_optimize_produces_blocks_within_budget():
    req = StudySessionRequest(
        user_id="u1",
        hour_of_day=10,
        day_of_week=1,
        available_minutes=30,
        average_attention_span_minutes=15,
        pending_new_portion=_range(),
        pending_review_portion=_range(),
    )
    plan = StudySessionOptimizer.optimize(req)
    assert plan.total_minutes > 0
    assert plan.total_minutes <= 40  # includes break padding
    assert any(b.block_type == SessionBlockType.REVIEW for b in plan.blocks)
    assert any(b.block_type == SessionBlockType.MEMORIZE for b in plan.blocks)
    assert any(b.block_type == SessionBlockType.BREAK for b in plan.blocks)
    assert plan.recommended_start_hour == 10


def test_optimize_respects_capacity_cap():
    req = StudySessionRequest(
        user_id="u1",
        hour_of_day=10,
        day_of_week=1,
        available_minutes=60,
        capacity=UserCapacity(daily_available_minutes=20),
    )
    plan = StudySessionOptimizer.optimize(req)
    assert plan.total_minutes <= 40


def test_optimize_without_new_portion_converts_to_review():
    req = StudySessionRequest(
        user_id="u1",
        hour_of_day=10,
        day_of_week=1,
        available_minutes=30,
        pending_review_portion=_range(),
    )
    plan = StudySessionOptimizer.optimize(req)
    assert not any(b.block_type == SessionBlockType.NEW for b in plan.blocks)


def test_fatigue_without_history_is_low_risk():
    s = _fatigue_from_history([], hour=10, fasting=False)
    assert s.detected is False


def test_fatigue_with_struggling_history_is_detected():
    logs = [
        SessionLog(date_str="2026-10-18", completed=False, performance=SessionPerformance.STRUGGLING),
        SessionLog(date_str="2026-10-19", completed=False, performance=SessionPerformance.STRUGGLING),
        SessionLog(date_str="2026-10-20", completed=True, performance=SessionPerformance.STRUGGLING),
    ]
    s = _fatigue_from_history(logs, hour=22, fasting=True)
    assert s.detected is True
    assert s.confidence >= 0.5


def test_api_optimize(client):
    payload = {
        "user_id": "u1",
        "hour_of_day": 10,
        "day_of_week": 1,
        "available_minutes": 30,
        "is_ramadan": False,
    }
    resp = client.post("/study/optimize", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert "blocks" in body
    assert body["total_minutes"] > 0
    assert body["energy_level"] in {"low", "medium", "high", "peak"}


def test_api_fatigue_check(client):
    resp = client.post("/study/fatigue-check", json={"hour_of_day": 14, "fasting": False})
    assert resp.status_code == 200
    assert "confidence" in resp.json()
