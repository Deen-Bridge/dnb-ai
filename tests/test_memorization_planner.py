"""Tests for the Personalized Quran Memorization Planner System (#252)."""

import pytest
from fastapi.testclient import TestClient

from errors import APIException
from memorization_planner import (
    TOTAL_QURAN_SURAHS,
    AdjustPlanRequest,
    AyahRange,
    CheckMilestoneRequest,
    DifficultyTier,
    GeneratePlanRequest,
    GoalType,
    MemorizationPlannerEngine,
    PaceLevel,
    PlanGoal,
    SessionLog,
    SessionPerformance,
    UserCapacity,
    get_juz_surahs,
    get_surah_info,
    router,
)


@pytest.fixture
def client():
    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


# ---------------------------------------------------------------------------
# Surah Data & Structure Tests
# ---------------------------------------------------------------------------


class TestSurahData:
    def test_total_surahs_count(self):
        surahs = [get_surah_info(i) for i in range(1, 115)]
        assert len(surahs) == TOTAL_QURAN_SURAHS
        assert all(s is not None for s in surahs)

    def test_surah_info_attributes(self):
        fatihah = get_surah_info(1)
        assert fatihah is not None
        assert fatihah.name == "Al-Fatihah"
        assert fatihah.ayah_count == 7
        assert fatihah.difficulty == DifficultyTier.EASY

        baqarah = get_surah_info(2)
        assert baqarah is not None
        assert baqarah.name == "Al-Baqarah"
        assert baqarah.ayah_count == 286
        assert baqarah.difficulty == DifficultyTier.HARD

        mulk = get_surah_info(67)
        assert mulk is not None
        assert mulk.name == "Al-Mulk"
        assert mulk.ayah_count == 30
        assert mulk.juz_start == 29

    def test_juz_surahs_retrieval(self):
        juz_30 = get_juz_surahs(30)
        assert len(juz_30) == 37  # Surahs 78 to 114
        assert any(s.name == "An-Naba" for s in juz_30)
        assert any(s.name == "An-Nas" for s in juz_30)

        juz_1 = get_juz_surahs(1)
        assert any(s.name == "Al-Fatihah" for s in juz_1)
        assert any(s.name == "Al-Baqarah" for s in juz_1)

    def test_invalid_juz(self):
        assert get_juz_surahs(0) == []
        assert get_juz_surahs(31) == []


# ---------------------------------------------------------------------------
# Model Validation Tests
# ---------------------------------------------------------------------------


class TestModelValidation:
    def test_valid_ayah_range(self):
        r = AyahRange(
            surah_number=1,
            surah_name="Al-Fatihah",
            start_ayah=1,
            end_ayah=7,
            ayah_count=7,
        )
        assert r.ayah_count == 7

    def test_invalid_ayah_range_inverted(self):
        with pytest.raises(ValueError, match="cannot be less than"):
            AyahRange(
                surah_number=1,
                surah_name="Al-Fatihah",
                start_ayah=5,
                end_ayah=2,
                ayah_count=1,
            )

    def test_invalid_ayah_range_exceeds_surah(self):
        with pytest.raises(ValueError, match="only has 7 ayahs"):
            AyahRange(
                surah_number=1,
                surah_name="Al-Fatihah",
                start_ayah=1,
                end_ayah=10,
                ayah_count=10,
            )

    def test_user_capacity_defaults_and_validation(self):
        cap = UserCapacity()
        assert cap.daily_available_minutes == 30
        assert cap.days_per_week == 6
        assert cap.pace_level == PaceLevel.MODERATE

        with pytest.raises(ValueError):
            UserCapacity(days_per_week=8)


# ---------------------------------------------------------------------------
# Plan Generation & Timeline Calculation Tests
# ---------------------------------------------------------------------------


class TestPlanGeneration:
    def test_generate_plan_juz_30(self):
        req = GeneratePlanRequest(
            user_id="ahmed_01",
            goal=PlanGoal(goal_type=GoalType.JUZ, target_juz=[30]),
            capacity=UserCapacity(
                daily_available_minutes=30,
                days_per_week=6,
                pace_level=PaceLevel.MODERATE,
            ),
            start_date="2026-11-01",
        )
        plan = MemorizationPlannerEngine.generate_plan(req)

        assert plan.user_id == "ahmed_01"
        assert plan.total_target_ayahs == 564  # Juz 30 ayahs
        assert plan.daily_capacity_ayahs > 0
        assert plan.estimated_study_days > 0
        assert plan.estimated_calendar_days >= plan.estimated_study_days
        assert len(plan.weekly_schedule) == 7
        assert len(plan.milestones) == 4

        # Check time split
        assert plan.daily_time_split["sabaq_new_minutes"] > 0
        assert plan.daily_time_split["sabqi_recent_revision_minutes"] > 0
        assert plan.daily_time_split["manzil_cumulative_minutes"] > 0

    def test_generate_plan_specific_surahs(self):
        req = GeneratePlanRequest(
            user_id="fatima_02",
            goal=PlanGoal(
                goal_type=GoalType.SPECIFIC_SURAHS,
                target_surahs=[36, 67],  # Ya-Sin and Al-Mulk
            ),
            capacity=UserCapacity(
                daily_available_minutes=45,
                days_per_week=5,
                pace_level=PaceLevel.INTENSIVE,
            ),
        )
        plan = MemorizationPlannerEngine.generate_plan(req)

        assert plan.total_target_ayahs == 83 + 30  # 113 ayahs
        assert plan.difficulty_profile["easy"] >= 1
        assert plan.estimated_study_days < 25

    def test_already_memorized_filter(self):
        req = GeneratePlanRequest(
            user_id="omar_03",
            goal=PlanGoal(
                goal_type=GoalType.SPECIFIC_SURAHS,
                target_surahs=[1, 112, 113, 114],
                already_memorized_surahs=[1, 112],
            ),
            capacity=UserCapacity(daily_available_minutes=20),
        )
        plan = MemorizationPlannerEngine.generate_plan(req)
        assert plan.total_target_ayahs == 5 + 6  # Falaq (5) + Nas (6) = 11

    def test_schedule_includes_rest_and_consolidation(self):
        req = GeneratePlanRequest(
            user_id="user_test",
            goal=PlanGoal(goal_type=GoalType.JUZ, target_juz=[30]),
            capacity=UserCapacity(days_per_week=5, daily_available_minutes=30),
        )
        plan = MemorizationPlannerEngine.generate_plan(req)
        week = plan.weekly_schedule

        rest_days = [d for d in week if d.is_rest_day]
        consolidation_days = [d for d in week if d.is_consolidation_day]
        active_days = [d for d in week if not d.is_rest_day and not d.is_consolidation_day]

        assert len(rest_days) == 2  # 7 - 5 = 2 rest days
        assert len(consolidation_days) == 1  # 5th day is consolidation
        assert len(active_days) == 4

    def test_empty_pending_surahs_raises_exception(self):
        req = GeneratePlanRequest(
            user_id="empty_user",
            goal=PlanGoal(
                goal_type=GoalType.SPECIFIC_SURAHS,
                target_surahs=[1],
                already_memorized_surahs=[1],
            ),
            capacity=UserCapacity(),
        )
        with pytest.raises(APIException) as exc_info:
            MemorizationPlannerEngine.generate_plan(req)
        assert exc_info.value.status_code == 400


# ---------------------------------------------------------------------------
# Adaptive Plan Adjustment Tests
# ---------------------------------------------------------------------------


class TestAdaptiveAdjustment:
    def test_adjustment_when_struggling_or_missed(self):
        logs = [
            SessionLog(date_str="2026-10-18", completed=False, performance=SessionPerformance.STRUGGLING),
            SessionLog(date_str="2026-10-19", completed=True, performance=SessionPerformance.STRUGGLING),
            SessionLog(date_str="2026-10-20", completed=False, performance=SessionPerformance.STRUGGLING),
        ]
        req = AdjustPlanRequest(
            plan_id="plan_123",
            completed_ayahs_count=45,
            recent_sessions=logs,
            days_missed=3,
        )
        adj = MemorizationPlannerEngine.adjust_plan(req)

        assert adj.retention_health == "Needs Consolidation"
        assert adj.days_added_or_subtracted > 0
        assert adj.recommended_pace_ayahs < 10
        assert any("Pause new Sabaq" in step for step in adj.immediate_next_steps)
        assert "double reward" in adj.encouragement  # Authentic hadith citation included

    def test_adjustment_when_excelling(self):
        logs = [
            SessionLog(date_str="2026-10-18", completed=True, performance=SessionPerformance.STRONG),
            SessionLog(date_str="2026-10-19", completed=True, performance=SessionPerformance.STRONG),
            SessionLog(date_str="2026-10-20", completed=True, performance=SessionPerformance.STRONG),
        ]
        req = AdjustPlanRequest(
            plan_id="plan_123",
            completed_ayahs_count=120,
            recent_sessions=logs,
            days_missed=0,
        )
        adj = MemorizationPlannerEngine.adjust_plan(req)

        assert adj.retention_health == "Excellent"
        assert adj.recommended_pace_ayahs >= 10
        assert adj.days_added_or_subtracted <= 0
        assert "camels" in adj.encouragement  # Authentic Bukhari 5033 reminder on retention

    def test_adjustment_with_updated_capacity(self):
        req = AdjustPlanRequest(
            plan_id="plan_123",
            completed_ayahs_count=50,
            recent_sessions=[],
            updated_capacity=UserCapacity(
                daily_available_minutes=60,
                pace_level=PaceLevel.INTENSIVE,
            ),
        )
        adj = MemorizationPlannerEngine.adjust_plan(req)
        assert adj.recommended_pace_ayahs > 15


# ---------------------------------------------------------------------------
# Milestone Tracking & Celebration Tests
# ---------------------------------------------------------------------------


class TestMilestoneTracking:
    def test_milestone_surah_completed(self):
        res = MemorizationPlannerEngine.check_milestone(
            CheckMilestoneRequest(
                plan_id="p1",
                current_memorized_ayahs=30,
                recently_finished_surah=67,  # Al-Mulk
            )
        )
        assert res.achieved is True
        assert "Al-Mulk" in res.milestone_title
        assert "Surah Finisher" in res.badge
        assert "Bukhari 5027" in res.authentic_citation

    def test_milestone_streak_30_days(self):
        res = MemorizationPlannerEngine.check_milestone(
            CheckMilestoneRequest(
                plan_id="p1",
                current_memorized_ayahs=150,
                current_streak_days=30,
            )
        )
        assert res.achieved is True
        assert "30-Day" in res.milestone_title
        assert "Istiqamah" in res.badge

    def test_milestone_halfway_mark(self):
        res = MemorizationPlannerEngine.check_milestone(
            CheckMilestoneRequest(
                plan_id="p1",
                current_memorized_ayahs=300,
            ),
            total_goal_ayahs=564,
        )
        assert res.achieved is True
        assert "Halfway" in res.milestone_title
        assert "Muslim 798" in res.authentic_citation


# ---------------------------------------------------------------------------
# FastApi API Endpoints Integration Tests
# ---------------------------------------------------------------------------


class TestEndpoints:
    def test_api_list_surahs(self, client):
        resp = client.get("/memorization/surahs")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 114
        assert data[0]["name"] == "Al-Fatihah"
        assert data[1]["name"] == "Al-Baqarah"

    def test_api_generate_plan(self, client):
        payload = {
            "user_id": "test_user_api",
            "goal": {
                "goal_type": "specific_surahs",
                "target_surahs": [36, 67],
                "already_memorized_surahs": [],
            },
            "capacity": {
                "daily_available_minutes": 35,
                "days_per_week": 6,
                "pace_level": "moderate",
            },
            "start_date": "2026-11-01",
        }
        resp = client.post("/memorization/plan", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["user_id"] == "test_user_api"
        assert data["total_target_ayahs"] == 113
        assert len(data["weekly_schedule"]) == 7

    def test_api_adjust_plan(self, client):
        payload = {
            "plan_id": "plan_test_api",
            "completed_ayahs_count": 30,
            "recent_sessions": [
                {"date_str": "2026-10-19", "completed": True, "performance": "struggling"},
                {"date_str": "2026-10-20", "completed": False, "performance": "struggling"},
            ],
            "days_missed": 2,
        }
        resp = client.post("/memorization/adjust", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["retention_health"] == "Needs Consolidation"
        assert len(data["immediate_next_steps"]) > 0

    def test_api_milestone_check(self, client):
        payload = {
            "plan_id": "plan_test_api",
            "current_memorized_ayahs": 50,
            "current_streak_days": 15,
            "recently_finished_surah": 114,
        }
        resp = client.post("/memorization/milestone-check", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["achieved"] is True
        assert "An-Nas" in data["milestone_title"]
