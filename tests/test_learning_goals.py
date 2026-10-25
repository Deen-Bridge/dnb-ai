"""Tests for learning goal tracking and recommendations (#256).

Everything is deterministic: the engine takes an explicit ``today`` and the HTTP layer
exposes it as ``as_of``, so no test depends on the real clock. No network or model is used.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import date
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import learning_goals
from learning_goals import (
    MILESTONE_MESSAGES,
    Goal,
    GoalConflictError,
    GoalHorizon,
    GoalStatus,
    GoalStore,
    GoalType,
    PaceStatus,
    ProgressEntry,
    build_achievements,
    build_reminder,
    build_summary,
    compute_pace,
    compute_streaks,
    effective_target_date,
    expected_percent,
    is_complete,
    milestones_reached,
    percent_complete,
    progress_bar,
    recent_daily_rate,
    recommend,
    reset_goal_store,
    router,
    suggest_timeline,
)

START = date(2026, 1, 1)
END = date(2026, 1, 11)  # a 10-day goal, so "today = Jan 6" is exactly halfway


def _goal(
    *,
    target: float = 100.0,
    start: date = START,
    end: date | None = END,
    horizon: GoalHorizon = GoalHorizon.SHORT_TERM,
    goal_type: GoalType = GoalType.MEMORIZATION,
    entries: list[tuple[date, float]] | None = None,
    weekly_capacity: float | None = None,
    title: str = "Memorize Juz Amma",
    user_id: str = "u1",
    goal_id: str = "g1",
) -> Goal:
    return Goal(
        goal_id=goal_id,
        user_id=user_id,
        title=title,
        goal_type=goal_type,
        horizon=horizon,
        target_amount=target,
        unit="pages",
        start_date=start,
        target_date=end,
        weekly_capacity=weekly_capacity,
        entries=[ProgressEntry(entry_date=d, amount=a) for d, a in (entries or [])],
    )


@pytest.fixture(autouse=True)
def _clean_store() -> Iterator[None]:
    reset_goal_store()
    yield
    reset_goal_store()


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def _create(client: TestClient, **overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "user_id": "u1",
        "title": "Memorize Juz Amma",
        "goal_type": "memorization",
        "target_amount": 100,
        "unit": "pages",
        "target_date": "2026-01-11",
    }
    body.update(overrides)
    response = client.post("/learning-goals", params={"as_of": "2026-01-01"}, json=body)
    assert response.status_code == 201, response.text
    data: dict[str, Any] = response.json()
    return data


def _progress(client: TestClient, goal_id: str, amount: float, as_of: str = "2026-01-06", **extra: Any) -> Any:
    return client.post(
        f"/learning-goals/{goal_id}/progress",
        params={"as_of": as_of},
        json={"user_id": "u1", "amount": amount, **extra},
    )


# ---------------------------------------------------------------------------
# Progress calculation
# ---------------------------------------------------------------------------


def test_progress_bar_rendering() -> None:
    assert progress_bar(0) == "[--------------------] 0%"
    assert progress_bar(50) == "[##########----------] 50%"
    assert progress_bar(100) == "[####################] 100%"


def test_percent_complete_never_rounds_up_to_100_before_the_goal_is_done() -> None:
    goal = _goal(entries=[(START, 99.96)])

    assert not is_complete(goal)
    assert percent_complete(goal) == 99.9


def test_goal_completes_despite_float_rounding() -> None:
    goal = _goal(target=0.3, entries=[(START, 0.1), (START, 0.2)])

    assert is_complete(goal)
    assert percent_complete(goal) == 100.0


def test_expected_percent_tracks_elapsed_time_and_is_clamped() -> None:
    goal = _goal()

    assert expected_percent(goal, date(2025, 12, 1)) == 0.0
    assert expected_percent(goal, date(2026, 1, 6)) == 50.0
    assert expected_percent(goal, date(2026, 3, 1)) == 100.0


def test_effective_target_date_defaults_by_horizon() -> None:
    assert effective_target_date(_goal(end=None, horizon=GoalHorizon.SHORT_TERM)) == date(2026, 1, 31)
    assert effective_target_date(_goal(end=None, horizon=GoalHorizon.LONG_TERM)) == date(2026, 6, 30)
    assert effective_target_date(_goal(end=date(2026, 2, 2))) == date(2026, 2, 2)


@pytest.mark.parametrize(
    ("done", "expected"),
    [
        (0, PaceStatus.NOT_STARTED),
        (40, PaceStatus.BEHIND),
        (46, PaceStatus.ON_TRACK),  # 92% of where it should be: still within tolerance
        (50, PaceStatus.ON_TRACK),
        (54, PaceStatus.ON_TRACK),
        (60, PaceStatus.AHEAD),
    ],
)
def test_pace_compares_actual_progress_with_the_plan(done: float, expected: PaceStatus) -> None:
    goal = _goal(entries=[(date(2026, 1, 5), done)] if done else [])

    assert compute_pace(goal, date(2026, 1, 6)) == expected


def test_progress_on_the_first_day_counts_as_ahead() -> None:
    goal = _goal(entries=[(START, 5)])

    assert compute_pace(goal, START) == PaceStatus.AHEAD


def test_completed_goal_reports_completed_pace() -> None:
    goal = _goal(entries=[(START, 100)])
    goal.status = GoalStatus.COMPLETED

    assert compute_pace(goal, date(2026, 1, 6)) == PaceStatus.COMPLETED


# ---------------------------------------------------------------------------
# Streaks and recent pace
# ---------------------------------------------------------------------------


def test_streaks_count_consecutive_days_and_report_the_longest() -> None:
    days = [date(2026, 1, d) for d in (1, 2, 3, 5)]
    goal = _goal(entries=[(d, 1) for d in days])

    assert compute_streaks(goal, date(2026, 1, 5)) == (1, 3)


def test_streak_survives_until_the_end_of_the_day_after_the_last_entry() -> None:
    goal = _goal(entries=[(date(2026, 1, 4), 1), (date(2026, 1, 5), 1)])

    assert compute_streaks(goal, date(2026, 1, 5)) == (2, 2)
    assert compute_streaks(goal, date(2026, 1, 6)) == (2, 2)  # nothing logged yet today
    assert compute_streaks(goal, date(2026, 1, 7)) == (0, 2)  # a full day was missed


def test_several_entries_on_one_day_count_as_one_day() -> None:
    goal = _goal(entries=[(START, 1), (START, 2), (START, 3)])

    assert compute_streaks(goal, START) == (1, 1)


def test_streaks_are_zero_without_entries() -> None:
    assert compute_streaks(_goal(), START) == (0, 0)


def test_recent_daily_rate_uses_a_14_day_window() -> None:
    goal = _goal(entries=[(date(2026, 1, 1), 100), (date(2026, 1, 10), 14)])

    # The Jan 1 entry is outside the window ending Jan 15 (Jan 2 - Jan 15).
    assert recent_daily_rate(goal, date(2026, 1, 15)) == 1.0


def test_recent_daily_rate_for_a_brand_new_goal_uses_a_one_day_window() -> None:
    assert recent_daily_rate(_goal(entries=[(START, 5)]), START) == 5.0
    assert recent_daily_rate(_goal(), START) == 0.0


# ---------------------------------------------------------------------------
# Milestones and achievements
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("percent", "expected"),
    [(0, []), (24.9, []), (25, [25]), (60, [25, 50]), (99.9, [25, 50, 75]), (100, [25, 50, 75, 100])],
)
def test_milestones_reached(percent: float, expected: list[int]) -> None:
    assert milestones_reached(percent) == expected


def test_every_milestone_has_a_celebration_message() -> None:
    assert set(MILESTONE_MESSAGES) == {25, 50, 75, 100}


def test_seven_day_streak_earns_an_achievement() -> None:
    goal = _goal(entries=[(date(2026, 1, d), 1) for d in range(1, 8)])

    assert "7-day streak" in build_achievements(goal, date(2026, 1, 7))
    assert "7-day streak" not in build_achievements(_goal(entries=[(START, 1)]), START)


# ---------------------------------------------------------------------------
# Timeline suggestions
# ---------------------------------------------------------------------------


def test_timeline_discounts_stated_capacity_for_realism() -> None:
    result = suggest_timeline(85, 10, START)

    assert result.realistic is True
    assert result.assumed_weekly_amount == 8.5
    assert result.days_needed == 70
    assert result.weeks_needed == 10
    assert result.completion_date == date(2026, 3, 12)


def test_timeline_uses_an_observed_pace_as_is() -> None:
    result = suggest_timeline(70, 10, START, apply_realism=False)

    assert result.assumed_weekly_amount == 10.0
    assert result.days_needed == 49


def test_timeline_that_would_take_over_ten_years_is_reported_unrealistic() -> None:
    result = suggest_timeline(1_000_000, 1, START)

    assert result.realistic is False
    assert result.days_needed is None
    assert result.completion_date is None
    assert "ten years" in result.note


def test_timeline_takes_at_least_one_day() -> None:
    assert suggest_timeline(0.001, 1000, START).days_needed == 1


# ---------------------------------------------------------------------------
# Recommendations
# ---------------------------------------------------------------------------


def test_recommendation_spreads_the_remainder_over_the_days_left() -> None:
    goal = _goal(entries=[(date(2026, 1, 5), 50)])

    rec = recommend(goal, date(2026, 1, 6))  # 50 left, 5 days left

    assert rec.pace == PaceStatus.ON_TRACK
    assert rec.required_daily_amount == 10.0
    assert rec.required_weekly_amount == 50.0  # capped at what is actually left
    assert rec.daily_action == "Aim for about 10 pages today."
    assert rec.weekly_action == "Aim for about 50 pages this week."
    assert rec.overdue is False
    assert rec.adjusted_timeline is None
    assert rec.tip == learning_goals.TYPE_TIPS[GoalType.MEMORIZATION]


def test_behind_learner_gets_a_timeline_based_on_their_real_pace() -> None:
    goal = _goal(entries=[(date(2026, 1, 5), 10)])

    rec = recommend(goal, date(2026, 1, 6))  # 10 pages over a 6-day window = 1.667/day

    assert rec.pace == PaceStatus.BEHIND
    assert rec.adjusted_timeline is not None
    assert rec.adjusted_timeline.assumed_weekly_amount == pytest.approx(11.667, abs=0.01)
    assert rec.adjusted_timeline.days_needed == 54  # 90 pages left at ~11.67 per week


def test_not_started_learner_falls_back_to_stated_capacity() -> None:
    goal = _goal(weekly_capacity=20)

    rec = recommend(goal, date(2026, 1, 6))

    assert rec.pace == PaceStatus.NOT_STARTED
    assert rec.adjusted_timeline is not None
    assert rec.adjusted_timeline.assumed_weekly_amount == 17.0  # 20 * 0.85


def test_not_started_learner_without_capacity_gets_no_timeline() -> None:
    assert recommend(_goal(), date(2026, 1, 6)).adjusted_timeline is None


def test_overdue_goal_gets_a_fresh_plan_instead_of_a_negative_one() -> None:
    goal = _goal(entries=[(date(2026, 1, 5), 20)])

    rec = recommend(goal, date(2026, 2, 1))

    assert rec.overdue is True
    assert rec.days_left < 0
    assert "has passed" in rec.message
    assert rec.required_daily_amount == pytest.approx(80 / 30, abs=0.001)  # fresh 30-day short-term window


def test_ahead_learner_keeps_the_same_plan() -> None:
    goal = _goal(entries=[(date(2026, 1, 5), 90)])

    rec = recommend(goal, date(2026, 1, 6))

    assert rec.pace == PaceStatus.AHEAD
    assert rec.adjusted_timeline is None


def test_completed_goal_has_no_further_actions() -> None:
    goal = _goal(entries=[(START, 100)])
    goal.status = GoalStatus.COMPLETED

    rec = recommend(goal, date(2026, 1, 6))

    assert rec.pace == PaceStatus.COMPLETED
    assert rec.daily_action is None
    assert rec.weekly_action is None
    assert rec.required_daily_amount == 0.0


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


def test_store_hides_goals_owned_by_other_users() -> None:
    store = GoalStore()
    store.add(_goal(user_id="alice", goal_id="a1"))

    assert store.get("a1", "alice").goal_id == "a1"
    with pytest.raises(KeyError):
        store.get("a1", "bob")
    with pytest.raises(KeyError):
        store.get("missing", "alice")


def test_store_enforces_a_per_user_goal_limit() -> None:
    store = GoalStore()
    for index in range(learning_goals.MAX_GOALS_PER_USER):
        store.add(_goal(goal_id=f"g{index}"))

    with pytest.raises(GoalConflictError):
        store.add(_goal(goal_id="one-too-many"))
    store.add(_goal(goal_id="other-user", user_id="someone-else"))  # the limit is per user


def test_store_delete_and_listing() -> None:
    store = GoalStore()
    store.add(_goal(goal_id="a", user_id="u1"))
    store.add(_goal(goal_id="b", user_id="u2"))

    assert [g.goal_id for g in store.list_for_user("u1")] == ["a"]
    store.delete("a", "u1")
    assert store.list_for_user("u1") == []
    with pytest.raises(KeyError):
        store.delete("a", "u1")


# ---------------------------------------------------------------------------
# Reminders and summary
# ---------------------------------------------------------------------------


def test_reminder_for_a_goal_with_no_progress_encourages_starting() -> None:
    reminder = build_reminder(_goal(), date(2026, 1, 6))

    assert reminder.kind == "get_started"
    assert reminder.days_since_last_entry is None
    assert "Memorize Juz Amma" in reminder.message


def test_reminder_warns_when_the_deadline_is_close_and_the_learner_is_behind() -> None:
    goal = _goal(entries=[(date(2026, 1, 8), 10)])

    reminder = build_reminder(goal, date(2026, 1, 9))  # 2 days left, only 10% done

    assert reminder.kind == "deadline"
    assert "2 day(s)" in reminder.message


def test_reminder_for_a_lapsed_learner_mentions_the_gap() -> None:
    goal = _goal(entries=[(START, 50)])

    reminder = build_reminder(goal, date(2026, 1, 6))

    assert reminder.kind == "inactive"
    assert reminder.days_since_last_entry == 5
    assert "5 days" in reminder.message


def test_reminder_celebrates_a_streak() -> None:
    goal = _goal(entries=[(date(2026, 1, 4), 20), (date(2026, 1, 5), 20), (date(2026, 1, 6), 20)])

    reminder = build_reminder(goal, date(2026, 1, 6))

    assert reminder.kind == "streak"
    assert reminder.streak == 3


def test_reminder_defaults_to_a_progress_update() -> None:
    goal = _goal(entries=[(date(2026, 1, 6), 50)])

    reminder = build_reminder(goal, date(2026, 1, 6))

    assert reminder.kind == "progress_update"
    assert "50% complete" in reminder.message


def test_reminder_for_a_completed_goal_suggests_a_next_goal() -> None:
    goal = _goal(entries=[(START, 100)])
    goal.status = GoalStatus.COMPLETED

    assert build_reminder(goal, date(2026, 1, 6)).kind == "completed"


def test_summary_aggregates_goals_and_lists_todays_actions() -> None:
    behind = _goal(goal_id="behind", entries=[(date(2026, 1, 5), 10)])
    done = _goal(goal_id="done", entries=[(START, 100)])
    done.status = GoalStatus.COMPLETED

    summary = build_summary("u1", [behind, done], date(2026, 1, 6))

    assert summary.total_goals == 2
    assert summary.active_goals == 1
    assert summary.completed_goals == 1
    assert summary.average_percent_complete == 55.0
    assert summary.goals_behind == ["behind"]
    assert summary.next_actions == ["Aim for about 18 pages today."]


def test_summary_for_a_learner_with_no_goals() -> None:
    summary = build_summary("nobody", [], date(2026, 1, 6))

    assert summary.total_goals == 0
    assert summary.average_percent_complete == 0.0
    assert summary.goals_behind == []
    assert summary.next_actions == []


# ---------------------------------------------------------------------------
# HTTP: creating goals
# ---------------------------------------------------------------------------


def test_create_goal_returns_a_goal_with_zero_progress(client: TestClient) -> None:
    data = _create(client)

    assert data["title"] == "Memorize Juz Amma"
    assert data["status"] == "active"
    assert data["target_date"] == "2026-01-11"
    assert data["progress"]["percent_complete"] == 0.0
    assert data["progress"]["pace"] == "not_started"
    assert data["progress"]["days_left"] == 10
    assert data["progress"]["progress_bar"] == "[--------------------] 0%"


def test_target_date_is_derived_from_weekly_capacity(client: TestClient) -> None:
    data = _create(client, target_date=None, target_amount=85, weekly_capacity=10)

    assert data["target_date"] == "2026-03-12"
    assert data["weekly_capacity"] == 10.0


def test_target_date_defaults_from_the_horizon(client: TestClient) -> None:
    assert _create(client, target_date=None, horizon="long_term")["target_date"] == "2026-06-30"
    assert _create(client, target_date=None, horizon="short_term")["target_date"] == "2026-01-31"


@pytest.mark.parametrize(
    "overrides",
    [
        {"target_amount": 0},
        {"target_amount": -5},
        {"title": ""},
        {"unit": ""},
        {"goal_type": "astrology"},
        {"target_date": "2025-12-01"},  # before the start
        {"start_date": "2026-02-01", "target_date": "2026-01-15"},  # target before explicit start
        {"title": "   "},  # blank after trimming
    ],
)
def test_invalid_goals_are_rejected(client: TestClient, overrides: dict[str, Any]) -> None:
    body: dict[str, Any] = {"user_id": "u1", "title": "Goal", "target_amount": 10, "target_date": "2026-01-11"}
    body.update(overrides)

    response = client.post("/learning-goals", params={"as_of": "2026-01-01"}, json=body)

    assert response.status_code == 422


def test_unrealistic_capacity_is_rejected_with_an_explanation(client: TestClient) -> None:
    response = client.post(
        "/learning-goals",
        params={"as_of": "2026-01-01"},
        json={"user_id": "u1", "title": "Huge", "target_amount": 1_000_000, "weekly_capacity": 1},
    )

    assert response.status_code == 422
    assert "ten years" in response.json()["detail"]


def test_as_of_outside_the_supported_years_is_rejected(client: TestClient) -> None:
    response = client.post(
        "/learning-goals",
        params={"as_of": "1999-01-01"},
        json={"user_id": "u1", "title": "Goal", "target_amount": 10},
    )

    assert response.status_code == 422


def test_a_learner_cannot_exceed_the_goal_limit(client: TestClient) -> None:
    for index in range(learning_goals.MAX_GOALS_PER_USER):
        _create(client, title=f"Goal {index}")

    response = client.post(
        "/learning-goals",
        params={"as_of": "2026-01-01"},
        json={"user_id": "u1", "title": "One too many", "target_amount": 10},
    )

    assert response.status_code == 409


# ---------------------------------------------------------------------------
# HTTP: logging progress, milestones and completion
# ---------------------------------------------------------------------------


def test_progress_unlocks_milestones_with_celebrations(client: TestClient) -> None:
    goal_id = _create(client)["goal_id"]

    first = _progress(client, goal_id, 30).json()
    assert first["new_milestones"] == [25]
    assert first["celebrations"] == [MILESTONE_MESSAGES[25]]
    assert first["goal"]["progress"]["percent_complete"] == 30.0
    assert first["goal"]["progress"]["pace"] == "behind"  # 30% done where 50% was expected

    second = _progress(client, goal_id, 30).json()
    assert second["new_milestones"] == [50]
    assert second["goal"]["progress"]["milestones_reached"] == [25, 50]


def test_one_large_entry_unlocks_every_milestone_it_crosses(client: TestClient) -> None:
    goal_id = _create(client)["goal_id"]

    body = _progress(client, goal_id, 80).json()

    assert body["new_milestones"] == [25, 50, 75]
    assert len(body["celebrations"]) == 3


def test_completing_a_goal_marks_it_done_and_blocks_more_progress(client: TestClient) -> None:
    goal_id = _create(client)["goal_id"]
    _progress(client, goal_id, 60)

    done = _progress(client, goal_id, 40).json()

    assert done["new_milestones"] == [75, 100]
    assert done["goal"]["status"] == "completed"
    assert done["goal"]["completed_on"] == "2026-01-06"
    assert done["goal"]["progress"]["percent_complete"] == 100.0
    assert "Goal completed" in done["goal"]["achievements"]
    assert _progress(client, goal_id, 1).status_code == 409


def test_progress_dates_are_validated(client: TestClient) -> None:
    goal_id = _create(client)["goal_id"]

    assert _progress(client, goal_id, 5, entry_date="2026-02-01").status_code == 422  # in the future
    assert _progress(client, goal_id, 5, entry_date="2025-12-01").status_code == 422  # before the goal began
    assert _progress(client, goal_id, 5, entry_date="2026-01-03").status_code == 200


@pytest.mark.parametrize("amount", [0, -1])
def test_non_positive_progress_is_rejected(client: TestClient, amount: float) -> None:
    goal_id = _create(client)["goal_id"]

    assert _progress(client, goal_id, amount).status_code == 422


# ---------------------------------------------------------------------------
# HTTP: reading goals, ownership, recommendations, reminders, summary
# ---------------------------------------------------------------------------


def test_get_goal_requires_the_owning_user(client: TestClient) -> None:
    goal_id = _create(client)["goal_id"]
    params = {"as_of": "2026-01-06"}

    assert client.get(f"/learning-goals/{goal_id}", params={"user_id": "u1", **params}).status_code == 200
    assert client.get(f"/learning-goals/{goal_id}", params={"user_id": "intruder", **params}).status_code == 404
    assert client.get("/learning-goals/missing", params={"user_id": "u1", **params}).status_code == 404


def test_another_user_cannot_log_progress_on_a_goal(client: TestClient) -> None:
    goal_id = _create(client)["goal_id"]

    response = client.post(
        f"/learning-goals/{goal_id}/progress",
        params={"as_of": "2026-01-06"},
        json={"user_id": "intruder", "amount": 5},
    )

    assert response.status_code == 404


def test_list_goals_only_returns_the_requesters_goals(client: TestClient) -> None:
    _create(client, title="Mine")
    _create(client, user_id="other", title="Theirs")

    mine = client.get("/learning-goals/user/u1", params={"as_of": "2026-01-06"}).json()

    assert [g["title"] for g in mine] == ["Mine"]
    assert client.get("/learning-goals/user/nobody").json() == []


def test_recommendations_endpoint_reflects_actual_progress(client: TestClient) -> None:
    goal_id = _create(client)["goal_id"]
    _progress(client, goal_id, 50, entry_date="2026-01-05")

    params = {"user_id": "u1", "as_of": "2026-01-06"}

    response = client.get(f"/learning-goals/{goal_id}/recommendations", params=params)

    body = response.json()
    assert response.status_code == 200
    assert body["pace"] == "on_track"
    assert body["daily_action"] == "Aim for about 10 pages today."
    assert body["overdue"] is False


def test_reminders_only_cover_active_goals(client: TestClient) -> None:
    active = _create(client, title="Active")["goal_id"]
    finished = _create(client, title="Finished")["goal_id"]
    _progress(client, finished, 100)

    reminders = client.get("/learning-goals/user/u1/reminders", params={"as_of": "2026-01-06"}).json()

    assert [r["goal_id"] for r in reminders] == [active]
    assert reminders[0]["kind"] == "get_started"


def test_summary_endpoint_aggregates_goals(client: TestClient) -> None:
    _create(client, title="A")
    finished = _create(client, title="B")["goal_id"]
    _progress(client, finished, 100)

    summary = client.get("/learning-goals/user/u1/summary", params={"as_of": "2026-01-06"}).json()

    assert summary["total_goals"] == 2
    assert summary["active_goals"] == 1
    assert summary["completed_goals"] == 1
    assert summary["average_percent_complete"] == 50.0


def test_delete_removes_the_goal(client: TestClient) -> None:
    goal_id = _create(client)["goal_id"]

    assert client.delete(f"/learning-goals/{goal_id}", params={"user_id": "intruder"}).status_code == 404
    assert client.delete(f"/learning-goals/{goal_id}", params={"user_id": "u1"}).status_code == 204
    assert client.get(f"/learning-goals/{goal_id}", params={"user_id": "u1"}).status_code == 404


# ---------------------------------------------------------------------------
# HTTP: reference endpoints
# ---------------------------------------------------------------------------


def test_types_endpoint_lists_goal_types_horizons_and_milestones(client: TestClient) -> None:
    body = client.get("/learning-goals/types").json()

    assert "memorization" in body["goal_types"]
    assert "hadith" in body["goal_types"]
    assert "fiqh" in body["goal_types"]
    assert body["horizons"] == ["short_term", "long_term"]
    assert body["milestones"] == ["25", "50", "75", "100"]


def test_timeline_endpoint_suggests_a_realistic_date(client: TestClient) -> None:
    response = client.post(
        "/learning-goals/timeline",
        json={"target_amount": 85, "weekly_capacity": 10, "start_date": "2026-01-01"},
    )

    body = response.json()
    assert response.status_code == 200
    assert body["realistic"] is True
    assert body["completion_date"] == "2026-03-12"
    assert body["weeks_needed"] == 10


def test_timeline_endpoint_validates_input(client: TestClient) -> None:
    assert client.post("/learning-goals/timeline", json={"target_amount": 0, "weekly_capacity": 5}).status_code == 422
    assert client.post("/learning-goals/timeline", json={"target_amount": 5, "weekly_capacity": 0}).status_code == 422
    too_early = {"target_amount": 5, "weekly_capacity": 5, "start_date": "1999-01-01"}
    assert client.post("/learning-goals/timeline", json=too_early).status_code == 422
