"""Learning goal tracking and recommendations (#256).

Why this exists
---------------
Learners have no structured way to say "I want to memorize Juz 30 in three months"
or "I want to finish a hadith collection this year" and then be guided toward it.
This module lets a learner define short-term and long-term goals of several types
(memorization, fiqh, hadith, tafsir, Arabic, ...), log progress against them, and
get back progress indicators, milestone celebrations, daily/weekly actions,
pace-adjusted recommendations, motivational reminders and realistic timelines.

Design constraints
------------------
Deterministic and offline: no language model, no network, no import-time side
effects. Every calculation takes an explicit ``today`` so behaviour is
reproducible and unit-testable; the HTTP layer exposes it as the optional
``as_of`` query parameter. Goals are held in an in-process store, matching the
other self-contained modules in this service; persistence belongs in
``dnb-backend`` and can replace ``GoalStore`` without touching the engine.

The service has no authentication layer, so ``user_id`` is a caller-supplied key,
exactly as in the other per-learner modules. It is not an access-control boundary.
"""

from __future__ import annotations

import logging
import math
import threading
import uuid
from datetime import date, timedelta
from enum import Enum

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, model_validator

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/learning-goals", tags=["learning-goals"])

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MILESTONES: tuple[int, ...] = (25, 50, 75, 100)
# People rarely sustain their full planned capacity, so timelines assume this share.
REALISM_FACTOR = 0.85
# Recent activity window used to measure the learner's actual pace.
PACE_WINDOW_DAYS = 14
# Within +/-10% of the expected amount counts as on track.
AHEAD_RATIO = 1.1
BEHIND_RATIO = 0.9
# A suggested timeline longer than this is reported as unrealistic rather than computed.
MAX_TIMELINE_DAYS = 3650
STREAK_ACHIEVEMENT_DAYS = 7
MAX_TARGET_AMOUNT = 1_000_000.0
MAX_GOALS_PER_USER = 25
MAX_TOTAL_GOALS = 20_000
MAX_ENTRIES_PER_GOAL = 1_000
# Float sums such as 0.1 + 0.2 must still complete a goal of 0.3, so completion is
# judged with a tiny relative tolerance rather than exact equality.
_REL_EPS = 1e-9
_EPSILON = 1e-9
_PERCENT_EPS = 1e-6


class GoalType(str, Enum):
    """What the learner is working toward."""

    MEMORIZATION = "memorization"
    FIQH = "fiqh"
    HADITH = "hadith"
    TAFSIR = "tafsir"
    ARABIC_LANGUAGE = "arabic_language"
    SEERAH = "seerah"
    AQEEDAH = "aqeedah"
    GENERAL = "general"


class GoalHorizon(str, Enum):
    SHORT_TERM = "short_term"
    LONG_TERM = "long_term"


class GoalStatus(str, Enum):
    ACTIVE = "active"
    COMPLETED = "completed"


class PaceStatus(str, Enum):
    """How actual progress compares with where the learner should be today."""

    NOT_STARTED = "not_started"
    BEHIND = "behind"
    ON_TRACK = "on_track"
    AHEAD = "ahead"
    COMPLETED = "completed"


# With no target date, a goal is paced over this many days by default.
DEFAULT_HORIZON_DAYS: dict[GoalHorizon, int] = {
    GoalHorizon.SHORT_TERM: 30,
    GoalHorizon.LONG_TERM: 180,
}

MILESTONE_MESSAGES: dict[int, str] = {
    25: "A quarter of the way there. A strong start; keep the rhythm going.",
    50: "Halfway. The hardest part, building the habit, is behind you.",
    75: "Three quarters done. The finish line is in sight.",
    100: "Goal complete. Well done. Revisit what you learned so it stays with you.",
}

TYPE_TIPS: dict[GoalType, str] = {
    GoalType.MEMORIZATION: "Revise what you learned yesterday before adding anything new.",
    GoalType.FIQH: "Study one ruling together with its evidence and note one practical application.",
    GoalType.HADITH: "Read the hadith, who narrated it, and write down one lesson from it.",
    GoalType.TAFSIR: "Read the verses first, then the commentary, then summarize in your own words.",
    GoalType.ARABIC_LANGUAGE: "Practice new vocabulary in a sentence you write yourself.",
    GoalType.SEERAH: "Read one episode and note the lesson it teaches.",
    GoalType.AQEEDAH: "Study one topic slowly and check each point against its source.",
    GoalType.GENERAL: "Keep sessions short and regular rather than long and rare.",
}

PACE_MESSAGES: dict[PaceStatus, str] = {
    PaceStatus.NOT_STARTED: "You have not logged any progress yet. A small first session is enough to begin.",
    PaceStatus.BEHIND: "You are behind your plan. That is recoverable: a slightly higher steady pace closes the gap.",
    PaceStatus.ON_TRACK: "You are on track. Consistency is doing the work; keep going.",
    PaceStatus.AHEAD: "You are ahead of your plan. Keep the pace or use the margin to revise.",
    PaceStatus.COMPLETED: "This goal is complete. Consider setting the next one.",
}

MIN_YEAR = 2000
MAX_YEAR = 2200


class GoalConflictError(Exception):
    """The request is well-formed but not allowed in the goal's current state."""


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class ProgressEntry(BaseModel):
    entry_date: date
    amount: float = Field(..., gt=0, le=MAX_TARGET_AMOUNT)
    note: str | None = Field(default=None, max_length=500)


class Goal(BaseModel):
    """A learner's goal together with the progress logged against it."""

    goal_id: str
    user_id: str
    title: str
    goal_type: GoalType
    horizon: GoalHorizon
    target_amount: float
    unit: str
    start_date: date
    target_date: date | None = None
    weekly_capacity: float | None = None
    status: GoalStatus = GoalStatus.ACTIVE
    completed_on: date | None = None
    entries: list[ProgressEntry] = Field(default_factory=list)
    achieved_milestones: list[int] = Field(default_factory=list)


class GoalCreateRequest(BaseModel):
    user_id: str = Field(..., min_length=1, max_length=100)
    title: str = Field(..., min_length=1, max_length=200)
    goal_type: GoalType = GoalType.GENERAL
    horizon: GoalHorizon = GoalHorizon.SHORT_TERM
    target_amount: float = Field(..., gt=0, le=MAX_TARGET_AMOUNT, description="Total amount to reach, in `unit`.")
    unit: str = Field(default="units", min_length=1, max_length=40, description="e.g. pages, ayahs, hadith, lessons.")
    start_date: date | None = Field(default=None, description="Defaults to today.")
    target_date: date | None = Field(default=None, description="Defaults to a horizon-based period after the start.")
    weekly_capacity: float | None = Field(
        default=None, gt=0, le=MAX_TARGET_AMOUNT, description="Amount the learner expects to do per week."
    )

    @model_validator(mode="after")
    def _dates_in_order(self) -> GoalCreateRequest:
        if self.start_date and self.target_date and self.target_date <= self.start_date:
            raise ValueError("target_date must be after start_date")
        return self


class ProgressRequest(BaseModel):
    user_id: str = Field(..., min_length=1, max_length=100)
    amount: float = Field(..., gt=0, le=MAX_TARGET_AMOUNT)
    entry_date: date | None = Field(default=None, description="Defaults to today.")
    note: str | None = Field(default=None, max_length=500)


class TimelineRequest(BaseModel):
    target_amount: float = Field(..., gt=0, le=MAX_TARGET_AMOUNT)
    weekly_capacity: float = Field(..., gt=0, le=MAX_TARGET_AMOUNT)
    start_date: date | None = Field(default=None, description="Defaults to today.")


class SuggestedTimeline(BaseModel):
    realistic: bool
    days_needed: int | None
    weeks_needed: int | None
    completion_date: date | None
    assumed_weekly_amount: float
    note: str


class GoalProgress(BaseModel):
    amount_done: float
    remaining: float
    percent_complete: float
    expected_percent: float
    pace: PaceStatus
    pace_message: str
    days_left: int
    current_streak: int
    recent_daily_rate: float
    milestones_reached: list[int]
    progress_bar: str


class GoalView(BaseModel):
    goal_id: str
    user_id: str
    title: str
    goal_type: GoalType
    horizon: GoalHorizon
    target_amount: float
    unit: str
    start_date: date
    target_date: date
    weekly_capacity: float | None
    status: GoalStatus
    completed_on: date | None
    entries_count: int
    achievements: list[str]
    recent_entries: list[ProgressEntry]
    progress: GoalProgress


class ProgressResponse(BaseModel):
    goal: GoalView
    new_milestones: list[int]
    celebrations: list[str]


class Recommendation(BaseModel):
    goal_id: str
    pace: PaceStatus
    message: str
    daily_action: str | None
    weekly_action: str | None
    required_daily_amount: float
    required_weekly_amount: float
    recent_daily_rate: float
    days_left: int
    overdue: bool
    adjusted_timeline: SuggestedTimeline | None
    tip: str


class Reminder(BaseModel):
    goal_id: str
    kind: str
    message: str
    streak: int
    days_since_last_entry: int | None


class LearningSummary(BaseModel):
    user_id: str
    total_goals: int
    active_goals: int
    completed_goals: int
    average_percent_complete: float
    goals_behind: list[str]
    next_actions: list[str]


# ---------------------------------------------------------------------------
# Engine: pure functions, each takes an explicit ``today``
# ---------------------------------------------------------------------------


def check_date_in_range(value: date, name: str = "date") -> date:
    """Reject dates far enough out that later date arithmetic could overflow."""
    if not MIN_YEAR <= value.year <= MAX_YEAR:
        raise ValueError(f"{name} must be between the years {MIN_YEAR} and {MAX_YEAR}")
    return value


def _fmt(amount: float) -> str:
    text = f"{amount:.2f}" if abs(amount) < 10 else f"{amount:.1f}"
    return text.rstrip("0").rstrip(".") if "." in text else text


def effective_target_date(goal: Goal) -> date:
    if goal.target_date is not None:
        return goal.target_date
    return goal.start_date + timedelta(days=DEFAULT_HORIZON_DAYS[goal.horizon])


def amount_done(goal: Goal) -> float:
    return sum(entry.amount for entry in goal.entries)


def _exact_percent(goal: Goal) -> float:
    return min(100.0, amount_done(goal) / goal.target_amount * 100.0)


def is_complete(goal: Goal) -> bool:
    return amount_done(goal) >= goal.target_amount * (1.0 - _REL_EPS)


def percent_complete(goal: Goal) -> float:
    if is_complete(goal):
        return 100.0
    # Never display "100" for a goal that is not actually finished.
    return min(round(_exact_percent(goal), 1), 99.9)


def expected_percent(goal: Goal, today: date) -> float:
    """Where the learner should be today if progress were spread evenly over the goal."""
    total_days = max(1, (effective_target_date(goal) - goal.start_date).days)
    elapsed = min(max((today - goal.start_date).days, 0), total_days)
    return round(elapsed / total_days * 100.0, 1)


def compute_pace(goal: Goal, today: date) -> PaceStatus:
    if goal.status == GoalStatus.COMPLETED:
        return PaceStatus.COMPLETED
    if amount_done(goal) <= 0:
        return PaceStatus.NOT_STARTED
    expected = expected_percent(goal, today)
    if expected <= _EPSILON:
        return PaceStatus.AHEAD
    ratio = _exact_percent(goal) / expected
    if ratio >= AHEAD_RATIO:
        return PaceStatus.AHEAD
    if ratio < BEHIND_RATIO:
        return PaceStatus.BEHIND
    return PaceStatus.ON_TRACK


def compute_streaks(goal: Goal, today: date) -> tuple[int, int]:
    """Return (current_streak, longest_streak) in consecutive days with an entry.

    A streak is still "current" if the learner has not logged yet today but did
    yesterday, so it does not read as broken before the day is over.
    """
    days = sorted({entry.entry_date for entry in goal.entries})
    if not days:
        return 0, 0
    longest = run = 1
    for previous, current in zip(days, days[1:], strict=False):
        run = run + 1 if (current - previous).days == 1 else 1
        longest = max(longest, run)
    last = days[-1]
    if (today - last).days > 1:
        return 0, longest
    current_streak = 1
    for index in range(len(days) - 1, 0, -1):
        if (days[index] - days[index - 1]).days == 1:
            current_streak += 1
        else:
            break
    return current_streak, longest


def recent_daily_rate(goal: Goal, today: date) -> float:
    """Average amount per day over the recent window (shorter for a brand-new goal)."""
    window = max(1, min(PACE_WINDOW_DAYS, (today - goal.start_date).days + 1))
    cutoff = today - timedelta(days=window)
    total = sum(e.amount for e in goal.entries if cutoff < e.entry_date <= today)
    return round(total / window, 3)


def milestones_reached(percent: float) -> list[int]:
    return [m for m in MILESTONES if percent + _PERCENT_EPS >= m]


def progress_bar(percent: float, width: int = 20) -> str:
    filled = max(0, min(width, round(percent / 100.0 * width)))
    return f"[{'#' * filled}{'-' * (width - filled)}] {percent:.0f}%"


def suggest_timeline(
    target_amount: float, weekly_amount: float, start: date, *, apply_realism: bool = True
) -> SuggestedTimeline:
    """Estimate when ``target_amount`` is reachable at ``weekly_amount`` per week.

    A stated capacity is discounted by ``REALISM_FACTOR`` because people rarely sustain
    their full plan; a pace that was actually observed is used as-is.
    """
    effective = weekly_amount * (REALISM_FACTOR if apply_realism else 1.0)
    raw_days = target_amount / effective * 7.0 if effective > 0 else math.inf
    if not math.isfinite(raw_days) or raw_days > MAX_TIMELINE_DAYS:
        return SuggestedTimeline(
            realistic=False,
            days_needed=None,
            weeks_needed=None,
            completion_date=None,
            assumed_weekly_amount=round(effective, 3),
            note=(
                "At this weekly amount the goal would take over ten years. "
                "Consider a smaller target or a higher weekly commitment."
            ),
        )
    days = max(1, math.ceil(raw_days))
    return SuggestedTimeline(
        realistic=True,
        days_needed=days,
        weeks_needed=math.ceil(days / 7),
        completion_date=start + timedelta(days=days),
        assumed_weekly_amount=round(effective, 3),
        note=(
            f"Planning on about {_fmt(effective)} per week "
            f"({'allowing for missed days' if apply_realism else 'your recent pace'}), "
            f"you could finish around {(start + timedelta(days=days)).isoformat()}."
        ),
    )


def build_achievements(goal: Goal, today: date) -> list[str]:
    earned = [f"{m}% milestone reached" for m in goal.achieved_milestones]
    _, longest = compute_streaks(goal, today)
    if longest >= STREAK_ACHIEVEMENT_DAYS:
        earned.append(f"{STREAK_ACHIEVEMENT_DAYS}-day streak")
    if goal.status == GoalStatus.COMPLETED:
        earned.append("Goal completed")
    return earned


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


class GoalStore:
    """Thread-safe in-process goal store, bounded per user and overall."""

    def __init__(self) -> None:
        self._goals: dict[str, Goal] = {}
        self._lock = threading.RLock()

    def add(self, goal: Goal) -> None:
        with self._lock:
            if len(self._goals) >= MAX_TOTAL_GOALS:
                raise GoalConflictError("The service has reached its goal capacity; try again later.")
            owned = sum(1 for g in self._goals.values() if g.user_id == goal.user_id)
            if owned >= MAX_GOALS_PER_USER:
                raise GoalConflictError(f"A learner can have at most {MAX_GOALS_PER_USER} goals; delete one first.")
            self._goals[goal.goal_id] = goal

    def get(self, goal_id: str, user_id: str) -> Goal:
        """Return the goal, or raise KeyError. A goal owned by someone else reads as missing."""
        with self._lock:
            goal = self._goals.get(goal_id)
            if goal is None or goal.user_id != user_id:
                raise KeyError(goal_id)
            return goal

    def list_for_user(self, user_id: str) -> list[Goal]:
        with self._lock:
            return [g for g in self._goals.values() if g.user_id == user_id]

    def delete(self, goal_id: str, user_id: str) -> None:
        with self._lock:
            self.get(goal_id, user_id)
            del self._goals[goal_id]

    def clear(self) -> None:
        with self._lock:
            self._goals.clear()


goal_store = GoalStore()


def reset_goal_store() -> None:
    """Empty the shared store (used by tests)."""
    goal_store.clear()


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


def create_goal(request: GoalCreateRequest, today: date) -> Goal:
    start = check_date_in_range(request.start_date or today, "start_date")
    target = request.target_date
    if target is not None:
        check_date_in_range(target, "target_date")
        if target <= start:
            raise ValueError("target_date must be after start_date")
    elif request.weekly_capacity is not None:
        timeline = suggest_timeline(request.target_amount, request.weekly_capacity, start)
        if not timeline.realistic or timeline.completion_date is None:
            raise ValueError(timeline.note)
        target = check_date_in_range(timeline.completion_date, "target_date")
    goal = Goal(
        goal_id=uuid.uuid4().hex,
        user_id=request.user_id,
        title=request.title.strip(),
        goal_type=request.goal_type,
        horizon=request.horizon,
        target_amount=request.target_amount,
        unit=request.unit.strip(),
        start_date=start,
        target_date=target,
        weekly_capacity=request.weekly_capacity,
    )
    if not goal.title or not goal.unit:
        raise ValueError("title and unit must not be blank")
    goal_store.add(goal)
    return goal


def log_progress(goal: Goal, amount: float, entry_date: date | None, note: str | None, today: date) -> list[int]:
    """Record progress and return the milestones newly reached by it."""
    if goal.status == GoalStatus.COMPLETED:
        raise GoalConflictError("This goal is already complete.")
    when = check_date_in_range(entry_date or today, "entry_date")
    if when > today:
        raise ValueError("entry_date cannot be in the future")
    if when < goal.start_date:
        raise ValueError("entry_date cannot be before the goal's start_date")
    if len(goal.entries) >= MAX_ENTRIES_PER_GOAL:
        raise GoalConflictError("This goal has reached its progress-entry limit.")
    goal.entries.append(ProgressEntry(entry_date=when, amount=amount, note=note.strip() if note else None))

    complete = is_complete(goal)
    reached = [m for m in milestones_reached(_exact_percent(goal)) if m < 100 or complete]
    new = [m for m in reached if m not in goal.achieved_milestones]
    goal.achieved_milestones.extend(new)
    if complete:
        goal.status = GoalStatus.COMPLETED
        goal.completed_on = today
    return new


def build_view(goal: Goal, today: date) -> GoalView:
    target = effective_target_date(goal)
    pct = percent_complete(goal)
    pace = compute_pace(goal, today)
    streak, _ = compute_streaks(goal, today)
    progress = GoalProgress(
        amount_done=round(amount_done(goal), 3),
        remaining=round(max(0.0, goal.target_amount - amount_done(goal)), 3),
        percent_complete=pct,
        expected_percent=expected_percent(goal, today),
        pace=pace,
        pace_message=PACE_MESSAGES[pace],
        days_left=(target - today).days,
        current_streak=streak,
        recent_daily_rate=recent_daily_rate(goal, today),
        milestones_reached=list(goal.achieved_milestones),
        progress_bar=progress_bar(pct),
    )
    return GoalView(
        goal_id=goal.goal_id,
        user_id=goal.user_id,
        title=goal.title,
        goal_type=goal.goal_type,
        horizon=goal.horizon,
        target_amount=goal.target_amount,
        unit=goal.unit,
        start_date=goal.start_date,
        target_date=target,
        weekly_capacity=goal.weekly_capacity,
        status=goal.status,
        completed_on=goal.completed_on,
        entries_count=len(goal.entries),
        achievements=build_achievements(goal, today),
        recent_entries=goal.entries[-5:][::-1],
        progress=progress,
    )


def recommend(goal: Goal, today: date) -> Recommendation:
    """Daily/weekly actions that reflect where the learner actually is, not just the plan."""
    pace = compute_pace(goal, today)
    tip = TYPE_TIPS[goal.goal_type]
    rate = recent_daily_rate(goal, today)
    days_left = (effective_target_date(goal) - today).days
    if pace == PaceStatus.COMPLETED:
        return Recommendation(
            goal_id=goal.goal_id,
            pace=pace,
            message=PACE_MESSAGES[pace],
            daily_action=None,
            weekly_action=None,
            required_daily_amount=0.0,
            required_weekly_amount=0.0,
            recent_daily_rate=rate,
            days_left=days_left,
            overdue=False,
            adjusted_timeline=None,
            tip=tip,
        )

    remaining = max(0.0, goal.target_amount - amount_done(goal))
    overdue = days_left < 0
    message = PACE_MESSAGES[pace]
    if overdue:
        # The date has passed, so pace the remainder over a fresh window of the default length.
        window = DEFAULT_HORIZON_DAYS[goal.horizon]
        message = f"The target date has passed. Here is a fresh {window}-day plan for what is left."
    else:
        window = max(1, days_left)
    required_daily = remaining / window
    required_weekly = min(remaining, required_daily * 7)

    adjusted: SuggestedTimeline | None = None
    if overdue or pace in (PaceStatus.BEHIND, PaceStatus.NOT_STARTED):
        if rate > 0:
            adjusted = suggest_timeline(remaining, rate * 7, today, apply_realism=False)
        elif goal.weekly_capacity:
            adjusted = suggest_timeline(remaining, goal.weekly_capacity, today)

    return Recommendation(
        goal_id=goal.goal_id,
        pace=pace,
        message=message,
        daily_action=f"Aim for about {_fmt(required_daily)} {goal.unit} today.",
        weekly_action=f"Aim for about {_fmt(required_weekly)} {goal.unit} this week.",
        required_daily_amount=round(required_daily, 3),
        required_weekly_amount=round(required_weekly, 3),
        recent_daily_rate=rate,
        days_left=days_left,
        overdue=overdue,
        adjusted_timeline=adjusted,
        tip=tip,
    )


def build_reminder(goal: Goal, today: date) -> Reminder:
    """One motivational nudge per goal, chosen by what the learner most needs to hear."""
    streak, _ = compute_streaks(goal, today)
    last = max((e.entry_date for e in goal.entries), default=None)
    since = (today - last).days if last else None
    pace = compute_pace(goal, today)
    days_left = (effective_target_date(goal) - today).days
    title = goal.title

    if goal.status == GoalStatus.COMPLETED:
        kind, message = "completed", f"You completed '{title}'. Consider setting your next goal."
    elif 0 <= days_left <= 3 and pace in (PaceStatus.BEHIND, PaceStatus.NOT_STARTED):
        kind = "deadline"
        message = f"'{title}' is due in {days_left} day(s). A focused push now can still make a difference."
    elif last is None:
        kind = "get_started"
        message = f"You set '{title}' but have not logged anything yet. Start with one small session."
    elif since is not None and since >= 3:
        kind = "inactive"
        message = f"It has been {since} days since you worked on '{title}'. A short session today restarts the habit."
    elif streak >= 3:
        kind, message = "streak", f"{streak} days in a row on '{title}'. Keep the chain going."
    else:
        kind = "progress_update"
        message = f"'{title}' is {percent_complete(goal):.0f}% complete. {PACE_MESSAGES[pace]}"
    return Reminder(goal_id=goal.goal_id, kind=kind, message=message, streak=streak, days_since_last_entry=since)


def build_summary(user_id: str, goals: list[Goal], today: date) -> LearningSummary:
    active = [g for g in goals if g.status == GoalStatus.ACTIVE]
    percents = [percent_complete(g) for g in goals]
    behind = [g.goal_id for g in active if compute_pace(g, today) == PaceStatus.BEHIND]
    actions: list[str] = []
    for goal in active:
        action = recommend(goal, today).daily_action
        if action:
            actions.append(action)
    return LearningSummary(
        user_id=user_id,
        total_goals=len(goals),
        active_goals=len(active),
        completed_goals=len(goals) - len(active),
        average_percent_complete=round(sum(percents) / len(percents), 1) if percents else 0.0,
        goals_behind=behind,
        next_actions=actions,
    )


# ---------------------------------------------------------------------------
# HTTP routes
# ---------------------------------------------------------------------------


def _today(as_of: date | None) -> date:
    if as_of is None:
        return date.today()
    try:
        return check_date_in_range(as_of, "as_of")
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def _find(goal_id: str, user_id: str) -> Goal:
    try:
        return goal_store.get(goal_id, user_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Goal not found") from None


@router.post("", response_model=GoalView, status_code=201)
async def create_goal_endpoint(body: GoalCreateRequest, as_of: date | None = None) -> GoalView:
    """Define a short-term or long-term learning goal."""
    today = _today(as_of)
    try:
        goal = create_goal(body, today)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except GoalConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return build_view(goal, today)


@router.get("/types")
async def list_goal_types() -> dict[str, list[str]]:
    """The supported goal types, horizons and milestone thresholds."""
    return {
        "goal_types": [t.value for t in GoalType],
        "horizons": [h.value for h in GoalHorizon],
        "milestones": [str(m) for m in MILESTONES],
    }


@router.post("/timeline", response_model=SuggestedTimeline)
async def suggest_timeline_endpoint(body: TimelineRequest) -> SuggestedTimeline:
    """Suggest a realistic completion date for a target at a given weekly capacity."""
    try:
        start = check_date_in_range(body.start_date or date.today(), "start_date")
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return suggest_timeline(body.target_amount, body.weekly_capacity, start)


@router.get("/user/{user_id}", response_model=list[GoalView])
async def list_goals(user_id: str, as_of: date | None = None) -> list[GoalView]:
    """All goals a learner has defined."""
    today = _today(as_of)
    return [build_view(g, today) for g in goal_store.list_for_user(user_id)]


@router.get("/user/{user_id}/summary", response_model=LearningSummary)
async def get_summary(user_id: str, as_of: date | None = None) -> LearningSummary:
    """Overall progress across a learner's goals, with today's suggested actions."""
    return build_summary(user_id, goal_store.list_for_user(user_id), _today(as_of))


@router.get("/user/{user_id}/reminders", response_model=list[Reminder])
async def get_reminders(user_id: str, as_of: date | None = None) -> list[Reminder]:
    """Motivational reminders and progress updates for a learner's active goals."""
    today = _today(as_of)
    return [build_reminder(g, today) for g in goal_store.list_for_user(user_id) if g.status == GoalStatus.ACTIVE]


@router.get("/{goal_id}", response_model=GoalView)
async def get_goal(goal_id: str, user_id: str, as_of: date | None = None) -> GoalView:
    """One goal with its progress, pace and achievements."""
    return build_view(_find(goal_id, user_id), _today(as_of))


@router.post("/{goal_id}/progress", response_model=ProgressResponse)
async def record_progress(goal_id: str, body: ProgressRequest, as_of: date | None = None) -> ProgressResponse:
    """Log progress toward a goal; reports any milestones it newly unlocks."""
    today = _today(as_of)
    goal = _find(goal_id, body.user_id)
    try:
        new = log_progress(goal, body.amount, body.entry_date, body.note, today)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except GoalConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return ProgressResponse(
        goal=build_view(goal, today),
        new_milestones=new,
        celebrations=[MILESTONE_MESSAGES[m] for m in new],
    )


@router.get("/{goal_id}/recommendations", response_model=Recommendation)
async def get_recommendations(goal_id: str, user_id: str, as_of: date | None = None) -> Recommendation:
    """Daily and weekly actions, adjusted to the learner's actual pace."""
    return recommend(_find(goal_id, user_id), _today(as_of))


# A 204 response must not carry a body, so no response model is declared explicitly. Letting FastAPI
# infer one from the return annotation made it reject this route at import time (see CI on #743).
@router.delete("/{goal_id}", status_code=204, response_model=None)
async def delete_goal(goal_id: str, user_id: str) -> None:
    """Delete a goal and its history."""
    _find(goal_id, user_id)
    goal_store.delete(goal_id, user_id)
