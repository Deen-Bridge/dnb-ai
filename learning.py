"""Personalized learning-path recommendation — turns a learner's profile,
progress, and a caller-supplied course catalog into an ordered, justified
study path using Gemini's JSON mode.

This service is **stateless about the catalog**. Deen Bridge's course/book data
lives in ``dnb-backend`` (see ``README.md`` service table); this AI service only
ever sees purchase *metadata* (``stellar.py``) and has no local catalog or DB.
The candidate courses are therefore **supplied in the request body** by the
caller (``dnb-backend``), and that catalog is the single source of truth for
what may be recommended. The endpoint must never invent course ids.

The structured-generation machinery (schema translation, generator seam, and
the bounded retry-to-502 loop) is reused from ``study.py`` rather than
duplicated. On top of it this module adds **deterministic grounding
guardrails**: after the model responds, any step whose ``course_id`` is not in
the submitted catalog, any already-completed course, and any prerequisite
violation are dropped in code — even if the prompt asked the model not to. The
endpoint is structurally unable to emit a path that recommends a non-catalog id
or a completed course.
"""

import json
import logging
import os
from enum import Enum

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, ValidationInfo, field_validator, model_validator

from study import (
    BaseGenerator,
    FakeGenerator,
    GeminiGenerator,
    pydantic_to_gemini_schema,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["learning"])

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_CATALOG_SIZE = 200
MAX_PROGRESS_RECORDS = 200
MAX_NOTES_LENGTH = 1_000
MAX_GOALS = 10
# A course counts as "already completed" (and is never re-recommended) once the
# learner has finished at least this share of it.
COMPLETION_THRESHOLD = 90.0
DEFAULT_RETRIES = 2

LEARNING_CONTEXT = """You are an AI academic advisor for Deen Bridge, an Islamic
e-learning platform. You design a personalized, ordered study path for a learner
from a fixed catalog of real courses.

Hard rules:
1. Recommend ONLY courses whose course_id appears in the provided catalog. Never
   invent, guess, or rename a course_id.
2. Never recommend a course the learner has already completed.
3. Never recommend a course before its prerequisites are satisfied — a
   prerequisite is satisfied only if the learner already completed it or an
   earlier step in this same path covers it.
4. Order steps so each builds on the previous; number them contiguously from 1.
5. Ground every reason in *this* learner's stated goals, level, and progress —
   not generic advice.
6. Maintain Islamic etiquette (adab) and scholarly humility: a study path is a
   suggestion, not a fatwa, and a qualified teacher should guide the learner.
"""


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class LearnerLevel(str, Enum):
    beginner = "beginner"
    intermediate = "intermediate"
    advanced = "advanced"


class LearningGoal(str, Enum):
    quran_reading = "quran_reading"
    tajweed = "tajweed"
    memorization = "memorization"
    arabic = "arabic"
    fiqh_basics = "fiqh_basics"
    aqeedah = "aqeedah"
    seerah = "seerah"
    hadith = "hadith"
    tafsir = "tafsir"


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class LearnerProfile(BaseModel):
    level: LearnerLevel = Field(..., description="Learner's current overall level")
    goals: list[LearningGoal] = Field(
        ...,
        min_length=1,
        max_length=MAX_GOALS,
        description="What the learner wants to achieve (at least one goal)",
    )
    time_per_week_hours: float | None = Field(
        default=None,
        gt=0,
        le=168,
        description="Optional weekly study budget in hours",
    )
    notes: str | None = Field(
        default=None,
        max_length=MAX_NOTES_LENGTH,
        description="Optional free-text notes about the learner (length-capped)",
    )


class ProgressRecord(BaseModel):
    course_id: str = Field(..., min_length=1, description="Catalog id of a course the learner engaged with")
    title: str = Field(..., min_length=1, description="Human-readable course title")
    category: str = Field(..., min_length=1, description="Course category, e.g. 'quran' or 'fiqh'")
    level: LearnerLevel = Field(..., description="Level of the course")
    completion_pct: float = Field(
        ...,
        ge=0,
        le=100,
        description="Percentage of the course the learner has completed (0-100)",
    )
    quiz_scores: list[float] | None = Field(
        default=None,
        description="Optional quiz scores (0-100) the learner earned in this course",
    )

    @field_validator("quiz_scores")
    @classmethod
    def scores_in_range(cls, v: list[float] | None) -> list[float] | None:
        if v is not None and any(not (0 <= s <= 100) for s in v):
            raise ValueError("quiz_scores must each be between 0 and 100")
        return v


class CourseCatalogItem(BaseModel):
    course_id: str = Field(..., min_length=1, description="Unique catalog id — the only id that may be recommended")
    title: str = Field(..., min_length=1, description="Course title")
    category: str = Field(..., min_length=1, description="Course category")
    level: LearnerLevel = Field(..., description="Difficulty level of the course")
    prerequisites: list[str] = Field(
        default_factory=list,
        description="course_ids that must be completed before this course",
    )
    description: str = Field(..., min_length=1, description="Short description of the course")


class LearningPathRequest(BaseModel):
    profile: LearnerProfile = Field(..., description="The learner's profile")
    progress: list[ProgressRecord] = Field(
        default_factory=list,
        max_length=MAX_PROGRESS_RECORDS,
        description="What the learner has studied so far",
    )
    catalog: list[CourseCatalogItem] = Field(
        ...,
        min_length=1,
        max_length=MAX_CATALOG_SIZE,
        description="Candidate courses supplied by the caller — the single source of truth for recommendations",
    )

    @model_validator(mode="after")
    def catalog_ids_unique(self) -> "LearningPathRequest":
        ids = [c.course_id for c in self.catalog]
        if len(set(ids)) != len(ids):
            raise ValueError("catalog course_ids must be unique")
        return self

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "profile": {
                        "level": "beginner",
                        "goals": ["quran_reading", "tajweed"],
                        "time_per_week_hours": 5,
                        "notes": "Reverted last year; can read Arabic letters slowly.",
                    },
                    "progress": [
                        {
                            "course_id": "arabic-101",
                            "title": "Arabic Alphabet",
                            "category": "arabic",
                            "level": "beginner",
                            "completion_pct": 100,
                        }
                    ],
                    "catalog": [
                        {
                            "course_id": "quran-101",
                            "title": "Quran Reading Basics",
                            "category": "quran",
                            "level": "beginner",
                            "prerequisites": ["arabic-101"],
                            "description": "Read the Quran from the mushaf with correct letters.",
                        },
                        {
                            "course_id": "tajweed-201",
                            "title": "Introduction to Tajweed",
                            "category": "quran",
                            "level": "intermediate",
                            "prerequisites": ["quran-101"],
                            "description": "Rules of recitation: makharij and basic ahkam.",
                        },
                    ],
                }
            ]
        }
    }


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class LearningStep(BaseModel):
    course_id: str = Field(..., min_length=1, description="Catalog id of the recommended course")
    # Optional on input: the authoritative title is filled from the catalog by
    # the grounding guardrails, so the model is never trusted to supply it.
    title: str = Field(default="", description="Course title (set from the catalog)")
    order: int = Field(..., ge=1, description="1-based position of this step in the path")
    reason: str = Field(..., min_length=1, description="Why this course is next for this learner")
    prerequisites_satisfied: bool = Field(
        ...,
        description="True when every prerequisite is completed or covered earlier in the path",
    )
    estimated_weeks: int = Field(..., ge=1, le=104, description="Rough time to complete at the learner's pace")

    @field_validator("reason")
    @classmethod
    def reason_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("reason must not be blank")
        return v


class LearningPath(BaseModel):
    steps: list[LearningStep] = Field(..., min_length=1, description="Ordered study path")
    summary: str = Field(..., min_length=1, description="Path-level summary for the learner")
    scholarly_note: str = Field(
        ...,
        min_length=1,
        description="Scholarly-humility note — a path is guidance, not a fatwa",
    )

    @model_validator(mode="after")
    def validate_structure(self, info: ValidationInfo) -> "LearningPath":
        course_ids = [s.course_id for s in self.steps]
        if len(set(course_ids)) != len(course_ids):
            raise ValueError("steps must not contain duplicate course_id")

        orders = sorted(s.order for s in self.steps)
        if orders != list(range(1, len(self.steps) + 1)):
            raise ValueError("step order must be contiguous starting from 1")

        # Grounding guarantee: when the caller passes the catalog ids via
        # validation context, a LearningPath simply cannot be constructed with a
        # course_id that is not in the catalog — belt-and-suspenders on top of
        # the deterministic guardrails.
        context = info.context or {}
        catalog_ids = context.get("catalog_ids")
        if catalog_ids is not None:
            unknown = [cid for cid in course_ids if cid not in catalog_ids]
            if unknown:
                raise ValueError(f"steps reference course_ids not in catalog: {sorted(unknown)}")
        return self

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "steps": [
                        {
                            "course_id": "quran-101",
                            "title": "Quran Reading Basics",
                            "order": 1,
                            "reason": "You finished the Arabic alphabet, so reading from the mushaf is the natural next step toward your quran_reading goal.",
                            "prerequisites_satisfied": True,
                            "estimated_weeks": 6,
                        },
                        {
                            "course_id": "tajweed-201",
                            "title": "Introduction to Tajweed",
                            "order": 2,
                            "reason": "Once you can read fluently, tajweed refines your recitation — directly serving your tajweed goal.",
                            "prerequisites_satisfied": True,
                            "estimated_weeks": 8,
                        },
                    ],
                    "summary": "Build fluent Quran reading first, then refine recitation with tajweed.",
                    "scholarly_note": "This path is a study suggestion, not a religious ruling; please learn recitation under a qualified teacher.",
                }
            ]
        }
    }


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------


def _format_catalog(catalog: list[CourseCatalogItem]) -> str:
    lines = []
    for c in catalog:
        prereqs = ", ".join(c.prerequisites) if c.prerequisites else "none"
        lines.append(
            f"- course_id={c.course_id} | title={c.title} | category={c.category} "
            f"| level={c.level.value} | prerequisites=[{prereqs}] | {c.description}"
        )
    return "\n".join(lines)


def _format_progress(progress: list[ProgressRecord]) -> str:
    if not progress:
        return "(none — this is a new learner)"
    lines = []
    for p in progress:
        scores = ""
        if p.quiz_scores:
            scores = f" | quiz_scores={p.quiz_scores}"
        lines.append(
            f"- course_id={p.course_id} | title={p.title} | category={p.category} "
            f"| level={p.level.value} | completion_pct={p.completion_pct}{scores}"
        )
    return "\n".join(lines)


def _build_prompt(request: LearningPathRequest) -> str:
    """Build the recommendation prompt from the learner and catalog."""
    profile = request.profile
    goals = ", ".join(g.value for g in profile.goals)
    time_line = (
        f"{profile.time_per_week_hours} hours/week" if profile.time_per_week_hours is not None else "not specified"
    )
    notes_line = profile.notes.strip() if profile.notes else "(none)"

    lines = [
        LEARNING_CONTEXT,
        "",
        "Learner profile:",
        f"- level: {profile.level.value}",
        f"- goals: {goals}",
        f"- weekly study time: {time_line}",
        f"- notes: {notes_line}",
        "",
        "Learner progress so far:",
        _format_progress(request.progress),
        "",
        f"Course catalog (recommend ONLY from these {len(request.catalog)} course_ids):",
        _format_catalog(request.catalog),
        "",
        "Instructions:",
        "- Produce an ordered learning path drawn only from the catalog above.",
        f"- Skip any course already completed (completion_pct >= {COMPLETION_THRESHOLD}).",
        "- Respect prerequisites: never place a course before a prerequisite it depends on.",
        "- Number steps contiguously starting at 1, with no duplicate course_id.",
        "- For each step give a concrete reason tied to the learner's goals and progress.",
        "- Set prerequisites_satisfied to true only when it genuinely holds.",
        "- Include a path-level summary and a short scholarly-humility note.",
        "",
        'Output ONLY valid JSON with keys "steps" (array), "summary" (string), and '
        '"scholarly_note" (string). No markdown, no code fences, no extra text.',
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Parsing and grounding guardrails
# ---------------------------------------------------------------------------


def _parse_steps(response_text: str) -> tuple[list[LearningStep], str, str]:
    """Parse and per-field validate a raw model response.

    Returns ``(steps, summary, scholarly_note)``. Cross-step structural checks
    and catalog grounding are applied later. Raises ``ValueError`` on any
    malformed or schema-invalid output.
    """
    try:
        data = json.loads(response_text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e}") from e

    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object, got {type(data).__name__}")

    raw_steps = data.get("steps")
    if not isinstance(raw_steps, list) or not raw_steps:
        raise ValueError("'steps' must be a non-empty array")

    steps: list[LearningStep] = []
    errors: list[str] = []
    for i, item in enumerate(raw_steps):
        try:
            steps.append(LearningStep.model_validate(item))
        except Exception as exc:
            errors.append(f"step[{i}]: {exc}")

    summary = data.get("summary")
    if not isinstance(summary, str) or not summary.strip():
        errors.append("'summary' must be a non-empty string")

    scholarly_note = data.get("scholarly_note")
    if not isinstance(scholarly_note, str) or not scholarly_note.strip():
        errors.append("'scholarly_note' must be a non-empty string")

    if errors:
        raise ValueError("; ".join(errors))

    assert isinstance(summary, str) and isinstance(scholarly_note, str)  # narrowed by the checks above
    return steps, summary, scholarly_note


def _apply_guardrails(
    steps: list[LearningStep],
    request: LearningPathRequest,
) -> list[LearningStep]:
    """Deterministically drop any step that is not grounded in the catalog.

    Removes steps whose ``course_id`` is not in the catalog, courses already
    completed by the learner, prerequisite violations, and duplicates — then
    renumbers the survivors' ``order`` contiguously from 1. Runs regardless of
    what the model returned, so the model cannot smuggle an ungrounded course
    into the path.
    """
    catalog = {c.course_id: c for c in request.catalog}
    completed = {p.course_id for p in request.progress if p.completion_pct >= COMPLETION_THRESHOLD}

    survivors: list[LearningStep] = []
    satisfied: set[str] = set(completed)
    seen: set[str] = set()

    for step in sorted(steps, key=lambda s: s.order):
        cid = step.course_id
        item = catalog.get(cid)
        if item is None:
            logger.warning("Dropping step %s: course_id not in catalog", cid)
            continue
        if cid in seen:
            logger.warning("Dropping step %s: duplicate course_id", cid)
            continue
        if cid in completed:
            logger.warning("Dropping step %s: course already completed", cid)
            continue
        if any(pr not in satisfied for pr in item.prerequisites):
            logger.warning("Dropping step %s: prerequisites not satisfied", cid)
            continue

        seen.add(cid)
        satisfied.add(cid)
        survivors.append(step)

    renumbered: list[LearningStep] = []
    for idx, step in enumerate(survivors, start=1):
        renumbered.append(
            step.model_copy(
                update={
                    "order": idx,
                    "title": catalog[step.course_id].title,
                    "prerequisites_satisfied": True,
                }
            )
        )
    return renumbered


# ---------------------------------------------------------------------------
# Core generation loop (with bounded retry)
# ---------------------------------------------------------------------------


def _generate_learning_path(
    generator: BaseGenerator,
    request: LearningPathRequest,
    max_retries: int = DEFAULT_RETRIES,
) -> LearningPath:
    """Run the generation loop, applying grounding guardrails to each attempt.

    Feeds validation and guardrail violations back to the model on retry.
    Raises ``HTTPException`` 502 with a structured error when no grounded path
    survives after ``max_retries`` retries.
    """
    base_prompt = _build_prompt(request)
    output_schema = pydantic_to_gemini_schema(LearningPath)
    catalog_ids = {c.course_id for c in request.catalog}

    last_error: str | None = None

    for attempt in range(max_retries + 1):
        prompt = base_prompt
        if last_error is not None:
            prompt = (
                f"{base_prompt}\n\n"
                f"Your previous output was rejected:\n{last_error}\n"
                "Recommend only catalog course_ids, skip completed courses, respect "
                "prerequisites, and try again."
            )

        try:
            response_text = generator.generate(prompt, output_schema)
            steps, summary, scholarly_note = _parse_steps(response_text)
            grounded = _apply_guardrails(steps, request)
            if not grounded:
                raise ValueError("no catalog-grounded steps survived after guardrails")

            path = LearningPath.model_validate(
                {
                    "steps": [s.model_dump() for s in grounded],
                    "summary": summary,
                    "scholarly_note": scholarly_note,
                },
                context={"catalog_ids": catalog_ids},
            )
            logger.info(
                "Learning path generated on attempt %d/%d (%d steps)",
                attempt + 1,
                max_retries + 1,
                len(path.steps),
            )
            return path
        except (json.JSONDecodeError, ValueError) as exc:
            last_error = str(exc)
            logger.warning(
                "Learning-path attempt %d/%d failed: %s",
                attempt + 1,
                max_retries + 1,
                last_error,
            )

    logger.error("All %d learning-path attempts exhausted", max_retries + 1)
    raise HTTPException(
        status_code=502,
        detail={
            "error": "Learning-path generation failed after multiple attempts",
            "last_validation_error": last_error,
            "retries_used": max_retries + 1,
        },
    )


# ---------------------------------------------------------------------------
# Generator seam (dependency-injected so tests can supply a FakeGenerator)
# ---------------------------------------------------------------------------


def get_generator() -> BaseGenerator:
    """Return the generator for a request.

    Uses a :class:`FakeGenerator` when ``USE_FAKE_GENERATOR=1`` (offline running)
    and the real :class:`GeminiGenerator` otherwise. Tests override this
    dependency to inject recorded responses.
    """
    if os.getenv("USE_FAKE_GENERATOR", "0") == "1":
        logger.info("Using FakeGenerator (USE_FAKE_GENERATOR=1)")
        return FakeGenerator()
    return GeminiGenerator()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.post("/learning-path", response_model=LearningPath)
async def recommend_learning_path(
    request: LearningPathRequest,
    generator: BaseGenerator = Depends(get_generator),
) -> LearningPath:
    """Recommend an ordered, justified learning path for a learner.

    **The catalog is passed in by the caller.** Deen Bridge's course data lives
    in ``dnb-backend``; this AI service is stateless about it and holds no local
    catalog or database. Every recommendation is drawn exclusively from the
    ``catalog`` in the request body — the endpoint never invents course ids and
    never recommends a course absent from that list.

    Given a ``profile`` (level, goals, optional weekly hours and notes),
    ``progress`` (courses already studied), and a non-empty ``catalog`` of
    candidate courses, returns a :class:`LearningPath` of ordered steps, each a
    concrete next course with a reason grounded in the learner's goals and
    progress, plus a path-level summary and a scholarly-humility note.

    Grounding is enforced in code, not just the prompt: after the model
    responds, any step referencing a non-catalog course, an already-completed
    course, or an unsatisfied prerequisite is dropped and the path renumbered.
    Empty catalog or oversized inputs return ``422`` before any model call; a
    model that cannot produce a grounded path after retries returns ``502``.
    """
    return _generate_learning_path(generator, request)
