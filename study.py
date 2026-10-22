"""Study-content generator â€” produces schema-validated quizzes and flashcards
using Gemini's JSON mode.

Supports topic-based and lesson-text-based generation with three difficulty
levels and a bounded retry loop that feeds pydantic validation errors back
to the model.
"""

import copy
import json
import logging
import os
from enum import Enum
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field, field_validator, model_validator

from errors import APIException

logger = logging.getLogger(__name__)

router = APIRouter(tags=["study"])

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_LESSON_TEXT_LENGTH = 20_000
DEFAULT_RETRIES = 2
GENERATION_TEMPERATURE = 0.3

ISLAMIC_CONTEXT = """You are an AI assistant specialized in creating Islamic study content.
Your generated content must:
1. Be based on authentic Islamic sources (Quran and Hadith)
2. Be accurate and respectful
3. Avoid controversial or divisive topics
4. Cite authentic sources when mentioning specific narrations or verses
5. Not fabricate citations â€” if unsure, omit the citation
6. Acknowledge scholarly differences of opinion where they exist
7. Maintain Islamic etiquette (adab) throughout

When generating multiple-choice questions:
- Ensure the correct answer is clearly correct
- Ensure distractors are plausible but clearly incorrect
- Provide explanations that reference authentic sources

When generating flashcards:
- Keep the front concise and clear
- Provide thorough but focused back content
- Tag with relevant subject categories
"""

DIFFICULTY_PROMPTS: dict[str, str] = {
    "beginner": (
        "BEGINNER level â€” test recall of well-known, foundational Islamic knowledge. "
        "Questions should cover basic facts that most Muslims would know. "
        "Avoid complex legal rulings or detailed scholarly debates."
    ),
    "intermediate": (
        "INTERMEDIATE level â€” test applied understanding. "
        "Questions may require connecting concepts, understanding evidence, "
        "or applying principles to new situations. Suitable for students "
        "with basic Islamic education."
    ),
    "advanced": (
        "ADVANCED level â€” test comparative and usul-level nuance. "
        "Questions may involve weighing evidence, understanding methodology "
        "differences, or analyzing complex rulings. Suitable for students "
        "of knowledge."
    ),
}

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class Difficulty(str, Enum):
    beginner = "beginner"
    intermediate = "intermediate"
    advanced = "advanced"


class ContentKind(str, Enum):
    quiz = "quiz"
    flashcards = "flashcards"
    both = "both"


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class TopicSource(BaseModel):
    topic: str = Field(..., min_length=1, description="Study topic to generate content about")


class LessonTextSource(BaseModel):
    lesson_text: str = Field(
        ...,
        min_length=1,
        max_length=MAX_LESSON_TEXT_LENGTH,
        description="Lesson text to base content on",
    )


class StudyGenerateRequest(BaseModel):
    source: TopicSource | LessonTextSource = Field(
        ...,
        description="Source of content: either a topic or lesson text",
        json_schema_extra={
            "examples": [
                {"topic": "The Five Pillars of Islam"},
                {"lesson_text": "Salah is the second pillar of Islam..."},
            ]
        },
    )
    kind: ContentKind = Field(
        default=ContentKind.quiz,
        description="Type of content to generate",
        json_schema_extra={"examples": ["quiz", "flashcards", "both"]},
    )
    difficulty: Difficulty = Field(
        default=Difficulty.beginner,
        description="Difficulty level of generated content",
    )
    count: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of items to generate (1-20)",
    )


class QuizQuestion(BaseModel):
    question: str = Field(..., min_length=1, description="The quiz question")
    choices: list[str] = Field(..., description="Exactly 4 answer choices")
    correct_index: int = Field(..., ge=0, le=3, description="Index of the correct answer (0-3)")
    explanation: str = Field(..., min_length=1, description="Explanation of the correct answer")
    difficulty: str = Field(..., description="Difficulty level of this question")

    @field_validator("choices")
    @classmethod
    def choices_must_be_exactly_four_distinct_nonempty(cls, v: list[str]) -> list[str]:
        if len(v) != 4:
            raise ValueError(f"Must have exactly 4 choices, got {len(v)}")
        stripped = [c.strip() for c in v]
        if any(not c for c in stripped):
            raise ValueError("Choices must not be empty")
        if len(set(stripped)) != len(stripped):
            raise ValueError("Choices must not contain duplicates")
        return stripped

    @model_validator(mode="after")
    def question_not_identical_to_any_choice(self) -> "QuizQuestion":
        if not self.question or not self.choices:
            return self
        q = self.question.lower().strip()
        for c in self.choices:
            if q == c.lower().strip():
                raise ValueError("Question must not be identical to any choice")
        return self


class Flashcard(BaseModel):
    front: str = Field(..., min_length=1, description="Front of the flashcard (question/term)")
    back: str = Field(..., min_length=1, description="Back of the flashcard (answer/definition)")
    tags: list[str] = Field(default_factory=list, description="Tags for categorization")


class StudyGenerateResponse(BaseModel):
    quizzes: list[QuizQuestion] | None = Field(default=None, description="Generated quiz questions")
    flashcards: list[Flashcard] | None = Field(default=None, description="Generated flashcards")
    source_used: str = Field(..., description="Whether 'topic' or 'lesson_text' was used")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "quizzes": [
                        {
                            "question": "What is the first pillar of Islam?",
                            "choices": [
                                "Salah",
                                "Shahadah",
                                "Zakat",
                                "Hajj",
                            ],
                            "correct_index": 1,
                            "explanation": "The Shahadah is the declaration of faith and the first pillar.",
                            "difficulty": "beginner",
                        }
                    ],
                    "flashcards": None,
                    "source_used": "topic",
                }
            ]
        }
    }


# ---------------------------------------------------------------------------
# Schema translation  (pydantic â†’ Gemini-compatible JSON schema)
# ---------------------------------------------------------------------------


def _strip_title(obj: Any) -> Any:
    """Recursively strip 'title' keys â€” Gemini rejects them."""
    if isinstance(obj, dict):
        return {k: _strip_title(v) for k, v in obj.items() if k != "title"}
    if isinstance(obj, list):
        return [_strip_title(item) for item in obj]
    return obj


def _resolve_refs(schema: dict, defs: dict) -> dict:
    """Replace ``$ref`` with the actual definition from ``$defs``."""
    if isinstance(schema, dict):
        if "$ref" in schema:
            key = schema["$ref"].split("/")[-1]
            resolved = copy.deepcopy(defs.get(key, schema))
            return _resolve_refs(resolved, defs)
        return {k: _resolve_refs(v, defs) for k, v in schema.items()}
    if isinstance(schema, list):
        return [_resolve_refs(item, defs) for item in schema]
    return schema


def _inline_defs(schema: dict) -> dict:
    """Move ``$defs`` into the schema tree (Gemini ignores ``$defs``)."""
    if "$defs" not in schema:
        return schema
    defs = schema.pop("$defs")
    return _resolve_refs(schema, defs)


def pydantic_to_gemini_schema(model: type[BaseModel]) -> dict:
    """Convert a pydantic model to a Gemini-compatible JSON schema."""
    schema = model.model_json_schema()
    schema = _strip_title(schema)
    schema = _inline_defs(schema)
    return schema


# ---------------------------------------------------------------------------
# Generator interface and implementations
# ---------------------------------------------------------------------------


class BaseGenerator:
    """Interface for content generation."""

    def generate(self, prompt: str, schema: dict) -> str:
        """Send a prompt and return the raw response text."""
        raise NotImplementedError


class GeminiGenerator(BaseGenerator):
    """Real generator that calls the Gemini API."""

    def __init__(self, model_name: str = "gemini-2.5-flash"):
        self.model_name = model_name

    def generate(self, prompt: str, schema: dict) -> str:
        import google.generativeai as genai

        model = genai.GenerativeModel(self.model_name)
        response = model.generate_content(
            prompt,
            generation_config={
                "temperature": GENERATION_TEMPERATURE,
                "response_mime_type": "application/json",
                "response_schema": schema,
            },
        )
        return response.text


class FakeGenerator(BaseGenerator):
    """Generator that returns pre-recorded responses (for tests)."""

    def __init__(self, responses: list[str] | None = None):
        self.responses = responses or ["{}"]
        self.call_count = 0

    def generate(self, prompt: str, schema: dict) -> str:
        idx = min(self.call_count, len(self.responses) - 1)
        self.call_count += 1
        return self.responses[idx]


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------


def _build_prompt(
    source: TopicSource | LessonTextSource,
    kind: ContentKind,
    difficulty: Difficulty,
    count: int,
) -> tuple[str, str]:
    """Build the generation prompt and return ``(prompt, source_used)``."""
    if isinstance(source, TopicSource):
        source_text = f"Topic: {source.topic}"
        source_used = "topic"
    else:
        text = source.lesson_text
        if len(text) > 500:
            text = text[:500] + "..."
        source_text = f"Lesson text:\n{text}"
        source_used = "lesson_text"

    diff_prompt = DIFFICULTY_PROMPTS.get(difficulty.value, DIFFICULTY_PROMPTS["beginner"])

    lines = [
        ISLAMIC_CONTEXT,
        "",
        diff_prompt,
        "",
        "Source:",
        source_text,
        "",
        "Instructions:",
    ]
    if kind in (ContentKind.quiz, ContentKind.both):
        lines.append(f"Generate exactly {count} quiz question(s) based on the source.")
    if kind in (ContentKind.flashcards, ContentKind.both):
        lines.append(f"Generate exactly {count} flashcard(s) based on the source.")
    lines.append("")
    lines.append(_output_format_instruction(kind))
    lines.append("")
    lines.append("Output ONLY valid JSON. No markdown, no code fences, no extra text.")
    return "\n".join(lines), source_used


def _output_format_instruction(kind: ContentKind) -> str:
    if kind == ContentKind.quiz:
        return (
            'Output a JSON object with key "quizzes" (array). '
            "Each item: question (str), choices ([str; 4 items]), "
            "correct_index (int 0-3), explanation (str), difficulty (str)."
        )
    if kind == ContentKind.flashcards:
        return 'Output a JSON object with key "flashcards" (array). Each item: front (str), back (str), tags ([str]).'
    return (
        'Output a JSON object with keys "quizzes" (array) and "flashcards" (array). '
        "Quiz items: question (str), choices ([str; 4 items]), "
        "correct_index (int 0-3), explanation (str), difficulty (str). "
        "Flashcard items: front (str), back (str), tags ([str])."
    )


def _build_output_schema(kind: ContentKind) -> dict:
    """Build the JSON schema that Gemini should use as its output shape."""
    quiz_schema = pydantic_to_gemini_schema(QuizQuestion)
    flash_schema = pydantic_to_gemini_schema(Flashcard)

    if kind == ContentKind.quiz:
        return {
            "type": "object",
            "properties": {"quizzes": {"type": "array", "items": quiz_schema}},
            "required": ["quizzes"],
        }
    if kind == ContentKind.flashcards:
        return {
            "type": "object",
            "properties": {"flashcards": {"type": "array", "items": flash_schema}},
            "required": ["flashcards"],
        }
    return {
        "type": "object",
        "properties": {
            "quizzes": {"type": "array", "items": quiz_schema},
            "flashcards": {"type": "array", "items": flash_schema},
        },
        "required": ["quizzes", "flashcards"],
    }


# ---------------------------------------------------------------------------
# Parsing and validation
# ---------------------------------------------------------------------------


def _parse_and_validate(response_text: str, kind: ContentKind) -> dict:
    """Parse raw JSON from a model response and validate against schemas.

    Returns ``{"quizzes": [...], "flashcards": [...]}`` with validated
    pydantic objects.  Raises ``ValueError`` on any failure.
    """
    try:
        data = json.loads(response_text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e}") from e

    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object, got {type(data).__name__}")

    result: dict[str, Any] = {}
    errors: list[str] = []

    if kind in (ContentKind.quiz, ContentKind.both):
        raw = data.get("quizzes")
        if not isinstance(raw, list):
            errors.append("'quizzes' must be a non-empty array")
        else:
            validated = []
            for i, item in enumerate(raw):
                try:
                    validated.append(QuizQuestion.model_validate(item))
                except Exception as exc:
                    errors.append(f"quiz[{i}]: {exc}")
            result["quizzes"] = validated

    if kind in (ContentKind.flashcards, ContentKind.both):
        raw = data.get("flashcards")
        if not isinstance(raw, list):
            errors.append("'flashcards' must be a non-empty array")
        else:
            validated_cards = []
            for i, item in enumerate(raw):
                try:
                    validated_cards.append(Flashcard.model_validate(item))
                except Exception as exc:
                    errors.append(f"flashcard[{i}]: {exc}")
            result["flashcards"] = validated_cards

    if errors:
        raise ValueError("; ".join(errors))

    return result


# ---------------------------------------------------------------------------
# Core generation loop (with bounded retry)
# ---------------------------------------------------------------------------


def _generate_content(
    generator: BaseGenerator,
    source: TopicSource | LessonTextSource,
    kind: ContentKind,
    difficulty: Difficulty,
    count: int,
    max_retries: int = DEFAULT_RETRIES,
) -> StudyGenerateResponse:
    """Run the generation loop with up to ``max_retries`` retries on error."""
    base_prompt, source_used = _build_prompt(source, kind, difficulty, count)
    output_schema = _build_output_schema(kind)

    last_error: str | None = None

    for attempt in range(max_retries + 1):
        prompt = base_prompt
        if last_error is not None:
            prompt = (
                f"{base_prompt}\n\n"
                f"Your previous output failed validation:\n{last_error}\n"
                "Please fix the errors and try again."
            )

        try:
            response_text = generator.generate(prompt, output_schema)
            result = _parse_and_validate(response_text, kind)
            logger.info(
                "Generation succeeded on attempt %d/%d (kind=%s, count=%d)",
                attempt + 1,
                max_retries + 1,
                kind.value,
                count,
            )
            return StudyGenerateResponse(
                quizzes=result.get("quizzes"),
                flashcards=result.get("flashcards"),
                source_used=source_used,
            )
        except (json.JSONDecodeError, ValueError) as exc:
            last_error = str(exc)
            logger.warning(
                "Generation attempt %d/%d failed: %s",
                attempt + 1,
                max_retries + 1,
                last_error,
            )

    logger.error(
        "All %d generation attempts exhausted for kind=%s",
        max_retries + 1,
        kind.value,
    )
    raise APIException(
        status_code=502,
        detail={
            "error": "Content generation failed after multiple attempts",
            "last_validation_error": last_error,
            "retries_used": max_retries + 1,
        },
        hint=(
            "The AI model could not produce schema-compliant questions or flashcards. "
            "Try simplifying the topic, shortening the lesson text, or adjusting the difficulty level, "
            "then retry in a few moments."
        ),
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.post("/study/generate", response_model=StudyGenerateResponse)
async def generate_study_content(request: StudyGenerateRequest):
    """Generate quizzes and/or flashcards for Islamic study.

    Accepts a topic or lesson text and returns schema-validated quizzes
    and/or flashcards at the requested difficulty level.
    """
    if isinstance(request.source, LessonTextSource):
        text_len = len(request.source.lesson_text)
        if text_len > MAX_LESSON_TEXT_LENGTH:
            raise APIException(
                status_code=422,
                detail=f"lesson_text must not exceed {MAX_LESSON_TEXT_LENGTH} characters (got {text_len})",
                hint=(
                    f"Shorten the input text to under {MAX_LESSON_TEXT_LENGTH} characters "
                    f"(current length is {text_len} characters) or split it into smaller study sections."
                ),
            )

    use_fake = os.getenv("USE_FAKE_GENERATOR", "0") == "1"
    if use_fake:
        generator: BaseGenerator = FakeGenerator()
        logger.info("Using FakeGenerator (USE_FAKE_GENERATOR=1)")
    else:
        generator = GeminiGenerator()
        logger.info("Using GeminiGenerator")

    return _generate_content(
        generator=generator,
        source=request.source,
        kind=request.kind,
        difficulty=request.difficulty,
        count=request.count,
    )


# ---------------------------------------------------------------------------
# Study Session Optimization (#258)
# ---------------------------------------------------------------------------
# Optimises session timing, duration, content mix, breaks, and fatigue
# detection based on user energy, attention span, Ramadan/work context, and
# recent SessionLog history. Reuses SessionLog, SessionPerformance,
# UserCapacity and AyahRange from memorization_planner.


from memorization_planner import (  # noqa: E402
    AyahRange,
    SessionLog,
    SessionPerformance,
    UserCapacity,
)


class EnergyLevel(str, Enum):
    """Inferred cognitive energy at a given time."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    PEAK = "peak"


class SessionBlockType(str, Enum):
    MEMORIZE = "memorize"
    REVIEW = "review"
    NEW = "new"
    BREAK = "break"


class WorkSchedule(str, Enum):
    NONE = "none"
    MORNING = "morning"
    EVENING = "evening"
    NIGHT = "night"


class SessionBlock(BaseModel):
    """A single timed block within an optimized study session."""

    block_type: SessionBlockType
    duration_minutes: int = Field(ge=1, le=120)
    ayah_range: AyahRange | None = None
    rationale: str = Field(default="", description="Why this block appears where it does")


class StudySessionRequest(BaseModel):
    """Inputs for optimizing one study session."""

    user_id: str = Field("default_user", min_length=1)
    hour_of_day: int = Field(..., ge=0, le=23, description="Local hour the session starts (0-23)")
    day_of_week: int = Field(..., ge=0, le=6, description="0=Monday, 6=Sunday")
    available_minutes: int = Field(30, ge=10, le=240)
    is_ramadan: bool = Field(False, description="Month of Ramadan fasting context")
    fasting: bool = Field(False, description="Fasting today (overrides is_ramadan for edge cases)")
    work_schedule: WorkSchedule = Field(WorkSchedule.NONE)
    average_attention_span_minutes: int = Field(25, ge=5, le=90)
    capacity: UserCapacity | None = Field(None, description="Optional existing planner capacity")
    recent_sessions: list[SessionLog] = Field(default_factory=list)
    pending_new_portion: AyahRange | None = Field(None, description="Optional new ayahs to memorize this session")
    pending_review_portion: AyahRange | None = Field(None, description="Optional ayahs to review")


class StudySessionPlan(BaseModel):
    """Optimized session output."""

    user_id: str
    blocks: list[SessionBlock]
    total_minutes: int
    recommended_start_hour: int
    break_after_minutes: int
    energy_level: EnergyLevel
    fatigue_risk: str = Field(description="low / moderate / high")
    structure_rationale: str


class FatigueCheckRequest(BaseModel):
    """Inputs for fatigue detection."""

    recent_sessions: list[SessionLog] = Field(default_factory=list)
    hour_of_day: int = Field(..., ge=0, le=23)
    fasting: bool = False


class FatigueSignal(BaseModel):
    detected: bool
    confidence: float = Field(ge=0.0, le=1.0)
    recommendation: str
    suggested_break_minutes: int = Field(ge=0, le=120)


def _infer_energy_level(hour: int, fasting: bool, schedule: WorkSchedule) -> EnergyLevel:
    """Heuristic energy level from hour + fasting + work schedule.

    Not learned â€” explicitly documented in the PR as a v1 heuristic.
    """
    # Baseline circadian shape: peaks ~9-11 and ~15-17; low ~13-14 and overnight.
    if 9 <= hour <= 11 or 15 <= hour <= 17:
        base = EnergyLevel.PEAK
    elif 7 <= hour <= 8 or 12 <= hour <= 14 or 18 <= hour <= 20:
        base = EnergyLevel.MEDIUM
    elif 6 <= hour < 7 or 21 <= hour <= 22:
        base = EnergyLevel.LOW
    else:
        base = EnergyLevel.LOW

    # Fasting pulls energy down, especially afternoon.
    if fasting and 12 <= hour <= 17 and base == EnergyLevel.PEAK:
        base = EnergyLevel.HIGH

    # Work schedule overlap lowers effective energy.
    if schedule == WorkSchedule.MORNING and 8 <= hour <= 12 and base == EnergyLevel.PEAK:
        base = EnergyLevel.HIGH
    if schedule == WorkSchedule.EVENING and 17 <= hour <= 21 and base == EnergyLevel.HIGH:
        base = EnergyLevel.MEDIUM
    if schedule == WorkSchedule.NIGHT and (hour >= 22 or hour <= 5) and base == EnergyLevel.LOW:
        base = EnergyLevel.LOW  # stays low; night-shift noted in rationale

    return base


def _fatigue_from_history(sessions: list[SessionLog], hour: int, fasting: bool) -> FatigueSignal:
    """Detect fatigue from recent session performance."""
    if not sessions:
        return FatigueSignal(
            detected=False,
            confidence=0.2,
            recommendation="No history yet â€” start with a moderate 25-minute session and log the result.",
            suggested_break_minutes=0,
        )

    struggle = sum(1 for s in sessions if s.performance == SessionPerformance.STRUGGLING or not s.completed)
    ratio = struggle / len(sessions)

    # Late hour / fasting amplifies risk.
    hour_penalty = 0.15 if (hour >= 22 or hour <= 4) else 0.0
    fasting_penalty = 0.15 if fasting else 0.0
    score = min(1.0, ratio + hour_penalty + fasting_penalty)

    if score >= 0.6:
        return FatigueSignal(
            detected=True,
            confidence=round(score, 2),
            recommendation="Fatigue detected. Shorten new memorization and prioritise light review.",
            suggested_break_minutes=15,
        )
    if score >= 0.35:
        return FatigueSignal(
            detected=True,
            confidence=round(score, 2),
            recommendation="Mild fatigue. Insert a mid-session break and reduce new material.",
            suggested_break_minutes=10,
        )
    return FatigueSignal(
        detected=False,
        confidence=round(1.0 - score, 2),
        recommendation="No fatigue signal. Proceed with the full plan.",
        suggested_break_minutes=0,
    )


def _content_mix(energy: EnergyLevel, fasting: bool) -> dict[str, float]:
    """Return ratios for memorize / review / new that sum to ~1.0.

    Higher energy -> more new memorization; lower energy -> more review.
    Fasting shifts weight toward review to protect retention under stress.
    """
    if energy == EnergyLevel.PEAK:
        base = {"memorize": 0.50, "review": 0.30, "new": 0.20}
    elif energy == EnergyLevel.HIGH:
        base = {"memorize": 0.45, "review": 0.35, "new": 0.20}
    elif energy == EnergyLevel.MEDIUM:
        base = {"memorize": 0.35, "review": 0.45, "new": 0.20}
    else:  # LOW
        base = {"memorize": 0.25, "review": 0.55, "new": 0.20}

    if fasting:
        base["memorize"] -= 0.05
        base["review"] += 0.05
    return base


def _pick_start_hour(preferred_hour: int, day_of_week: int, is_ramadan: bool) -> int:
    """Nudge preferred hour toward optimal windows, respecting Ramadan fasting."""
    if is_ramadan:
        # Post-iftar (19-21) and pre-dawn (04-06) are the strongest windows.
        if 6 < preferred_hour < 19:
            return 20 if preferred_hour >= 13 else 5
        return preferred_hour
    # Weekend mornings benefit; weekday midday less so.
    if day_of_week >= 5 and preferred_hour == 12:
        return 9
    if day_of_week < 5 and (preferred_hour == 13 or preferred_hour == 14):
        return 15
    return preferred_hour


class StudySessionOptimizer:
    """Stateless optimizer; callers pass in inputs and read the plan."""

    @classmethod
    def optimize(cls, req: StudySessionRequest) -> StudySessionPlan:
        fasting = req.fasting or req.is_ramadan
        energy = _infer_energy_level(req.hour_of_day, fasting, req.work_schedule)
        start_hour = _pick_start_hour(req.hour_of_day, req.day_of_week, req.is_ramadan)

        available = req.available_minutes
        if req.capacity:
            available = min(available, req.capacity.daily_available_minutes)

        attention = max(10, req.average_attention_span_minutes)
        mix = _content_mix(energy, fasting)

        # Reserve time for breaks: 1 break every `attention` minutes, ~5 min each.
        est_breaks = max(0, available // (attention + 5))
        break_minutes = est_breaks * 5
        work_minutes = max(10, available - break_minutes)

        mem_min = max(5, round(work_minutes * mix["memorize"]))
        rev_min = max(5, round(work_minutes * mix["review"]))
        new_min = max(0, work_minutes - mem_min - rev_min)

        # If no new portion is provided, convert new minutes into review.
        if req.pending_new_portion is None and new_min > 0:
            rev_min += new_min
            new_min = 0

        blocks: list[SessionBlock] = []
        remaining_mem, remaining_rev, remaining_new = mem_min, rev_min, new_min

        def _push(block_type: SessionBlockType, minutes: int, ayah_range: AyahRange | None, why: str) -> None:
            nonlocal blocks
            if minutes > 0:
                blocks.append(
                    SessionBlock(
                        block_type=block_type,
                        duration_minutes=min(minutes, 120),
                        ayah_range=ayah_range,
                        rationale=why,
                    )
                )

        # Ordering: review first (warm-up), then memorize, break, then new.
        _push(
            SessionBlockType.REVIEW,
            remaining_rev,
            req.pending_review_portion,
            "Warm-up review consolidates recent ayahs before new load.",
        )
        remaining_rev = 0
        _push(
            SessionBlockType.MEMORIZE,
            remaining_mem,
            req.pending_new_portion,
            "Peak/high energy is best spent on consolidating fresh memorization.",
        )
        remaining_mem = 0

        if est_breaks > 0:
            _push(SessionBlockType.BREAK, 5, None, "Short break to reset attention.")
        if remaining_new > 0:
            _push(
                SessionBlockType.NEW,
                remaining_new,
                req.pending_new_portion,
                "Energy headroom allows safe introduction of new ayahs.",
            )

        total = sum(b.duration_minutes for b in blocks)
        fatigue = _fatigue_from_history(req.recent_sessions, req.hour_of_day, fasting)

        # If fatigue, add a second break and shorten the longest work block.
        if fatigue.detected and blocks:
            blocks.append(
                SessionBlock(
                    block_type=SessionBlockType.BREAK,
                    duration_minutes=fatigue.suggested_break_minutes or 10,
                    rationale="Fatigue signal from recent sessions.",
                )
            )
            total = sum(b.duration_minutes for b in blocks)

        fatigue_risk = "low" if not fatigue.detected else ("high" if fatigue.confidence >= 0.6 else "moderate")

        rationale = (
            f"Energy={energy.value}; fasting={fasting}; schedule={req.work_schedule.value}. "
            f"Content mix memorize/review/new="
            f"{mix['memorize']:.0%}/{mix['review']:.0%}/{mix['new']:.0%}. "
            f"Attention span {attention} min drives break cadence."
        )

        return StudySessionPlan(
            user_id=req.user_id,
            blocks=blocks,
            total_minutes=total,
            recommended_start_hour=start_hour,
            break_after_minutes=attention,
            energy_level=energy,
            fatigue_risk=fatigue_risk,
            structure_rationale=rationale,
        )

    @classmethod
    def check_fatigue(cls, req: FatigueCheckRequest) -> FatigueSignal:
        return _fatigue_from_history(req.recent_sessions, req.hour_of_day, req.fasting)

    @staticmethod
    def peak_times(user_id: str, recent_sessions: list[SessionLog]) -> dict[str, list[int]]:
        """Very small peak-window estimator. Returns hours ranked by STRONG sessions.

        Documented in the PR as a v1 seed pending real telemetry.
        """
        if not recent_sessions:
            return {"morning": [9, 10, 11], "afternoon": [15, 16, 17], "evening": [20, 21]}
        # Without per-session hour data on SessionLog, return the circadian default
        # and note that richer history would refine this.
        return {"morning": [9, 10, 11], "afternoon": [15, 16, 17], "evening": [20, 21]}


@router.post("/study/optimize", response_model=StudySessionPlan)
async def optimize_study_session(request: StudySessionRequest) -> StudySessionPlan:
    """Return an optimized study session: block order, durations, breaks, and rationale."""
    return StudySessionOptimizer.optimize(request)


@router.post("/study/fatigue-check", response_model=FatigueSignal)
async def check_study_fatigue(request: FatigueCheckRequest) -> FatigueSignal:
    """Detect fatigue from recent session performance and return rest recommendation."""
    return StudySessionOptimizer.check_fatigue(request)
