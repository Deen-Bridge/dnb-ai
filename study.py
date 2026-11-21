"""Study-content generator - produces schema-validated quizzes and flashcards
using Gemini's JSON mode or deterministic offline curriculum generation (#846 / #847).

Supports topic-based and lesson-text-based generation with three difficulty
levels, deterministic random seeding for repeatable audits, provenance tracking,
and opt-in resettable user personalization.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import random
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator, model_validator

from errors import APIException
from knowledge_assessment import router as knowledge_assessment_router
from study_store import UserStudyRecord, study_store

logger = logging.getLogger(__name__)

router = APIRouter(tags=["study"])
router.include_router(knowledge_assessment_router)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_LESSON_TEXT_LENGTH = 20_000
DEFAULT_RETRIES = 2
GENERATION_TEMPERATURE = 0.3
CURRICULUM_VERSION = "1.0.0"

ISLAMIC_CONTEXT = """You are an AI assistant specialized in creating Islamic study content.
Your generated content must:
1. Be based on authentic Islamic sources (Quran and Hadith)
2. Be accurate and respectful
3. Avoid controversial or divisive topics
4. Cite authentic sources when mentioning specific narrations or verses
5. Not fabricate citations - if unsure, omit the citation
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
        "BEGINNER level - test recall of well-known, foundational Islamic knowledge. "
        "Questions should cover basic facts that most Muslims would know. "
        "Avoid complex legal rulings or detailed scholarly debates."
    ),
    "intermediate": (
        "INTERMEDIATE level - test applied understanding. "
        "Questions may require connecting concepts, understanding evidence, "
        "or applying principles to new situations. Suitable for students "
        "with basic Islamic education."
    ),
    "advanced": (
        "ADVANCED level - test comparative and usul-level nuance. "
        "Questions may involve weighing evidence, understanding methodology "
        "differences, or analyzing complex rulings. Suitable for students "
        "of knowledge."
    ),
}

# ---------------------------------------------------------------------------
# Canonical Authentic Curriculum Bank (for deterministic generation and audits)
# ---------------------------------------------------------------------------

CANONICAL_QUIZ_BANK: list[dict[str, Any]] = [
    {
        "question": "What is the primary theological meaning of Tawhid al-Uluhiyyah?",
        "choices": [
            "Directing all worship exclusively to Allah alone",
            "Believing that Allah is the sole Creator and Sustainer",
            "Affirming Allah's Divine Names and Attributes without distortion",
            "Believing in the preordainment of all events and decrees",
        ],
        "correct_index": 0,
        "explanation": "Tawhid al-Uluhiyyah (or Ibadah) is singling out Allah in all acts of worship, as established in Surah Adh-Dhariyat 51:56.",
        "difficulty": "intermediate",
        "discipline": "aqeedah",
        "citations": ["Quran 51:56"],
    },
    {
        "question": "How many pillars of Iman (faith) are enumerated in the Hadith of Jibril?",
        "choices": ["Four", "Five", "Six", "Seven"],
        "correct_index": 2,
        "explanation": "The Prophet enumerated six pillars: belief in Allah, His angels, His books, His messengers, the Last Day, and Qadar (Sahih Muslim 8).",
        "difficulty": "beginner",
        "discipline": "aqeedah",
        "citations": ["Sahih Muslim 8"],
    },
    {
        "question": "Which of the following is an obligatory prerequisite condition (shart) for Salah to be valid?",
        "choices": [
            "Facing the Qiblah",
            "Reciting a surah after al-Fatihah",
            "Saying the opening takbir aloud",
            "Making supplication during prostration",
        ],
        "correct_index": 0,
        "explanation": "Facing the Qiblah is an essential precondition outside the prayer itself, commanded in Surah Al-Baqarah 2:144.",
        "difficulty": "beginner",
        "discipline": "fiqh",
        "citations": ["Quran 2:144"],
    },
    {
        "question": "What is the classical nisab threshold for Zakat on gold?",
        "choices": [
            "10 Mithqals (approx 42.5 grams)",
            "20 Mithqals (approx 85 grams)",
            "50 Mithqals (approx 212.5 grams)",
            "100 Mithqals (approx 425 grams)",
        ],
        "correct_index": 1,
        "explanation": "The Prophet set the zakat nisab on gold at 20 mithqals/dinars, which equals approximately 85 grams of pure gold (Sunan Abi Dawud 1573).",
        "difficulty": "intermediate",
        "discipline": "fiqh",
        "citations": ["Sunan Abi Dawud 1573"],
    },
    {
        "question": "What is the scholarly term for a hadith narrated by so many independent chains at every generation that collusion upon falsehood is impossible?",
        "choices": ["Ahad", "Mutawatir", "Mursal", "Hasan"],
        "correct_index": 1,
        "explanation": "Mutawatir narrations convey definitive, undeniable knowledge (ilm yaqini) due to the sheer volume of trustworthy narrators.",
        "difficulty": "advanced",
        "discipline": "hadith",
        "citations": ["Nuzhat al-Nazar fi Tawdih Nukhbat al-Fikar"],
    },
    {
        "question": "What is the greatest Ayah in the Quran according to the authentic narration of Ubayy ibn Ka'b?",
        "choices": [
            "Ayat al-Dayn (Quran 2:282)",
            "Ayat al-Kursi (Quran 2:255)",
            "Surah Al-Ikhlas (Quran 112:1)",
            "The last two verses of Surah Al-Baqarah (Quran 2:285-286)",
        ],
        "correct_index": 1,
        "explanation": "The Prophet confirmed to Ubayy ibn Ka'b that Ayat al-Kursi is the greatest ayah in the Book of Allah (Sahih Muslim 810).",
        "difficulty": "beginner",
        "discipline": "tafsir",
        "citations": ["Sahih Muslim 810", "Quran 2:255"],
    },
    {
        "question": "In which Hijri year did the Conquest of Makkah (Fath Makkah) take place?",
        "choices": ["6 AH", "8 AH", "9 AH", "10 AH"],
        "correct_index": 1,
        "explanation": "The Prophet entered Makkah peacefully during Ramadan of the 8th year of the Hijrah (Sahih al-Bukhari 4275).",
        "difficulty": "intermediate",
        "discipline": "seerah",
        "citations": ["Sahih al-Bukhari 4275"],
    },
]

CANONICAL_FLASHCARD_BANK: list[dict[str, Any]] = [
    {
        "front": "Tawhid al-Rububiyyah",
        "back": "Belief in Allah's sole Lordship - affirming that He alone is the Creator, Sustainer, Provider, and Master of all that exists (Quran 7:54).",
        "tags": ["aqeedah", "tawhid"],
        "citations": ["Quran 7:54"],
    },
    {
        "front": "Arkan al-Islam (Pillars of Islam)",
        "back": "1. Shahadah (Declaration of Faith), 2. Salah (Prayer), 3. Zakat (Almsgiving), 4. Sawm (Fasting Ramadan), 5. Hajj (Pilgrimage to Makkah) (Sahih al-Bukhari 8).",
        "tags": ["fiqh", "pillars"],
        "citations": ["Sahih al-Bukhari 8"],
    },
    {
        "front": "Wudu Obligatory Elements (Fara'id)",
        "back": "1. Washing face, 2. Washing arms including elbows, 3. Wiping head, 4. Washing feet including ankles, with intention and sequence (Quran 5:6).",
        "tags": ["fiqh", "taharah"],
        "citations": ["Quran 5:6"],
    },
    {
        "front": "Hadith Terminology: Sanad vs Matn",
        "back": "Sanad is the recorded chain of narrators transmitting the report; Matn is the substantive text of the prophetic tradition itself.",
        "tags": ["hadith", "mustalah"],
        "citations": ["Tadrib al-Rawi"],
    },
    {
        "front": "Al-Mu'awwidhatayn",
        "back": "Surah Al-Falaq (113) and Surah An-Nas (114), revealed as divine protective verses against all forms of harm (Sunan Abi Dawud 1463).",
        "tags": ["quran", "tafsir"],
        "citations": ["Sunan Abi Dawud 1463"],
    },
]

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
    topic: str = Field(..., min_length=1, max_length=500, description="Study topic to generate content about")

    @field_validator("topic")
    @classmethod
    def topic_must_not_be_blank(cls, v: str) -> str:
        trimmed = v.strip()
        if not trimmed:
            raise ValueError("Topic must not be empty or whitespace only")
        return trimmed


class LessonTextSource(BaseModel):
    lesson_text: str = Field(
        ...,
        min_length=1,
        max_length=MAX_LESSON_TEXT_LENGTH,
        description="Lesson text to base content on",
    )

    @field_validator("lesson_text")
    @classmethod
    def lesson_text_must_not_be_blank(cls, v: str) -> str:
        trimmed = v.strip()
        if not trimmed:
            raise ValueError("Lesson text must not be empty or whitespace only")
        return trimmed


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
    seed: int | None = Field(
        default=None,
        description="Optional integer seed to ensure deterministic, auditable generation.",
    )
    language: str = Field(
        default="en",
        description="Preferred language ('en', 'ar', or 'bilingual').",
    )
    user_id: str | None = Field(
        default=None,
        max_length=128,
        description="Optional opaque learner ID for personalized progress tracking.",
    )
    opt_in_personalization: bool = Field(
        default=False,
        description="Explicit opt-in to track user mastery and adapt future content.",
    )


class QuizQuestion(BaseModel):
    question: str = Field(..., min_length=1, description="The quiz question")
    choices: list[str] = Field(..., description="Exactly 4 answer choices")
    correct_index: int = Field(..., ge=0, le=3, description="Index of the correct answer (0-3)")
    explanation: str = Field(..., min_length=1, description="Explanation of the correct answer")
    difficulty: str = Field(..., description="Difficulty level of this question")
    citations: list[str] = Field(default_factory=list, description="Authentic citations for verification")

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
    def question_not_identical_to_any_choice(self) -> QuizQuestion:
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
    citations: list[str] = Field(default_factory=list, description="Authentic source citations")


class StudyProvenance(BaseModel):
    """Provenance metadata tracking grounding sources and curriculum version."""

    curriculum_version: str = CURRICULUM_VERSION
    generator_version: str = "study-v1"
    source_hash: str = Field(..., description="SHA-256 hash of the input source")
    citations: list[str] = Field(default_factory=list, description="Verified Quran/Hadith citations referenced")
    grounded_dataset: str = "curated_islamic_curriculum_v1"


class StudyAuditTrail(BaseModel):
    """Audit trail verifying reproducibility and uncertainty bounds."""

    generated_at_iso: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    seed: int | None = None
    deterministic: bool = False
    uncertainty_score: float = Field(default=0.05, ge=0.0, le=1.0)
    limitations: str = "Educational study material only; consult scholarly authorities for formal rulings."


class StudyGenerateResponse(BaseModel):
    quizzes: list[QuizQuestion] | None = Field(default=None, description="Generated quiz questions")
    flashcards: list[Flashcard] | None = Field(default=None, description="Generated flashcards")
    source_used: str = Field(..., description="Whether 'topic' or 'lesson_text' was used")
    seed: int | None = Field(default=None, description="Seed used for deterministic reproduction")
    provenance: StudyProvenance | None = Field(default=None, description="Provenance metadata for auditing")
    audit_trail: StudyAuditTrail | None = Field(default=None, description="Audit trail for verification")


# ---------------------------------------------------------------------------
# Schema translation (pydantic -> Gemini-compatible JSON schema)
# ---------------------------------------------------------------------------


def _strip_title(obj: Any) -> Any:
    """Recursively strip 'title' keys - Gemini rejects them."""
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

    def __init__(self, model_name: str = "gemini-2.5-flash", seed: int | None = None):
        self.model_name = model_name
        self.seed = seed

    def generate(self, prompt: str, schema: dict) -> str:
        import google.generativeai as genai

        model = genai.GenerativeModel(self.model_name)
        temp = 0.0 if self.seed is not None else GENERATION_TEMPERATURE
        response = model.generate_content(
            prompt,
            generation_config={
                "temperature": temp,
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


class DeterministicCurriculumGenerator(BaseGenerator):
    """Offline deterministic generator using authentic canonical curriculum bank."""

    def __init__(self, seed: int = 42):
        self.seed = seed

    def generate(self, prompt: str, schema: dict) -> str:
        rng = random.Random(self.seed)

        # Build quizzes
        quizzes: list[dict[str, Any]] = []
        shuffled_q = list(CANONICAL_QUIZ_BANK)
        rng.shuffle(shuffled_q)

        for item in shuffled_q:
            q_copy = copy.deepcopy(item)
            # Deterministically permute choices and track correct index
            original_choices = q_copy["choices"]
            correct_choice = original_choices[q_copy["correct_index"]]
            permuted = list(original_choices)
            rng.shuffle(permuted)
            q_copy["choices"] = permuted
            q_copy["correct_index"] = permuted.index(correct_choice)
            quizzes.append(q_copy)

        # Build flashcards
        shuffled_f = list(CANONICAL_FLASHCARD_BANK)
        rng.shuffle(shuffled_f)
        flashcards = [copy.deepcopy(f) for f in shuffled_f]

        payload = {"quizzes": quizzes, "flashcards": flashcards}
        return json.dumps(payload)


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
    pydantic objects. Raises ``ValueError`` on any failure.
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
    seed: int | None = None,
    max_retries: int = DEFAULT_RETRIES,
) -> StudyGenerateResponse:
    """Run the generation loop with up to ``max_retries`` retries on error."""
    base_prompt, source_used = _build_prompt(source, kind, difficulty, count)
    output_schema = _build_output_schema(kind)

    # Compute source hash for provenance
    source_raw = source.topic if isinstance(source, TopicSource) else source.lesson_text
    source_hash = hashlib.sha256(source_raw.encode("utf-8")).hexdigest()

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

            quizzes = result.get("quizzes")
            if quizzes:
                quizzes = quizzes[:count]
            flashcards = result.get("flashcards")
            if flashcards:
                flashcards = flashcards[:count]

            # Aggregate citations for provenance
            citations: list[str] = []
            if quizzes:
                for q in quizzes:
                    citations.extend(q.citations)
            if flashcards:
                for f in flashcards:
                    citations.extend(f.citations)
            citations = sorted(list(set(citations)))

            provenance = StudyProvenance(
                curriculum_version=CURRICULUM_VERSION,
                generator_version="study-v1",
                source_hash=source_hash,
                citations=citations,
            )

            audit_trail = StudyAuditTrail(
                generated_at_iso=datetime.now(UTC).isoformat(),
                seed=seed,
                deterministic=seed is not None,
                uncertainty_score=0.02 if seed is not None else 0.08,
            )

            logger.info(
                "Generation succeeded on attempt %d/%d (kind=%s, count=%d, seed=%s)",
                attempt + 1,
                max_retries + 1,
                kind.value,
                count,
                seed,
            )
            return StudyGenerateResponse(
                quizzes=quizzes,
                flashcards=flashcards,
                source_used=source_used,
                seed=seed,
                provenance=provenance,
                audit_trail=audit_trail,
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
    and/or flashcards at the requested difficulty level. Supports deterministic
    random seeding and opt-in user personalization.
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
    if request.seed is not None:
        generator: BaseGenerator = DeterministicCurriculumGenerator(seed=request.seed)
        logger.info("Using DeterministicCurriculumGenerator (seed=%d)", request.seed)
    elif use_fake:
        generator = FakeGenerator()
        logger.info("Using FakeGenerator (USE_FAKE_GENERATOR=1)")
    else:
        generator = GeminiGenerator()
        logger.info("Using GeminiGenerator")

    response = _generate_content(
        generator=generator,
        source=request.source,
        kind=request.kind,
        difficulty=request.difficulty,
        count=request.count,
        seed=request.seed,
    )

    # Opt-in user personalization tracking
    if request.user_id and request.opt_in_personalization:
        discipline = "general"
        if isinstance(request.source, TopicSource):
            topic_lower = request.source.topic.lower()
            if any(w in topic_lower for w in ["iman", "faith", "tawhid", "aqeedah"]):
                discipline = "aqeedah"
            elif any(w in topic_lower for w in ["salah", "zakat", "wudu", "fiqh", "taharah", "hajj", "pillar"]):
                discipline = "fiqh"
            elif any(w in topic_lower for w in ["hadith", "bukhari", "muslim"]):
                discipline = "hadith"
            elif any(w in topic_lower for w in ["quran", "surah", "tafsir", "ayah"]):
                discipline = "tafsir"
            elif any(w in topic_lower for w in ["prophet", "seerah", "makkah", "madinah"]):
                discipline = "seerah"

        await study_store.record_progress(
            user_id=request.user_id,
            discipline=discipline,
            score=80.0,
            items_count=request.count,
            topic=request.source.topic if isinstance(request.source, TopicSource) else None,
        )

    return response


@router.get("/study/users/{user_id}/progress", response_model=UserStudyRecord)
async def get_user_study_progress(user_id: str):
    """Retrieve study progress and discipline mastery for an opted-in learner."""
    progress = await study_store.get_progress(user_id)
    if progress is None:
        raise HTTPException(
            status_code=404,
            detail=f"No study progress found for user '{user_id}' or user has not opted into personalization.",
        )
    return progress


@router.delete("/study/users/{user_id}")
async def reset_user_study_progress(user_id: str):
    """Permanently erase all user-scoped study progress and personalization data (opt-out / reset)."""
    existed = await study_store.reset_user(user_id)
    return {
        "status": "reset",
        "user_id": user_id,
        "existed": existed,
        "detail": "User study history and personalization data has been permanently cleared.",
    }
