"""Study-content generator — produces schema-validated quizzes and flashcards
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
from typing import Any, Dict, List, Optional, Tuple, Union

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator, model_validator

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
5. Not fabricate citations — if unsure, omit the citation
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

DIFFICULTY_PROMPTS: Dict[str, str] = {
    "beginner": (
        "BEGINNER level — test recall of well-known, foundational Islamic knowledge. "
        "Questions should cover basic facts that most Muslims would know. "
        "Avoid complex legal rulings or detailed scholarly debates."
    ),
    "intermediate": (
        "INTERMEDIATE level — test applied understanding. "
        "Questions may require connecting concepts, understanding evidence, "
        "or applying principles to new situations. Suitable for students "
        "with basic Islamic education."
    ),
    "advanced": (
        "ADVANCED level — test comparative and usul-level nuance. "
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
    source: Union[TopicSource, LessonTextSource] = Field(
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
    choices: List[str] = Field(..., description="Exactly 4 answer choices")
    correct_index: int = Field(..., ge=0, le=3, description="Index of the correct answer (0-3)")
    explanation: str = Field(..., min_length=1, description="Explanation of the correct answer")
    difficulty: str = Field(..., description="Difficulty level of this question")

    @field_validator("choices")
    @classmethod
    def choices_must_be_exactly_four_distinct_nonempty(cls, v: List[str]) -> List[str]:
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
    tags: List[str] = Field(default_factory=list, description="Tags for categorization")


class StudyGenerateResponse(BaseModel):
    quizzes: Optional[List[QuizQuestion]] = Field(default=None, description="Generated quiz questions")
    flashcards: Optional[List[Flashcard]] = Field(default=None, description="Generated flashcards")
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
# Schema translation  (pydantic → Gemini-compatible JSON schema)
# ---------------------------------------------------------------------------


def _strip_title(obj: Any) -> Any:
    """Recursively strip 'title' keys — Gemini rejects them."""
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

    def __init__(self, model_name: str = "gemini-2.5-flash-preview-05-20"):
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

    def __init__(self, responses: Optional[List[str]] = None):
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
    source: Union[TopicSource, LessonTextSource],
    kind: ContentKind,
    difficulty: Difficulty,
    count: int,
) -> Tuple[str, str]:
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
    lines.append(
        "Output ONLY valid JSON. No markdown, no code fences, no extra text."
    )
    return "\n".join(lines), source_used


def _output_format_instruction(kind: ContentKind) -> str:
    if kind == ContentKind.quiz:
        return (
            'Output a JSON object with key "quizzes" (array). '
            "Each item: question (str), choices ([str; 4 items]), "
            "correct_index (int 0-3), explanation (str), difficulty (str)."
        )
    if kind == ContentKind.flashcards:
        return (
            'Output a JSON object with key "flashcards" (array). '
            "Each item: front (str), back (str), tags ([str])."
        )
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
            "properties": {
                "quizzes": {"type": "array", "items": quiz_schema}
            },
            "required": ["quizzes"],
        }
    if kind == ContentKind.flashcards:
        return {
            "type": "object",
            "properties": {
                "flashcards": {"type": "array", "items": flash_schema}
            },
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
        raise ValueError(f"Invalid JSON: {e}")

    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object, got {type(data).__name__}")

    result: Dict[str, Any] = {}
    errors: List[str] = []

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
            validated = []
            for i, item in enumerate(raw):
                try:
                    validated.append(Flashcard.model_validate(item))
                except Exception as exc:
                    errors.append(f"flashcard[{i}]: {exc}")
            result["flashcards"] = validated

    if errors:
        raise ValueError("; ".join(errors))

    return result


# ---------------------------------------------------------------------------
# Core generation loop (with bounded retry)
# ---------------------------------------------------------------------------


def _generate_content(
    generator: BaseGenerator,
    source: Union[TopicSource, LessonTextSource],
    kind: ContentKind,
    difficulty: Difficulty,
    count: int,
    max_retries: int = DEFAULT_RETRIES,
) -> StudyGenerateResponse:
    """Run the generation loop with up to ``max_retries`` retries on error."""
    base_prompt, source_used = _build_prompt(source, kind, difficulty, count)
    output_schema = _build_output_schema(kind)

    last_error: Optional[str] = None

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
    raise HTTPException(
        status_code=502,
        detail={
            "error": "Content generation failed after multiple attempts",
            "last_validation_error": last_error,
            "retries_used": max_retries + 1,
        },
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
            raise HTTPException(
                status_code=422,
                detail=f"lesson_text must not exceed {MAX_LESSON_TEXT_LENGTH} characters (got {text_len})",
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
