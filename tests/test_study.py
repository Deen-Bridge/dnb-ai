"""Tests for the study-content generator.

All tests run offline against recorded fixtures — no GEMINI_API_KEY needed.
"""

import json
import os
from pathlib import Path
from typing import List

import pytest
from pydantic import ValidationError

from study import (
    MAX_LESSON_TEXT_LENGTH,
    ContentKind,
    Difficulty,
    FakeGenerator,
    Flashcard,
    LessonTextSource,
    QuizQuestion,
    StudyGenerateRequest,
    StudyGenerateResponse,
    TopicSource,
    _build_prompt,
    _generate_content,
    _parse_and_validate,
    pydantic_to_gemini_schema,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_fixture(name: str) -> str:
    return (FIXTURE_DIR / f"{name}.json").read_text(encoding="utf-8")


def make_topic_source(topic: str = "The Five Pillars") -> TopicSource:
    return TopicSource(topic=topic)


def make_lesson_source(text: str = "Salah is the second pillar.") -> LessonTextSource:
    return LessonTextSource(lesson_text=text)


# ---------------------------------------------------------------------------
# Pydantic model validation
# ---------------------------------------------------------------------------


class TestQuizQuestion:
    def test_valid_quiz_question(self):
        q = QuizQuestion(
            question="What is Zakat?",
            choices=["Charity", "Fasting", "Prayer", "Pilgrimage"],
            correct_index=0,
            explanation="Zakat is obligatory charity.",
            difficulty="beginner",
        )
        assert q.question == "What is Zakat?"
        assert len(q.choices) == 4
        assert q.correct_index == 0

    def test_invalid_choice_count(self):
        with pytest.raises(ValidationError, match="exactly 4 choices"):
            QuizQuestion(
                question="Q?",
                choices=["A", "B", "C"],
                correct_index=0,
                explanation="E",
                difficulty="beginner",
            )

    def test_duplicate_choices(self):
        with pytest.raises(ValidationError, match="duplicates"):
            QuizQuestion(
                question="Q?",
                choices=["Same", "Same", "C", "D"],
                correct_index=0,
                explanation="E",
                difficulty="beginner",
            )

    def test_empty_choice(self):
        with pytest.raises(ValidationError):
            QuizQuestion(
                question="Q?",
                choices=["A", "", "C", "D"],
                correct_index=0,
                explanation="E",
                difficulty="beginner",
            )

    def test_question_identical_to_choice(self):
        with pytest.raises(ValidationError, match="identical to any choice"):
            QuizQuestion(
                question="What is Salah?",
                choices=["What is Salah?", "Fasting", "Charity", "Hajj"],
                correct_index=1,
                explanation="E",
                difficulty="beginner",
            )

    def test_question_empty(self):
        with pytest.raises(ValidationError):
            QuizQuestion(
                question="",
                choices=["A", "B", "C", "D"],
                correct_index=0,
                explanation="E",
                difficulty="beginner",
            )

    def test_explanation_empty(self):
        with pytest.raises(ValidationError):
            QuizQuestion(
                question="Q?",
                choices=["A", "B", "C", "D"],
                correct_index=0,
                explanation="",
                difficulty="beginner",
            )

    def test_correct_index_out_of_range(self):
        with pytest.raises(ValidationError):
            QuizQuestion(
                question="Q?",
                choices=["A", "B", "C", "D"],
                correct_index=5,
                explanation="E",
                difficulty="beginner",
            )


class TestFlashcard:
    def test_valid_flashcard(self):
        fc = Flashcard(front="Term?", back="Definition.", tags=["tag1"])
        assert fc.front == "Term?"
        assert fc.back == "Definition."
        assert fc.tags == ["tag1"]

    def test_empty_front(self):
        with pytest.raises(ValidationError):
            Flashcard(front="", back="Definition.", tags=[])

    def test_empty_back(self):
        with pytest.raises(ValidationError):
            Flashcard(front="Term?", back="", tags=[])


# ---------------------------------------------------------------------------
# Request model tests
# ---------------------------------------------------------------------------


class TestStudyGenerateRequest:
    def test_topic_source(self):
        req = StudyGenerateRequest(source={"topic": "Fiqh"})
        assert isinstance(req.source, TopicSource)
        assert req.source.topic == "Fiqh"

    def test_lesson_text_source(self):
        req = StudyGenerateRequest(source={"lesson_text": "Some lesson"})
        assert isinstance(req.source, LessonTextSource)
        assert req.source.lesson_text == "Some lesson"

    def test_default_values(self):
        req = StudyGenerateRequest(source={"topic": "Tawheed"})
        assert req.kind == ContentKind.quiz
        assert req.difficulty == Difficulty.beginner
        assert req.count == 5

    def test_count_bounds(self):
        with pytest.raises(ValidationError):
            StudyGenerateRequest(source={"topic": "X"}, count=0)
        with pytest.raises(ValidationError):
            StudyGenerateRequest(source={"topic": "X"}, count=21)

    def test_oversized_lesson_text_is_rejected_by_pydantic(self):
        long_text = "x" * (MAX_LESSON_TEXT_LENGTH + 1)
        with pytest.raises(ValidationError):
            StudyGenerateRequest(source={"lesson_text": long_text})


# ---------------------------------------------------------------------------
# Schema translation
# ---------------------------------------------------------------------------


class TestPydanticToGeminiSchema:
    def test_no_title_keys(self):
        schema = pydantic_to_gemini_schema(QuizQuestion)
        schema_str = json.dumps(schema)
        assert '"title"' not in schema_str

    def test_no_defs(self):
        schema = pydantic_to_gemini_schema(QuizQuestion)
        assert "$defs" not in schema

    def test_has_required_fields(self):
        schema = pydantic_to_gemini_schema(Flashcard)
        assert "properties" in schema
        assert "front" in schema["properties"]
        assert "back" in schema["properties"]


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------


class TestBuildPrompt:
    def test_topic_in_prompt(self):
        prompt, source_used = _build_prompt(
            make_topic_source("Tawheed"),
            ContentKind.quiz,
            Difficulty.beginner,
            5,
        )
        assert "Topic: Tawheed" in prompt
        assert source_used == "topic"

    def test_lesson_text_in_prompt(self):
        prompt, source_used = _build_prompt(
            make_lesson_source("Some lesson text here"),
            ContentKind.quiz,
            Difficulty.beginner,
            5,
        )
        assert "Lesson text:" in prompt
        assert source_used == "lesson_text"

    def test_difficulty_in_prompt(self):
        prompt, _ = _build_prompt(
            make_topic_source("X"),
            ContentKind.quiz,
            Difficulty.advanced,
            3,
        )
        assert "ADVANCED" in prompt

    def test_count_in_prompt(self):
        prompt, _ = _build_prompt(
            make_topic_source("X"),
            ContentKind.flashcards,
            Difficulty.beginner,
            10,
        )
        assert "exactly 10" in prompt


# ---------------------------------------------------------------------------
# Parsing and validation
# ---------------------------------------------------------------------------


class TestParseAndValidate:
    def test_valid_quiz(self):
        raw = load_fixture("quiz_valid")
        result = _parse_and_validate(raw, ContentKind.quiz)
        assert "quizzes" in result
        assert len(result["quizzes"]) == 3
        assert all(isinstance(q, QuizQuestion) for q in result["quizzes"])

    def test_valid_flashcards(self):
        raw = load_fixture("flashcards_valid")
        result = _parse_and_validate(raw, ContentKind.flashcards)
        assert "flashcards" in result
        assert len(result["flashcards"]) == 2
        assert all(isinstance(f, Flashcard) for f in result["flashcards"])

    def test_invalid_json_raises(self):
        raw = load_fixture("invalid_json")
        with pytest.raises(ValueError, match="Invalid JSON"):
            _parse_and_validate(raw, ContentKind.quiz)

    def test_schema_invalid_raises(self):
        raw = load_fixture("schema_invalid")
        with pytest.raises(ValueError, match="quiz"):
            _parse_and_validate(raw, ContentKind.quiz)

    def test_semantic_invalid_raises(self):
        raw = load_fixture("semantic_invalid")
        with pytest.raises(ValueError, match="duplicates"):
            _parse_and_validate(raw, ContentKind.quiz)

    def test_both_kind(self):
        data = {
            "quizzes": [
                {
                    "question": "Q1?",
                    "choices": ["A", "B", "C", "D"],
                    "correct_index": 0,
                    "explanation": "E",
                    "difficulty": "beginner",
                }
            ],
            "flashcards": [
                {"front": "F", "back": "B", "tags": ["t"]}
            ],
        }
        result = _parse_and_validate(json.dumps(data), ContentKind.both)
        assert "quizzes" in result
        assert "flashcards" in result
        assert len(result["quizzes"]) == 1
        assert len(result["flashcards"]) == 1


# ---------------------------------------------------------------------------
# Generation loop with retry
# ---------------------------------------------------------------------------


class TestGenerateContent:
    def test_happy_path_quiz(self):
        raw = load_fixture("quiz_valid")
        gen = FakeGenerator(responses=[raw])
        result = _generate_content(
            generator=gen,
            source=make_topic_source(),
            kind=ContentKind.quiz,
            difficulty=Difficulty.beginner,
            count=3,
        )
        assert isinstance(result, StudyGenerateResponse)
        assert result.quizzes is not None
        assert len(result.quizzes) == 3
        assert result.source_used == "topic"

    def test_happy_path_flashcards(self):
        raw = load_fixture("flashcards_valid")
        gen = FakeGenerator(responses=[raw])
        result = _generate_content(
            generator=gen,
            source=make_lesson_source(),
            kind=ContentKind.flashcards,
            difficulty=Difficulty.intermediate,
            count=2,
        )
        assert isinstance(result, StudyGenerateResponse)
        assert result.flashcards is not None
        assert len(result.flashcards) == 2
        assert result.source_used == "lesson_text"

    def test_happy_path_both(self):
        data = json.dumps({
            "quizzes": [
                {
                    "question": "Q?",
                    "choices": ["A", "B", "C", "D"],
                    "correct_index": 0,
                    "explanation": "E",
                    "difficulty": "beginner",
                }
            ],
            "flashcards": [
                {"front": "F", "back": "B", "tags": ["t"]}
            ],
        })
        gen = FakeGenerator(responses=[data])
        result = _generate_content(
            generator=gen,
            source=make_topic_source(),
            kind=ContentKind.both,
            difficulty=Difficulty.advanced,
            count=1,
        )
        assert result.quizzes is not None
        assert result.flashcards is not None
        assert len(result.quizzes) == 1
        assert len(result.flashcards) == 1

    def test_retry_on_invalid_json_then_succeeds(self):
        """First response is invalid JSON, second is valid — retry loop should recover."""
        invalid = load_fixture("invalid_json")
        valid = load_fixture("quiz_valid")
        gen = FakeGenerator(responses=[invalid, valid])
        result = _generate_content(
            generator=gen,
            source=make_topic_source("Tawheed"),
            kind=ContentKind.quiz,
            difficulty=Difficulty.beginner,
            count=3,
            max_retries=2,
        )
        assert len(result.quizzes) == 3

    def test_retry_on_schema_invalid_then_succeeds(self):
        invalid = load_fixture("schema_invalid")
        valid = load_fixture("quiz_valid")
        gen = FakeGenerator(responses=[invalid, valid])
        result = _generate_content(
            generator=gen,
            source=make_topic_source(),
            kind=ContentKind.quiz,
            difficulty=Difficulty.beginner,
            count=3,
            max_retries=2,
        )
        assert len(result.quizzes) == 3

    def test_all_retries_exhausted_raises_502(self):
        from fastapi import HTTPException

        invalid = load_fixture("invalid_json")
        gen = FakeGenerator(responses=[invalid, invalid, invalid])
        with pytest.raises(HTTPException) as excinfo:
            _generate_content(
                generator=gen,
                source=make_topic_source(),
                kind=ContentKind.quiz,
                difficulty=Difficulty.beginner,
                count=3,
                max_retries=2,
            )
        assert excinfo.value.status_code == 502
        detail = excinfo.value.detail
        assert "retries_used" in detail
        assert detail["retries_used"] == 3  # max_retries+1 = 3

    def test_all_retries_exhausted_returns_structured_error(self):
        from fastapi import HTTPException

        bad_data = json.dumps({"quizzes": "not an array"})
        gen = FakeGenerator(responses=[bad_data, bad_data, bad_data])
        with pytest.raises(HTTPException) as excinfo:
            _generate_content(
                generator=gen,
                source=make_topic_source(),
                kind=ContentKind.quiz,
                difficulty=Difficulty.beginner,
                count=3,
                max_retries=2,
            )
        detail = excinfo.value.detail
        assert isinstance(detail, dict)
        assert "last_validation_error" in detail
        assert "retries_used" in detail


# ---------------------------------------------------------------------------
# Response model
# ---------------------------------------------------------------------------


class TestStudyGenerateResponse:
    def test_valid_response(self):
        resp = StudyGenerateResponse(
            quizzes=[],
            flashcards=None,
            source_used="topic",
        )
        assert resp.quizzes == []
        assert resp.flashcards is None
        assert resp.source_used == "topic"

    def test_with_flashcards_only(self):
        resp = StudyGenerateResponse(
            quizzes=None,
            flashcards=[Flashcard(front="F", back="B", tags=[])],
            source_used="lesson_text",
        )
        assert resp.quizzes is None
        assert len(resp.flashcards) == 1
