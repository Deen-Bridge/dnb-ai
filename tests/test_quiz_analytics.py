"""Tests for the Quiz Performance Analytics subsystem (#257)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from main import app
from quiz_analytics import (
    QuestionType,
    QuizAnalyticsRequest,
    QuizQuestionAttempt,
    QuizSessionRecord,
    analyze_quiz_performance,
)

client = TestClient(app)


def _sample_request() -> QuizAnalyticsRequest:
    return QuizAnalyticsRequest(
        learner_id="learner-42",
        sessions=[
            QuizSessionRecord(
                session_id="s1",
                timestamp_iso="2026-10-01T10:00:00Z",
                attempts=[
                    QuizQuestionAttempt(
                        question_id="q1",
                        topic="zakat",
                        question_type=QuestionType.scenario_ruling,
                        madhhab="shafii",
                        is_correct=False,
                        time_spent_seconds=55.0,
                        difficulty=4,
                    ),
                    QuizQuestionAttempt(
                        question_id="q2",
                        topic="salah",
                        question_type=QuestionType.multiple_choice,
                        madhhab="hanafi",
                        is_correct=True,
                        time_spent_seconds=18.0,
                        difficulty=2,
                    ),
                ],
            ),
            QuizSessionRecord(
                session_id="s2",
                timestamp_iso="2026-10-08T10:00:00Z",
                attempts=[
                    QuizQuestionAttempt(
                        question_id="q3",
                        topic="zakat",
                        question_type=QuestionType.scenario_ruling,
                        madhhab="shafii",
                        is_correct=True,
                        time_spent_seconds=48.0,
                        difficulty=4,
                    ),
                    QuizQuestionAttempt(
                        question_id="q4",
                        topic="salah",
                        question_type=QuestionType.multiple_choice,
                        madhhab="hanafi",
                        is_correct=True,
                        time_spent_seconds=14.0,
                        difficulty=2,
                    ),
                ],
            ),
        ],
    )


def test_analyze_quiz_performance_computes_trends_topics_and_madhhabs() -> None:
    res = analyze_quiz_performance(_sample_request())
    assert res.overall_accuracy == 0.75
    assert res.total_questions == 4
    assert res.trend_direction == "improving"
    assert res.trend_slope > 0

    topics = {t.topic: t for t in res.topic_insights}
    assert topics["zakat"].accuracy == 0.5
    assert topics["zakat"].needs_attention is True
    assert topics["salah"].accuracy == 1.0
    assert topics["salah"].mastery_tier == "mastered"

    qtypes = {q.question_type: q for q in res.challenging_question_types}
    assert qtypes[QuestionType.scenario_ruling].is_challenging is True
    assert len(res.madhhab_comparisons) == 2
    assert res.peer_comparison.percentile_estimate > 50.0
    assert any("zakat" in r for r in res.recommendations)


def test_quiz_analytics_endpoint_roundtrip() -> None:
    resp = client.post("/quiz-analytics/analyze", json=_sample_request().model_dump())
    assert resp.status_code == 200
    data = resp.json()
    assert data["learner_id"] == "learner-42"
    assert "topic_accuracy_bars" in data["chart_series"]
