"""Quiz Performance Analytics Subsystem (#257).

Provides topic-level accuracy insights, temporal performance trend analysis,
challenging question-type detection, madhhab comparison breakdowns, time-spent
efficiency metrics, anonymized peer percentile comparison, and personalized
study recommendations.
"""

from __future__ import annotations

import logging
import math
from enum import Enum
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/quiz-analytics", tags=["quiz-analytics"])


class QuestionType(str, Enum):
    multiple_choice = "multiple_choice"
    true_false = "true_false"
    scenario_ruling = "scenario_ruling"
    evidence_matching = "evidence_matching"
    arabic_term = "arabic_term"


class QuizQuestionAttempt(BaseModel):
    question_id: str = Field(..., min_length=1, max_length=120)
    topic: str = Field(..., min_length=1, max_length=80)
    question_type: QuestionType = QuestionType.multiple_choice
    madhhab: str | None = Field(default=None, max_length=40)
    is_correct: bool
    time_spent_seconds: float = Field(..., ge=0.0, le=3600.0)
    difficulty: int = Field(default=2, ge=1, le=5)


class QuizSessionRecord(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=120)
    timestamp_iso: str = Field(..., min_length=4, max_length=64)
    attempts: list[QuizQuestionAttempt] = Field(..., min_length=1, max_length=200)


class QuizAnalyticsRequest(BaseModel):
    learner_id: str = Field(..., min_length=1, max_length=120)
    sessions: list[QuizSessionRecord] = Field(..., min_length=1, max_length=100)
    peer_average_accuracy: float | None = Field(default=0.72, ge=0.0, le=1.0)
    peer_std_dev: float | None = Field(default=0.14, gt=0.0, le=1.0)


class TopicPerformanceInsight(BaseModel):
    topic: str
    total_questions: int
    correct_questions: int
    accuracy: float
    avg_time_seconds: float
    mastery_tier: str
    needs_attention: bool


class QuestionTypeInsight(BaseModel):
    question_type: QuestionType
    attempts: int
    accuracy: float
    avg_time_seconds: float
    is_challenging: bool


class MadhhabPerformanceComparison(BaseModel):
    madhhab: str
    attempts: int
    accuracy: float
    avg_time_seconds: float


class TrendPoint(BaseModel):
    session_id: str
    timestamp_iso: str
    accuracy: float
    avg_time_seconds: float


class PeerComparisonSummary(BaseModel):
    learner_accuracy: float
    peer_mean_accuracy: float
    percentile_estimate: float
    standing: str


class QuizAnalyticsResponse(BaseModel):
    learner_id: str
    overall_accuracy: float
    total_questions: int
    total_time_seconds: float
    trend_direction: str
    trend_slope: float
    trend_series: list[TrendPoint]
    topic_insights: list[TopicPerformanceInsight]
    challenging_question_types: list[QuestionTypeInsight]
    madhhab_comparisons: list[MadhhabPerformanceComparison]
    peer_comparison: PeerComparisonSummary
    recommendations: list[str]
    chart_series: dict[str, Any]


def _mastery_tier(accuracy: float) -> str:
    if accuracy >= 0.85:
        return "mastered"
    if accuracy >= 0.70:
        return "proficient"
    if accuracy >= 0.50:
        return "developing"
    return "needs_review"


def _normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def analyze_quiz_performance(req: QuizAnalyticsRequest) -> QuizAnalyticsResponse:
    all_attempts: list[QuizQuestionAttempt] = []
    trend_series: list[TrendPoint] = []

    for session in req.sessions:
        s_total = len(session.attempts)
        s_correct = sum(1 for a in session.attempts if a.is_correct)
        s_time = sum(a.time_spent_seconds for a in session.attempts)
        trend_series.append(
            TrendPoint(
                session_id=session.session_id,
                timestamp_iso=session.timestamp_iso,
                accuracy=round(s_correct / s_total, 4),
                avg_time_seconds=round(s_time / s_total, 2),
            )
        )
        all_attempts.extend(session.attempts)

    total_q = len(all_attempts)
    total_correct = sum(1 for a in all_attempts if a.is_correct)
    total_time = round(sum(a.time_spent_seconds for a in all_attempts), 2)
    overall_acc = round(total_correct / total_q, 4) if total_q else 0.0

    # Trend slope via linear regression over session index
    n_sess = len(trend_series)
    if n_sess >= 2:
        mean_x = (n_sess - 1) / 2.0
        mean_y = sum(p.accuracy for p in trend_series) / n_sess
        num = sum((i - mean_x) * (p.accuracy - mean_y) for i, p in enumerate(trend_series))
        den = sum((i - mean_x) ** 2 for i in range(n_sess))
        slope = round(num / den, 4) if den > 0 else 0.0
    else:
        slope = 0.0

    if slope > 0.03:
        trend_direction = "improving"
    elif slope < -0.03:
        trend_direction = "declining"
    else:
        trend_direction = "stable"

    # Topic breakdown
    by_topic: dict[str, list[QuizQuestionAttempt]] = {}
    for att in all_attempts:
        by_topic.setdefault(att.topic, []).append(att)

    topic_insights: list[TopicPerformanceInsight] = []
    for topic, items in sorted(by_topic.items()):
        t_total = len(items)
        t_corr = sum(1 for a in items if a.is_correct)
        t_acc = round(t_corr / t_total, 4)
        t_time = round(sum(a.time_spent_seconds for a in items) / t_total, 2)
        topic_insights.append(
            TopicPerformanceInsight(
                topic=topic,
                total_questions=t_total,
                correct_questions=t_corr,
                accuracy=t_acc,
                avg_time_seconds=t_time,
                mastery_tier=_mastery_tier(t_acc),
                needs_attention=t_acc < 0.70,
            )
        )

    # Question type breakdown
    by_qtype: dict[QuestionType, list[QuizQuestionAttempt]] = {}
    for att in all_attempts:
        by_qtype.setdefault(att.question_type, []).append(att)

    qtype_insights: list[QuestionTypeInsight] = []
    for qtype, items in sorted(by_qtype.items(), key=lambda kv: kv[0].value):
        q_total = len(items)
        q_corr = sum(1 for a in items if a.is_correct)
        q_acc = round(q_corr / q_total, 4)
        q_time = round(sum(a.time_spent_seconds for a in items) / q_total, 2)
        qtype_insights.append(
            QuestionTypeInsight(
                question_type=qtype,
                attempts=q_total,
                accuracy=q_acc,
                avg_time_seconds=q_time,
                is_challenging=q_acc < 0.65 or q_time > 45.0,
            )
        )

    # Madhhab breakdown
    by_madhhab: dict[str, list[QuizQuestionAttempt]] = {}
    for att in all_attempts:
        if att.madhhab:
            by_madhhab.setdefault(att.madhhab.lower(), []).append(att)

    madhhab_comparisons: list[MadhhabPerformanceComparison] = []
    for madhhab, items in sorted(by_madhhab.items()):
        m_total = len(items)
        m_corr = sum(1 for a in items if a.is_correct)
        madhhab_comparisons.append(
            MadhhabPerformanceComparison(
                madhhab=madhhab,
                attempts=m_total,
                accuracy=round(m_corr / m_total, 4),
                avg_time_seconds=round(sum(a.time_spent_seconds for a in items) / m_total, 2),
            )
        )

    peer_mean = req.peer_average_accuracy if req.peer_average_accuracy is not None else 0.72
    peer_sd = req.peer_std_dev if req.peer_std_dev is not None else 0.14
    z_score = (overall_acc - peer_mean) / peer_sd
    percentile = round(min(99.9, max(0.1, _normal_cdf(z_score) * 100.0)), 1)
    if percentile >= 75.0:
        standing = "above_average"
    elif percentile >= 40.0:
        standing = "on_par"
    else:
        standing = "below_average"

    recommendations: list[str] = []
    weak_topics = [t.topic for t in topic_insights if t.needs_attention]
    if weak_topics:
        recommendations.append(f"Prioritize targeted review in {', '.join(weak_topics)} where accuracy is below 70%.")
    hard_types = [q.question_type.value for q in qtype_insights if q.is_challenging]
    if hard_types:
        recommendations.append(
            f"Practice untimed drills on {', '.join(hard_types)} items to improve recall speed and precision."
        )
    if trend_direction == "declining":
        recommendations.append(
            "Recent session scores are trending downward; revisit foundational lessons before attempting advanced quizzes."
        )
    if not recommendations:
        recommendations.append(
            "Strong mastery across current topics; advance to higher-difficulty comparative fiqh and usul assessments."
        )

    chart_series = {
        "topic_accuracy_bars": {t.topic: t.accuracy for t in topic_insights},
        "session_trend_line": [{"session": p.session_id, "accuracy": p.accuracy} for p in trend_series],
        "time_by_topic_seconds": {t.topic: t.avg_time_seconds for t in topic_insights},
    }

    return QuizAnalyticsResponse(
        learner_id=req.learner_id,
        overall_accuracy=overall_acc,
        total_questions=total_q,
        total_time_seconds=total_time,
        trend_direction=trend_direction,
        trend_slope=slope,
        trend_series=trend_series,
        topic_insights=topic_insights,
        challenging_question_types=qtype_insights,
        madhhab_comparisons=madhhab_comparisons,
        peer_comparison=PeerComparisonSummary(
            learner_accuracy=overall_acc,
            peer_mean_accuracy=peer_mean,
            percentile_estimate=percentile,
            standing=standing,
        ),
        recommendations=recommendations,
        chart_series=chart_series,
    )


@router.post("/analyze", response_model=QuizAnalyticsResponse)
async def analyze_quiz_endpoint(body: QuizAnalyticsRequest) -> QuizAnalyticsResponse:
    """Compute topic, trend, madhhab, and peer analytics over a learner's quiz sessions."""
    return analyze_quiz_performance(body)
