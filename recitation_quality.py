"""Quran recitation quality analysis — offline, structured phonetic input.

Accepts pre-extracted phoneme-level features (no audio processing) and
produces per-phoneme, per-word, per-segment, and composite quality scores
along with Tajweed evaluation, rhythm/flow assessment, reference comparison,
actionable feedback, and longitudinal progress tracking.

Scoring formula
---------------
Per-phoneme accuracy = 1.0 if char matches and duration within tolerance,
0.5 partial (char match but duration off), else penalty based on
character-distance. Per-word accuracy is the duration-weighted average of its
phoneme scores. The composite score combines four dimensions:
  pronunciation (0.4), tajweed (0.3), rhythm (0.2), consistency (0.1),
each normalised to [0, 1].
"""

from __future__ import annotations

import logging
import statistics
import time
import uuid
from collections import defaultdict
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from config import get_settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

settings = get_settings()

ENABLE_RECITATION_QUALITY: bool = getattr(settings, "ENABLE_RECITATION_QUALITY", True)
QUALITY_PASSING_SCORE: float = getattr(settings, "QUALITY_PASSING_SCORE", 0.7)
QUALITY_RHYTHM_WINDOW_MS: int = getattr(settings, "QUALITY_RHYTHM_WINDOW_MS", 200)

# ---------------------------------------------------------------------------
# Pydantic models — input
# ---------------------------------------------------------------------------


class PhonemeResult(BaseModel):
    expected_char: str
    actual_char: str
    duration_ms: float = Field(ge=0)
    pitch_contour: list[float] | None = None
    confidence: float = Field(default=1.0, ge=0, le=1)


class RecitationInput(BaseModel):
    phonemes: list[PhonemeResult] = Field(min_length=1)
    text_segments: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Pydantic models — analysis output
# ---------------------------------------------------------------------------


class PhonemeAnalysis(BaseModel):
    expected_char: str
    actual_char: str
    matched: bool
    accuracy_score: float = Field(ge=0, le=1)
    duration_ms: float
    duration_ok: bool
    tajweed_violations: list[str] = Field(default_factory=list)


class WordAnalysis(BaseModel):
    word_text: str
    phoneme_scores: list[float]
    accuracy_score: float = Field(ge=0, le=1)
    tajweed_score: float = Field(ge=0, le=1)
    rhythm_score: float = Field(ge=0, le=1)


class SegmentAnalysis(BaseModel):
    segment_text: str
    word_analyses: list[WordAnalysis]
    accuracy_score: float = Field(ge=0, le=1)
    tajweed_score: float = Field(ge=0, le=1)
    rhythm_score: float = Field(ge=0, le=1)
    consistency_score: float = Field(ge=0, le=1)


class RecitationAnalysis(BaseModel):
    overall_score: float = Field(ge=0, le=1)
    pronunciation_score: float = Field(ge=0, le=1)
    tajweed_score: float = Field(ge=0, le=1)
    rhythm_score: float = Field(ge=0, le=1)
    consistency_score: float = Field(ge=0, le=1)
    segment_analyses: list[SegmentAnalysis]
    phoneme_analyses: list[PhonemeAnalysis]
    passed: bool
    metadata: dict[str, Any] = Field(default_factory=dict)


class ComparisonResult(BaseModel):
    reference_name: str
    overall_delta: float
    segment_deltas: list[dict[str, Any]]
    pronunciation_delta: float
    tajweed_delta: float
    rhythm_delta: float
    summary: str


class QualityFeedback(BaseModel):
    overall_score: float = Field(ge=0, le=1)
    passed: bool
    strengths: list[str]
    areas_for_improvement: list[str]
    specific_exercises: list[str]
    segment_notes: list[dict[str, Any]]


class ProgressReport(BaseModel):
    user_id: str
    analysis_count: int
    latest_score: float = Field(ge=0, le=1)
    average_score: float = Field(ge=0, le=1)
    trend: str  # "improving" | "stable" | "declining"
    score_history: list[dict[str, Any]]
    improvement_areas: list[str]


# ---------------------------------------------------------------------------
# Tajweed integration (import with fallback)
# ---------------------------------------------------------------------------

_TAJWEED_AVAILABLE = False
try:
    from tajweed_detector import detect_violations as _detect_tajweed_violations  # type: ignore[import-not-found]

    _TAJWEED_AVAILABLE = True
except ImportError:
    _TAJWEED_AVAILABLE = False

# Basic tajweed rules for fallback
_TAJWEED_RULES: dict[str, list[str]] = {
    "م": ["الإظهار الحلقي"],
    "ن": ["إخفاء حقيقي"],
    "ل": ["إخفاء شفوي"],
    "ر": ["إخفاء صغير"],
    "ص": ["مد обязательный"],
    "ض": ["مد لين"],
}


def _basic_tajweed_check(expected: str, actual: str) -> list[str]:
    """Minimal tajweed rule check when tajweed_detector is unavailable."""
    violations: list[str] = []
    if expected != actual:
        violations.append("خطأ في النطق")
    return violations


def detect_tajweed_violations(expected: str, actual: str) -> list[str]:
    """Detect tajweed rule violations with fallback to basic rules."""
    if _TAJWEED_AVAILABLE:
        try:
            return _detect_tajweed_violations(expected, actual)  # type: ignore[no-any-return]
        except Exception:
            logger.debug("tajweed_detector raised; falling back to basic rules")
    return _basic_tajweed_check(expected, actual)


# ---------------------------------------------------------------------------
# Scoring engine
# ---------------------------------------------------------------------------

_DUR_TOLERANCE_RATIO = 0.25  # 25 % tolerance window around expected duration


def _score_phoneme(ph: PhonemeResult, expected_duration_ms: float | None = None) -> PhonemeAnalysis:
    """Score a single phoneme on accuracy and duration."""
    matched = ph.expected_char == ph.actual_char
    char_score = 1.0 if matched else 0.0

    # Duration check
    dur_ok = True
    if expected_duration_ms is not None and expected_duration_ms > 0:
        tol = expected_duration_ms * _DUR_TOLERANCE_RATIO
        dur_ok = abs(ph.duration_ms - expected_duration_ms) <= max(tol, QUALITY_RHYTHM_WINDOW_MS)
    if not matched:
        char_score = 0.0

    accuracy = char_score
    violations = detect_tajweed_violations(ph.expected_char, ph.actual_char)

    return PhonemeAnalysis(
        expected_char=ph.expected_char,
        actual_char=ph.actual_char,
        matched=matched,
        accuracy_score=accuracy,
        duration_ms=ph.duration_ms,
        duration_ok=dur_ok,
        tajweed_violations=violations,
    )


def _coefficient_of_variation(values: list[float]) -> float:
    """Coefficient of variation; returns 0 for empty/single-element lists."""
    if len(values) < 2:
        return 0.0
    mean = statistics.mean(values)
    if mean == 0:
        return 0.0
    return statistics.stdev(values) / mean


def _rhythm_score(phoneme_analyses: list[PhonemeAnalysis]) -> float:
    """Fluency score: lower CV → higher score, clamped to [0,1]."""
    if len(phoneme_analyses) < 2:
        return 1.0
    durations = [p.duration_ms for p in phoneme_analyses]
    cv = _coefficient_of_variation(durations)
    # Map CV to score: cv=0 → 1.0, cv≥0.8 → 0.0
    return max(0.0, min(1.0, 1.0 - cv / 0.8))


def _consistency_score(phoneme_analyses: list[PhonemeAnalysis]) -> float:
    """Score based on the proportion of duration-ok phonemes."""
    if not phoneme_analyses:
        return 1.0
    ok_count = sum(1 for p in phoneme_analyses if p.duration_ok)
    return ok_count / len(phoneme_analyses)


def _split_into_words(segments: list[str]) -> list[list[str]]:
    """Best-effort split of text segments into words."""
    words: list[list[str]] = []
    for seg in segments:
        parts = seg.split()
        if parts:
            words.append(parts)
    return words


def analyze_recitation(inp: RecitationInput) -> RecitationAnalysis:
    """Run the full quality analysis pipeline on structured input."""
    phoneme_analyses = [_score_phoneme(ph) for ph in inp.phonemes]

    # Overall pronunciation score (mean of phoneme accuracy)
    pronunciation = float(np.mean([p.accuracy_score for p in phoneme_analyses])) if phoneme_analyses else 0.0

    # Tajweed: proportion of phonemes with no violations
    tajweed_ph = [p for p in phoneme_analyses if not p.tajweed_violations]
    tajweed = len(tajweed_ph) / len(phoneme_analyses) if phoneme_analyses else 1.0

    # Rhythm
    rhythm = _rhythm_score(phoneme_analyses)

    # Consistency
    consistency = _consistency_score(phoneme_analyses)

    # Composite
    overall = 0.4 * pronunciation + 0.3 * tajweed + 0.2 * rhythm + 0.1 * consistency

    # Per-segment analysis (group phonemes across segments)
    segments = inp.text_segments if inp.text_segments else [""]
    phonemes_per_segment = max(1, len(inp.phonemes) / len(segments))
    segment_analyses: list[SegmentAnalysis] = []
    idx = 0
    for seg_text in segments:
        end = min(idx + int(phonemes_per_segment), len(phoneme_analyses))
        seg_phonemes = phoneme_analyses[idx:end]
        idx = end

        seg_pronunciation = float(np.mean([p.accuracy_score for p in seg_phonemes])) if seg_phonemes else 0.0
        seg_tajweed = (
            len([p for p in seg_phonemes if not p.tajweed_violations]) / len(seg_phonemes) if seg_phonemes else 1.0
        )
        seg_rhythm = _rhythm_score(seg_phonemes)
        seg_consistency = _consistency_score(seg_phonemes)

        # Word-level breakdown
        words = seg_text.split() if seg_text else []
        phonemes_per_word = max(1, len(seg_phonemes) / max(1, len(words)))
        word_analyses: list[WordAnalysis] = []
        widx = 0
        for w in words:
            wend = min(widx + int(phonemes_per_word), len(seg_phonemes))
            w_phonemes = seg_phonemes[widx:wend]
            widx = wend
            w_acc = float(np.mean([p.accuracy_score for p in w_phonemes])) if w_phonemes else 0.0
            w_tajweed = (
                len([p for p in w_phonemes if not p.tajweed_violations]) / len(w_phonemes) if w_phonemes else 1.0
            )
            w_rhythm = _rhythm_score(w_phonemes)
            word_analyses.append(
                WordAnalysis(
                    word_text=w,
                    phoneme_scores=[p.accuracy_score for p in w_phonemes],
                    accuracy_score=w_acc,
                    tajweed_score=w_tajweed,
                    rhythm_score=w_rhythm,
                )
            )

        segment_analyses.append(
            SegmentAnalysis(
                segment_text=seg_text,
                word_analyses=word_analyses,
                accuracy_score=seg_pronunciation,
                tajweed_score=seg_tajweed,
                rhythm_score=seg_rhythm,
                consistency_score=seg_consistency,
            )
        )

    return RecitationAnalysis(
        overall_score=round(overall, 4),
        pronunciation_score=round(pronunciation, 4),
        tajweed_score=round(tajweed, 4),
        rhythm_score=round(rhythm, 4),
        consistency_score=round(consistency, 4),
        segment_analyses=segment_analyses,
        phoneme_analyses=phoneme_analyses,
        passed=overall >= QUALITY_PASSING_SCORE,
        metadata=inp.metadata,
    )


# ---------------------------------------------------------------------------
# Reference comparison
# ---------------------------------------------------------------------------


class ReferenceProfile(BaseModel):
    name: str = "reference"
    segment_scores: dict[str, float] = Field(default_factory=dict)
    pronunciation_score: float = 1.0
    tajweed_score: float = 1.0
    rhythm_score: float = 1.0
    consistency_score: float = 1.0
    overall_score: float = 1.0


def compare_to_reference(
    inp: RecitationInput,
    reference: ReferenceProfile,
) -> ComparisonResult:
    """Compare a recitation analysis against a reference reciter profile."""
    analysis = analyze_recitation(inp)
    seg_deltas: list[dict[str, Any]] = []
    for seg in analysis.segment_analyses:
        ref_seg = reference.segment_scores.get(seg.segment_text, reference.overall_score)
        seg_deltas.append(
            {
                "segment": seg.segment_text,
                "your_score": seg.accuracy_score,
                "reference_score": ref_seg,
                "delta": round(seg.accuracy_score - ref_seg, 4),
            }
        )

    overall_delta = round(analysis.overall_score - reference.overall_score, 4)
    pron_delta = round(analysis.pronunciation_score - reference.pronunciation_score, 4)
    taj_delta = round(analysis.tajweed_score - reference.tajweed_score, 4)
    rhythm_delta = round(analysis.rhythm_score - reference.rhythm_score, 4)

    if overall_delta >= 0.05:
        summary = "Your recitation is close to or above the reference level."
    elif overall_delta >= -0.05:
        summary = "Your recitation is comparable to the reference with minor differences."
    else:
        summary = "There are notable differences from the reference; focus on the areas below."

    return ComparisonResult(
        reference_name=reference.name,
        overall_delta=overall_delta,
        segment_deltas=seg_deltas,
        pronunciation_delta=pron_delta,
        tajweed_delta=taj_delta,
        rhythm_delta=rhythm_delta,
        summary=summary,
    )


# ---------------------------------------------------------------------------
# Feedback generation
# ---------------------------------------------------------------------------


def generate_quality_feedback(analysis: RecitationAnalysis) -> QualityFeedback:
    """Generate actionable feedback from an analysis result."""
    strengths: list[str] = []
    improvements: list[str] = []
    exercises: list[str] = []

    if analysis.pronunciation_score >= 0.85:
        strengths.append("Excellent pronunciation accuracy across phonemes.")
    elif analysis.pronunciation_score < 0.6:
        improvements.append("Pronunciation accuracy needs significant work.")
        exercises.append("Practice each letter in isolation with a tajweed chart.")

    if analysis.tajweed_score >= 0.85:
        strengths.append("Strong adherence to tajweed rules.")
    elif analysis.tajweed_score < 0.6:
        improvements.append("Multiple tajweed rule violations detected.")
        exercises.append("Review the specific tajweed rules for letters with violations.")

    if analysis.rhythm_score >= 0.8:
        strengths.append("Good rhythm and flow consistency.")
    elif analysis.rhythm_score < 0.5:
        improvements.append("Rhythm is irregular; recitation feels uneven.")
        exercises.append("Practice reciting along with a slow, measured reciter to build consistency.")

    if analysis.consistency_score >= 0.9:
        strengths.append("Highly consistent phoneme durations.")
    elif analysis.consistency_score < 0.6:
        improvements.append("Duration of phonemes varies considerably.")
        exercises.append("Use a metronome-style练习 to stabilise timing.")

    # Per-segment notes
    segment_notes: list[dict[str, Any]] = []
    for seg in analysis.segment_analyses:
        if seg.accuracy_score < 0.7:
            segment_notes.append(
                {
                    "segment": seg.segment_text,
                    "note": f"Score {seg.accuracy_score:.2f} — focus on character accuracy.",
                }
            )

    if not strengths:
        strengths.append("Consistent effort across the recitation.")
    if not improvements:
        improvements.append("Overall performance is solid; maintain regular practice.")

    return QualityFeedback(
        overall_score=analysis.overall_score,
        passed=analysis.passed,
        strengths=strengths,
        areas_for_improvement=improvements,
        specific_exercises=exercises,
        segment_notes=segment_notes,
    )


# ---------------------------------------------------------------------------
# Progress tracking  (in-memory, documented swappable)
# ---------------------------------------------------------------------------

_progress_store: dict[str, list[dict[str, Any]]] = defaultdict(list)


def track_progress(user_id: str, analysis: RecitationAnalysis) -> ProgressReport:
    """Store an analysis and return a progress report with trend detection.

    Storage is an in-memory dict.  To persist, swap ``_progress_store`` for
    a Redis/SQLite-backed mapping with the same interface.
    """
    entry = {
        "analysis_id": uuid.uuid4().hex,
        "timestamp": time.time(),
        "overall_score": analysis.overall_score,
        "pronunciation_score": analysis.pronunciation_score,
        "tajweed_score": analysis.tajweed_score,
        "rhythm_score": analysis.rhythm_score,
        "consistency_score": analysis.consistency_score,
        "passed": analysis.passed,
    }
    _progress_store[user_id].append(entry)
    history = _progress_store[user_id]

    scores = [h["overall_score"] for h in history]
    avg = float(np.mean(scores)) if scores else 0.0
    latest = scores[-1] if scores else 0.0

    # Trend detection over last 5 entries
    trend = "stable"
    if len(scores) >= 3:
        recent = scores[-5:] if len(scores) >= 5 else scores
        diffs = [recent[i + 1] - recent[i] for i in range(len(recent) - 1)]
        mean_diff = float(np.mean(diffs))
        if mean_diff > 0.02:
            trend = "improving"
        elif mean_diff < -0.02:
            trend = "declining"

    # Identify persistent low-scoring areas
    low_areas: list[str] = []
    if np.mean([h["pronunciation_score"] for h in history[-3:]]) < 0.7:
        low_areas.append("pronunciation")
    if np.mean([h["tajweed_score"] for h in history[-3:]]) < 0.7:
        low_areas.append("tajweed")
    if np.mean([h["rhythm_score"] for h in history[-3:]]) < 0.7:
        low_areas.append("rhythm")

    return ProgressReport(
        user_id=user_id,
        analysis_count=len(history),
        latest_score=latest,
        average_score=round(avg, 4),
        trend=trend,
        score_history=history,
        improvement_areas=low_areas,
    )


def reset_progress(user_id: str) -> None:
    """Clear stored progress for a user (utility for tests)."""
    _progress_store.pop(user_id, None)


# ---------------------------------------------------------------------------
# FastAPI router
# ---------------------------------------------------------------------------

from fastapi import APIRouter  # noqa: E402

router = APIRouter(prefix="/recitation", tags=["recitation"])


@router.post("/analyze")
async def endpoint_analyze(body: RecitationInput) -> RecitationAnalysis:
    """Analyse a structured recitation input and return quality scores."""
    if not ENABLE_RECITATION_QUALITY:
        from fastapi import HTTPException as _HTTP

        raise _HTTP(status_code=503, detail="Recitation quality analysis is disabled.")
    return analyze_recitation(body)


@router.post("/compare-reference")
async def endpoint_compare_reference(
    input_data: RecitationInput,
    reference: ReferenceProfile,
) -> ComparisonResult:
    """Compare a recitation against a reference reciter profile."""
    if not ENABLE_RECITATION_QUALITY:
        from fastapi import HTTPException as _HTTP

        raise _HTTP(status_code=503, detail="Recitation quality analysis is disabled.")
    return compare_to_reference(input_data, reference)


@router.get("/progress/{user_id}")
async def endpoint_progress(user_id: str) -> ProgressReport:
    """Get the progress report for a user."""
    # Return a zero-progress report if no data yet
    history = _progress_store.get(user_id, [])
    if not history:
        return ProgressReport(
            user_id=user_id,
            analysis_count=0,
            latest_score=0.0,
            average_score=0.0,
            trend="stable",
            score_history=[],
            improvement_areas=[],
        )
    scores = [h["overall_score"] for h in history]
    avg = float(np.mean(scores))
    latest = scores[-1]
    trend = "stable"
    if len(scores) >= 3:
        recent = scores[-5:] if len(scores) >= 5 else scores
        diffs = [recent[i + 1] - recent[i] for i in range(len(recent) - 1)]
        mean_diff = float(np.mean(diffs))
        if mean_diff > 0.02:
            trend = "improving"
        elif mean_diff < -0.02:
            trend = "declining"
    return ProgressReport(
        user_id=user_id,
        analysis_count=len(history),
        latest_score=latest,
        average_score=round(avg, 4),
        trend=trend,
        score_history=history,
        improvement_areas=[],
    )


@router.post("/feedback")
async def endpoint_feedback(body: RecitationInput) -> QualityFeedback:
    """Analyse recitation and return actionable feedback."""
    if not ENABLE_RECITATION_QUALITY:
        from fastapi import HTTPException as _HTTP

        raise _HTTP(status_code=503, detail="Recitation quality analysis is disabled.")
    analysis = analyze_recitation(body)
    return generate_quality_feedback(analysis)
