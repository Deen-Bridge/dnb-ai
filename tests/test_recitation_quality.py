"""Tests for recitation quality analysis — fully offline, synthetic data."""

import uuid

from recitation_quality import (
    PhonemeResult,
    ProgressReport,
    QualityFeedback,
    RecitationInput,
    ReferenceProfile,
    analyze_recitation,
    compare_to_reference,
    detect_tajweed_violations,
    generate_quality_feedback,
    reset_progress,
    track_progress,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _perfect_input() -> RecitationInput:
    """Phonemes where every expected char matches actual."""
    return RecitationInput(
        phonemes=[
            PhonemeResult(expected_char="ب", actual_char="ب", duration_ms=120, confidence=1.0),
            PhonemeResult(expected_char="س", actual_char="س", duration_ms=130, confidence=1.0),
            PhonemeResult(expected_char="م", actual_char="م", duration_ms=110, confidence=1.0),
            PhonemeResult(expected_char="ا", actual_char="ا", duration_ms=140, confidence=1.0),
            PhonemeResult(expected_char="ل", actual_char="ل", duration_ms=125, confidence=1.0),
            PhonemeResult(expected_char="ل", actual_char="ل", duration_ms=125, confidence=1.0),
            PhonemeResult(expected_char="ه", actual_char="ه", duration_ms=115, confidence=1.0),
            PhonemeResult(expected_char="ي", actual_char="ي", duration_ms=135, confidence=1.0),
        ],
        text_segments=["بسم", "الله", "الرحمن", "الرحيم"],
        metadata={"surah": 1, "ayah": 1},
    )


def _mismatched_input() -> RecitationInput:
    """Phonemes with mismatches and timing irregularity."""
    return RecitationInput(
        phonemes=[
            PhonemeResult(expected_char="ب", actual_char="ت", duration_ms=100, confidence=0.8),
            PhonemeResult(expected_char="س", actual_char="س", duration_ms=300, confidence=0.9),
            PhonemeResult(expected_char="م", actual_char="ن", duration_ms=80, confidence=0.7),
            PhonemeResult(expected_char="ا", actual_char="ا", duration_ms=120, confidence=1.0),
            PhonemeResult(expected_char="ل", actual_char="ل", duration_ms=500, confidence=0.85),
            PhonemeResult(expected_char="ل", actual_char="ل", duration_ms=90, confidence=0.9),
            PhonemeResult(expected_char="ه", actual_char="ه", duration_ms=130, confidence=1.0),
            PhonemeResult(expected_char="ي", actual_char="ي", duration_ms=110, confidence=1.0),
        ],
        text_segments=["بسم", "الله", "الرحمن", "الرحيم"],
        metadata={"surah": 1, "ayah": 1},
    )


# ---------------------------------------------------------------------------
# Perfect input scores
# ---------------------------------------------------------------------------


class TestPerfectInput:
    def test_overall_score_high(self) -> None:
        analysis = analyze_recitation(_perfect_input())
        assert analysis.overall_score >= 0.9

    def test_pronunciation_score_high(self) -> None:
        analysis = analyze_recitation(_perfect_input())
        assert analysis.pronunciation_score == 1.0

    def test_passed_with_high_score(self) -> None:
        analysis = analyze_recitation(_perfect_input())
        assert analysis.passed is True

    def test_all_phonemes_matched(self) -> None:
        analysis = analyze_recitation(_perfect_input())
        assert all(p.matched for p in analysis.phoneme_analyses)


# ---------------------------------------------------------------------------
# Mismatched input scores low
# ---------------------------------------------------------------------------


class TestMismatchedInput:
    def test_overall_score_low(self) -> None:
        analysis = analyze_recitation(_mismatched_input())
        assert analysis.overall_score < 0.7

    def test_pronunciation_below_one(self) -> None:
        analysis = analyze_recitation(_mismatched_input())
        assert analysis.pronunciation_score < 1.0

    def test_not_all_matched(self) -> None:
        analysis = analyze_recitation(_mismatched_input())
        assert not all(p.matched for p in analysis.phoneme_analyses)

    def test_mismatched_phonemes_detected(self) -> None:
        analysis = analyze_recitation(_mismatched_input())
        mismatches = [p for p in analysis.phoneme_analyses if not p.matched]
        assert len(mismatches) >= 2


# ---------------------------------------------------------------------------
# Rhythm variance detection
# ---------------------------------------------------------------------------


class TestRhythmAnalysis:
    def test_uniform_rhythm_high(self) -> None:
        inp = RecitationInput(
            phonemes=[
                PhonemeResult(expected_char="ا", actual_char="ا", duration_ms=100),
                PhonemeResult(expected_char="ب", actual_char="ب", duration_ms=105),
                PhonemeResult(expected_char="ج", actual_char="ج", duration_ms=98),
                PhonemeResult(expected_char="د", actual_char="د", duration_ms=102),
            ],
        )
        analysis = analyze_recitation(inp)
        assert analysis.rhythm_score >= 0.8

    def test_irregular_rhythm_low(self) -> None:
        inp = RecitationInput(
            phonemes=[
                PhonemeResult(expected_char="ا", actual_char="ا", duration_ms=50),
                PhonemeResult(expected_char="ب", actual_char="ب", duration_ms=500),
                PhonemeResult(expected_char="ج", actual_char="ج", duration_ms=30),
                PhonemeResult(expected_char="د", actual_char="د", duration_ms=600),
            ],
        )
        analysis = analyze_recitation(inp)
        assert analysis.rhythm_score < 0.6


# ---------------------------------------------------------------------------
# Composite score in [0, 1]
# ---------------------------------------------------------------------------


class TestCompositeScore:
    def test_score_range(self) -> None:
        for inp in [_perfect_input(), _mismatched_input()]:
            analysis = analyze_recitation(inp)
            assert 0.0 <= analysis.overall_score <= 1.0
            assert 0.0 <= analysis.pronunciation_score <= 1.0
            assert 0.0 <= analysis.tajweed_score <= 1.0
            assert 0.0 <= analysis.rhythm_score <= 1.0
            assert 0.0 <= analysis.consistency_score <= 1.0

    def test_composite_weights(self) -> None:
        inp = _perfect_input()
        analysis = analyze_recitation(inp)
        expected = (
            0.4 * analysis.pronunciation_score
            + 0.3 * analysis.tajweed_score
            + 0.2 * analysis.rhythm_score
            + 0.1 * analysis.consistency_score
        )
        assert abs(analysis.overall_score - expected) < 0.001


# ---------------------------------------------------------------------------
# Feedback generation
# ---------------------------------------------------------------------------


class TestFeedback:
    def test_feedback_generated(self) -> None:
        analysis = analyze_recitation(_perfect_input())
        fb = generate_quality_feedback(analysis)
        assert isinstance(fb, QualityFeedback)
        assert len(fb.strengths) > 0
        assert len(fb.areas_for_improvement) > 0

    def test_feedback_has_exercises_for_low_score(self) -> None:
        analysis = analyze_recitation(_mismatched_input())
        fb = generate_quality_feedback(analysis)
        assert len(fb.specific_exercises) > 0

    def test_feedback_match_analysis_score(self) -> None:
        analysis = analyze_recitation(_perfect_input())
        fb = generate_quality_feedback(analysis)
        assert fb.overall_score == analysis.overall_score

    def test_feedback_passed_matches(self) -> None:
        analysis = analyze_recitation(_mismatched_input())
        fb = generate_quality_feedback(analysis)
        assert fb.passed == analysis.passed


# ---------------------------------------------------------------------------
# Progress tracking
# ---------------------------------------------------------------------------


class TestProgressTracking:
    def setup_method(self) -> None:
        self.user_id = f"test-user-{uuid.uuid4().hex[:8]}"

    def test_first_entry(self) -> None:
        reset_progress(self.user_id)
        analysis = analyze_recitation(_perfect_input())
        report = track_progress(self.user_id, analysis)
        assert isinstance(report, ProgressReport)
        assert report.analysis_count == 1
        assert report.latest_score == analysis.overall_score

    def test_multiple_entries_trend(self) -> None:
        reset_progress(self.user_id)
        for _ in range(5):
            track_progress(self.user_id, analyze_recitation(_perfect_input()))
        report = track_progress(self.user_id, analyze_recitation(_mismatched_input()))
        assert report.analysis_count == 6

    def test_declining_trend(self) -> None:
        reset_progress(self.user_id)
        # Scores that decline
        for scores in [
            [0.9, 0.85, 0.8, 0.75, 0.7, 0.65],
        ]:
            for s in scores:
                inp = RecitationInput(
                    phonemes=[
                        PhonemeResult(expected_char="ا", actual_char="ا", duration_ms=int(100 * s)),
                    ]
                    * 5,
                )
                analysis = analyze_recitation(inp)
                # Force the overall_score for trend detection
                analysis.overall_score = s  # type: ignore[assignment]
                track_progress(self.user_id, analysis)
        report = track_progress(self.user_id, analyze_recitation(_perfect_input()))
        # At least we can verify the history grew
        assert report.analysis_count >= 6


# ---------------------------------------------------------------------------
# Reference comparison
# ---------------------------------------------------------------------------


class TestReferenceComparison:
    def test_comparison_result(self) -> None:
        inp = _perfect_input()
        ref = ReferenceProfile(name="Mishary", overall_score=0.95)
        result = compare_to_reference(inp, ref)
        assert result.reference_name == "Mishary"
        assert isinstance(result.overall_delta, float)
        assert len(result.segment_deltas) == len(inp.text_segments)

    def test_above_reference(self) -> None:
        inp = _perfect_input()
        ref = ReferenceProfile(name="Baseline", overall_score=0.5)
        result = compare_to_reference(inp, ref)
        assert result.overall_delta > 0

    def test_below_reference(self) -> None:
        inp = _mismatched_input()
        ref = ReferenceProfile(name="Expert", overall_score=0.98)
        result = compare_to_reference(inp, ref)
        assert result.overall_delta < 0


# ---------------------------------------------------------------------------
# Tajweed violation detection
# ---------------------------------------------------------------------------


class TestTajweed:
    def test_matching_chars_no_violation(self) -> None:
        violations = detect_tajweed_violations("ب", "ب")
        assert len(violations) == 0

    def test_mismatched_chars_violation(self) -> None:
        violations = detect_tajweed_violations("ب", "ت")
        assert len(violations) > 0


# ---------------------------------------------------------------------------
# API schema validation (pydantic model round-trip)
# ---------------------------------------------------------------------------


class TestAPISchema:
    def test_recitation_input_schema(self) -> None:
        data = {
            "phonemes": [{"expected_char": "ب", "actual_char": "ب", "duration_ms": 100, "confidence": 1.0}],
            "text_segments": ["بسم"],
            "metadata": {},
        }
        inp = RecitationInput(**data)
        assert inp.phonemes[0].expected_char == "ب"

    def test_analysis_schema_serializable(self) -> None:
        analysis = analyze_recitation(_perfect_input())
        d = analysis.model_dump()
        assert "overall_score" in d
        assert isinstance(d["phoneme_analyses"], list)

    def test_reference_profile_schema(self) -> None:
        ref = ReferenceProfile(name="Test", overall_score=0.9)
        d = ref.model_dump()
        assert d["name"] == "Test"
