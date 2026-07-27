"""Tests for the confidence score and the abstain / hedge / answer policy.

All offline — no model calls, no GEMINI_API_KEY.
"""

import pytest

import confidence
from confidence import (
    ABSTENTION_MESSAGE,
    NO_SIGNAL_PRIOR,
    SCHOLAR_QUEUED_NOTE,
    SIGNAL_WEIGHTS,
    UNCERTAINTY_NOTE,
    ConfidenceBand,
    ConfidenceSignals,
    apply_policy,
    assess,
    band_for,
    build_signals,
    compute_confidence,
    count_hedges,
    expressed_certainty,
    should_queue_for_scholar,
    thresholds,
)


CONFIDENT_ANSWER = (
    "The five daily prayers are Fajr, Dhuhr, Asr, Maghrib and Isha. "
    "They are obligatory upon every adult Muslim."
)

HEDGED_ANSWER = (
    "I think this might be permissible, though I'm not sure, and it could be "
    "that scholars differ — possibly on the details."
)


# ---------------------------------------------------------------------------
# Hedging detection
# ---------------------------------------------------------------------------


class TestHedging:
    def test_confident_answer_has_no_hedges(self):
        assert count_hedges(CONFIDENT_ANSWER) == 0
        assert expressed_certainty(CONFIDENT_ANSWER) == 1.0

    def test_hedged_answer_is_detected(self):
        assert count_hedges(HEDGED_ANSWER) >= 3
        assert expressed_certainty(HEDGED_ANSWER) == 0.0

    def test_certainty_decreases_with_hedges(self):
        one = expressed_certainty("I think the ruling is clear.")
        two = expressed_certainty("I think it is unclear.")
        assert 0.0 < one < 1.0
        assert two < one

    @pytest.mark.parametrize("text", ["", "   ", None])
    def test_empty_answer_has_no_certainty(self, text):
        assert expressed_certainty(text or "") == 0.0

    def test_advising_a_scholar_is_not_hedging(self):
        """Good adab is not doubt — pointing to a scholar must not cost score."""
        text = (
            "Zakat is due at 2.5% once the nisab is met. "
            "Please consult a qualified local scholar for your situation."
        )
        assert count_hedges(text) == 0


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


class TestComputeConfidence:
    def test_no_signals_uses_the_documented_prior(self):
        assert compute_confidence(ConfidenceSignals()) == pytest.approx(NO_SIGNAL_PRIOR)

    def test_single_signal_is_that_signal(self):
        signals = ConfidenceSignals(self_consistency=0.8)
        assert compute_confidence(signals) == pytest.approx(0.8)

    def test_weighted_mean_over_present_signals(self):
        signals = ConfidenceSignals(self_consistency=1.0, citation_verification=0.0)
        expected = SIGNAL_WEIGHTS["self_consistency"] / (
            SIGNAL_WEIGHTS["self_consistency"] + SIGNAL_WEIGHTS["citation_verification"]
        )
        assert compute_confidence(signals) == pytest.approx(expected, abs=1e-4)

    def test_absent_signal_is_skipped_not_treated_as_zero(self):
        with_one = compute_confidence(ConfidenceSignals(self_consistency=0.9))
        with_zero = compute_confidence(
            ConfidenceSignals(self_consistency=0.9, citation_verification=0.0)
        )
        assert with_one > with_zero

    def test_high_stakes_lowers_the_same_evidence(self):
        base = ConfidenceSignals(self_consistency=0.9, expressed_certainty=0.9)
        stakes = ConfidenceSignals(
            self_consistency=0.9, expressed_certainty=0.9, is_high_stakes=True
        )
        assert compute_confidence(stakes) < compute_confidence(base)

    def test_high_stakes_penalty_applies_once(self):
        signals = ConfidenceSignals(self_consistency=1.0, is_high_stakes=True)
        assert compute_confidence(signals) == pytest.approx(
            1.0 - confidence.HIGH_STAKES_PENALTY, abs=1e-4
        )

    def test_self_reported_certainty_alone_cannot_reach_confident(self):
        """A fluent answer nothing checked must not certify itself."""
        signals = ConfidenceSignals(expressed_certainty=1.0)
        score = compute_confidence(signals)
        assert score <= confidence.UNVERIFIED_CEILING
        assert score < confidence.CONFIDENCE_HIGH_THRESHOLD

    def test_one_external_signal_lifts_the_ceiling(self):
        capped = compute_confidence(ConfidenceSignals(expressed_certainty=1.0))
        corroborated = compute_confidence(
            ConfidenceSignals(expressed_certainty=1.0, self_consistency=1.0)
        )
        assert corroborated > capped

    def test_ceiling_does_not_raise_a_low_score(self):
        score = compute_confidence(ConfidenceSignals(expressed_certainty=0.1))
        assert score == pytest.approx(0.1)

    def test_score_is_bounded(self):
        for value in (0.0, 0.5, 1.0):
            score = compute_confidence(
                ConfidenceSignals(self_consistency=value, citation_verification=value)
            )
            assert 0.0 <= score <= 1.0

    def test_religious_flag_alone_does_not_change_the_score(self):
        """is_religious routes the answer; it is not evidence about correctness."""
        plain = ConfidenceSignals(self_consistency=0.8)
        religious = ConfidenceSignals(self_consistency=0.8, is_religious=True)
        assert compute_confidence(plain) == compute_confidence(religious)

    def test_signals_out_of_range_are_rejected(self):
        with pytest.raises(ValueError):
            ConfidenceSignals(self_consistency=1.5)


# ---------------------------------------------------------------------------
# Bands
# ---------------------------------------------------------------------------


class TestConfiguration:
    """Bad configuration must degrade to defaults, never crash on import."""

    @pytest.mark.parametrize("raw", ["not-a-number", "", "1.5", "-0.2"])
    def test_invalid_threshold_falls_back(self, monkeypatch, raw):
        monkeypatch.setenv("CONFIDENCE_LOW_THRESHOLD", raw)
        assert confidence._env_float("CONFIDENCE_LOW_THRESHOLD", 0.4) == 0.4

    @pytest.mark.parametrize("raw", ["not-a-number", "", "0", "-3"])
    def test_invalid_hedge_saturation_falls_back(self, monkeypatch, raw):
        monkeypatch.setenv("CONFIDENCE_HEDGE_SATURATION", raw)
        assert confidence._env_int("CONFIDENCE_HEDGE_SATURATION", 3) == 3

    def test_valid_values_are_read(self, monkeypatch):
        monkeypatch.setenv("CONFIDENCE_LOW_THRESHOLD", "0.25")
        monkeypatch.setenv("CONFIDENCE_HEDGE_SATURATION", "5")
        assert confidence._env_float("CONFIDENCE_LOW_THRESHOLD", 0.4) == 0.25
        assert confidence._env_int("CONFIDENCE_HEDGE_SATURATION", 3) == 5


class TestBands:
    def test_band_boundaries(self):
        low = confidence.CONFIDENCE_LOW_THRESHOLD
        high = confidence.CONFIDENCE_HIGH_THRESHOLD
        assert band_for(low - 0.01) is ConfidenceBand.ABSTAIN
        assert band_for(low) is ConfidenceBand.UNCERTAIN
        assert band_for(high - 0.01) is ConfidenceBand.UNCERTAIN
        assert band_for(high) is ConfidenceBand.CONFIDENT
        assert band_for(1.0) is ConfidenceBand.CONFIDENT

    def test_thresholds_are_configurable(self, monkeypatch):
        monkeypatch.setattr(confidence, "CONFIDENCE_LOW_THRESHOLD", 0.9)
        monkeypatch.setattr(confidence, "CONFIDENCE_HIGH_THRESHOLD", 0.95)
        assert band_for(0.8) is ConfidenceBand.ABSTAIN

    def test_thresholds_report_current_policy(self):
        policy = thresholds()
        assert policy["low"] == confidence.CONFIDENCE_LOW_THRESHOLD
        assert policy["high"] == confidence.CONFIDENCE_HIGH_THRESHOLD


# ---------------------------------------------------------------------------
# Queueing
# ---------------------------------------------------------------------------


class TestQueueing:
    def test_low_confidence_religious_answer_is_queued(self):
        signals = ConfidenceSignals(self_consistency=0.1, is_religious=True)
        assert should_queue_for_scholar(compute_confidence(signals), signals) is True

    def test_low_confidence_non_religious_answer_is_not_queued(self):
        """A scholar's time is for religious content, not general trivia."""
        signals = ConfidenceSignals(self_consistency=0.1, is_religious=False)
        assert should_queue_for_scholar(compute_confidence(signals), signals) is False

    def test_threshold_boundary_does_not_queue_silently(self):
        """Queued must never be true while the user is shown a plain hedge.

        At exactly the low threshold the band is UNCERTAIN, so with the
        default (equal) thresholds the item must not be queued either —
        otherwise the answer goes to a scholar without the user being told.
        """
        signals = ConfidenceSignals(
            self_consistency=confidence.CONFIDENCE_LOW_THRESHOLD,
            citation_verification=confidence.CONFIDENCE_LOW_THRESHOLD,
            is_religious=True,
        )
        assessment = assess(signals)
        assert assessment.score == pytest.approx(confidence.CONFIDENCE_LOW_THRESHOLD)
        assert assessment.band is ConfidenceBand.UNCERTAIN
        assert assessment.queued is False

    def test_queueing_above_the_abstain_band_still_tells_the_user(self, monkeypatch):
        """If an operator queues hedged answers too, the user is still told."""
        monkeypatch.setattr(confidence, "SCHOLAR_QUEUE_THRESHOLD", 0.9)
        signals = ConfidenceSignals(
            self_consistency=0.6, citation_verification=0.6, is_religious=True
        )
        assessment = assess(signals)
        assert assessment.band is ConfidenceBand.UNCERTAIN
        assert assessment.queued is True
        text = apply_policy("An answer.", assessment)
        assert UNCERTAINTY_NOTE in text
        assert SCHOLAR_QUEUED_NOTE.strip() in text

    def test_high_confidence_religious_answer_is_not_queued(self):
        signals = ConfidenceSignals(self_consistency=0.95, is_religious=True)
        assert should_queue_for_scholar(compute_confidence(signals), signals) is False


# ---------------------------------------------------------------------------
# Assessment and policy
# ---------------------------------------------------------------------------


class TestAssessAndPolicy:
    def test_low_confidence_religious_answer_abstains_and_queues(self):
        assessment = assess(
            ConfidenceSignals(
                self_consistency=0.1, expressed_certainty=0.0, is_religious=True
            )
        )
        assert assessment.band is ConfidenceBand.ABSTAIN
        assert assessment.abstained is True
        assert assessment.queued is True

        text = apply_policy("Some doubtful ruling.", assessment)
        assert "Some doubtful ruling" not in text
        assert text.startswith(ABSTENTION_MESSAGE[:40])
        assert "scholar" in text.lower()
        assert SCHOLAR_QUEUED_NOTE.strip() in text

    def test_low_confidence_non_religious_answer_abstains_without_queueing(self):
        assessment = assess(
            ConfidenceSignals(self_consistency=0.1, is_religious=False)
        )
        assert assessment.abstained is True
        assert assessment.queued is False
        text = apply_policy("Some doubtful trivia.", assessment)
        assert SCHOLAR_QUEUED_NOTE.strip() not in text

    def test_middle_band_answers_with_an_uncertainty_note(self):
        assessment = assess(ConfidenceSignals(self_consistency=0.5))
        assert assessment.band is ConfidenceBand.UNCERTAIN
        text = apply_policy(CONFIDENT_ANSWER, assessment)
        assert CONFIDENT_ANSWER in text
        assert UNCERTAINTY_NOTE in text

    def test_high_band_answers_unchanged(self):
        assessment = assess(
            ConfidenceSignals(self_consistency=0.95, citation_verification=1.0)
        )
        assert assessment.band is ConfidenceBand.CONFIDENT
        assert apply_policy(CONFIDENT_ANSWER, assessment) == CONFIDENT_ANSWER
        assert assessment.abstained is False
        assert assessment.queued is False

    def test_abstention_replaces_rather_than_prefixes(self):
        """An abstention that still shows the doubtful answer is not an abstention."""
        assessment = assess(ConfidenceSignals(self_consistency=0.0, is_religious=True))
        text = apply_policy("The ruling is definitely X.", assessment)
        assert "definitely X" not in text

    def test_assessment_reports_the_signals_it_used(self):
        assessment = assess(
            ConfidenceSignals(self_consistency=0.8, expressed_certainty=0.9)
        )
        assert assessment.signals_used == ["expressed_certainty", "self_consistency"]
        assert "citation_verification" not in assessment.signals

    def test_review_id_starts_unset(self):
        assert assess(ConfidenceSignals()).review_id is None


# ---------------------------------------------------------------------------
# End-to-end signal assembly
# ---------------------------------------------------------------------------


class TestBuildSignals:
    def test_hedged_religious_answer_lands_in_abstain(self):
        signals = build_signals(HEDGED_ANSWER, is_religious=True, is_high_stakes=True)
        assessment = assess(signals)
        assert assessment.band is ConfidenceBand.ABSTAIN
        assert assessment.queued is True

    def test_confident_answer_with_agreement_passes_through(self):
        signals = build_signals(
            CONFIDENT_ANSWER,
            is_religious=True,
            self_consistency=0.95,
            citation_verification=1.0,
        )
        assert assess(signals).band is ConfidenceBand.CONFIDENT

    def test_shared_signals_are_used_not_recomputed(self):
        """#ai-18 and #40 supply their own numbers; this module just consumes them."""
        signals = build_signals(
            CONFIDENT_ANSWER,
            is_religious=False,
            self_consistency=0.2,
            citation_verification=0.1,
        )
        assert signals.self_consistency == 0.2
        assert signals.citation_verification == 0.1
        # A textually confident answer cannot outvote poor external signals.
        assert assess(signals).band is not ConfidenceBand.CONFIDENT

    def test_unverified_answer_hedges_rather_than_passing(self):
        """With no external signals, a fluent answer must not sail through."""
        signals = build_signals(CONFIDENT_ANSWER, is_religious=True)
        assert assess(signals).band is ConfidenceBand.UNCERTAIN
