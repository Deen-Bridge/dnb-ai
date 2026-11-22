"""Tajweed result calibration and scholar review boundary (#856).

Unit tests pin the calibration formula and review reasons, integration tests go
through the HTTP route, and the evaluation tests run the labelled set from
``scripts.eval_recitation_calibration``, including latency and the no-network
(zero-cost) guarantee. Offline; no secrets, no ASR.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from quran_recitation import (
    CALIBRATION_VERSION,
    NOT_ASSESSED_SCOPE,
    SCHOLAR_REVIEW_BOUNDARY,
    UNKNOWN_ASR_PRIOR,
    AnalyzeRequest,
    AnalyzeResponse,
    RiwayahInfo,
    analyze_recitation,
    calibrate_confidence,
    router,
    scholar_review_reasons,
)
from scripts.eval_recitation_calibration import EVAL_CASES, IKHLAS, evaluate_case, run_evaluation

# Generous ceiling for shared CI runners; measured p95 is ~10 ms locally.
LATENCY_BUDGET_MS = 250.0


def _analyze(transcript: str | None, **kwargs: object) -> AnalyzeResponse:
    return analyze_recitation(AnalyzeRequest(transcript=transcript, **kwargs))  # type: ignore[arg-type]


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


# ---------------------------------------------------------------------------
# Calibration formula
# ---------------------------------------------------------------------------


class TestCalibrateConfidence:
    def test_product_of_accuracy_and_asr(self) -> None:
        assert calibrate_confidence(1.0, 0.9) == 0.9
        assert calibrate_confidence(0.8, 0.5) == 0.4

    def test_unknown_asr_uses_prior_below_high_threshold(self) -> None:
        assert calibrate_confidence(1.0, None) == UNKNOWN_ASR_PRIOR
        assert UNKNOWN_ASR_PRIOR < 0.75  # shared "high" threshold

    @pytest.mark.parametrize("accuracy,asr", [(0.0, 1.0), (1.0, 0.0), (0.0, None)])
    def test_zero_inputs(self, accuracy: float, asr: float | None) -> None:
        assert calibrate_confidence(accuracy, asr) == 0.0

    def test_clamped_to_unit_interval(self) -> None:
        assert calibrate_confidence(1.5, 1.0) == 1.0
        assert calibrate_confidence(-0.2, 1.0) == 0.0

    def test_monotonic_in_both_inputs(self) -> None:
        grid = [i / 10 for i in range(11)]
        for asr in grid:
            scores = [calibrate_confidence(acc, asr) for acc in grid]
            assert scores == sorted(scores)
        for acc in grid:
            scores = [calibrate_confidence(acc, asr) for asr in grid]
            assert scores == sorted(scores)


class TestReviewReasons:
    def test_clean_high_confidence_needs_no_review(self) -> None:
        assert scholar_review_reasons("accurate", 0.9, None) == []

    def test_reason_order_is_stable(self) -> None:
        riwayah = RiwayahInfo(name="Warsh 'an Nafi'", basis="text_variant", note="", hint_conflict=True)
        assert scholar_review_reasons("inaccurate", 0.3, riwayah) == [
            "asr_confidence_below_high",
            "text_deviations",
            "non_hafs_reading",
            "riwayah_hint_conflict",
        ]

    def test_hint_or_default_riwayah_is_not_a_review_reason(self) -> None:
        for basis in ("hint", "default"):
            riwayah = RiwayahInfo(name="Warsh 'an Nafi'", basis=basis, note="")
            assert scholar_review_reasons("accurate", 0.9, riwayah) == []


# ---------------------------------------------------------------------------
# Analysis responses
# ---------------------------------------------------------------------------


class TestAnalyzeCalibration:
    def test_every_response_carries_the_boundary(self) -> None:
        for transcript in (IKHLAS, "ذهب الولد إلى المدرسة", ""):
            result = _analyze(transcript, asr_confidence=0.95)
            assert result.calibration_version == CALIBRATION_VERSION
            assert result.scholar_review.boundary == SCHOLAR_REVIEW_BOUNDARY
            assert "tajwid_rules" in result.scholar_review.not_assessed
            assert result.scholar_review.not_assessed == list(NOT_ASSESSED_SCOPE)

    def test_unidentified_is_low_with_fallback_not_high(self) -> None:
        # Regression: unidentified and empty results used to report confidence_band="high".
        for transcript in ("ذهب الولد إلى المدرسة صباحا", "hello world", ""):
            result = _analyze(transcript, asr_confidence=0.99)
            assert result.identified is False
            assert result.confidence_band == "low"
            assert result.calibrated_confidence == 0.0
            assert result.fallback_recommended is True
            assert result.review_required is False
            assert result.scholar_review.reasons == ["unidentified"]

    def test_unknown_asr_is_never_high(self) -> None:
        # Regression: omitting asr_confidence used to yield "high" with no review.
        for case in EVAL_CASES:
            result = _analyze(case.transcript, riwayah_hint=case.riwayah_hint)
            assert result.confidence_band != "high", case.name
            if result.identified:
                assert "asr_confidence_unknown" in result.scholar_review.reasons

    def test_inaccurate_recitation_is_not_high_even_with_perfect_asr(self) -> None:
        heavy = "قل هو الله أحد كتاب الصمد لم سماء ولم بحر ولم يكن له قلم أحد"
        result = _analyze(heavy, asr_confidence=1.0)
        assert result.verification == "inaccurate"
        assert result.confidence_band != "high"
        assert result.review_required is True

    def test_deviation_requires_review_even_in_high_band(self) -> None:
        result = _analyze(IKHLAS.replace(" الصمد", ""), asr_confidence=0.95)
        assert result.confidence_band == "high"
        assert result.review_required is True
        assert result.scholar_review.required is True
        assert result.scholar_review.reasons == ["text_deviations"]

    def test_review_flags_agree(self) -> None:
        for case in EVAL_CASES:
            result = _analyze(case.transcript, asr_confidence=case.asr_confidence, riwayah_hint=case.riwayah_hint)
            assert result.scholar_review.required == result.review_required, case.name

    def test_low_asr_note_mentions_fallback(self) -> None:
        result = _analyze(IKHLAS, asr_confidence=0.2)
        assert "Low ASR confidence detected; safe fallback mode recommended." in result.note
        assert "Scholar review required" in result.note

    def test_low_calibrated_confidence_from_text_mentions_fallback(self) -> None:
        heavy = "قل هو الله أحد كتاب الصمد لم سماء ولم بحر ولم يكن له قلم أحد"
        result = _analyze(heavy, asr_confidence=0.5)
        assert result.fallback_recommended is True
        assert "Low calibrated confidence; safe fallback mode recommended." in result.note

    def test_riwayah_hint_conflict_is_structured(self) -> None:
        fatihah = "الحمد لله رب العالمين الرحمن الرحيم مالك يوم الدين"
        result = _analyze(fatihah, asr_confidence=0.95, riwayah_hint="warsh")
        assert result.riwayah is not None and result.riwayah.hint_conflict is True
        assert "riwayah_hint_conflict" in result.scholar_review.reasons

    def test_diacritised_input_calibrates_like_plain(self) -> None:
        plain = _analyze(IKHLAS, asr_confidence=0.9)
        voweled = _analyze("قُلْ هُوَ ٱللَّهُ أَحَدٌ ٱللَّهُ ٱلصَّمَدُ لَمْ يَلِدْ وَلَمْ يُولَدْ وَلَمْ يَكُن لَّهُۥ كُفُوًا أَحَدٌۢ", asr_confidence=0.9)
        assert voweled.calibrated_confidence == plain.calibrated_confidence
        assert voweled.confidence_band == plain.confidence_band


# ---------------------------------------------------------------------------
# HTTP integration
# ---------------------------------------------------------------------------


class TestRoute:
    def test_response_contract(self, client: TestClient) -> None:
        resp = client.post("/quran-recitation/analyze", json={"transcript": IKHLAS, "asr_confidence": 0.95})
        assert resp.status_code == 200
        body = resp.json()
        assert body["confidence_band"] == "high"
        assert body["calibrated_confidence"] == 0.95
        assert body["calibration_version"] == CALIBRATION_VERSION
        review = body["scholar_review"]
        assert review["required"] is False and review["reasons"] == []
        assert set(review) == {"required", "reasons", "assessed", "not_assessed", "boundary"}

    def test_missing_asr_confidence_requires_review(self, client: TestClient) -> None:
        body = client.post("/quran-recitation/analyze", json={"transcript": IKHLAS}).json()
        assert body["confidence_band"] == "medium"
        assert body["review_required"] is True
        assert body["scholar_review"]["reasons"] == ["asr_confidence_unknown"]

    def test_consent_still_enforced(self, client: TestClient) -> None:
        resp = client.post("/quran-recitation/analyze", json={"transcript": IKHLAS, "consent_to_process": False})
        assert resp.status_code == 400

    def test_out_of_range_asr_confidence_rejected(self, client: TestClient) -> None:
        resp = client.post("/quran-recitation/analyze", json={"transcript": IKHLAS, "asr_confidence": 1.5})
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Evaluation: labelled set, latency, cost
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", EVAL_CASES, ids=[c.name for c in EVAL_CASES])
def test_eval_case_matches_label(case: object) -> None:
    result = evaluate_case(case, repeats=3)  # type: ignore[arg-type]
    assert result.passed, result.mismatches


def test_evaluation_latency_and_zero_cost() -> None:
    report = run_evaluation(repeats=5)  # also fails if any socket is opened
    assert report["passed"] == report["cases"] == len(EVAL_CASES)
    assert report["provider_calls"] == 0
    assert report["estimated_cost_usd"] == 0.0
    assert report["max_p95_ms"] < LATENCY_BUDGET_MS
