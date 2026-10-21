"""Tests for the Automated Tajweed Error Detection subsystem (#246)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from main import app
from tajweed_detector import (
    RecitedSegmentObservation,
    TajweedDetectionRequest,
    TajweedErrorCategory,
    detect_tajweed_errors,
)

client = TestClient(app)


def test_detect_tajweed_errors_identifies_makharij_madd_ghunnah_and_qalqalah() -> None:
    req = TajweedDetectionRequest(
        surah=1,
        ayah=7,
        segments=[
            RecitedSegmentObservation(
                word_index=0,
                expected_word="الصِّرَاطَ",
                observed_word="السِّرَاطَ",
            ),
            RecitedSegmentObservation(
                word_index=1,
                expected_word="الضَّالِّينَ",
                observed_word="الضَّالِّينَ",
                expected_duration_counts=6.0,
                observed_duration_counts=2.0,
            ),
            RecitedSegmentObservation(
                word_index=2,
                expected_word="مِنْ بَعْدِ",
                observed_word="مِنْ بَعْدِ",
                rule_expected=TajweedErrorCategory.ghunnah,
                nasalization_ratio=0.2,
            ),
            RecitedSegmentObservation(
                word_index=3,
                expected_word="أَحَدْ",
                observed_word="أَحَدْ",
                rule_expected=TajweedErrorCategory.qalqalah,
                plosive_bounce_detected=False,
            ),
        ],
    )
    res = detect_tajweed_errors(req)
    categories = [e.category for e in res.errors]
    assert TajweedErrorCategory.makharij in categories
    assert TajweedErrorCategory.madd_duration in categories
    assert TajweedErrorCategory.ghunnah in categories
    assert TajweedErrorCategory.qalqalah in categories
    assert res.overall_tajweed_score < 100.0


def test_tajweed_detector_endpoint() -> None:
    resp = client.post(
        "/recitation/tajweed-detector/analyze",
        json={
            "surah": 112,
            "ayah": 1,
            "segments": [
                {
                    "word_index": 0,
                    "expected_word": "قُلْ",
                    "observed_word": "قُلْ",
                }
            ],
        },
    )
    assert resp.status_code == 200
    assert resp.json()["overall_tajweed_score"] == 100.0
