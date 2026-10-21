"""Tests for the Arabic Speech-to-Text Optimization subsystem (#237)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_arabic_stt_optimizer_restores_islamic_orthography_and_picks_best_candidate() -> None:
    resp = client.post(
        "/arabic-dialect/stt-optimizer/optimize",
        json={
            "candidates": [
                {
                    "transcript": "ما حكم الصلاه والزكاه في سوره البقرة ان شاء الله",
                    "acoustic_confidence": 0.85,
                },
                {
                    "transcript": "كلام عام غير مرتبط",
                    "acoustic_confidence": 0.80,
                },
            ]
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "الصلاة" in data["optimized_transcript"]
    assert "الزكاة" in data["optimized_transcript"]
    assert "سورة" in data["optimized_transcript"]
    assert "إن شاء الله" in data["optimized_transcript"]
    assert len(data["corrections"]) >= 4
