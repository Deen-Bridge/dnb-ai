"""Tests for Quranic Recitation Audio-to-Verse Identification & Tafsir Matcher (#238)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_recitation_verse_sync_identifies_ayat_al_kursi() -> None:
    resp = client.post(
        "/concordance/recitation-sync/analyze",
        json={
            "recited_arabic_transcript": "الله لا إله إلا هو الحي القيوم",
            "duration_seconds": 5.0,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["verse_key"] == "2:255"
    assert data["surah_name"] == "Al-Baqarah"
    assert data["word_completeness_score"] == 100.0
    assert data["missing_words"] == []
