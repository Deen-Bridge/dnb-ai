"""Tests for the Screen Recording Islamic Content Analysis subsystem (#248)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_screen_recording_deduplicates_frames_and_extracts_references() -> None:
    payload = {
        "recording_id": "rec-101",
        "title": "Usul al-Tafsir Online Seminar",
        "frames": [
            {
                "frame_index": 0,
                "timestamp_seconds": 0.0,
                "ocr_text": "Slide 1: Principles of Tafsir — Surah Al-Baqarah 2:255",
                "scene_type": "slide",
            },
            {
                "frame_index": 1,
                "timestamp_seconds": 15.0,
                "ocr_text": "Slide 1: Principles of Tafsir — Surah Al-Baqarah 2:255",
                "scene_type": "slide",
            },
            {
                "frame_index": 2,
                "timestamp_seconds": 30.0,
                "ocr_text": "Mushaf View: Ayah 3:190 and Hadith in Sahih Bukhari 1",
                "scene_type": "mushaf",
            },
        ],
    }
    resp = client.post("/image-analysis/screen-recording/analyze", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["unique_slide_count"] == 2
    assert data["slides"][0]["end_seconds"] == 15.0
    assert data["all_quran_references"] == ["2:255", "3:190"]
    assert data["all_hadith_references"] == ["Sahih Bukhari 1"]
