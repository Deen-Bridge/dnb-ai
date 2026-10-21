"""Tests for Voice Note Upload, Transcription, and Islamic Q&A Routing (#243)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_voice_note_transcribe_and_route_english_and_arabic() -> None:
    resp = client.post(
        "/worship/voice-note/transcribe-and-route",
        json={
            "note_id": "vn-101",
            "audio_format": "ogg",
            "duration_seconds": 14.5,
            "raw_utterance": "Assalamu alaikum. Does bleeding from a small cut break wudhu before salat?",
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["detected_language"] == "en"
    assert data["suggested_topic"] == "prayer"
    assert "wudu" in data["normalized_transcript"]
    assert "Does bleeding" in data["extracted_question"]
