"""Tests for the Islamic Lecture Audio-to-Text & Structuring subsystem (#249)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_lecture_transcription_extracts_sections_quran_hadith_and_speakers() -> None:
    payload = {
        "lecture_id": "khutbah-01",
        "title": "Patience and Gratitude in Light of the Quran",
        "duration_seconds": 3600.0,
        "chunks": [
            {
                "speaker": "Imam Ahmad",
                "start_seconds": 0.0,
                "end_seconds": 120.0,
                "text": "Alhamdulillah, assalamu alaikum. Today we reflect on gratitude.",
            },
            {
                "speaker": "Imam Ahmad",
                "start_seconds": 120.0,
                "end_seconds": 600.0,
                "text": "Allah reminds us in Surah Al-Baqarah 2:152 and 14:7 regarding shukr.",
            },
            {
                "speaker": "Moderator",
                "start_seconds": 600.0,
                "end_seconds": 900.0,
                "text": "As narrated in Sahih Muslim 2999, the affair of the believer is amazing.",
            },
        ],
    }
    resp = client.post("/audio-hadith/lecture-transcription/transcribe", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["sections"]) == 3
    assert [q["reference"] for q in data["quran_references"]] == ["2:152", "14:7"]
    assert data["hadith_references"][0]["reference_text"] == "Sahih Muslim 2999"
    assert len(data["speakers"]) == 2
