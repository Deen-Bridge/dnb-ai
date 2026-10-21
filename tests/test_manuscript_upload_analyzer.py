"""Tests for the Islamic Manuscript & Calligraphy Image Upload Analyzer (#240)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_manuscript_upload_identifies_kufic_and_quran_reference() -> None:
    resp = client.post(
        "/calligraphy/manuscript-upload/analyze",
        json={
            "manuscript_id": "ms-kufic-01",
            "mime_type": "image/jpeg",
            "ocr_arabic_text": "بسم الله الرحمن الرحيم قل هو الله أحد",
            "angular_stroke_ratio": 0.80,
            "estimated_century_ah": 2,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["detected_script_style"] == "Kufic"
    assert "Quran 1:1 (Basmala)" in data["identified_references"]
    assert "Quran 112:1 (Surah Al-Ikhlas)" in data["identified_references"]
