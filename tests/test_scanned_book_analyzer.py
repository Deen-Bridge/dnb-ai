"""Tests for the Scanned Islamic Book Page Layout & Citation Analyzer (#242)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_scanned_book_page_separates_matn_footnotes_marginalia_and_citations() -> None:
    resp = client.post(
        "/page-analysis/scanned-book/analyze",
        json={
            "book_id": "al-umm-vol1",
            "page_number": 42,
            "blocks": [
                {
                    "block_id": "b1",
                    "y_norm": 0.30,
                    "x_norm": 0.50,
                    "font_scale": 1.0,
                    "text": "قال الشافعي رحمه الله في قوله تعالى 2:43 وأقيموا الصلاة وآتوا الزكاة.",
                },
                {
                    "block_id": "b2",
                    "y_norm": 0.85,
                    "x_norm": 0.50,
                    "font_scale": 0.75,
                    "text": "(1) أخرجه في صحيح البخاري كتاب الإيمان.",
                },
                {
                    "block_id": "b3",
                    "y_norm": 0.40,
                    "x_norm": 0.08,
                    "font_scale": 0.80,
                    "text": "حاشية: قوله وأقيموا الصلاة نص في الوجوب.",
                },
            ],
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "2:43" in data["quran_verses"]
    assert "صحيح البخاري" in data["hadith_citations"]
    assert len(data["footnotes"]) == 1
    assert len(data["marginalia"]) == 1
    assert "قال الشافعي" in data["main_text"]
