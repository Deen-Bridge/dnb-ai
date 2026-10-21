"""Quranic Recitation Audio-to-Verse Identification & Tafsir Matcher (#238).

Identifies the recited Surah and Ayah range from transcribed Quranic audio,
computes word-level completeness against the canonical verse text, and
returns the matched verse metadata and concise thematic summary.
"""

from __future__ import annotations

import logging
import re

from fastapi import APIRouter
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/recitation-sync", tags=["recitation-sync"])


class CanonicalVerseEntry(BaseModel):
    surah_number: int
    surah_name: str
    ayah_number: int
    arabic_text: str
    english_translation: str
    tafsir_summary: str


_CANONICAL_VERSES: tuple[CanonicalVerseEntry, ...] = (
    CanonicalVerseEntry(
        surah_number=1,
        surah_name="Al-Fatihah",
        ayah_number=1,
        arabic_text="بسم الله الرحمن الرحيم",
        english_translation="In the name of Allah, the Entirely Merciful, the Especially Merciful.",
        tafsir_summary="Opens the Quran by invoking Allah's comprehensive mercy (Ar-Rahman, Ar-Rahim).",
    ),
    CanonicalVerseEntry(
        surah_number=1,
        surah_name="Al-Fatihah",
        ayah_number=2,
        arabic_text="الحمد لله رب العالمين",
        english_translation="All praise is due to Allah, Lord of the worlds.",
        tafsir_summary="Establishes praise and gratitude exclusively for Allah as Sustainer of all creation.",
    ),
    CanonicalVerseEntry(
        surah_number=2,
        surah_name="Al-Baqarah",
        ayah_number=255,
        arabic_text="الله لا إله إلا هو الحي القيوم",
        english_translation="Allah - there is no deity except Him, the Ever-Living, the Sustainer of all existence.",
        tafsir_summary="Ayat al-Kursi: the greatest verse in the Quran affirming Allah's eternal life and sovereignty.",
    ),
    CanonicalVerseEntry(
        surah_number=112,
        surah_name="Al-Ikhlas",
        ayah_number=1,
        arabic_text="قل هو الله أحد",
        english_translation="Say, He is Allah, [who is] One.",
        tafsir_summary="Affirms pure monotheism (Tawhid) and the absolute uniqueness of Allah.",
    ),
)

_DIACRITICS_RE = re.compile(r"[\u064B-\u065F\u0670]")


def _strip_tashkeel(text: str) -> str:
    return _DIACRITICS_RE.sub("", text)


class RecitationAudioAnalysisRequest(BaseModel):
    recited_arabic_transcript: str = Field(..., min_length=1, max_length=4000)
    duration_seconds: float = Field(default=6.0, gt=0.1, le=600.0)


class RecitationVerseMatchResponse(BaseModel):
    surah_number: int
    surah_name: str
    ayah_number: int
    verse_key: str
    canonical_arabic: str
    english_translation: str
    tafsir_summary: str
    word_completeness_score: float
    words_per_minute: float
    missing_words: list[str]


def analyze_recitation_verse(req: RecitationAudioAnalysisRequest) -> RecitationVerseMatchResponse:
    norm_input = _strip_tashkeel(req.recited_arabic_transcript)
    input_words = [w for w in norm_input.split() if w]

    best_entry = _CANONICAL_VERSES[0]
    best_overlap = -1.0
    best_missing: list[str] = []

    for entry in _CANONICAL_VERSES:
        canon_words = _strip_tashkeel(entry.arabic_text).split()
        matched = [w for w in canon_words if w in input_words]
        ratio = len(matched) / max(1, len(canon_words))
        if ratio > best_overlap:
            best_overlap = ratio
            best_entry = entry
            best_missing = [w for w in canon_words if w not in input_words]

    wpm = round((len(input_words) / req.duration_seconds) * 60.0, 2)
    return RecitationVerseMatchResponse(
        surah_number=best_entry.surah_number,
        surah_name=best_entry.surah_name,
        ayah_number=best_entry.ayah_number,
        verse_key=f"{best_entry.surah_number}:{best_entry.ayah_number}",
        canonical_arabic=best_entry.arabic_text,
        english_translation=best_entry.english_translation,
        tafsir_summary=best_entry.tafsir_summary,
        word_completeness_score=round(best_overlap * 100.0, 2),
        words_per_minute=wpm,
        missing_words=best_missing,
    )


@router.post("/analyze", response_model=RecitationVerseMatchResponse)
async def analyze_recitation_verse_endpoint(
    body: RecitationAudioAnalysisRequest,
) -> RecitationVerseMatchResponse:
    """Identify the Surah and Ayah from recited Quranic audio and return Tafsir context."""
    return analyze_recitation_verse(body)
