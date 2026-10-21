"""Voice Note Upload, Transcription, and Islamic Q&A Routing (#243).

Accepts uploaded voice notes (OGG/MP3/WAV/M4A metadata + transcript/utterance
frames), detects the spoken language, normalizes Islamic terminology across
Arabic and English, extracts the core religious question, and returns a
structured Q&A routing payload.
"""

from __future__ import annotations

import logging
import re
from enum import Enum

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/worship/voice-note", tags=["voice-note"])


class AudioContainerFormat(str, Enum):
    ogg = "ogg"
    mp3 = "mp3"
    wav = "wav"
    m4a = "m4a"


class VoiceNoteUploadRequest(BaseModel):
    note_id: str = Field(..., min_length=1, max_length=120)
    audio_format: AudioContainerFormat = AudioContainerFormat.ogg
    duration_seconds: float = Field(..., gt=0.0, le=900.0)
    sample_rate_hz: int = Field(default=16000, ge=8000, le=96000)
    raw_utterance: str = Field(..., min_length=1, max_length=10000)
    language_hint: str | None = Field(default=None, max_length=16)


class VoiceNoteQAResponse(BaseModel):
    note_id: str
    audio_format: AudioContainerFormat
    duration_seconds: float
    detected_language: str
    normalized_transcript: str
    extracted_question: str
    suggested_topic: str
    qa_pipeline_payload: dict[str, str]


_ARABIC_CHAR_RE = re.compile(r"[\u0600-\u06FF]")

_TOPIC_KEYWORDS: list[tuple[str, tuple[str, ...]]] = [
    ("prayer", ("salah", "prayer", "wudu", "tayammum", "صلاة", "الصلاة", "وضوء", "تيمم")),
    ("zakat", ("zakat", "nisab", "fitrah", "زكاة", "الزكاة", "نصاب")),
    ("fasting", ("fasting", "ramadan", "sawm", "suhoor", "iftar", "صيام", "رمضان", "إفطار")),
    ("hajj", ("hajj", "umrah", "ihram", "tawaf", "حج", "عمرة", "إحرام", "طواف")),
    ("quran", ("quran", "surah", "ayah", "tafsir", "قرآن", "سورة", "آية", "تفسير")),
]


def process_voice_note(req: VoiceNoteUploadRequest) -> VoiceNoteQAResponse:
    text = re.sub(r"\s+", " ", req.raw_utterance).strip()
    if not text:
        raise HTTPException(status_code=422, detail="Voice note transcript cannot be empty.")

    arabic_chars = len(_ARABIC_CHAR_RE.findall(text))
    detected_lang = req.language_hint or ("ar" if arabic_chars > len(text) * 0.25 else "en")

    normalized = text
    replacements = {
        r"\bwudhu\b": "wudu",
        r"\bsalat\b": "salah",
        r"\bzakaat\b": "zakat",
        r"\bramadhaan\b": "ramadan",
    }
    for pat, repl in replacements.items():
        normalized = re.sub(pat, repl, normalized, flags=re.IGNORECASE)

    lower = normalized.lower()
    topic = "general_fiqh"
    for candidate_topic, kw_list in _TOPIC_KEYWORDS:
        if any(kw in lower for kw in kw_list):
            topic = candidate_topic
            break

    sentences = [s.strip() for s in re.split(r"(?<=[.!?؟])\s+", normalized) if s.strip()]
    question_sents = [
        s
        for s in sentences
        if "?" in s or "؟" in s or re.match(r"^(what|how|is|can|does|when|حكم|ما|هل|كيف)\b", s, re.IGNORECASE)
    ]
    extracted_q = question_sents[0] if question_sents else normalized

    return VoiceNoteQAResponse(
        note_id=req.note_id,
        audio_format=req.audio_format,
        duration_seconds=req.duration_seconds,
        detected_language=detected_lang,
        normalized_transcript=normalized,
        extracted_question=extracted_q,
        suggested_topic=topic,
        qa_pipeline_payload={
            "query": extracted_q,
            "language": detected_lang,
            "topic": topic,
            "source": f"voice_note:{req.note_id}",
        },
    )


@router.post("/transcribe-and-route", response_model=VoiceNoteQAResponse)
async def transcribe_and_route_voice_note_endpoint(
    body: VoiceNoteUploadRequest,
) -> VoiceNoteQAResponse:
    """Process an uploaded voice note, normalize its transcript, and build a Q&A pipeline payload."""
    return process_voice_note(body)
