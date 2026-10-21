"""Long-Form Islamic Lecture & Khutbah Transcription & Structuring Engine (#249).

Provides structured transcription for Islamic lectures (up to 2 hours),
automatic topic/section boundary detection, Quranic verse reference extraction,
Hadith citation extraction, speaker turn attribution, and searchable indexing.
"""

from __future__ import annotations

import logging
import re

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/lecture-transcription", tags=["lecture-transcription"])

MAX_LECTURE_DURATION_SECONDS = 7200.0  # 2 hours


class UtteranceChunk(BaseModel):
    speaker: str = Field(default="Speaker 1", min_length=1, max_length=80)
    start_seconds: float = Field(..., ge=0.0, le=MAX_LECTURE_DURATION_SECONDS)
    end_seconds: float = Field(..., gt=0.0, le=MAX_LECTURE_DURATION_SECONDS)
    text: str = Field(..., min_length=1, max_length=4000)


class LectureTranscriptionRequest(BaseModel):
    lecture_id: str = Field(..., min_length=1, max_length=120)
    title: str = Field(..., min_length=1, max_length=240)
    duration_seconds: float = Field(..., gt=0.0, le=MAX_LECTURE_DURATION_SECONDS)
    chunks: list[UtteranceChunk] = Field(..., min_length=1, max_length=500)


class LectureSection(BaseModel):
    section_index: int
    topic: str
    start_seconds: float
    end_seconds: float
    summary_text: str


class ExtractedQuranReference(BaseModel):
    reference: str
    context_snippet: str
    timestamp_seconds: float


class ExtractedHadithReference(BaseModel):
    collection: str
    reference_text: str
    context_snippet: str
    timestamp_seconds: float


class SpeakerAttributionSummary(BaseModel):
    speaker: str
    utterance_count: int
    speaking_time_seconds: float


class LectureTranscriptionResponse(BaseModel):
    lecture_id: str
    title: str
    duration_seconds: float
    full_transcript: str
    sections: list[LectureSection]
    quran_references: list[ExtractedQuranReference]
    hadith_references: list[ExtractedHadithReference]
    speakers: list[SpeakerAttributionSummary]


_TOPIC_KEYWORDS: list[tuple[str, tuple[str, ...]]] = [
    ("Introduction & Khutbat al-Hajah", ("الحمد لله", "in the name of allah", "alhamdulillah", "assalamu alaikum")),
    ("Quranic Exegesis (Tafsir)", ("surah", "ayah", "verse", "سورة", "آية", "tafsir")),
    ("Prophetic Sunnah & Hadith", ("bukhari", "muslim", "tirmidhi", "narrated", "prophet", "حديث", "رواه")),
    ("Fiqh & Practical Rulings", ("ruling", "wajib", "halal", "haram", "zakat", "salah", "حكم", "زكاة", "صلاة")),
    ("Spiritual Reminder & Du'a", ("dua", "repentance", "taqwa", "اللهم", "دعاء", "تقوى")),
]

_QURAN_REF_RE = re.compile(r"\b(?:Surah\s+[A-Za-z\-']+\s+)?(\d{1,3}:\d{1,3})\b", re.IGNORECASE)
_HADITH_REF_RE = re.compile(
    r"\b(Sahih\s+al-Bukhari|Sahih\s+Muslim|Sunan\s+Abi\s+Dawud|Jami`\s+at-Tirmidhi|Bukhari|Muslim)\s*(?:#|\bno\.?\s*)?(\d+)?",
    re.IGNORECASE,
)


def _detect_topic(text: str) -> str:
    lowered = text.lower()
    for label, cues in _TOPIC_KEYWORDS:
        if any(cue in lowered for cue in cues):
            return label
    return "General Discourse"


def process_lecture_chunks(req: LectureTranscriptionRequest) -> LectureTranscriptionResponse:
    for chunk in req.chunks:
        if chunk.end_seconds < chunk.start_seconds:
            raise HTTPException(status_code=422, detail="end_seconds must be >= start_seconds")
        if chunk.end_seconds > req.duration_seconds:
            raise HTTPException(status_code=422, detail="chunk end_seconds exceeds lecture duration_seconds")

    full_transcript = "\n".join(f"[{c.speaker} @ {c.start_seconds:.1f}s] {c.text}" for c in req.chunks)

    sections: list[LectureSection] = []
    for chunk in req.chunks:
        topic = _detect_topic(chunk.text)
        if sections and sections[-1].topic == topic:
            sections[-1].end_seconds = chunk.end_seconds
            sections[-1].summary_text += " " + chunk.text
        else:
            sections.append(
                LectureSection(
                    section_index=len(sections) + 1,
                    topic=topic,
                    start_seconds=chunk.start_seconds,
                    end_seconds=chunk.end_seconds,
                    summary_text=chunk.text,
                )
            )

    quran_refs: list[ExtractedQuranReference] = []
    hadith_refs: list[ExtractedHadithReference] = []
    speaker_stats: dict[str, list[float]] = {}

    for chunk in req.chunks:
        dur = max(0.0, chunk.end_seconds - chunk.start_seconds)
        if chunk.speaker not in speaker_stats:
            speaker_stats[chunk.speaker] = [0.0, 0.0]
        speaker_stats[chunk.speaker][0] += 1
        speaker_stats[chunk.speaker][1] += dur

        for m in _QURAN_REF_RE.finditer(chunk.text):
            quran_refs.append(
                ExtractedQuranReference(
                    reference=m.group(1),
                    context_snippet=chunk.text[:160],
                    timestamp_seconds=chunk.start_seconds,
                )
            )
        for hm in _HADITH_REF_RE.finditer(chunk.text):
            col = hm.group(1)
            num = hm.group(2) or ""
            hadith_refs.append(
                ExtractedHadithReference(
                    collection=col,
                    reference_text=f"{col} {num}".strip(),
                    context_snippet=chunk.text[:160],
                    timestamp_seconds=chunk.start_seconds,
                )
            )

    speakers = [
        SpeakerAttributionSummary(
            speaker=spk,
            utterance_count=int(vals[0]),
            speaking_time_seconds=round(vals[1], 2),
        )
        for spk, vals in sorted(speaker_stats.items())
    ]

    return LectureTranscriptionResponse(
        lecture_id=req.lecture_id,
        title=req.title,
        duration_seconds=req.duration_seconds,
        full_transcript=full_transcript,
        sections=sections,
        quran_references=quran_refs,
        hadith_references=hadith_refs,
        speakers=speakers,
    )


@router.post("/transcribe", response_model=LectureTranscriptionResponse)
async def transcribe_lecture_endpoint(body: LectureTranscriptionRequest) -> LectureTranscriptionResponse:
    """Structure and index a long-form Islamic lecture with topic, Quran, Hadith, and speaker extraction."""
    return process_lecture_chunks(body)
