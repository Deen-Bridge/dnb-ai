"""Screen Recording Islamic Content Analysis Subsystem (#248).

Processes frame sequences from recorded Islamic educational classes, digital
Mushaf readings, and slide presentations to deduplicate slide transitions,
extract visible Arabic and English text, detect Quranic verses and Hadith
citations across frames, and build a timestamped study outline.
"""

from __future__ import annotations

import logging
import re

from fastapi import APIRouter
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/screen-recording", tags=["screen-recording"])


class FrameObservation(BaseModel):
    frame_index: int = Field(..., ge=0)
    timestamp_seconds: float = Field(..., ge=0.0, le=14400.0)
    ocr_text: str = Field(..., min_length=1, max_length=4000)
    scene_type: str = Field(default="slide", max_length=60)


class ScreenRecordingAnalysisRequest(BaseModel):
    recording_id: str = Field(..., min_length=1, max_length=120)
    title: str = Field(..., min_length=1, max_length=240)
    frames: list[FrameObservation] = Field(..., min_length=1, max_length=500)


class ExtractedSlideSegment(BaseModel):
    slide_number: int
    start_seconds: float
    end_seconds: float
    scene_type: str
    extracted_text: str
    quran_references: list[str]
    hadith_references: list[str]


class ScreenRecordingAnalysisResponse(BaseModel):
    recording_id: str
    title: str
    unique_slide_count: int
    slides: list[ExtractedSlideSegment]
    all_quran_references: list[str]
    all_hadith_references: list[str]
    study_outline: list[str]


_VERSE_RE = re.compile(r"\b(\d{1,3}:\d{1,3})\b")
_HADITH_RE = re.compile(
    r"\b((?:Sahih\s+)?(?:al-)?Bukhari|(?:Sahih\s+)?Muslim|Tirmidhi|Abu\s+Dawud)\s*(?:#|\bno\.?\s*)?(\d+)?",
    re.IGNORECASE,
)


def _normalize_frame_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def analyze_screen_recording(req: ScreenRecordingAnalysisRequest) -> ScreenRecordingAnalysisResponse:
    slides: list[ExtractedSlideSegment] = []
    all_quran: list[str] = []
    all_hadith: list[str] = []

    for frame in sorted(req.frames, key=lambda f: f.timestamp_seconds):
        norm = _normalize_frame_text(frame.ocr_text)
        if slides and _normalize_frame_text(slides[-1].extracted_text) == norm:
            slides[-1].end_seconds = frame.timestamp_seconds
            continue

        q_refs = [m.group(1) for m in _VERSE_RE.finditer(norm)]
        h_refs = [f"{m.group(1)} {m.group(2) or ''}".strip() for m in _HADITH_RE.finditer(norm)]

        for q in q_refs:
            if q not in all_quran:
                all_quran.append(q)
        for h in h_refs:
            if h not in all_hadith:
                all_hadith.append(h)

        slides.append(
            ExtractedSlideSegment(
                slide_number=len(slides) + 1,
                start_seconds=frame.timestamp_seconds,
                end_seconds=frame.timestamp_seconds,
                scene_type=frame.scene_type,
                extracted_text=norm,
                quran_references=q_refs,
                hadith_references=h_refs,
            )
        )

    outline = [
        f"Slide {s.slide_number} ({s.start_seconds:.0f}s-{s.end_seconds:.0f}s) [{s.scene_type}]: {s.extracted_text[:90]}"
        for s in slides
    ]

    return ScreenRecordingAnalysisResponse(
        recording_id=req.recording_id,
        title=req.title,
        unique_slide_count=len(slides),
        slides=slides,
        all_quran_references=all_quran,
        all_hadith_references=all_hadith,
        study_outline=outline,
    )


@router.post("/analyze", response_model=ScreenRecordingAnalysisResponse)
async def analyze_screen_recording_endpoint(
    body: ScreenRecordingAnalysisRequest,
) -> ScreenRecordingAnalysisResponse:
    """Analyze screen-recording frames from Islamic classes or digital Mushaf sessions."""
    return analyze_screen_recording(body)
