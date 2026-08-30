"""
Islamic Lecture Audio Summarization System

This module provides automated processing of Islamic lecture audio with
transcription, topic extraction, reference detection, and structured
summarization while maintaining theological accuracy.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

logger = logging.getLogger(__name__)


class Language(str, Enum):
    """Supported languages for transcription."""
    ARABIC = "ar"
    ENGLISH = "en"
    URDU = "ur"
    INDONESIAN = "id"
    MALAY = "ms"
    TURKISH = "tr"
    FRENCH = "fr"


class ReferenceType(str, Enum):
    """Types of Islamic references."""
    QURAN = "quran"
    HADITH = "hadith"
    SCHOLARLY = "scholarly"
    HISTORICAL = "historical"


@dataclass
class Timestamp:
    """Represents a timestamp in the audio."""
    start_seconds: float
    end_seconds: float

    def format(self) -> str:
        """Format as HH:MM:SS."""
        def _fmt(secs: float) -> str:
            hours = int(secs // 3600)
            mins = int((secs % 3600) // 60)
            secs_rem = int(secs % 60)
            return f"{hours:02d}:{mins:02d}:{secs_rem:02d}"
        return f"{_fmt(self.start_seconds)} - {_fmt(self.end_seconds)}"

    def to_dict(self) -> dict:
        return {
            "start_seconds": self.start_seconds,
            "end_seconds": self.end_seconds,
            "formatted": self.format(),
        }


@dataclass
class Speaker:
    """Represents an identified speaker."""
    id: str
    label: str  # e.g., "Speaker 1", "Sheikh Ahmad"
    speaking_time: float = 0.0  # Total seconds

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "label": self.label,
            "speaking_time": self.speaking_time,
        }


@dataclass
class TranscriptSegment:
    """A segment of transcribed text with metadata."""
    text: str
    timestamp: Timestamp
    speaker: Optional[Speaker] = None
    language: Language = Language.ARABIC
    confidence: float = 0.0

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "timestamp": self.timestamp.to_dict(),
            "speaker": self.speaker.to_dict() if self.speaker else None,
            "language": self.language.value,
            "confidence": self.confidence,
        }


@dataclass
class DetectedReference:
    """A detected Islamic reference in the lecture."""
    reference_type: ReferenceType
    text: str
    timestamp: Timestamp
    surah: Optional[int] = None
    ayah: Optional[int] = None
    hadith_source: Optional[str] = None
    hadith_number: Optional[str] = None
    scholar: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "reference_type": self.reference_type.value,
            "text": self.text,
            "timestamp": self.timestamp.to_dict(),
            "surah": self.surah,
            "ayah": self.ayah,
            "hadith_source": self.hadith_source,
            "hadith_number": self.hadith_number,
            "scholar": self.scholar,
        }


@dataclass
class Topic:
    """An extracted topic from the lecture."""
    title: str
    description: str
    timestamp: Timestamp
    keywords: list[str] = field(default_factory=list)
    related_references: list[DetectedReference] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "description": self.description,
            "timestamp": self.timestamp.to_dict(),
            "keywords": self.keywords,
            "related_references": [r.to_dict() for r in self.related_references],
        }


@dataclass
class SummarySection:
    """A section of the lecture summary."""
    title: str
    content: str
    timestamp: Timestamp
    key_points: list[str] = field(default_factory=list)
    references: list[DetectedReference] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "content": self.content,
            "timestamp": self.timestamp.to_dict(),
            "key_points": self.key_points,
            "references": [r.to_dict() for r in self.references],
        }


@dataclass
class LectureSummary:
    """Complete summary of an Islamic lecture."""
    title: str
    duration_seconds: float
    primary_language: Language
    speakers: list[Speaker] = field(default_factory=list)
    abstract: str = ""
    topics: list[Topic] = field(default_factory=list)
    sections: list[SummarySection] = field(default_factory=list)
    all_references: list[DetectedReference] = field(default_factory=list)
    key_takeaways: list[str] = field(default_factory=list)
    transcript_segments: list[TranscriptSegment] = field(default_factory=list)
    full_transcript: str = ""
    processing_warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "duration_seconds": self.duration_seconds,
            "duration_formatted": self._format_duration(),
            "primary_language": self.primary_language.value,
            "speakers": [s.to_dict() for s in self.speakers],
            "abstract": self.abstract,
            "topics": [t.to_dict() for t in self.topics],
            "sections": [s.to_dict() for s in self.sections],
            "all_references": [r.to_dict() for r in self.all_references],
            "key_takeaways": self.key_takeaways,
            "transcript_segments": [t.to_dict() for t in self.transcript_segments],
            "full_transcript": self.full_transcript,
            "processing_warnings": self.processing_warnings,
        }

    def _format_duration(self) -> str:
        hours = int(self.duration_seconds // 3600)
        mins = int((self.duration_seconds % 3600) // 60)
        secs = int(self.duration_seconds % 60)
        if hours > 0:
            return f"{hours}h {mins}m {secs}s"
        return f"{mins}m {secs}s"


# Islamic terminology patterns for reference detection
QURAN_PATTERNS = [
    r"سورة\s+(\w+)",
    r"الآية\s+(\d+)",
    r"﴿.*?﴾",
    r"قال\s+الله\s+تعالى",
    r"قوله\s+تعالى",
    r"surah\s+(\w+)",
    r"verse\s+(\d+)",
    r"ayah\s+(\d+)",
]

HADITH_PATTERNS = [
    r"قال\s+رسول\s+الله",
    r"قال\s+النبي",
    r"رواه\s+(\w+)",
    r"أخرجه\s+(\w+)",
    r"صحيح\s+(البخاري|مسلم)",
    r"the\s+prophet\s+said",
    r"hadith\s+narrated\s+by",
]

SCHOLAR_PATTERNS = [
    r"قال\s+الإمام\s+(\w+)",
    r"قال\s+الشيخ\s+(\w+)",
    r"ذكر\s+ابن\s+(\w+)",
    r"imam\s+(\w+)\s+said",
    r"sheikh\s+(\w+)",
]


def detect_language(text: str) -> Language:
    """Detect the primary language of the text."""
    # Simple heuristic based on character ranges
    arabic_chars = len(re.findall(r'[\u0600-\u06FF]', text))
    latin_chars = len(re.findall(r'[a-zA-Z]', text))

    if arabic_chars > latin_chars:
        return Language.ARABIC
    return Language.ENGLISH


def detect_references(
    segments: list[TranscriptSegment]
) -> list[DetectedReference]:
    """
    Detect Islamic references (Quran, Hadith, Scholarly) in transcript.
    """
    references = []

    for segment in segments:
        text = segment.text

        # Detect Quranic references
        for pattern in QURAN_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                references.append(DetectedReference(
                    reference_type=ReferenceType.QURAN,
                    text=match.group(0),
                    timestamp=segment.timestamp,
                ))

        # Detect Hadith references
        for pattern in HADITH_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                source = match.group(1) if match.lastindex else None
                references.append(DetectedReference(
                    reference_type=ReferenceType.HADITH,
                    text=match.group(0),
                    timestamp=segment.timestamp,
                    hadith_source=source,
                ))

        # Detect scholarly references
        for pattern in SCHOLAR_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                scholar = match.group(1) if match.lastindex else None
                references.append(DetectedReference(
                    reference_type=ReferenceType.SCHOLARLY,
                    text=match.group(0),
                    timestamp=segment.timestamp,
                    scholar=scholar,
                ))

    return references


def extract_topics(
    segments: list[TranscriptSegment],
    min_segment_count: int = 3,
) -> list[Topic]:
    """
    Extract main topics from the lecture using topic segmentation.
    """
    topics = []

    # Group segments by time windows
    window_size = 300  # 5 minutes
    current_window_start = 0
    current_segments = []

    for segment in segments:
        if segment.timestamp.start_seconds - current_window_start > window_size:
            # Process current window
            if current_segments:
                topic = _extract_topic_from_segments(current_segments)
                if topic:
                    topics.append(topic)

            current_window_start = segment.timestamp.start_seconds
            current_segments = []

        current_segments.append(segment)

    # Process final window
    if current_segments:
        topic = _extract_topic_from_segments(current_segments)
        if topic:
            topics.append(topic)

    return topics


def _extract_topic_from_segments(segments: list[TranscriptSegment]) -> Optional[Topic]:
    """Extract a topic from a group of segments."""
    if not segments:
        return None

    combined_text = " ".join(s.text for s in segments)

    # Extract keywords (simplified - in production would use NLP)
    # Look for Islamic terminology
    keywords = []
    islamic_terms = [
        "صلاة", "زكاة", "صوم", "حج", "توحيد", "إيمان", "تقوى",
        "prayer", "fasting", "charity", "pilgrimage", "faith"
    ]
    for term in islamic_terms:
        if term in combined_text.lower():
            keywords.append(term)

    # Generate title (simplified)
    title = f"Topic at {segments[0].timestamp.format().split(' - ')[0]}"

    return Topic(
        title=title,
        description=combined_text[:200] + "..." if len(combined_text) > 200 else combined_text,
        timestamp=Timestamp(
            start_seconds=segments[0].timestamp.start_seconds,
            end_seconds=segments[-1].timestamp.end_seconds,
        ),
        keywords=keywords[:5],
    )


def generate_summary(
    segments: list[TranscriptSegment],
    topics: list[Topic],
    references: list[DetectedReference],
) -> tuple[str, list[SummarySection], list[str]]:
    """
    Generate structured summary of the lecture.

    Returns:
        Tuple of (abstract, sections, key_takeaways)
    """
    # Generate abstract (first ~100 words)
    full_text = " ".join(s.text for s in segments)
    words = full_text.split()
    abstract = " ".join(words[:100]) + "..." if len(words) > 100 else full_text

    # Generate sections based on topics
    sections = []
    for topic in topics:
        section = SummarySection(
            title=topic.title,
            content=topic.description,
            timestamp=topic.timestamp,
            key_points=topic.keywords,
            references=[r for r in references
                       if topic.timestamp.start_seconds <= r.timestamp.start_seconds <= topic.timestamp.end_seconds],
        )
        sections.append(section)

    # Generate key takeaways
    key_takeaways = []
    if references:
        quran_count = len([r for r in references if r.reference_type == ReferenceType.QURAN])
        hadith_count = len([r for r in references if r.reference_type == ReferenceType.HADITH])
        if quran_count:
            key_takeaways.append(f"Contains {quran_count} Quranic references")
        if hadith_count:
            key_takeaways.append(f"Contains {hadith_count} Hadith references")

    return abstract, sections, key_takeaways


async def transcribe_audio(
    audio_data: bytes,
    language_hint: Optional[Language] = None,
) -> list[TranscriptSegment]:
    """
    Transcribe audio with Islamic terminology awareness.

    In production, this would use:
    - Google Speech-to-Text with Arabic model
    - Azure Speech Services
    - OpenAI Whisper
    - Custom fine-tuned models for Islamic terminology
    """
    # Simulate transcription result
    # In production, this would call actual speech-to-text API

    simulated_segments = [
        TranscriptSegment(
            text="بسم الله الرحمن الرحيم، الحمد لله رب العالمين",
            timestamp=Timestamp(start_seconds=0, end_seconds=10),
            language=Language.ARABIC,
            confidence=0.95,
        ),
        TranscriptSegment(
            text="اليوم نتحدث عن فضل الصلاة وأهميتها في حياة المسلم",
            timestamp=Timestamp(start_seconds=10, end_seconds=25),
            language=Language.ARABIC,
            confidence=0.92,
        ),
        TranscriptSegment(
            text="قال الله تعالى في سورة البقرة: ﴿وَأَقِيمُوا الصَّلَاةَ﴾",
            timestamp=Timestamp(start_seconds=25, end_seconds=40),
            language=Language.ARABIC,
            confidence=0.94,
        ),
        TranscriptSegment(
            text="وقال رسول الله صلى الله عليه وسلم: الصلاة عماد الدين",
            timestamp=Timestamp(start_seconds=40, end_seconds=55),
            language=Language.ARABIC,
            confidence=0.91,
        ),
    ]

    return simulated_segments


async def summarize_lecture(
    audio_data: bytes,
    title: Optional[str] = None,
    language_hint: Optional[Language] = None,
) -> LectureSummary:
    """
    Process and summarize an Islamic lecture audio.

    Args:
        audio_data: Raw audio bytes
        title: Optional title for the lecture
        language_hint: Hint for primary language

    Returns:
        Complete lecture summary with transcription and analysis
    """
    warnings = []

    # Transcribe audio
    segments = await transcribe_audio(audio_data, language_hint)

    if not segments:
        warnings.append("No speech detected in audio")
        return LectureSummary(
            title=title or "Unknown Lecture",
            duration_seconds=0,
            primary_language=language_hint or Language.ARABIC,
            processing_warnings=warnings,
        )

    # Detect primary language
    combined_text = " ".join(s.text for s in segments)
    primary_language = detect_language(combined_text)

    # Detect references
    references = detect_references(segments)

    # Extract topics
    topics = extract_topics(segments)

    # Generate summary
    abstract, sections, key_takeaways = generate_summary(segments, topics, references)

    # Calculate duration
    duration = segments[-1].timestamp.end_seconds if segments else 0

    return LectureSummary(
        title=title or "Islamic Lecture",
        duration_seconds=duration,
        primary_language=primary_language,
        speakers=[Speaker(id="speaker_1", label="Primary Speaker", speaking_time=duration)],
        abstract=abstract,
        topics=topics,
        sections=sections,
        all_references=references,
        key_takeaways=key_takeaways,
        transcript_segments=segments,
        full_transcript=combined_text,
        processing_warnings=warnings,
    )


# FastAPI router
from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/audio", tags=["Audio Summarization"])


class SummarizationResponse(BaseModel):
    """Response model for lecture summarization."""
    success: bool
    data: dict


@router.post("/summarize", response_model=SummarizationResponse)
async def summarize_audio_lecture(
    file: UploadFile = File(...),
    title: Optional[str] = None,
    language: Optional[str] = None,
):
    """
    Summarize an Islamic lecture audio file.

    Supports:
    - Arabic and multilingual speech recognition
    - Islamic terminology-aware transcription
    - Quranic and Hadith reference detection
    - Topic extraction and segmentation
    - Timestamped summaries
    """
    # Validate file type
    allowed_types = ["audio/mpeg", "audio/wav", "audio/mp3", "audio/m4a", "audio/ogg"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported audio format. Allowed: {', '.join(allowed_types)}"
        )

    # Read audio data
    audio_data = await file.read()

    # Parse language hint
    lang_hint = None
    if language:
        try:
            lang_hint = Language(language)
        except ValueError:
            pass

    # Process lecture
    result = await summarize_lecture(
        audio_data=audio_data,
        title=title,
        language_hint=lang_hint,
    )

    return SummarizationResponse(
        success=True,
        data=result.to_dict(),
    )
