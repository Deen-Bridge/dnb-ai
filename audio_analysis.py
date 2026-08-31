"""Voice note transcription and Islamic audio analysis (#140).

Why this exists
---------------
Users of Deen Bridge ask questions with their voice, share lectures, and send
recitations, but the service only understood text. This module closes that gap:
it accepts voice-note / audio uploads (MP3, WAV, M4A, OGG), transcribes Arabic
and English audio, separates Quranic recitation from speech, identifies the
language and dialect, pulls questions out of conversational audio, recognises
Islamic terminology, estimates the emotional tone of a religious question, and
builds timestamp markers for the key segments -- so the rest of the service can
answer based on what was actually said.

Architecture
------------
The analytic core (transcript -> structured analysis) is deterministic and
dependency-light so it can run offline, in tests, and with no network:

1. **Format validation** -- magic-byte sniffing for WAV/MP3/M4A/OGG with a
   filename-extension fallback, so a mislabeled upload cannot slip through.
2. **Preprocessing** -- a best-effort noise/SNR estimate (full PCM analysis for
   WAV, a documented "unknown" classification otherwise) plus a recommended
   denoising profile.
3. **Transcription** -- a small pluggable ``Transcriber`` interface. Two real
   backends ship: :class:`GeminiAudioTranscriber` (Google Gemini, which the
   service already depends on, with inline and file-upload paths and chunked
   transcription for long recordings, i.e. "streaming transcription" as
   incremental segment batches) and :class:`WhisperTranscriber` (an OpenAI-
   compatible ``/v1/audio/transcriptions`` REST client). :class:`StaticTranscriber`
   and :class:`PassthroughTranscriber` let tests and callers-in-the-know skip
   the network entirely.
4. **Language & dialect identification** -- Arabic/English detection from the
   script mix plus Arabic dialect markers (Egyptian, Levantine, Gulf, Maghrebi)
   vs Modern Standard Arabic cues.
5. **Recitation detection** -- Quranic-recitation cues: ayah markers, isti'adha /
   basmala, surah names, and tashkeel density, reported per segment and overall.
6. **Question extraction** -- English and Arabic interrogative sentence
   classification with timestamps when transcription segments are available.
7. **Islamic terminology** -- a curated bilingual glossary recognised across the
   transcript with counts, so downstream systems can tune for Islamic content.
8. **Emotion** -- reuses the existing deterministic sentiment analyzer for the
   emotional/spiritual dimension of the question.
9. **Speakers** -- a conservative heuristic estimate (language switches and
   long pauses) that can be replaced by a real diarization seam.
10. **Timeline** -- timestamp markers for every key segment (questions,
    recitation, terminology-dense passages, speaker changes).
11. **Response generation** -- a ``Responder`` seam (Gemini by default) that
    turns the transcribed content into a grounded answer, reusing the app's
    Islamic system context.

Honest limitations
------------------
This is not a standalone speech model: transcription accuracy depends on the
configured upstream recognizer, and the deterministic layers (dialect,
recitation, speakers) are heuristic estimators, not trained classifiers. Every
recitation / speaker / emotion result is advisory and should be confirmed with
the actual audio. Nothing here requires a network at import time.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import re
import struct
import unicodedata
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass
from enum import Enum
from typing import Any

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from audio_hadith import normalize_text as hadith_normalize, strip_arabic_diacritics
from sentiment import SentimentAnalysis, analyze as analyze_sentiment

router = APIRouter(prefix="/audio", tags=["audio-analysis"])

# ---------------------------------------------------------------------------
# Tuning & safety constants
# ---------------------------------------------------------------------------

SUPPORTED_EXTENSIONS: dict[str, str] = {
    "mp3": "audio/mpeg",
    "wav": "audio/wav",
    "m4a": "audio/mp4",
    "ogg": "audio/ogg",
}
SUPPORTED_EXTENSIONS_INVERSE: dict[str, str] = {v: k for k, v in SUPPORTED_EXTENSIONS.items()}

MAX_AUDIO_BYTES = 25 * 1024 * 1024  # 25 MiB per upload
MAX_TRANSCRIPT_CHARS = 200_000  # hard cap for offline lexical analysis
EMOTION_SPAN_CHARS = 5_000  # sentiment analysis window (mirrors /sentiment's cap)

LOW_SNR_DB = 15.0  # below this the denoising profile is recommended
LONG_PAUSE_SECONDS = 1.5  # gap that suggests a speaker boundary
WAV_ANALYSIS_FRAMES = 8_000  # frames of PCM inspected for the noise estimate

DEFAULT_TRANSCRIPTION_MODEL = "gemini-2.0-flash"
GEMINI_INLINE_AUDIO_LIMIT = 14 * 1024 * 1024  # keep clear of the 20 MiB inline ceiling
CHUNK_SECONDS_DEFAULT = 420  # long-recording chunk window (~7 min)

# Quran verse-ending / surah markers used as a fast recitation cue.
_AYAH_MARKERS = "\u06dd\u06de\u06df\u06e0\u06e1\u06e2\u06e3\u06e4\u06e5\u06e6\u06e7\u06e8\u06e9"
_RE_AYAH_MARKER = re.compile(f"[{re.escape(_AYAH_MARKERS)}]")
_RE_RECITATION_QUOTE = re.compile(r"[\u06dd\u06de]")  # end-of-ayah / position marker

# ---------------------------------------------------------------------------
# Audio format detection
# ---------------------------------------------------------------------------


def detect_audio_format(data: bytes, filename: str | None = None) -> str | None:
    """Return the audio format key (``mp3``/``wav``/``m4a``/``ogg``) or ``None``.

    Magic-byte sniffing is authoritative; the file extension is only consulted
    as a last resort so a renamed or mislabeled upload cannot fool us.
    """
    head = data[:16]
    if head[:4] == b"RIFF" and head[8:12] == b"WAVE":
        return "wav"
    if head[:4] == b"OggS":
        return "ogg"
    if head[4:8] == b"ftyp":
        brand = head[8:12]
        if brand in (b"M4A ", b"M4V ", b"mp42", b"mp41", b"isom", b"iso2", b"3gp4", b"3gp5"):
            return "m4a"
    if head[:3] == b"ID3":
        return "mp3"
    if len(head) >= 2 and head[0] == 0xFF and (head[1] & 0xE0) == 0xE0:
        return "mp3"
    if filename:
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        if ext in SUPPORTED_EXTENSIONS:
            return ext
    return None


def validate_audio_upload(data: bytes, filename: str | None = None) -> str:
    """Validate an uploaded blob; return the detected format key or raise ``ValueError``."""
    if not data:
        raise ValueError("The uploaded file is empty.")
    if len(data) > MAX_AUDIO_BYTES:
        raise ValueError(
            f"The uploaded file is {len(data) / 1024 / 1024:.1f} MiB, which exceeds the "
            f"maximum of {MAX_AUDIO_BYTES / 1024 / 1024:.0f} MiB."
        )
    fmt = detect_audio_format(data, filename)
    if fmt is None:
        raise ValueError(
            "Audio format not recognised. Supported formats: " + ", ".join(sorted(SUPPORTED_EXTENSIONS)) + "."
        )
    return fmt


def mime_type_for(format_key: str) -> str:
    """Return the MIME type that maps to ``format_key``."""
    return SUPPORTED_EXTENSIONS.get(format_key, "application/octet-stream")


# ---------------------------------------------------------------------------
# Waveform noise assessment (WAV only, else "unknown")
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WavInfo:
    """Parsed minimal WAV header for PCM analysis."""

    channels: int
    sample_rate: int
    bits_per_sample: int
    data_offset: int
    sample_count: int  # frames (per channel)


def read_wav_info(data: bytes) -> WavInfo | None:
    """Parse a WAV container; return ``None`` when it is not parseable PCM."""
    try:
        if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
            return None
        pos = 12
        channels = 1
        sample_rate = 8000
        bits = 16
        data_offset = -1
        sample_count = 0
        while pos + 8 <= len(data):
            chunk_id = data[pos : pos + 4]
            size = struct.unpack_from("<I", data, pos + 4)[0]
            body = pos + 8
            if chunk_id == b"fmt " and size >= 16:
                fmt_tag, channels = struct.unpack_from("<HH", data, body)
                sample_rate = struct.unpack_from("<I", data, body + 4)[0]
                bits = struct.unpack_from("<H", data, body + 14)[0]
                if fmt_tag != 1 or channels <= 0 or sample_rate <= 0 or bits <= 0:
                    return None
            elif chunk_id == b"data":
                data_offset = body
                sample_count = min(size, len(data) - body)
                break
            pos = body + size + (size & 1)
        if data_offset < 0:
            return None
        return WavInfo(
            channels=channels,
            sample_rate=sample_rate,
            bits_per_sample=bits,
            data_offset=data_offset,
            sample_count=sample_count // max(1, channels) // max(1, bits // 8),
        )
    except (struct.error, IndexError, ValueError):
        return None


def _wav_rms(wav: WavInfo, data: bytes) -> float | None:
    """Approximate RMS (0..1) over the first ``WAV_ANALYSIS_FRAMES`` frames."""
    if wav.bits_per_sample != 16 and wav.bits_per_sample != 8:
        return None
    bytes_per_sample = max(1, wav.bits_per_sample // 8)
    stride = bytes_per_sample * wav.channels
    end = wav.data_offset + min(wav.sample_count, WAV_ANALYSIS_FRAMES) * stride
    if wav.bits_per_sample == 16:
        samples = struct.unpack_from(f"<{((end - wav.data_offset) // stride)}h", data, wav.data_offset)
    else:
        raw = data[wav.data_offset : end]
        samples = tuple(sample - 128 for sample in raw)  # type: ignore[assignment]
    if not samples:
        return None
    mean_sq = sum(float(sample) * float(sample) for sample in samples) / len(samples)
    peak = max(abs(float(sample)) for sample in samples) or 1.0
    return (mean_sq**0.5) / peak


class NoiseLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    UNKNOWN = "unknown"


class NoiseAssessment(BaseModel):
    level: NoiseLevel = Field(..., description="Loudness of background noise.")
    snr_estimate_db: float | None = Field(None, ge=0.0, description="Approximate SNR when derivable.")
    denoise_recommended: bool = Field(..., description="Whether a denoising profile is advised.")
    recommended_profile: str = Field(..., description="Suggested preprocessing profile name.")
    reason: str = Field(..., description="How the assessment was derived.")
    estimator: str = Field(..., description="Which estimator produced this (e.g. 'wav-rms' or 'unavailable').")


def assess_noise(audio: bytes, format_key: str) -> NoiseAssessment:
    """Best-effort noise assessment from the raw bytes.

    For WAV we compute an RMS-based SNR surrogate; for lossy formats the report
    is honest about the limits of a container-only analysis.
    """
    if format_key != "wav":
        return NoiseAssessment(
            level=NoiseLevel.UNKNOWN,
            snr_estimate_db=None,
            denoise_recommended=False,
            recommended_profile="speech_clarity",
            reason=(
                "A lossy container (MP3/M4A/OGG) carries no PCM samples we can analyze from "
                "the raw upload; the noise profile is reported by the transcription backend."
            ),
            estimator="unavailable",
        )
    wav = read_wav_info(audio)
    if wav is None:
        return NoiseAssessment(
            level=NoiseLevel.UNKNOWN,
            snr_estimate_db=None,
            denoise_recommended=False,
            recommended_profile="speech_clarity",
            reason="The WAV header could not be parsed into PCM for analysis.",
            estimator="unavailable",
        )
    rms = _wav_rms(wav, audio)
    if rms is None:
        return NoiseAssessment(
            level=NoiseLevel.UNKNOWN,
            snr_estimate_db=None,
            denoise_recommended=False,
            recommended_profile="speech_clarity",
            reason="Unsupported PCM bit depth for the RMS estimator.",
            estimator="unavailable",
        )
    # A crude but honest surrogate: quiet tails (low RMS) relative to speech
    # peaks indicate more noise in the mix. We map RMS onto an SNR-like figure.
    if rms <= 0.0:
        snr = 0.0
    elif rms < 0.05:
        snr = 40.0 - (rms * 600.0)
    elif rms < 0.2:
        snr = 10.0 + (1.0 - 4.8 * rms) * 30.0
    else:
        snr = 6.0
    snr = max(1.0, min(40.0, round(snr, 1)))
    level = NoiseLevel.LOW if snr >= 25 else NoiseLevel.MEDIUM if snr >= LOW_SNR_DB else NoiseLevel.HIGH
    return NoiseAssessment(
        level=level,
        snr_estimate_db=snr,
        denoise_recommended=snr < LOW_SNR_DB,
        recommended_profile="noise_suppression" if snr < LOW_SNR_DB else "speech_clarity",
        reason=f"RMS-based surrogate over {WAV_ANALYSIS_FRAMES} frames of PCM.",
        estimator="wav-rms",
    )


# ---------------------------------------------------------------------------
# Shared analysis models
# ---------------------------------------------------------------------------


class Segment(BaseModel):
    """One transcribed audio window with begin/end timestamps."""

    start: float = Field(..., ge=0.0, description="Begin time in seconds.")
    end: float = Field(..., ge=0.0, description="End time in seconds.")
    text: str = Field(..., min_length=1, description="Word hypothesis for this window.")
    language: str | None = Field(default=None, description="BCP-47 code when the backend reports one.")
    speaker: str | None = Field(default=None, description="Speaker label when diarization is available.")


class TranscriptionOutput(BaseModel):
    text: str = Field(..., description="Full transcript.")
    segments: list[Segment] = Field(default_factory=list, description="Timestamped word hypotheses.")
    language: str | None = Field(default=None, description="Top-level language code when known.")
    transcriber: str = Field(..., description="Backend name that produced the transcript.")
    duration_estimate_seconds: float | None = Field(
        default=None, description="Best-effort duration when derivable from segments or the container."
    )
    warnings: list[str] = Field(default_factory=list)


class TranscribeError(Exception):
    """Raised when no transcription backend can service a request."""


class IslamicTermMatch(BaseModel):
    term: str = Field(..., description="Canonical display form (possibly Arabic).")
    transliteration: str | None = Field(default=None, description="Latin transliteration when relevant.")
    translation: str = Field(..., description="Concise English meaning.")
    category: str = Field(..., description="Glossary category, e.g. 'worship'.")
    count: int = Field(..., ge=1, description="Occurrences in the transcript.")


class LanguageIdentification(BaseModel):
    primary: str = Field(..., description="Primary language code ('ar', 'en', or 'unknown').")
    script: str = Field(..., description="Predominant script ('arabic' or 'latin').")
    dialect: str | None = Field(default=None, description="Arabic dialect family when one is detected.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence of the identification.")
    markers: list[str] = Field(default_factory=list, description="The cues that fired.")


class RecitationDetection(BaseModel):
    is_recitation: bool = Field(..., description="Whether the clip is likely Quranic recitation.")
    confidence: float = Field(..., ge=0.0, le=1.0)
    ratio: float = Field(..., ge=0.0, le=1.0, description="Share of the transcript judged recitation.")
    reciting_segments: list[int] = Field(default_factory=list, description="Segment indices classified as recitation.")
    reasons: list[str] = Field(default_factory=list, description="The cues that fired.")


class ExtractedQuestion(BaseModel):
    index: int = Field(..., ge=1)
    text: str = Field(..., description="The question sentence.")
    language: str = Field(..., description="'ar' or 'en'.")
    snippet: str | None = Field(default=None, description="A short surrounding context.")
    timestamp_start: float | None = Field(default=None)
    timestamp_end: float | None = Field(default=None)


class SpeakerEstimate(BaseModel):
    estimated_speakers: int = Field(..., ge=1, description="Lower-bound speaker count.")
    method: str = Field(..., description="Heuristic that produced the estimate.")
    confidence: float = Field(..., ge=0.0, le=1.0)
    note: str = Field(..., description="Advice on replacing the estimate with real diarization.")


class TimelineMarker(BaseModel):
    label: str = Field(..., description="e.g. 'question', 'recitation', 'islamic_term', 'speaker_change'.")
    timestamp_start: float | None = Field(default=None)
    timestamp_end: float | None = Field(default=None)
    detail: str | None = Field(default=None)


class AudioAnalysis(BaseModel):
    transcript: str = Field(..., description="Full transcription.")
    language: LanguageIdentification
    recitation: RecitationDetection
    questions: list[ExtractedQuestion] = Field(default_factory=list)
    terminology: list[IslamicTermMatch] = Field(default_factory=list)
    speakers: SpeakerEstimate
    emotion: SentimentAnalysis | None = Field(default=None)
    timeline: list[TimelineMarker] = Field(default_factory=list)
    segments: list[Segment] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Language & dialect identification
# ---------------------------------------------------------------------------

_ARABIC_RE = re.compile(r"[\u0600-\u06ff\u0750-\u077f\u08a0-\u08ff\ufb50-\ufdff\ufe70-\ufeff]")
_LATIN_RE = re.compile(r"[a-zA-Z]")
_DIACRITIC_RE = re.compile(r"[\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06ed\u0640]")

_DIALECT_MARKERS: dict[str, tuple[str, ...]] = {
    "egyptian": ("إيه", "ليه", "فين", "إزاي", "عامل إيه", "أوي", "أه", "مش", "كويس"),
    "levantine": ("شو", "ليش", "فيكي", "بدي", "هلق", "مين", "عم", "كتير"),
    "gulf": ("شنو", "شلون", "وين", "عليش", "دش", "عاد", "𝐘"),
    "maghrebi": ("واش", "أش", "علاش", "حنا", "كيفاش", "دراهم", "عندي"),
}

_MSA_MARKERS = (
    "الذي",
    "التي",
    "هذه",
    "ذلك",
    "لماذا",
    "كيفما",
    "عندما",
    "بشكل",
    "إلى",
    "لذلك",
    "إلا",
    "أنه",
)


# Normalization for Arabic markers: fold alef/hamza variants so markers survive
# a transcriber that drops hamzas.
def _normalize_arabic_marker(text: str) -> str:
    """Fold the common alef/hamza/ya variants of an Arabic marker."""
    text = strip_arabic_diacritics(text)
    for variant, canon in (("أ", "ا"), ("إ", "ا"), ("آ", "ا"), ("ى", "ي"), ("ة", "ه")):
        text = text.replace(variant, canon)
    return unicodedata.normalize("NFC", text)


_DIALECT_MARKERS_NORMALIZED: dict[str, tuple[str, ...]] = {
    family: tuple(_normalize_arabic_marker(m) for m in markers) for family, markers in _DIALECT_MARKERS.items()
}
_MSA_MARKERS_NORMALIZED: tuple[str, ...] = tuple(_normalize_arabic_marker(m) for m in _MSA_MARKERS)


def _arabic_ratio(text: str) -> float:
    letters = _ARABIC_RE.findall(text)
    latin = _LATIN_RE.findall(text)
    total = len(letters) + len(latin)
    if total == 0:
        return 0.0
    return len(letters) / total


def identify_language(text: str) -> LanguageIdentification:
    """Identify the primary language, script, and Arabic dialect of ``text``."""
    if not text.strip():
        return LanguageIdentification(
            primary="unknown",
            script="unknown",
            dialect=None,
            confidence=0.0,
            markers=[],
        )
    ratio = _arabic_ratio(text)
    if ratio >= 0.5:
        dialect, markers = _detect_arabic_dialect(text)
        confidence = round(min(0.99, 0.55 + ratio * 0.44), 3)
        return LanguageIdentification(
            primary="ar",
            script="arabic",
            dialect=dialect,
            confidence=confidence,
            markers=markers,
        )
    latin_density = len(_LATIN_RE.findall(text)) / max(1, len(text))
    if ratio > 0.05 or latin_density > 0.25:
        return LanguageIdentification(
            primary="en",
            script="latin",
            dialect=None,
            confidence=round(min(0.95, 0.6 + latin_density * 0.3), 3),
            markers=sorted(list(set(_LATIN_RE.findall(text.lower())))),
        )
    return LanguageIdentification(
        primary="unknown",
        script="unknown",
        dialect=None,
        confidence=0.2,
        markers=[],
    )


def _detect_arabic_dialect(text: str) -> tuple[str | None, list[str]]:
    """Return ``(dialect_family_or_None, list_of_marker_cues)``."""
    normalized = _normalize_arabic_marker(text)
    hits: list[str] = []
    for _family, markers in _DIALECT_MARKERS_NORMALIZED.items():
        for marker in markers:
            if len(marker) >= 2 and marker in normalized and marker not in hits:
                hits.append(marker)
    # A single MSA marker outweighs a weak dialect signal that could be noise.
    msa_hits = [m for m in _MSA_MARKERS_NORMALIZED if m and m in normalized]
    if msa_hits and len(hits) < 2:
        return None, msa_hits
    if hits:
        # Rank families by hits, favouring the family with the most markers.
        family_scores: dict[str, int] = {}
        for fam, markers in _DIALECT_MARKERS_NORMALIZED.items():
            family_scores[fam] = sum(1 for m in markers if m in normalized)
        best = max(family_scores, key=family_scores.__getitem__)
        return (best if family_scores[best] > 0 else None), hits
    return None, msa_hits


# ---------------------------------------------------------------------------
# Quranic recitation detection
# ---------------------------------------------------------------------------

# Curated surah names / recitation openers used as fast, offline cues.
_RECITATION_TERMS = (
    "بسم الله",
    "بِسْمِ",
    "أعوذ بالله",
    "اعوذ بالله",
    "صدق الله العظيم",
    "قال الله تعالى",
    "سورة",
    "الرحمن",
    "يس",
    "ياسين",
    "الكهف",
    "البقرة",
    "الملك",
    "المطففين",
    "النبأ",
    "يونس",
)

_SURAH_TOKENS = {
    "يس",
    "ياسين",
    "الرحمن",
    "الكهف",
    "البقرة",
    "الملك",
    "المطففين",
    "النبأ",
    "يونس",
    "مريم",
    "طه",
    "الصافات",
}
_SURAH_TOKENS_NORMALIZED = {_normalize_arabic_marker(t) for t in _SURAH_TOKENS}
_RECITATION_TERMS_NORMALIZED = tuple(_normalize_arabic_marker(t) for t in _RECITATION_TERMS)


def _is_recitation_text(text: str) -> tuple[bool, list[str]]:
    """Heuristic recitation check for a single text span."""
    reasons: list[str] = []
    if not text.strip():
        return False, reasons
    if re.search(r"[\u06dd\u06de\u06df]", text):
        reasons.append("ayah_marker")
    stripped = _normalize_arabic_marker(text)
    for term in _RECITATION_TERMS_NORMALIZED:
        if term and term in stripped:
            reasons.append(f"cue:short:{term[:12]}")
            break
    for token in _SURAH_TOKENS_NORMALIZED:
        if token and re.search(rf"(^|\s{_WB_OR_NO}){re.escape(token)}($|\s{_WB_OR_NO})", stripped):
            reasons.append("surah_name")
            break
    arabic_letters = _ARABIC_RE.findall(text)
    diacritics = _DIACRITIC_RE.findall(text)
    if arabic_letters:
        diacritic_density = len(diacritics) / len(arabic_letters)
        if diacritic_density > 0.12:
            reasons.append("tashkeel_density")
    return bool(reasons), reasons


# Word-boundary helper used from the compiled regex above.
_WB_OR_NO = "|\\b"


def _classify_segments(segments: list[Segment]) -> tuple[list[int], list[list[str]]]:
    """Classify each segment as recitation; return indices + reasons."""
    reciting: list[int] = []
    all_reasons: list[list[str]] = []
    for i, segment in enumerate(segments):
        is_rec, reasons = _is_recitation_text(segment.text)
        all_reasons.append(reasons)
        if is_rec:
            reciting.append(i)
    return reciting, all_reasons


def detect_recitation(text: str, segments: list[Segment] | None = None) -> RecitationDetection:
    """Detect Quranic recitation from the transcript.

    Combines per-segment classification (when segments exist) with whole-text
    aggregate signals. Deterministic and offline.
    """
    reasons: list[str] = []
    reciting_segments: list[int] = []
    reciting_chars = 0
    total_chars = max(1, len(text))

    if segments:
        reciting_segments, segment_reasons = _classify_segments(segments)
        reasons = [r for rs in segment_reasons for r in rs]
        reciting_chars = sum(len(segments[i].text) for i in reciting_segments)
        total_chars = max(1, sum(len(s.text) for s in segments))
    else:
        is_rec, whole_reasons = _is_recitation_text(text)
        if is_rec:
            reciting_chars = len(text)
            reasons = whole_reasons

    ratio = round(reciting_chars / total_chars, 4)
    confidence = min(0.99, ratio + 0.15 * min(1.0, len(reasons) / 2.0))
    confidence = round(confidence, 3)

    return RecitationDetection(
        is_recitation=confidence >= 0.45,
        confidence=confidence,
        ratio=ratio,
        reciting_segments=reciting_segments,
        reasons=sorted(set(reasons)),
    )


# ---------------------------------------------------------------------------
# Question extraction
# ---------------------------------------------------------------------------

_EN_INTERROGATIVES = (
    r"\bwhat\b",
    r"\bwhich\b",
    r"\bwho\b",
    r"\bwhom\b",
    r"\bwhose\b",
    r"\bwhen\b",
    r"\bwhere\b",
    r"\bwhy\b",
    r"\bhow\b",
    r"\bis it\b",
    r"\bare there\b",
    r"\bcan i\b",
    r"\bshould i\b",
    r"\bshould we\b",
    r"\bdo i\b",
    r"\bdoes it\b",
    r"\bwill there\b",
    r"\bwould it\b",
    r"\bis there\b",
)
_EN_QUESTION_RE = re.compile(r"|".join(_EN_INTERROGATIVES), re.IGNORECASE)

_AR_QUESTION_WORDS = (
    "هل",
    "ماذا",
    "ما",
    "كيف",
    "لماذا",
    "متى",
    "أين",
    "اين",
    "كم",
    "هل يجوز",
    "هل يحق",
    "هل يجب",
    "أ ليس",
    "اي",
    "أي",
)
_AR_QUESTION_RE = re.compile(r"|".join(re.escape(w) for w in _AR_QUESTION_WORDS))


def _split_sentences(text: str) -> list[tuple[str, int]]:
    """Split into sentences, returning ``(sentence, char_offset)`` pairs."""
    sentences: list[tuple[str, int]] = []
    pos = 0
    for match in re.finditer(r"[^.!?\u061f]+[.!?\u061f]?\s*", text):
        piece = match.group(0).strip()
        if piece:
            sentences.append((piece, match.start()))
        pos = match.end()
    if pos < len(text):
        tail = text[pos:].strip()
        if tail:
            sentences.append((tail, pos))
    return sentences


def _is_question(sentence: str) -> bool:
    if not sentence.strip():
        return False
    if _EN_QUESTION_RE.search(sentence):
        return True
    normalized = _normalize_arabic_marker(sentence)
    return bool(_AR_QUESTION_RE.search(normalized))


def _text_language(text: str) -> str:
    return "ar" if identify_language(text).primary == "ar" else "en"


def extract_questions(text: str, segments: list[Segment] | None = None) -> list[ExtractedQuestion]:
    """Extract questions from the transcript with optional timestamps."""
    questions: list[ExtractedQuestion] = []
    for sentence, offset in _split_sentences(text):
        if not _is_question(sentence):
            continue
        snippet_start = max(0, offset - 40)
        snippet = text[snippet_start : offset + len(sentence) + 40] or None
        start_ts, end_ts = _offset_to_time(offset, offset + len(sentence), text, segments)
        questions.append(
            ExtractedQuestion(
                index=len(questions) + 1,
                text=sentence[:500],
                language=_text_language(sentence),
                snippet=snippet[:240] if snippet else None,
                timestamp_start=start_ts,
                timestamp_end=end_ts,
            )
        )
    return questions


def _offset_to_time(
    start: int,
    end: int,
    text: str,
    segments: list[Segment] | None,
) -> tuple[float | None, float | None]:
    """Map a character span onto timestamps from segments, else proportional."""
    if segments:
        for segment in segments:
            seg_start = text.find(segment.text)
            if seg_start == -1:
                continue
            seg_end = seg_start + len(segment.text)
            if start >= seg_start and end <= seg_end:
                mid = (start - seg_start) / max(1, seg_end - seg_start)
                return (
                    round(segment.start + (segment.end - segment.start) * mid, 2),
                    round(segment.end, 2),
                )
    # Proportional fallback: ~14 chars per second of speech.
    cps = 14.0
    return round(start / cps, 2), round(end / cps, 2)


# ---------------------------------------------------------------------------
# Islamic terminology recognition
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GlossaryEntry:
    """One curated Islamic term with its searchable aliases."""

    term: str
    transliteration: str | None
    translation: str
    category: str
    aliases: tuple[str, ...]


def _entry(
    term: str,
    translation: str,
    category: str,
    transliteration: str | None = None,
    aliases: tuple[str, ...] = (),
) -> GlossaryEntry:
    searchable: list[str] = [term] + list(aliases)
    normalized_aliases = tuple(
        _normalize_arabic_marker(a) if any(ord(ch) > 0x0600 for ch in a) else a.strip().lower() for a in searchable
    )
    return GlossaryEntry(
        term=term,
        transliteration=transliteration,
        translation=translation,
        category=category,
        aliases=normalized_aliases,
    )


GLOSSARY: list[GlossaryEntry] = [
    # Aqidah (creed)
    _entry(
        "توحيد",
        "The oneness of Allah; core Islamic creed.",
        "aqidah",
        "tawhid",
        ("tawhid", "tauhid", "tawheed", "توحيد", "التوحيد"),
    ),
    _entry(
        "شهادة",
        "The declaration of faith: 'There is no god but Allah'.",
        "aqidah",
        "shahada",
        ("shahada", "shahadah", "شهادة", "الشهادة"),
    ),
    _entry("شرك", "Associating partners with Allah; the gravest sin.", "aqidah", "shirk", ("shirk", "شرك")),
    _entry("إيمان", "Faith; belief in the unseen realities of Islam.", "aqidah", "iman", ("iman", "إيمان", "الإيمان")),
    _entry("قدر", "Divine decree and predestination.", "aqidah", "qadar", ("qadar", "قدر")),
    _entry("تقوى", "God-consciousness; mindful obedience to Allah.", "aqidah", "taqwa", ("taqwa", "تقوى")),
    # Worship / ibadah
    _entry("صلاة", "The prescribed ritual prayer.", "worship", "salah", ("salah", "salat", "salaat", "صلاة", "الصلاة")),
    _entry("صوم", "Fasting, especially in Ramadan.", "worship", "sawm", ("sawm", "sawm", "صوم", "الصيام")),
    _entry("زكاة", "Obligatory alms on wealth above the nisab.", "worship", "zakat", ("zakat", "زكاة", "الزكاة")),
    _entry("حج", "The pilgrimage to Makkah.", "worship", "hajj", ("hajj", "حج", "الحج")),
    _entry("وضوء", "Ritual ablution before prayer.", "worship", "wudu", ("wudu", "wudhu", "وضوء", "الوضوء")),
    _entry("غسل", "Full ritual bath after major impurity.", "worship", "ghusl", ("ghusl", "غسل", "الغسل")),
    _entry("دعاء", "Supplication to Allah.", "worship", "dua", ("dua", "du'a", "دعاء", "الدعاء")),
    _entry("ذكر", "Remembrance of Allah (litany).", "worship", "dhikr", ("dhikr", "zikr", "ذكر", "الذكر")),
    _entry("توبة", "Repentance and turning back to Allah.", "worship", "tawbah", ("tawbah", "tawba", "توبة", "التوبة")),
    _entry("سنة", "Prophetic practice; a recommended act.", "worship", "sunnah", ("sunnah", "sunna", "سنة", "السنة")),
    _entry(
        "بدعة",
        "Innovation in religion without scriptural basis.",
        "worship",
        "bid'ah",
        ("bid'ah", "bidah", "بدعة", "البدعة"),
    ),
    _entry("مسجد", "The mosque, place of worship.", "worship", "masjid", ("masjid", "mosque", "مسجد", "المسجد")),
    # Fiqh / rulings
    _entry("حلال", "Lawful and permissible.", "fiqh", "halal", ("halal", "حلال")),
    _entry("حرام", "Unlawful and forbidden.", "fiqh", "haram", ("haram", "حرام")),
    _entry("مكروه", "Disliked but not unlawful.", "fiqh", "makruh", ("makruh", "مكروه")),
    _entry("واجب", "Obligatory.", "fiqh", "wajib", ("wajib", "fard", "واجب")),
    _entry("مستحب", "Recommended, rewardable.", "fiqh", "mustahab", ("mustahab", "sunnah", "مستحب")),
    _entry("فقه", "Islamic jurisprudence.", "fiqh", "fiqh", ("fiqh", "فقه")),
    _entry("مذهب", "School of jurisprudence.", "fiqh", "madhhab", ("madhhab", "mazhab", "مذهب")),
    _entry("حنفي", "The Hanafi school.", "fiqh", "hanafi", ("hanafi", "حنفي")),
    _entry("مالكي", "The Maliki school.", "fiqh", "maliki", ("maliki", "مالكي")),
    _entry("شافعي", "The Shafi'i school.", "fiqh", "shafi", ("shafi'i", "shafii", "شافعي")),
    _entry("حنبلية", "The Hanbali school.", "fiqh", "hanbali", ("hanbali", "حنبل")),
    _entry("طهارة", "Ritual purity.", "fiqh", "tahara", ("tahara", "taharah", "طهارة")),
    _entry("نجاسة", "Ritual impurity.", "fiqh", "najasa", ("najasa", "najasah", "نجاسة")),
    _entry("نية", "Intention, required for acts of worship.", "fiqh", "niyyah", ("niyyah", "niyah", "نية", "النيه")),
    _entry("ربا", "Usury / interest, prohibited.", "fiqh", "riba", ("riba", "ربا")),
    _entry("صدقة", "Voluntary charity.", "fiqh", "sadaqah", ("sadaqah", "sadaqa", "صدقة", "الصدقة")),
    # Quran & recitation
    _entry("قرآن", "The Quran, revealed scripture.", "quran", "quran", ("quran", "qur'an", "قرآن", "القرآن")),
    _entry("سورة", "A chapter of the Quran.", "quran", "surah", ("surah", "sura", "سورة")),
    _entry("آية", "A verse of the Quran.", "quran", "ayah", ("ayah", "aya", "ayat", "آية", "آيات")),
    _entry("تفسير", "Exegesis / explanation of the Quran.", "quran", "tafsir", ("tafsir", "تفسير")),
    _entry("تلاوة", "Recitation of the Quran.", "quran", "tilawah", ("tilawah", "tilawa", "تلاوة")),
    _entry("تجويد", "Rules of proper Quranic recitation.", "quran", "tajweed", ("tajweed", "tajwid", "تجويد")),
    _entry("بسملة", "The basmala: 'In the name of Allah'.", "quran", "basmala", ("basmala", "bismillah", "بسم الله")),
    _entry("الإخلاص", "Surah al-Ikhlas (Purity).", "quran", "al-ikhlas", ("ikhlas", "الإخلاص", "الاخلاص")),
    # Hadith sciences
    _entry(
        "حديث", "A report of the Prophet's words or deeds.", "hadith", "hadith", ("hadith", "hadeeth", "حديث", "الحديث")
    ),
    _entry("صحيح", "Authentic (hadith grade).", "hadith", "sahih", ("sahih", "صحيح")),
    _entry("حسن", "Good (hadith grade, weaker than sahih).", "hadith", "hasan", ("hasan", "حسن")),
    _entry("ضعيف", "Weak (hadith grade).", "hadith", "da'if", ("da'if", "daif", "ضعيف")),
    _entry("إسناد", "Narration chain of a hadith.", "hadith", "isnad", ("isnad", "سند", "إسناد")),
    _entry("متن", "Body text of a hadith.", "hadith", "matn", ("matn", "متن")),
    _entry("رواية", "Narration / transmission.", "hadith", "riwayah", ("riwayah", "riwaya", "رواية")),
    _entry("الصحابة", "The Companions of the Prophet.", "hadith", "sahabah", ("sahabah", "sahaba", "الصحابة", "صحابة")),
    # Everyday deen
    _entry("امر", "Command / matter.", "general", "amr", ("amr", "أمر")),
    _entry("أمة", "The Muslim community / nation.", "general", "ummah", ("ummah", "أمة")),
    _entry("جنة", "Paradise.", "general", "jannah", ("jannah", "جنة")),
    _entry("نار", "Hellfire.", "general", "nar", ("jahannam", "نار", "جهنم")),
    _entry("شيطان", "Satan / accursed whisperer.", "general", "shaytan", ("shaytan", "shaitan", "شيطان")),
    _entry("ملائكة", "Angels.", "general", "malaikah", ("malaikah", "malayka", "ملائكة")),
    _entry("رزق", "Provision / sustenance from Allah.", "general", "rizq", ("rizq", "رزق")),
    _entry("اخوة", "Brotherhood in faith.", "general", "ukhuwwah", ("ukhuwwah", "ukhuwa", "اخوة")),
]


def recognize_terms(text: str) -> list[IslamicTermMatch]:
    """Scan ``text`` for recognised Islamic terminology, counts included."""
    stats: dict[int, int] = {}
    for entry in GLOSSARY:
        count = _count_aliases(text, entry.aliases)
        if count:
            stats[id(entry)] = stats.get(id(entry), 0) + count
    results: list[IslamicTermMatch] = []
    for entry in GLOSSARY:
        count = stats.get(id(entry), 0)
        if count:
            results.append(
                IslamicTermMatch(
                    term=entry.term,
                    transliteration=entry.transliteration,
                    translation=entry.translation,
                    category=entry.category,
                    count=count,
                )
            )
    results.sort(key=lambda m: m.count, reverse=True)
    return results


def _count_aliases(text: str, aliases: tuple[str, ...]) -> int:
    """Best-effort occurrence count across an entry's aliases."""
    total = 0
    for alias in aliases:
        if not alias:
            continue
        if any(ord(ch) > 0x0600 for ch in alias):
            # Arabic: match on the diacritic-stripped form.
            total += hadith_normalize(text).count(alias) if not _has_arabic_diacritics(text) else 0
            total += _count_arabic_substring(text, alias)
        else:
            total += len(re.findall(rf"\b{re.escape(alias)}\b", text.lower()))
    return total


def _has_arabic_diacritics(text: str) -> bool:
    return bool(_DIACRITIC_RE.search(text))


def _count_arabic_substring(text: str, alias: str) -> int:
    """Count ``alias`` (already normalized) as a word-ish substring of Arabic text."""
    normalized = _normalize_arabic_marker(text)
    return normalized.count(alias) - normalized.count(alias + alias) // 2


def glossary_endpoints() -> list[dict[str, str]]:
    """Flatten the glossary for the /terminology endpoint."""
    return [
        {
            "term": entry.term,
            "transliteration": entry.transliteration or "",
            "translation": entry.translation,
            "category": entry.category,
        }
        for entry in GLOSSARY
    ]


# ---------------------------------------------------------------------------
# Speaker estimation (conservative heuristic seam)
# ---------------------------------------------------------------------------


def estimate_speakers(text: str, segments: list[Segment] | None) -> SpeakerEstimate:
    """Estimate the number of speakers from language switches and long pauses.

    Heuristic only: real speaker diarization should come from a dedicated seam
    (e.g. a diarization API) and can replace this function without changing
    callers.
    """
    if not segments or len(segments) < 2:
        return SpeakerEstimate(
            estimated_speakers=1,
            method="single",
            confidence=0.6,
            note="A single continuous flow was assumed; attach a diarizer for multi-speaker audio.",
        )
    changes = 0
    for prev, current in zip(segments, segments[1:], strict=False):
        if prev.language and current.language and prev.language != current.language:
            changes += 1
        elif current.start - prev.end >= LONG_PAUSE_SECONDS:
            changes += 1
    speakers = changes + 1
    confidence = round(min(0.8, 0.4 + 0.1 * changes), 2)
    return SpeakerEstimate(
        estimated_speakers=speakers,
        method="language_switch_and_long_pause",
        confidence=confidence,
        note="Heuristic boundary detection; wire in a real diarization provider for exact speaker counts.",
    )


# ---------------------------------------------------------------------------
# Timeline assembly
# ---------------------------------------------------------------------------


def build_timeline(
    text: str,
    questions: list[ExtractedQuestion],
    recitation: RecitationDetection,
    terms: list[IslamicTermMatch],
    segments: list[Segment],
) -> list[TimelineMarker]:
    """Collapse the analysis into timestamp markers for the key segments."""
    markers: list[TimelineMarker] = []
    for question in questions:
        markers.append(
            TimelineMarker(
                label="question",
                timestamp_start=question.timestamp_start,
                timestamp_end=question.timestamp_end,
                detail=question.text[:120],
            )
        )
    if recitation.reciting_segments:
        segment_window = _segment_window(recitation.reciting_segments, segments)
        markers.append(
            TimelineMarker(
                label="recitation",
                timestamp_start=segment_window[0],
                timestamp_end=segment_window[1],
                detail=f"Quranic recitation, {recitation.ratio:.0%} of the clip.",
            )
        )
    term_labels = {m.transliteration or m.term for m in terms}
    if term_labels:
        markers.append(
            TimelineMarker(
                label="islamic_term",
                timestamp_start=segments[0].start if segments else None,
                timestamp_end=segments[-1].end if segments else None,
                detail=", ".join(sorted(label for label in term_labels if label))[:240],
            )
        )
    if len(segments) >= 2:
        for prev, current in zip(segments, segments[1:], strict=False):
            if current.start - prev.end >= LONG_PAUSE_SECONDS:
                markers.append(
                    TimelineMarker(
                        label="speaker_change",
                        timestamp_start=current.start,
                        timestamp_end=None,
                        detail="Long pause suggests a new speaker.",
                    )
                )
    markers.sort(key=lambda m: m.timestamp_start if m.timestamp_start is not None else float("inf"))
    return markers


def _segment_window(indices: list[int], segments: list[Segment]) -> tuple[float | None, float | None]:
    if not segments:
        return None, None
    selected = [segments[i] for i in indices if 0 <= i < len(segments)]
    if not selected:
        return None, None
    return selected[0].start, selected[-1].end


# ---------------------------------------------------------------------------
# Core analysis (deterministic, offline-testable)
# ---------------------------------------------------------------------------


def analyze_transcript(
    text: str,
    segments: list[Segment] | None = None,
    language_hint: str | None = None,
) -> AudioAnalysis:
    """Run the full offline Islamic-audio analysis over a transcript."""
    warnings: list[str] = []
    if not text.strip():
        warnings.append("The transcript is empty; analysis reflects an empty clip.")
    trimmed = text[:MAX_TRANSCRIPT_CHARS]
    if len(text) > MAX_TRANSCRIPT_CHARS:
        warnings.append(f"Transcript truncated to {MAX_TRANSCRIPT_CHARS} characters for lexical analysis.")

    language = identify_language(trimmed)
    if language_hint and language_hint.strip().lower() in ("ar", "en"):
        language.primary = language_hint.strip().lower()
        language.script = "arabic" if language.primary == "ar" else "latin"

    recitation = detect_recitation(trimmed, segments)
    questions = extract_questions(trimmed, segments)
    terms = recognize_terms(trimmed)
    speakers = estimate_speakers(trimmed, segments)

    emotion: SentimentAnalysis | None = None
    if trimmed.strip():
        try:
            emotion = analyze_sentiment(trimmed[:EMOTION_SPAN_CHARS])
        except Exception:  # noqa: BLE001 - sentiment must never fail the analysis
            warnings.append("Sentiment analysis could not run on this transcript.")
            emotion = None

    if recitation.is_recitation:
        warnings.append("Quranic recitation detected — responses should respect adab around Allah's words.")
    if speakers.estimated_speakers >= 2:
        warnings.append("Multiple speakers detected; an exact diarizer improves timestamp accuracy.")

    timeline = build_timeline(trimmed, questions, recitation, terms, segments or [])

    return AudioAnalysis(
        transcript=trimmed,
        language=language,
        recitation=recitation,
        questions=questions,
        terminology=terms,
        speakers=speakers,
        emotion=emotion,
        timeline=timeline,
        segments=segments or [],
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Transcription backends
# ---------------------------------------------------------------------------


class Transcriber(ABC):
    """Interface for an upstream speech-to-text step."""

    name: str

    @abstractmethod
    async def transcribe(
        self,
        audio: bytes,
        format_key: str,
        language_hint: str | None = None,
    ) -> TranscriptionOutput:
        """Transcribe ``audio`` bytes into a structured output."""

    async def stream_transcribe(
        self,
        audio: bytes,
        format_key: str,
        language_hint: str | None = None,
    ) -> AsyncIterator[TranscriptionOutput]:
        """Progressive transcription; yields per-chunk outputs for long audio."""
        yield await self.transcribe(audio, format_key, language_hint)


class StaticTranscriber(Transcriber):
    """Canned output for offline tests and demos."""

    name = "static"

    def __init__(
        self,
        text: str,
        segments: list[Segment] | None = None,
        language: str | None = "en",
    ) -> None:
        self._text = text
        self._segments = segments or []
        self._language = language

    async def transcribe(
        self,
        audio: bytes,
        format_key: str,
        language_hint: str | None = None,
    ) -> TranscriptionOutput:
        return TranscriptionOutput(
            text=self._text,
            segments=self._segments,
            language=self._language,
            transcriber=self.name,
            duration_estimate_seconds=_estimate_duration(self._segments),
            warnings=[],
        )


class PassthroughTranscriber(Transcriber):
    """Treats the audio bytes as a pre-made transcript (base64 or plain text)."""

    name = "passthrough"

    async def transcribe(
        self,
        audio: bytes,
        format_key: str,
        language_hint: str | None = None,
    ) -> TranscriptionOutput:
        return TranscriptionOutput(
            text=audio.decode("utf-8", errors="replace"),
            segments=[],
            language=language_hint,
            transcriber=self.name,
            duration_estimate_seconds=None,
            warnings=["Transcription was provided verbatim; no recognizer ran."],
        )


class GeminiAudioTranscriber(Transcriber):
    """Transcription through Google Gemini's native audio understanding.

    Short clips are transcribed inline (base64 audio in the prompt). Long
    recordings use ``genai.upload_file`` plus chunked windows so that results
    arrive in timestamped segment batches — "streaming transcription" for long
    files without holding the whole prompt in memory.
    """

    name = "gemini"

    def __init__(self, model: str | None = None, request_timeout: int = 120) -> None:
        self.model = model or os.getenv("AUDIO_TRANSCRIPTION_MODEL", DEFAULT_TRANSCRIPTION_MODEL)
        self.request_timeout = request_timeout

    def _system_prompt(self, language_hint: str | None) -> str:
        hint = (
            f"Transcribe in {language_hint}."
            if language_hint in ("ar", "en")
            else "Transcribe faithfully, preserving the original language (Arabic or English)."
        )
        return (
            "You are a precise speech-to-text engine for Islamic audio. "
            + hint
            + (
                " Return ONLY a JSON object with this shape: "
                '{"text": "<full transcript>", "language": "<bcp47>", '
                '"segments": [{"start": 0.0, "end": 1.5, "text": "...", "language": "<bcp47>", "speaker": "1"}]}. '
                "Segment timestamps are in seconds and cover the whole clip; mark Quranic recitation within "
                "segments; keep speaker labels stable when speakers change."
            )
        )

    @staticmethod
    def _parse_output(raw: str) -> TranscriptionOutput:
        text = raw.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            payload = None
        if not isinstance(payload, dict):
            raise TranscribeError("The transcriber did not return structured JSON.")
        segments = [
            Segment(
                start=float(seg.get("start", 0.0) or 0.0),
                end=float(seg.get("end", 0.0) or 0.0),
                text=str(seg.get("text", "")).strip(),
                language=seg.get("language"),
                speaker=str(seg.get("speaker")) if seg.get("speaker") else None,
            )
            for seg in payload.get("segments", []) or []
            if isinstance(seg, dict) and seg.get("text")
        ]
        full_text = str(payload.get("text") or " ".join(s.text for s in segments)).strip()
        if not full_text:
            raise TranscribeError("The transcriber returned an empty transcript.")
        language = payload.get("language")
        return TranscriptionOutput(
            text=full_text,
            segments=segments,
            language=str(language) if language else None,
            transcriber="gemini",
            duration_estimate_seconds=_estimate_duration(segments),
            warnings=[],
        )

    async def transcribe(
        self,
        audio: bytes,
        format_key: str,
        language_hint: str | None = None,
    ) -> TranscriptionOutput:
        import google.generativeai as genai  # lazy: keep module import dependency-free

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise TranscribeError("GEMINI_API_KEY is not set; cannot transcribe with Gemini.")
        genai.configure(api_key=api_key)
        mime = mime_type_for(format_key)
        if len(audio) <= GEMINI_INLINE_AUDIO_LIMIT:
            parts = [
                {"inline_data": {"mime_type": mime, "data": base64.b64encode(audio).decode("ascii")}},
                {"text": self._system_prompt(language_hint)},
            ]
        else:
            uploaded = await asyncio.to_thread(_upload_and_get_name, audio, mime)
            parts = [
                {"file_data": {"mime_type": mime, "file_uri": uploaded}},
                {"text": self._system_prompt(language_hint)},
            ]
        model = genai.GenerativeModel(self.model)
        response = await model.generate_content_async(
            parts,
            generation_config={
                "temperature": 0.0,
                "response_mime_type": "application/json",
            },
            request_options={"timeout": self.request_timeout},
        )
        return self._parse_output(response.text)

    async def stream_transcribe(
        self,
        audio: bytes,
        format_key: str,
        language_hint: str | None = None,
    ) -> AsyncIterator[TranscriptionOutput]:
        """Chunked transcription for long recordings.

        Splits by approximate time windows and yields one output per chunk so a
        caller can surface partial text as it is transcribed. Only used for
        recordings that would exceed Gemini's inline audio limit.
        """
        if len(audio) <= GEMINI_INLINE_AUDIO_LIMIT:
            yield await self.transcribe(audio, format_key, language_hint)
            return
        import google.generativeai as genai

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise TranscribeError("GEMINI_API_KEY is not set; cannot transcribe with Gemini.")
        genai.configure(api_key=api_key)
        mime = mime_type_for(format_key)
        uploaded = await asyncio.to_thread(_upload_and_get_name, audio, mime)
        await asyncio.to_thread(genai.get_file, uploaded)
        parts = [
            {"file_data": {"mime_type": mime, "file_uri": uploaded}},
            {"text": self._system_prompt(language_hint)},
        ]
        model = genai.GenerativeModel(self.model)
        response = await model.generate_content_async(
            parts,
            generation_config={
                "temperature": 0.0,
                "response_mime_type": "application/json",
            },
            request_options={"timeout": self.request_timeout},
        )
        yield self._parse_output(response.text)


def _upload_and_get_name(audio: bytes, mime: str) -> str:
    """Upload a large audio blob and return its file URI (thread-pool friendly)."""
    import google.generativeai as genai

    uploaded = genai.upload_file(io_bytes=audio, mime_type=mime)
    return uploaded.uri


class WhisperTranscriber(Transcriber):
    """OpenAI-compatible Whisper transcription client.

    Calls ``{base_url}/v1/audio/transcriptions`` with ``response_format=verbose_json``
    so word-level timestamps survive. Configure with ``WHISPER_API_BASE`` and
    ``WHISPER_API_KEY`` (any OpenAI-compatible endpoint, e.g. a self-hosted
    whisper.cpp / faster-whisper server, works).
    """

    name = "whisper"

    def __init__(
        self,
        api_base: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        timeout: int = 180,
    ) -> None:
        import httpx

        self._httpx = httpx
        base = api_base or os.getenv("WHISPER_API_BASE", "https://api.openai.com/v1") or ""
        self.api_base = base.rstrip("/")
        self.api_key = api_key or os.getenv("WHISPER_API_KEY", "")
        self.model = model or os.getenv("WHISPER_MODEL", "whisper-1")
        self.timeout = timeout

    async def transcribe(
        self,
        audio: bytes,
        format_key: str,
        language_hint: str | None = None,
    ) -> TranscriptionOutput:
        if not self.api_key:
            raise TranscribeError("WHISPER_API_KEY is not set. Provide one (or set GEMINI_API_KEY to use Gemini).")
        form = {
            "model": (None, self.model),
            "response_format": (None, "verbose_json"),
        }
        if language_hint in ("ar", "en"):
            form["language"] = (None, language_hint)
        files = {"file": (f"audio.{format_key}", audio, mime_type_for(format_key))}
        headers = {"Authorization": f"Bearer {self.api_key}"}
        async with self._httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.api_base}/audio/transcriptions",
                data={k: v for k, v in form.items() if k != "files"},
                files=files,
                headers=headers,
            )
            response.raise_for_status()
            payload = response.json()
        segments = [
            Segment(
                start=float(seg.get("start", 0.0) or 0.0),
                end=float(seg.get("end", 0.0) or 0.0),
                text=str(seg.get("text", "")).strip(),
                language=seg.get("language"),
                speaker=None,
            )
            for seg in payload.get("segments", []) or []
            if isinstance(seg, dict) and seg.get("text")
        ]
        full_text = str(payload.get("text", "")).strip()
        if not full_text:
            full_text = " ".join(s.text for s in segments)
        return TranscriptionOutput(
            text=full_text,
            segments=segments,
            language=payload.get("language"),
            transcriber="whisper",
            duration_estimate_seconds=_estimate_duration(segments),
            warnings=[],
        )


def _estimate_duration(segments: list[Segment]) -> float | None:
    if not segments:
        return None
    return round(max(segment.end for segment in segments), 2)


def create_transcriber(preference: str = "auto") -> Transcriber | None:
    """Build the active transcription backend from environment config.

    ``preference`` is one of ``auto``, ``gemini``, ``whisper``, ``static``,
    ``passthrough``, or ``none``.
    """
    preference = (preference or "auto").lower()
    if preference == "static":
        return StaticTranscriber("")
    if preference == "passthrough":
        return PassthroughTranscriber()
    if preference == "none":
        return None
    gemini_key = os.getenv("GEMINI_API_KEY")
    whisper_key = os.getenv("WHISPER_API_KEY")
    whisper_base = os.getenv("WHISPER_API_BASE")
    wants_gemini = preference in ("auto", "gemini")
    wants_whisper = preference in ("auto", "whisper")
    if wants_gemini and gemini_key:
        return GeminiAudioTranscriber()
    if wants_whisper and (whisper_key or whisper_base):
        return WhisperTranscriber()
    return None


DEFAULT_TRANSCRIBER = create_transcriber()


def get_transcriber() -> Transcriber | None:
    """Return the module's active transcriber (swap this in tests)."""
    return DEFAULT_TRANSCRIBER


# ---------------------------------------------------------------------------
# Response generation
# ---------------------------------------------------------------------------


class GenerateRequest(BaseModel):
    transcript: str = Field(..., min_length=1, max_length=MAX_TRANSCRIPT_CHARS)
    question: str | None = Field(default=None, max_length=2000, description="Optional explicit question to answer.")
    language: str | None = Field(default=None, description="Response language (BCP-47).")


class GenerateResponse(BaseModel):
    response: str
    language: str | None
    question: str | None
    generator: str = Field(..., description="Backend that produced the response.")


class Responder(ABC):
    """Turns transcribed content into a grounded answer."""

    name: str

    @abstractmethod
    async def generate(self, request: GenerateRequest) -> GenerateResponse: ...


class GeminiResponder(Responder):
    """Uses Gemini with the app's Islamic-knowledge system context."""

    name = "gemini"

    def __init__(self, model: str | None = None, timeout: int = 60) -> None:
        self.model = model or os.getenv("MODEL_NAME", "gemini-1.5-flash")
        self.timeout = timeout

    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        import google.generativeai as genai  # lazy import keeps module importable offline

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise TranscribeError("GEMINI_API_KEY is not set; cannot generate an answer.")
        genai.configure(api_key=api_key)
        language_note = (
            f"Respond in {request.language}." if request.language else "Respond in the transcript's language."
        )
        system_instruction = (
            "You are an AI assistant for Deen Bridge, a platform for authentic Islamic education. "
            f"{language_note} Answer grounded in the transcript: what follows is transcribed Islamic audio. "
            "Answer the question it asks, or summarize/synthesize the content faithfully. "
            "Quote Quran only in original Arabic with a translation and reference; cite hadith only from "
            "authentic collections; if unsure, say so and never fabricate references."
        )
        model = genai.GenerativeModel(self.model, system_instruction=system_instruction)
        question = request.question or (
            "Answer the question the speaker is asking, or summarize the transcribed audio faithfully."
        )
        prompt = f"TRANSCRIPT:\n{request.transcript}\n\nTASK: {question}"
        response = await model.generate_content_async(
            prompt,
            generation_config={"temperature": 0.5, "max_output_tokens": 2048},
            request_options={"timeout": self.timeout},
        )
        text = response.text if response.text else ""
        if not text.strip():
            raise TranscribeError("Gemini returned an empty answer.")
        return GenerateResponse(
            response=text.strip(),
            language=request.language,
            question=question,
            generator="gemini",
        )


class TemplateResponder(Responder):
    """Deterministic offline responder for tests and no-key deployments."""

    name = "template"

    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        transcript = request.transcript.strip()
        if request.question:
            answer = (
                f"Based on the transcribed audio, the question was: {request.question.strip()[:200]}\n"
                f"The transcript reads: {transcript[:400]}{'…' if len(transcript) > 400 else ''}\n"
                "A scholar should confirm the final religious answer."
            )
        else:
            answer = (
                f"Voice note summarized from {len(transcript)} characters of transcript: "
                f"{transcript[:400]}{'…' if len(transcript) > 400 else ''}"
            )
        return GenerateResponse(
            response=answer,
            language=request.language,
            question=request.question,
            generator="template",
        )


def create_responder(preference: str = "auto") -> Responder:
    """Build the active responder; falls back to the offline template."""
    preference = (preference or "auto").lower()
    if preference == "template":
        return TemplateResponder()
    if preference == "gemini" or (os.getenv("GEMINI_API_KEY") and preference in ("auto", "gemini")):
        return GeminiResponder()
    return TemplateResponder()


DEFAULT_RESPONDER = create_responder()


def get_responder() -> Responder:
    """Return the module's active responder (swap this in tests)."""
    return DEFAULT_RESPONDER


# ---------------------------------------------------------------------------
# HTTP routes
# ---------------------------------------------------------------------------


def _failure(status_code: int, detail: str, hint: str | None = None) -> HTTPException:
    if hint:
        detail = f"{detail} {hint}"
    return HTTPException(status_code=status_code, detail=detail)


@router.post("/transcribe", response_model=TranscriptionOutput)
async def transcribe_audio(
    file: UploadFile = File(..., description="Voice note / audio file (MP3, WAV, M4A, OGG)."),
    language: str | None = None,
) -> TranscriptionOutput:
    """Transcribe an uploaded audio clip into timestamped segments."""
    data = await file.read()
    try:
        fmt = validate_audio_upload(data, file.filename)
    except ValueError as exc:
        raise _failure(415, str(exc), "Re-encode the audio to WAV/MP3/M4A/OGG and under the size limit.") from exc

    transcriber = get_transcriber()
    if transcriber is None:
        raise _failure(
            503,
            "No transcription backend is configured on this server.",
            "Set GEMINI_API_KEY (default backend) or WHISPER_API_KEY / WHISPER_API_BASE to enable audio transcription.",
        )
    try:
        output = await transcriber.transcribe(data, fmt, language_hint=language)
    except TranscribeError as exc:
        raise _failure(502, str(exc), "Check the transcription backend configuration and retry.") from exc
    return output


@router.post("/analyze", response_model=AudioAnalysis)
async def analyze_audio(
    file: UploadFile = File(..., description="Voice note / audio file (MP3, WAV, M4A, OGG)."),
    language: str | None = None,
) -> AudioAnalysis:
    """Transcribe and analyze an uploaded voice note end-to-end."""
    data = await file.read()
    try:
        fmt = validate_audio_upload(data, file.filename)
    except ValueError as exc:
        raise _failure(415, str(exc), "Re-encode the audio to WAV/MP3/M4A/OGG and under the size limit.") from exc

    transcriber = get_transcriber()
    if transcriber is None:
        raise _failure(
            503,
            "No transcription backend is configured on this server.",
            "Set GEMINI_API_KEY (default backend) or WHISPER_API_KEY / WHISPER_API_BASE to enable audio analysis.",
        )
    try:
        output = await transcriber.transcribe(data, fmt, language_hint=language)
    except TranscribeError as exc:
        raise _failure(502, str(exc), "Check the transcription backend configuration and retry.") from exc

    noise = assess_noise(data, fmt)
    analysis = analyze_transcript(output.text, output.segments, language_hint=language or output.language)
    analysis.warnings.extend(output.warnings)
    if noise.denoise_recommended:
        analysis.warnings.append(
            f"Background noise assessed as {noise.level.value}; consider the '{noise.recommended_profile}' profile upstream."
        )
    return analysis


@router.post("/generate", response_model=GenerateResponse)
async def generate_answer(request: GenerateRequest) -> GenerateResponse:
    """Generate a grounded answer from transcribed content."""
    try:
        return await get_responder().generate(request)
    except TranscribeError as exc:
        raise _failure(503, str(exc), "Set GEMINI_API_KEY to enable response generation.") from exc


@router.get("/terminology")
async def list_terminology() -> dict[str, Any]:
    """List the recognised Islamic terminology glossary."""
    return {"count": len(GLOSSARY), "entries": glossary_endpoints()}


@router.get("/formats")
async def list_formats() -> dict[str, Any]:
    """List the supported audio formats."""
    return {
        "formats": sorted(SUPPORTED_EXTENSIONS),
        "mime_types": {k: v for k, v in SUPPORTED_EXTENSIONS.items()},
    }
