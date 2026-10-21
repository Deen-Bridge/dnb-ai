"""Audio Hadith verification, authentication, and video lecture transcripts.

Why this exists
----------------
Hadith audio lectures circulate widely, and a misquoted or fabricated
narration presented on air can spread fast. This module takes the *transcript*
of an audio narration (the output of an upstream Arabic speech-recognition
step) and checks it against a small, bundled corpus of well-known authenticated
narrations: it normalizes the wording, finds the best textual match, extracts
and reports the chain of narration (isnad) with narrator biographies when they
are named, surfaces the authenticity grade of the matched hadith, and flags
likely misquotations when the transcript drifts too far from any known text.

Scope and honest limitations
----------------------------
This is a deterministic, dependency-free heuristic layer — **not** a trained
speech model and **not** an exhaustive hadith database. It ships no real ASR;
audio is represented by its transcript, and a small pluggable ``Transcriber``
interface lets a real recognizer be injected later without changing callers.
The same pluggable interface also backs the video-lecture transcript pipeline
below: a ``VideoTranscriber`` extracts audio from MP4/MOV/AVI/WebM files and
produces a diarized, timestamped, topic-segmented transcript with Quranic and
hadith citation detection. No real media decoding or ASR is bundled — the
default implementation is deterministic and dependency-free, and a real
recognizer/decoder can be injected without changing callers.
The bundled corpus holds only a handful of famous narrations for demonstration
and testing. Every result is advisory and should be confirmed against a full
hadith reference (e.g. Sunnah.com) and a qualified scholar. The module has no
import-time side effects and needs no network or live services.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Iterable

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(prefix="/audio-hadith", tags=["audio-hadith"])

# Below this best-match score a transcript is treated as unmatched / a likely
# misquotation rather than a confident hit against the corpus.
MATCH_THRESHOLD = 0.45
# Video-lecture transcript limits and supported container formats.
MAX_LECTURE_LENGTH = 200_000
SUPPORTED_VIDEO_FORMATS = ("mp4", "mov", "avi", "webm")
# A confident, near-verbatim match.
STRONG_MATCH_THRESHOLD = 0.75
MAX_TRANSCRIPT_LENGTH = 20_000


# ---------------------------------------------------------------------------
# Text normalization
# ---------------------------------------------------------------------------

# Arabic combining diacritics (tashkeel) — stripped before matching so that a
# recognizer that drops or adds harakat still lines up with the corpus.
_ARABIC_DIACRITICS = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭ]")
_NON_WORD = re.compile(r"[^\w\s]", re.UNICODE)
_WS = re.compile(r"\s+")

# Transliteration / synonym folding: many equivalent English renderings of the
# same term should collapse to one token before matching.
_SYNONYMS: dict[str, str] = {
    "messenger": "prophet",
    "rasool": "prophet",
    "rasul": "prophet",
    "nabi": "prophet",
    "muhammad": "prophet",
    "allahs": "allah",
    "god": "allah",
    "deeds": "deed",
    "actions": "deed",
    "action": "deed",
    "intentions": "intention",
    "niyyah": "intention",
    "niyah": "intention",
    "hadeeth": "hadith",
    "narrated": "narrate",
    "reported": "narrate",
    "related": "narrate",
}

_STOPWORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "of",
        "to",
        "is",
        "are",
        "was",
        "were",
        "and",
        "or",
        "by",
        "for",
        "in",
        "on",
        "that",
        "this",
        "he",
        "she",
        "it",
        "his",
        "her",
        "him",
        "who",
        "whom",
        "be",
        "will",
        "shall",
        "with",
        "as",
        "so",
        "but",
        "from",
    }
)


def strip_arabic_diacritics(text: str) -> str:
    """Remove Arabic tashkeel and normalize alef/hamza variants."""
    text = _ARABIC_DIACRITICS.sub("", text)
    # Fold common alef and ya/hamza variants to a canonical form.
    for variant, canon in (
        ("أ", "ا"),  # alef with hamza above -> alef
        ("إ", "ا"),  # alef with hamza below -> alef
        ("آ", "ا"),  # alef madda -> alef
        ("ى", "ي"),  # alef maqsura -> ya
        ("ة", "ه"),  # ta marbuta -> ha
    ):
        text = text.replace(variant, canon)
    return text


def normalize_text(text: str) -> str:
    """Lowercase, strip diacritics/punctuation, and collapse whitespace."""
    text = unicodedata.normalize("NFC", text)
    text = strip_arabic_diacritics(text)
    text = text.lower()
    text = _NON_WORD.sub(" ", text)
    text = _WS.sub(" ", text)
    return text.strip()


def tokenize(text: str) -> list[str]:
    """Normalize, fold synonyms, and drop stopwords into a token list."""
    tokens: list[str] = []
    for tok in normalize_text(text).split():
        tok = _SYNONYMS.get(tok, tok)
        if tok in _STOPWORDS:
            continue
        tokens.append(tok)
    return tokens


def match_score(a_tokens: list[str], b_tokens: list[str]) -> float:
    """Blended token similarity in ``[0, 1]``.

    Combines Jaccard overlap (penalizes extra/missing words in either text)
    with containment of the shorter token set in the longer one (rewards a
    transcript that quotes a hadith faithfully even if it adds framing words).
    """
    a, b = set(a_tokens), set(b_tokens)
    if not a or not b:
        return 0.0
    inter = len(a & b)
    jaccard = inter / len(a | b)
    containment = inter / min(len(a), len(b))
    return round(0.5 * jaccard + 0.5 * containment, 4)


# ---------------------------------------------------------------------------
# Narrator biographies and isnad extraction
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Narrator:
    """A known narrator with a one-line biographical note."""

    name: str
    bio: str


# A small, well-known set of Companions and early narrators.
_NARRATORS: dict[str, Narrator] = {
    "umar ibn al-khattab": Narrator(
        "Umar ibn al-Khattab",
        "The second Rightly-Guided Caliph and a senior Companion (d. 23 AH).",
    ),
    "abu hurayrah": Narrator(
        "Abu Hurayrah",
        "A Companion who narrated the largest number of hadith (d. 59 AH).",
    ),
    "aishah": Narrator(
        "Aishah bint Abi Bakr",
        "Wife of the Prophet (peace be upon him) and a major narrator of hadith (d. 58 AH).",
    ),
    "anas ibn malik": Narrator(
        "Anas ibn Malik",
        "A Companion who served the Prophet (peace be upon him) as a youth (d. 93 AH).",
    ),
    "abdullah ibn umar": Narrator(
        "Abdullah ibn Umar",
        "Son of Umar ibn al-Khattab and a careful, prolific narrator (d. 73 AH).",
    ),
    "ibn abbas": Narrator(
        "Abdullah ibn Abbas",
        "Cousin of the Prophet (peace be upon him), called the scholar of the Ummah (d. 68 AH).",
    ),
    "al-nu'man ibn bashir": Narrator(
        "Al-Nu'man ibn Bashir",
        "A Companion and later governor; narrator of the 'lawful and unlawful are clear' hadith.",
    ),
}

# Phrases that introduce a narrator in a transcript, in English and common
# transliterations of the classical isnad verbs.
_ISNAD_MARKERS = (
    "narrated by",
    "reported by",
    "related by",
    "on the authority of",
    "reported from",
    "narrated from",
    "haddathana",
    "akhbarana",
    "haddathani",
)

_NARRATOR_ALIASES: dict[str, str] = {
    "umar": "umar ibn al-khattab",
    "umar ibn khattab": "umar ibn al-khattab",
    "abu huraira": "abu hurayrah",
    "aisha": "aishah",
    "ibn umar": "abdullah ibn umar",
    "abdullah ibn abbas": "ibn abbas",
    "nu'man ibn bashir": "al-nu'man ibn bashir",
}


def _canonical_narrator_key(name: str) -> str | None:
    # Normalize both the candidate and every known form the same way so that
    # punctuation (hyphens, apostrophes) and "bin"/"ibn" spelling do not matter.
    key = normalize_text(name).replace(" bin ", " ibn ").strip()
    if not key:
        return None
    for canon in _NARRATORS:
        if normalize_text(canon) == key:
            return canon
    for alias, canon in _NARRATOR_ALIASES.items():
        if normalize_text(alias) == key:
            return canon
    # Substring fall-back: the transcript may embed a name in a longer phrase.
    for canon in _NARRATORS:
        nk = normalize_text(canon)
        if nk in key or key in nk:
            return canon
    return None


def extract_isnad(transcript: str) -> list[Narrator]:
    """Pull out narrators introduced by isnad markers, de-duplicated in order."""
    lowered = transcript.lower()
    found: list[Narrator] = []
    seen: set[str] = set()
    for marker in _ISNAD_MARKERS:
        start = 0
        while True:
            idx = lowered.find(marker, start)
            if idx == -1:
                break
            start = idx + len(marker)
            # Grab the next few words after the marker as a candidate name.
            tail = transcript[start : start + 60]
            candidate = re.split(r"[,.;:]|\bthat\b|\bsaid\b|\bwho\b", tail, maxsplit=1)[0]
            key = _canonical_narrator_key(candidate)
            if key and key not in seen:
                seen.add(key)
                found.append(_NARRATORS[key])
    return found


# ---------------------------------------------------------------------------
# Authenticated-text corpus
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HadithEntry:
    """A bundled, authenticated narration used as a matching reference."""

    id: str
    text: str
    collection: str
    reference: str
    grade: str
    narrator: str
    tokens: list[str] = field(default_factory=list)


def _entry(id: str, text: str, collection: str, reference: str, grade: str, narrator: str) -> HadithEntry:
    return HadithEntry(
        id=id,
        text=text,
        collection=collection,
        reference=reference,
        grade=grade,
        narrator=narrator,
        tokens=tokenize(text),
    )


_CORPUS: list[HadithEntry] = [
    _entry(
        "intentions",
        "Actions are but by intentions, and every person will have only what they intended.",
        "Sahih al-Bukhari",
        "Bukhari 1",
        "sahih",
        "Umar ibn al-Khattab",
    ),
    _entry(
        "muslim-tongue-hand",
        "The Muslim is the one from whose tongue and hand the Muslims are safe.",
        "Sahih al-Bukhari",
        "Bukhari 10",
        "sahih",
        "Abdullah ibn Amr",
    ),
    _entry(
        "love-for-brother",
        "None of you truly believes until he loves for his brother what he loves for himself.",
        "Sahih al-Bukhari",
        "Bukhari 13",
        "sahih",
        "Anas ibn Malik",
    ),
    _entry(
        "good-word-charity",
        "A good word is charity.",
        "Sahih al-Bukhari",
        "Bukhari 2989",
        "sahih",
        "Abu Hurayrah",
    ),
    _entry(
        "lawful-unlawful-clear",
        "The lawful is clear and the unlawful is clear, and between them are doubtful matters.",
        "Sahih al-Bukhari",
        "Bukhari 52",
        "sahih",
        "Al-Nu'man ibn Bashir",
    ),
    _entry(
        "seeking-knowledge",
        "Seeking knowledge is an obligation upon every Muslim.",
        "Sunan Ibn Majah",
        "Ibn Majah 224",
        "hasan",
        "Anas ibn Malik",
    ),
]


class MatchResult(BaseModel):
    """One candidate corpus match with its score and metadata."""

    hadith_id: str
    matched_text: str
    collection: str
    reference: str
    grade: str
    corpus_narrator: str
    score: float


def find_matches(transcript: str, limit: int = 3) -> list[MatchResult]:
    """Score the transcript against every corpus entry, best first."""
    q_tokens = tokenize(transcript)
    scored = [
        MatchResult(
            hadith_id=e.id,
            matched_text=e.text,
            collection=e.collection,
            reference=e.reference,
            grade=e.grade,
            corpus_narrator=e.narrator,
            score=match_score(q_tokens, e.tokens),
        )
        for e in _CORPUS
    ]
    scored.sort(key=lambda m: m.score, reverse=True)
    return scored[:limit]


# ---------------------------------------------------------------------------
# Pluggable transcription (no real ASR bundled)
# ---------------------------------------------------------------------------


class Transcriber:
    """Interface for an upstream speech-to-text step.

    The default implementation is a pass-through used when a caller already has
    a transcript; a real Arabic recognizer can subclass this without changing
    any downstream verification logic.
    """

    def transcribe(self, audio_ref: str) -> str:  # pragma: no cover - interface
        raise NotImplementedError


class PassthroughTranscriber(Transcriber):
    """Treats the supplied ``audio_ref`` as the transcript itself."""

    def transcribe(self, audio_ref: str) -> str:
        return audio_ref


# ---------------------------------------------------------------------------
# Verification core
# ---------------------------------------------------------------------------


class NarratorInfo(BaseModel):
    name: str
    bio: str


class VerifyRequest(BaseModel):
    transcript: str = Field(
        ...,
        min_length=1,
        max_length=MAX_TRANSCRIPT_LENGTH,
        description="Transcribed text of the audio narration (output of an ASR step).",
    )
    language: str = Field("en", description="ISO code of the transcript language.")


class VerifyResponse(BaseModel):
    verified: bool
    confidence: float
    grade: str | None
    best_match: MatchResult | None
    candidates: list[MatchResult]
    isnad: list[NarratorInfo]
    flagged_misquotation: bool
    note: str


def verify_transcript(transcript: str, language: str = "en") -> VerifyResponse:
    """Match a narration transcript against the corpus and assemble a verdict."""
    candidates = find_matches(transcript)
    best = candidates[0] if candidates and candidates[0].score > 0 else None
    isnad = [NarratorInfo(name=n.name, bio=n.bio) for n in extract_isnad(transcript)]

    confidence = best.score if best else 0.0
    verified = best is not None and confidence >= MATCH_THRESHOLD
    grade = best.grade if (verified and best is not None) else None

    # A misquotation is a partial-but-weak match: the transcript clearly
    # gestures at a known hadith yet does not line up cleanly with it.
    flagged_misquotation = best is not None and 0.2 <= confidence < STRONG_MATCH_THRESHOLD

    if verified and best is not None:
        note = (
            f"Matched {best.reference} ({best.grade}) with confidence "
            f"{confidence:.2f}. Confirm against a full hadith reference before relying on it."
        )
        if flagged_misquotation:
            note += " Wording differs from the authenticated text — possible misquotation; verify the exact narration."
    else:
        note = (
            "No authenticated narration in the bundled corpus matched this transcript with "
            "sufficient confidence. Do not treat it as a verified hadith; check a full "
            "reference (e.g. Sunnah.com) and a qualified scholar."
        )

    return VerifyResponse(
        verified=verified,
        confidence=round(confidence, 4),
        grade=grade,
        best_match=best,
        candidates=candidates,
        isnad=isnad,
        flagged_misquotation=flagged_misquotation,
        note=note,
    )


@router.post("/verify", response_model=VerifyResponse)
async def verify_audio_hadith(request: VerifyRequest) -> VerifyResponse:
    """Verify a transcribed audio Hadith narration against the bundled corpus."""
    if not request.transcript.strip():
        raise HTTPException(status_code=422, detail="Transcript must not be empty.")
    return verify_transcript(request.transcript, request.language)


# ---------------------------------------------------------------------------
# Video lecture transcript extraction
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TranscriptSegment:
    """A timestamped, speaker-attributed span of a lecture transcript."""

    start: float
    end: float
    speaker: str
    text: str


@dataclass(frozen=True)
class TopicSegment:
    """A contiguous run of transcript segments grouped under one topic."""

    title: str
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class Citation:
    """A Quranic verse or hadith reference detected in the spoken text."""

    kind: str  # "quran" or "hadith"
    reference: str
    timestamp: float
    excerpt: str


class VideoTranscriber:
    """Interface for extracting a diarized transcript from a video file.

    A real implementation would decode the container, run ASR, and perform
    speaker diarization. The default is a deterministic, dependency-free
    pass-through that treats the supplied reference as pre-segmented text so
    the downstream pipeline (topics, citations, summaries) is testable without
    a media decoder or network service.
    """

    def transcribe(self, video_ref: str, language: str = "en") -> list[TranscriptSegment]:
        raise NotImplementedError


class PassthroughVideoTranscriber(VideoTranscriber):
    """Splits ``video_ref`` on blank lines into speaker-tagged segments.

    Each non-empty line is treated as one utterance; an optional ``Speaker:``
    prefix sets the speaker, otherwise ``Speaker 1`` is assumed. Timestamps are
    assigned sequentially at a fixed cadence so downstream logic has stable
    anchors.
    """

    _CADENCE_SECONDS = 5.0

    def transcribe(self, video_ref: str, language: str = "en") -> list[TranscriptSegment]:
        segments: list[TranscriptSegment] = []
        cursor = 0.0
        for raw in video_ref.splitlines():
            line = raw.strip()
            if not line:
                continue
            speaker = "Speaker 1"
            text = line
            if ":" in line:
                head, _, tail = line.partition(":")
                if head.strip() and len(head.strip().split()) <= 4:
                    speaker = head.strip()
                    text = tail.strip()
            if not text:
                continue
            segments.append(
                TranscriptSegment(
                    start=cursor,
                    end=cursor + self._CADENCE_SECONDS,
                    speaker=speaker,
                    text=text,
                )
            )
            cursor += self._CADENCE_SECONDS
        return segments


_VIDEO_EXT = re.compile(r"\.([A-Za-z0-9]+)$")


def _detect_format(video_ref: str) -> str:
    match = _VIDEO_EXT.search(video_ref.strip())
    if not match:
        return ""
    return match.group(1).lower()


_QURAN_REF = re.compile(
    r"\b(?:quran|qur'an|surah|surat)\s+([A-Za-z0-9' -]{2,40}?)\s+(\d{1,3})\s*[:.]\s*(\d{1,3})\b",
    re.IGNORECASE,
)
_HADITH_REF = re.compile(
    r"\b(bukhari|muslim|abu dawud|tirmidhi|nasai|ibn majah)\s+(\d{1,5})\b",
    re.IGNORECASE,
)


def _detect_citations(segments: Iterable[TranscriptSegment]) -> list[Citation]:
    citations: list[Citation] = []
    for seg in segments:
        for m in _QURAN_REF.finditer(seg.text):
            citations.append(
                Citation(
                    kind="quran",
                    reference=f"{m.group(1).strip()} {m.group(2)}:{m.group(3)}",
                    timestamp=seg.start,
                    excerpt=seg.text.strip(),
                )
            )
        for m in _HADITH_REF.finditer(seg.text):
            citations.append(
                Citation(
                    kind="hadith",
                    reference=f"{m.group(1).title()} {m.group(2)}",
                    timestamp=seg.start,
                    excerpt=seg.text.strip(),
                )
            )
    return citations


_TOPIC_MARKERS = (
    "now let us",
    "next topic",
    "moving on",
    "the next point",
    "let's discuss",
    "let us discuss",
    "another important",
    "finally",
)


def _segment_topics(segments: list[TranscriptSegment]) -> list[TopicSegment]:
    if not segments:
        return []
    buckets: list[list[TranscriptSegment]] = [[]]
    for seg in segments:
        lowered = seg.text.lower()
        if buckets[-1] and any(marker in lowered for marker in _TOPIC_MARKERS):
            buckets.append([])
        buckets[-1].append(seg)
    topics: list[TopicSegment] = []
    for idx, bucket in enumerate(buckets, start=1):
        if not bucket:
            continue
        first = bucket[0].text.strip().split()
        title = " ".join(first[:6]) or f"Topic {idx}"
        topics.append(
            TopicSegment(
                title=title,
                start=bucket[0].start,
                end=bucket[-1].end,
                text=" ".join(s.text for s in bucket),
            )
        )
    return topics


def _summarize(segments: list[TranscriptSegment], topics: list[TopicSegment]) -> tuple[str, list[str]]:
    if not segments:
        return "", []
    speakers = sorted({s.speaker for s in segments})
    summary = (
        f"Lecture with {len(segments)} utterances across {len(topics)} topic(s) "
        f"and {len(speakers)} speaker(s): {', '.join(speakers)}."
    )
    takeaways = [f"{t.title} ({t.start:.0f}s–{t.end:.0f}s)" for t in topics[:5]]
    return summary, takeaways


class SegmentModel(BaseModel):
    start: float
    end: float
    speaker: str
    text: str


class TopicModel(BaseModel):
    title: str
    start: float
    end: float
    text: str


class CitationModel(BaseModel):
    kind: str
    reference: str
    timestamp: float
    excerpt: str


class LectureRequest(BaseModel):
    video_ref: str = Field(
        ...,
        min_length=1,
        max_length=MAX_LECTURE_LENGTH,
        description="Video file reference or pre-segmented transcript text.",
    )
    language: str = Field("en", description="ISO code of the lecture language.")


class LectureResponse(BaseModel):
    format: str
    language: str
    segments: list[SegmentModel]
    topics: list[TopicModel]
    citations: list[CitationModel]
    summary: str
    key_takeaways: list[str]
    searchable_text: str


def extract_lecture_transcript(
    video_ref: str,
    language: str = "en",
    transcriber: VideoTranscriber | None = None,
) -> LectureResponse:
    """Run the full video-lecture transcript pipeline."""
    fmt = _detect_format(video_ref)
    if fmt and fmt not in SUPPORTED_VIDEO_FORMATS:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported video format '.{fmt}'. Supported: {', '.join(SUPPORTED_VIDEO_FORMATS)}.",
        )
    engine = transcriber or PassthroughVideoTranscriber()
    segments = engine.transcribe(video_ref, language)
    topics = _segment_topics(segments)
    citations = _detect_citations(segments)
    summary, takeaways = _summarize(segments, topics)
    searchable = "\n".join(
        f"[{s.start:.1f}-{s.end:.1f}] {s.speaker}: {s.text}" for s in segments
    )
    return LectureResponse(
        format=fmt or "unknown",
        language=language,
        segments=[SegmentModel(**s.__dict__) for s in segments],
        topics=[TopicModel(**t.__dict__) for t in topics],
        citations=[CitationModel(**c.__dict__) for c in citations],
        summary=summary,
        key_takeaways=takeaways,
        searchable_text=searchable,
    )


@router.post("/lecture-transcript", response_model=LectureResponse)
async def lecture_transcript(request: LectureRequest) -> LectureResponse:
    """Extract a diarized, topic-segmented transcript from a video lecture."""
    if not request.video_ref.strip():
        raise HTTPException(status_code=422, detail="Video reference must not be empty.")
    return extract_lecture_transcript(request.video_ref, request.language)
