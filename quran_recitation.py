"""Quranic recitation analysis.

Why this exists
----------------
Learners and moderators receive recitation clips and want to know *what* is
being recited (surah and ayah), whether it matches the Mushaf, and a little
about the style of delivery. This module takes the *transcript* of a recitation
(the output of an upstream Arabic speech-recognition step, optionally with
per-word timings) and aligns it word-by-word against bundled Mushaf text. It
identifies the surah and ayah range, handles partial recitations and mixed
content (basmalah, commentary, other speech before or after the recitation),
reports omitted / substituted / inserted words, attaches word-level timestamps,
and surfaces riwayah, reciter and maqam metadata.

Scope and honest limitations
----------------------------
This is a deterministic, dependency-free heuristic layer — **not** an acoustic
model. It ships no ASR, no voice fingerprinting and no pitch analysis, so:

* reciter identity and maqam are only reported when the caller supplies them
  (as a hint or in file metadata such as a title); they are never guessed from
  sound;
* riwayah is inferred only from a handful of known textual differences, and
  otherwise falls back to the caller's hint or the Hafs 'an 'Asim text the
  bundled Mushaf follows;
* timestamps come from caller-supplied word timings, or are estimated
  proportionally from the clip duration (and labelled as estimated);
* the bundled Mushaf covers only a few short surahs. Everything is advisory;
  verify against a printed Mushaf and a qualified teacher of tajwid.

A pluggable ``Transcriber`` lets a real recognizer be injected later without
changing callers. The module has no import-time side effects and needs no
network.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher, get_close_matches

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(prefix="/quran-recitation", tags=["quran-recitation"])

MAX_TRANSCRIPT_LENGTH = 20_000
MAX_WORD_TIMINGS = 5_000
# Minimum aligned words (and share of the transcript) to claim an identification.
MIN_MATCHED_WORDS = 2
MIN_COVERAGE = 0.3
# Fuzzy-snap cutoffs: noisy audio produces noisier transcripts, so be lenient.
SNAP_CUTOFF_CLEAN = 0.85
SNAP_CUTOFF_NOISY = 0.75
NOISY_ASR_CONFIDENCE = 0.6


# ---------------------------------------------------------------------------
# Text normalization
# ---------------------------------------------------------------------------

_DIACRITICS = re.compile("[ؐ-ًؚ-ٰٟۖ-ۭـ]")
_NON_WORD = re.compile(r"[^\w\s]", re.UNICODE)
_WS = re.compile(r"\s+")
_FOLDS = (
    ("ٱ", "ا"),  # alef wasla
    ("أ", "ا"),  # alef with hamza above
    ("إ", "ا"),  # alef with hamza below
    ("آ", "ا"),  # alef madda
    ("ى", "ي"),  # alef maqsura -> ya
    ("ة", "ه"),  # ta marbuta -> ha
)


def normalize_arabic(text: str) -> str:
    """Strip tashkeel/tatweel/punctuation and fold alef, ya and ta-marbuta variants."""
    text = unicodedata.normalize("NFC", text)
    text = _DIACRITICS.sub("", text)
    for variant, canon in _FOLDS:
        text = text.replace(variant, canon)
    text = _NON_WORD.sub(" ", text)
    return _WS.sub(" ", text).strip()


def tokenize(text: str) -> list[str]:
    return normalize_arabic(text).split()


# ---------------------------------------------------------------------------
# Bundled Mushaf text (Hafs 'an 'Asim, short surahs only)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Surah:
    number: int
    name: str
    ayat: tuple[str, ...]

    @property
    def words(self) -> list[tuple[int, int, str]]:
        """Flat ``(ayah, word_index, normalized_word)`` list, 1-based ayah."""
        return [
            (a, i, w)
            for a, text in enumerate(self.ayat, start=1)
            for i, w in enumerate(tokenize(text), start=1)
        ]


_SURAHS: dict[int, Surah] = {
    s.number: s
    for s in (
        Surah(
            1,
            "Al-Fatihah",
            (
                "بسم الله الرحمن الرحيم",
                "الحمد لله رب العالمين",
                "الرحمن الرحيم",
                "مالك يوم الدين",
                "إياك نعبد وإياك نستعين",
                "اهدنا الصراط المستقيم",
                "صراط الذين أنعمت عليهم غير المغضوب عليهم ولا الضالين",
            ),
        ),
        Surah(
            103,
            "Al-'Asr",
            (
                "والعصر",
                "إن الإنسان لفي خسر",
                "إلا الذين آمنوا وعملوا الصالحات وتواصوا بالحق وتواصوا بالصبر",
            ),
        ),
        Surah(
            108,
            "Al-Kawthar",
            (
                "إنا أعطيناك الكوثر",
                "فصل لربك وانحر",
                "إن شانئك هو الأبتر",
            ),
        ),
        Surah(
            112,
            "Al-Ikhlas",
            (
                "قل هو الله أحد",
                "الله الصمد",
                "لم يلد ولم يولد",
                "ولم يكن له كفوا أحد",
            ),
        ),
        Surah(
            113,
            "Al-Falaq",
            (
                "قل أعوذ برب الفلق",
                "من شر ما خلق",
                "ومن شر غاسق إذا وقب",
                "ومن شر النفاثات في العقد",
                "ومن شر حاسد إذا حسد",
            ),
        ),
        Surah(
            114,
            "An-Nas",
            (
                "قل أعوذ برب الناس",
                "ملك الناس",
                "إله الناس",
                "من شر الوسواس الخناس",
                "الذي يوسوس في صدور الناس",
                "من الجنة والناس",
            ),
        ),
    )
}

_BASMALAH = tokenize(_SURAHS[1].ayat[0])

# ---------------------------------------------------------------------------
# Riwayah, reciters and maqamat
# ---------------------------------------------------------------------------

# Display name -> aliases accepted in a hint.
RIWAYAT: dict[str, tuple[str, ...]] = {
    "Hafs 'an 'Asim": ("hafs", "hafs an asim", "hafs 'an 'asim", "حفص"),
    "Shu'bah 'an 'Asim": ("shubah", "shu'bah", "shuba", "شعبة"),
    "Warsh 'an Nafi'": ("warsh", "ورش"),
    "Qalun 'an Nafi'": ("qalun", "qaloon", "قالون"),
    "Ad-Duri 'an Abi 'Amr": ("duri", "al-duri", "ad-duri", "الدوري"),
    "Khalaf 'an Hamzah": ("khalaf", "خلف"),
}

# Words whose spelling differs between riwayat: Mushaf word -> {heard: readers}.
# Al-Fatihah 1:4 "Maliki" (long mim, Hafs/Shu'bah) vs "Maliki" (short, Warsh etc.).
_RIWAYAH_VARIANTS: dict[str, dict[str, tuple[str, ...]]] = {
    "مالك": {"ملك": ("Warsh 'an Nafi'", "Qalun 'an Nafi'", "Ad-Duri 'an Abi 'Amr", "Khalaf 'an Hamzah")},
}
_LONG_MIM_RIWAYAT = ("Hafs 'an 'Asim", "Shu'bah 'an 'Asim")


@dataclass(frozen=True)
class Reciter:
    name: str
    country: str
    style: str
    aliases: tuple[str, ...]


# Style notes only; no claim is made about a given clip being by them.
_RECITERS: tuple[Reciter, ...] = (
    Reciter("Mishary Rashid Alafasy", "Kuwait", "murattal", ("alafasy", "afasy", "العفاسي")),
    Reciter("Abdul Basit Abdus-Samad", "Egypt", "mujawwad", ("abdul basit", "abdulbasit", "عبد الباسط")),
    Reciter("Mahmoud Khalil Al-Husary", "Egypt", "murattal", ("husary", "hussary", "الحصري")),
    Reciter("Muhammad Siddiq Al-Minshawi", "Egypt", "mujawwad", ("minshawi", "menshawy", "المنشاوي")),
    Reciter("Abdurrahman As-Sudais", "Saudi Arabia", "murattal", ("sudais", "السديس")),
    Reciter("Saud Ash-Shuraim", "Saudi Arabia", "murattal", ("shuraim", "shuraym", "الشريم")),
)

MAQAMAT: dict[str, tuple[str, ...]] = {
    "Bayati": ("bayati", "bayyati", "بياتي"),
    "Hijaz": ("hijaz", "hijazi", "حجاز"),
    "Rast": ("rast", "راست"),
    "Sika": ("sika", "siga", "سيكا"),
    "Saba": ("saba", "صبا"),
    "Nahawand": ("nahawand", "nahawend", "نهاوند"),
    "Ajam": ("ajam", "عجم"),
    "Kurd": ("kurd", "كرد"),
}


def _resolve_alias(value: str | None, table: dict[str, tuple[str, ...]]) -> str | None:
    if not value:
        return None
    key = value.strip().lower()
    for canon, aliases in table.items():
        if key == canon.lower() or key in aliases:
            return canon
    return None


def find_reciter(*texts: str | None) -> Reciter | None:
    """Look for a known reciter's name in caller-supplied hint or metadata text."""
    haystack = " ".join(t.lower() for t in texts if t)
    if not haystack:
        return None
    for reciter in _RECITERS:
        names = (reciter.name.lower(), *reciter.aliases)
        if any(n in haystack for n in names):
            return reciter
    return None


# ---------------------------------------------------------------------------
# Pluggable transcription (no real ASR bundled)
# ---------------------------------------------------------------------------


class Transcriber:
    """Interface for an upstream Arabic speech-to-text step."""

    def transcribe(self, audio_ref: str) -> str:  # pragma: no cover - interface
        raise NotImplementedError


class PassthroughTranscriber(Transcriber):
    """Treats the supplied ``audio_ref`` as the transcript itself."""

    def transcribe(self, audio_ref: str) -> str:
        return audio_ref


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------


@dataclass
class _Alignment:
    surah: Surah
    # per transcript token: corpus word position (index into flat list) or None
    token_to_word: list[int | None]
    exact: list[bool]
    variants: list[tuple[str, str]]  # (heard, mushaf)
    matched: int


def _snap(tokens: list[str], vocab: list[str], cutoff: float) -> tuple[list[str], list[bool], list[tuple[str, str]]]:
    """Map each token onto the surah vocabulary: riwayah variants, then near-misses."""
    out: list[str] = []
    exact: list[bool] = []
    variants: list[tuple[str, str]] = []
    vocab_set = set(vocab)
    for tok in tokens:
        if tok in vocab_set:
            out.append(tok)
            exact.append(True)
            continue
        alt = next(
            (m for m in vocab_set if tok in _RIWAYAH_VARIANTS.get(m, {})),
            None,
        )
        if alt is not None:
            out.append(alt)
            exact.append(True)
            variants.append((tok, alt))
            continue
        close = get_close_matches(tok, vocab, n=1, cutoff=cutoff)
        if close:
            out.append(close[0])
            exact.append(False)
        else:
            out.append(tok)
            exact.append(False)
    return out, exact, variants


def _align(tokens: list[str], surah: Surah, cutoff: float) -> _Alignment:
    words = surah.words
    corpus = [w for _, _, w in words]
    snapped, exact, variants = _snap(tokens, sorted(set(corpus)), cutoff)
    sm = SequenceMatcher(None, snapped, corpus, autojunk=False)
    token_to_word: list[int | None] = [None] * len(tokens)
    matched = 0
    for block in sm.get_matching_blocks():
        for k in range(block.size):
            token_to_word[block.a + k] = block.b + k
            matched += 1
    return _Alignment(surah, token_to_word, exact, variants, matched)


def _best_alignment(tokens: list[str], cutoff: float) -> _Alignment | None:
    best: _Alignment | None = None
    for surah in _SURAHS.values():
        cand = _align(tokens, surah, cutoff)
        if best is None or cand.matched > best.matched:
            best = cand
    if best is None or best.matched < MIN_MATCHED_WORDS:
        return None
    if best.matched / max(len(tokens), 1) < MIN_COVERAGE:
        return None
    return best


# ---------------------------------------------------------------------------
# Timings
# ---------------------------------------------------------------------------


class WordTiming(BaseModel):
    word: str = Field(..., min_length=1, max_length=100)
    start: float = Field(..., ge=0)
    end: float = Field(..., ge=0)


def _token_times(
    tokens: list[str],
    timings: list[WordTiming] | None,
    duration: float | None,
) -> tuple[list[tuple[float, float] | None], str]:
    """Return a ``(start, end)`` per transcript token and where the times came from."""
    if timings and len(timings) == len(tokens):
        return [(t.start, max(t.end, t.start)) for t in timings], "provided"
    if duration and tokens:
        total = sum(len(t) for t in tokens)
        times: list[tuple[float, float] | None] = []
        cursor = 0.0
        for tok in tokens:
            span = duration * len(tok) / total
            times.append((round(cursor, 3), round(cursor + span, 3)))
            cursor += span
        return times, "estimated"
    return [None] * len(tokens), "none"


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class AlignedWord(BaseModel):
    ayah: int
    word_index: int
    expected: str
    heard: str | None
    status: str  # exact | approximate | missing
    start: float | None = None
    end: float | None = None


class AyahResult(BaseModel):
    ayah: int
    text: str
    expected_words: int
    matched_words: int
    complete: bool
    start: float | None = None
    end: float | None = None


class RecitationIssue(BaseModel):
    type: str  # omission | substitution | insertion
    ayah: int | None = None
    expected: str | None = None
    heard: str | None = None


class SurahIdentification(BaseModel):
    number: int
    name: str
    ayah_start: int
    ayah_end: int
    reference: str


class RiwayahInfo(BaseModel):
    name: str
    basis: str  # text_variant | hint | default
    note: str


class ReciterInfo(BaseModel):
    name: str
    country: str
    style: str
    basis: str


class AnalyzeRequest(BaseModel):
    transcript: str | None = Field(
        None,
        max_length=MAX_TRANSCRIPT_LENGTH,
        description="Arabic transcript of the recitation (output of an ASR step).",
    )
    word_timings: list[WordTiming] | None = Field(
        None, max_length=MAX_WORD_TIMINGS, description="Per-word ASR timings in seconds."
    )
    duration_seconds: float | None = Field(
        None, gt=0, description="Clip length; used to estimate timings when none are given."
    )
    asr_confidence: float | None = Field(
        None, ge=0, le=1, description="ASR confidence; low values loosen word matching."
    )
    riwayah_hint: str | None = Field(None, max_length=100)
    reciter_hint: str | None = Field(None, max_length=200)
    maqam_hint: str | None = Field(None, max_length=100)
    metadata_text: str | None = Field(
        None, max_length=500, description="Title or filename, searched for a known reciter."
    )


class AnalyzeResponse(BaseModel):
    identified: bool
    surah: SurahIdentification | None
    verification: str  # accurate | minor_deviations | inaccurate | unidentified
    accuracy: float
    coverage: float
    partial: bool
    ayat: list[AyahResult]
    words: list[AlignedWord]
    issues: list[RecitationIssue]
    leading_content: list[str]
    trailing_content: list[str]
    basmalah_detected: bool
    timing_source: str
    riwayah: RiwayahInfo | None
    reciter: ReciterInfo | None
    maqam: str | None
    note: str


# ---------------------------------------------------------------------------
# Analysis core
# ---------------------------------------------------------------------------


def _riwayah(alignment: _Alignment, hint: str | None) -> RiwayahInfo:
    hinted = _resolve_alias(hint, RIWAYAT)
    if alignment.variants:
        readers: tuple[str, ...] = ()
        for heard, mushaf in alignment.variants:
            readers = _RIWAYAH_VARIANTS[mushaf][heard]
        name = hinted if hinted in readers else readers[0]
        note = "Short-vowel reading of 'Maliki' (1:4), shared by " + ", ".join(readers) + "; not Hafs."
        if hinted and hinted not in readers:
            note += f" The hinted riwayah ({hinted}) reads it with a long mim, so the hint conflicts."
        return RiwayahInfo(name=name, basis="text_variant", note=note)
    surah_words = alignment.surah.words
    covers_14 = alignment.surah.number == 1 and any(
        surah_words[w][:2] == (4, 1) for w in alignment.token_to_word if w is not None
    )
    if covers_14:
        names = ", ".join(_LONG_MIM_RIWAYAT)
        chosen = hinted if hinted in _LONG_MIM_RIWAYAT else _LONG_MIM_RIWAYAT[0]
        note = f"Long-mim 'Maliki' (1:4) is consistent with {names}."
        if hinted and hinted not in _LONG_MIM_RIWAYAT:
            note += f" The hinted riwayah ({hinted}) differs, so the hint conflicts."
        return RiwayahInfo(name=chosen, basis="text_variant", note=note)
    if hinted:
        return RiwayahInfo(
            name=hinted,
            basis="hint",
            note="Taken from the caller's hint; this passage has no distinguishing textual variant.",
        )
    return RiwayahInfo(
        name="Hafs 'an 'Asim",
        basis="default",
        note="Assumed: the bundled Mushaf follows Hafs and this passage has no distinguishing variant.",
    )


def _empty_response(note: str, timing_source: str = "none") -> AnalyzeResponse:
    return AnalyzeResponse(
        identified=False,
        surah=None,
        verification="unidentified",
        accuracy=0.0,
        coverage=0.0,
        partial=False,
        ayat=[],
        words=[],
        issues=[],
        leading_content=[],
        trailing_content=[],
        basmalah_detected=False,
        timing_source=timing_source,
        riwayah=None,
        reciter=None,
        maqam=None,
        note=note,
    )


def analyze_recitation(request: AnalyzeRequest) -> AnalyzeResponse:
    """Identify, align and verify a recitation transcript against the bundled Mushaf."""
    transcript = request.transcript
    if not (transcript and transcript.strip()) and request.word_timings:
        transcript = " ".join(t.word for t in request.word_timings)
    tokens = tokenize(transcript or "")
    if not tokens:
        return _empty_response("No Arabic text to analyze.")

    noisy = request.asr_confidence is not None and request.asr_confidence < NOISY_ASR_CONFIDENCE
    cutoff = SNAP_CUTOFF_NOISY if noisy else SNAP_CUTOFF_CLEAN

    # Timings only line up when they cover the transcript token-for-token.
    timings = request.word_timings
    if timings and request.transcript and request.transcript.strip():
        timings = timings if len(timings) == len(tokens) else None
    token_times, timing_source = _token_times(tokens, timings, request.duration_seconds)

    alignment = _best_alignment(tokens, cutoff)
    if alignment is None:
        return _empty_response(
            "No bundled Quranic text matched this transcript with sufficient confidence. "
            "The bundled Mushaf covers only a few short surahs; do not treat this as a "
            "verified recitation.",
            timing_source,
        )

    surah = alignment.surah
    words = surah.words
    matched_idx = [i for i, w in enumerate(alignment.token_to_word) if w is not None]
    first_t, last_t = matched_idx[0], matched_idx[-1]
    first_w = alignment.token_to_word[first_t]
    last_w = alignment.token_to_word[last_t]
    assert first_w is not None and last_w is not None

    leading = tokens[:first_t]
    trailing = tokens[last_t + 1 :]
    basmalah = bool(leading) and leading[-len(_BASMALAH) :] == _BASMALAH and surah.number != 1
    if basmalah:
        leading = leading[: -len(_BASMALAH)]

    # Walk the span once: aligned words, issues, and transcript tokens per corpus word.
    word_to_token = {w: t for t, w in enumerate(alignment.token_to_word) if w is not None}
    aligned: list[AlignedWord] = []
    issues: list[RecitationIssue] = []
    for w in range(first_w, last_w + 1):
        ayah, widx, expected = words[w]
        t = word_to_token.get(w)
        if t is None:
            aligned.append(
                AlignedWord(ayah=ayah, word_index=widx, expected=expected, heard=None, status="missing")
            )
            continue
        t_time = token_times[t]
        aligned.append(
            AlignedWord(
                ayah=ayah,
                word_index=widx,
                expected=expected,
                heard=tokens[t],
                status="exact" if alignment.exact[t] else "approximate",
                start=t_time[0] if t_time else None,
                end=t_time[1] if t_time else None,
            )
        )

    # Pair unmatched transcript words inside the span with unmatched corpus words
    # (substitutions); leftovers are insertions or omissions.
    inside_extra = [t for t in range(first_t, last_t + 1) if alignment.token_to_word[t] is None]
    missing = [a for a in aligned if a.status == "missing"]
    for a in aligned:
        if a.status == "approximate":
            issues.append(RecitationIssue(type="substitution", ayah=a.ayah, expected=a.expected, heard=a.heard))
    for k, a in enumerate(missing):
        if k < len(inside_extra):
            issues.append(
                RecitationIssue(
                    type="substitution", ayah=a.ayah, expected=a.expected, heard=tokens[inside_extra[k]]
                )
            )
        else:
            issues.append(RecitationIssue(type="omission", ayah=a.ayah, expected=a.expected))
    for t in inside_extra[len(missing) :]:
        issues.append(RecitationIssue(type="insertion", heard=tokens[t]))

    span_len = last_w - first_w + 1
    exact_count = sum(1 for a in aligned if a.status == "exact")
    denominator = span_len + max(len(inside_extra) - len(missing), 0)
    accuracy = round(exact_count / denominator, 4)

    # Per-ayah summary (only ayat the recitation touched).
    ayat: list[AyahResult] = []
    for ayah in range(words[first_w][0], words[last_w][0] + 1):
        ayah_words = [a for a in aligned if a.ayah == ayah]
        total = len(tokenize(surah.ayat[ayah - 1]))
        got = sum(1 for a in ayah_words if a.status != "missing")
        starts = [a.start for a in ayah_words if a.start is not None]
        ends = [a.end for a in ayah_words if a.end is not None]
        ayat.append(
            AyahResult(
                ayah=ayah,
                text=surah.ayat[ayah - 1],
                expected_words=total,
                matched_words=got,
                complete=got == total and len(ayah_words) == total,
                start=min(starts) if starts else None,
                end=max(ends) if ends else None,
            )
        )

    ayah_start, ayah_end = words[first_w][0], words[last_w][0]
    # A whole-surah recitation starts at ayah 1 and reaches the last ayah.
    partial = not (ayah_start == 1 and ayah_end == len(surah.ayat) and all(a.complete for a in ayat))
    coverage = round(len(matched_idx) / len(tokens), 4)

    if accuracy >= 0.999 and not issues:
        verification = "accurate"
    elif accuracy >= 0.8:
        verification = "minor_deviations"
    else:
        verification = "inaccurate"

    reference = f"{surah.number}:{ayah_start}" if ayah_start == ayah_end else f"{surah.number}:{ayah_start}-{ayah_end}"

    reciter_match = find_reciter(request.reciter_hint, request.metadata_text)
    reciter = (
        ReciterInfo(
            name=reciter_match.name,
            country=reciter_match.country,
            style=reciter_match.style,
            basis="hint" if find_reciter(request.reciter_hint) else "metadata",
        )
        if reciter_match
        else None
    )

    note = (
        f"Aligned to {surah.name} {reference} (Hafs Mushaf text) with accuracy {accuracy:.2f}. "
        "Reciter and maqam are reported only from supplied hints/metadata — no voice or pitch "
        "analysis is performed. Verify against a printed Mushaf and a qualified teacher of tajwid."
    )
    if timing_source == "estimated":
        note += " Timestamps are estimated from clip duration, not measured."

    return AnalyzeResponse(
        identified=True,
        surah=SurahIdentification(
            number=surah.number,
            name=surah.name,
            ayah_start=ayah_start,
            ayah_end=ayah_end,
            reference=reference,
        ),
        verification=verification,
        accuracy=accuracy,
        coverage=coverage,
        partial=partial,
        ayat=ayat,
        words=aligned,
        issues=issues,
        leading_content=leading,
        trailing_content=trailing,
        basmalah_detected=basmalah,
        timing_source=timing_source,
        riwayah=_riwayah(alignment, request.riwayah_hint),
        reciter=reciter,
        maqam=_resolve_alias(request.maqam_hint, MAQAMAT),
        note=note,
    )


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze_quran_recitation(request: AnalyzeRequest) -> AnalyzeResponse:
    """Identify and verify a transcribed Quranic recitation against the bundled Mushaf."""
    has_text = bool(request.transcript and request.transcript.strip())
    if not has_text and not request.word_timings:
        raise HTTPException(status_code=422, detail="Provide a transcript or word_timings.")
    return analyze_recitation(request)


@router.get("/surahs")
async def list_supported_surahs() -> dict:
    """Surahs bundled for matching, plus recognised riwayat and maqamat."""
    return {
        "surahs": [{"number": s.number, "name": s.name, "ayat": len(s.ayat)} for s in _SURAHS.values()],
        "riwayat": list(RIWAYAT),
        "maqamat": list(MAQAMAT),
    }
