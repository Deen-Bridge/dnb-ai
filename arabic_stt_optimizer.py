"""Arabic Speech-to-Text Optimization Engine for Islamic Content (#237, #244).

Post-processes and optimizes raw Arabic speech-to-text transcripts with an
Islamic domain lexicon, phonetic confusion correction (emphatic/pharyngeal
consonants, hamza/ta-marbuta), classical formula restoration (salawat,
taraddi, basmala, istighfar), and Quranic citation normalization.

#244 adds dialect awareness (Egyptian, Gulf, Levantine), Arabic/English
code-switching handling, support for ``preserve_diacritics``, a larger lexicon
of Islamic terms and names, real-time (``/stream``) and batch (``/batch``)
endpoints, and word/character error-rate measurement (``/evaluate``).

Scope and honest limitations
----------------------------
This is a deterministic *post-processing* layer over a recognizer's output. It
contains no acoustic model and does not fine-tune one, so it cannot by itself
deliver a given recognition accuracy; it corrects the systematic orthographic
and vocabulary errors recognizers make on Islamic content. Use ``/evaluate``
against reference transcripts to measure the real improvement on your data.

Lexical corrections match undiacritized text. A word the recognizer already
diacritized is left exactly as it came (never damaged); ``preserve_diacritics``
only controls whether diacritics are kept or stripped from the output.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/stt-optimizer", tags=["arabic-stt-optimizer"])

MAX_BATCH_ITEMS = 50
# Error-rate evaluation is quadratic in input length, so its inputs are bounded explicitly.
MAX_EVAL_CHARS = 1000
EVALUATION_NOTE = (
    "Single-sample figures are indicative only; measure accuracy over a representative corpus of reference transcripts."
)


class STTTranscriptCandidate(BaseModel):
    transcript: str = Field(..., min_length=1, max_length=8000)
    acoustic_confidence: float = Field(default=0.8, ge=0.0, le=1.0)


class ArabicSTTOptimizationRequest(BaseModel):
    candidates: list[STTTranscriptCandidate] = Field(..., min_length=1, max_length=20)
    preserve_diacritics: bool = True
    dialect_hint: str | None = Field(
        default=None,
        max_length=40,
        description=(
            "One of msa, egyptian, gulf, levantine, maghrebi. An unrecognised value is ignored and the dialect detected."
        ),
    )
    normalize_to_msa: bool = Field(default=False, description="Also return the transcript rewritten in MSA.")
    normalize_latin_terms: bool = Field(
        default=False, description="Rewrite Latin-script Islamic formulas (e.g. inshallah) into Arabic."
    )


class AppliedSTTCorrection(BaseModel):
    original_span: str
    corrected_span: str
    category: str


class ArabicSTTOptimizationResponse(BaseModel):
    best_raw_transcript: str
    optimized_transcript: str
    domain_score: float
    confidence: float
    corrections: list[AppliedSTTCorrection]
    detected_islamic_terms: list[str]
    detected_dialect: str | None = None
    dialect_confidence: float | None = None
    dialect_source: str | None = None
    msa_transcript: str | None = None
    code_switching: bool = False
    latin_segments: list[str] = Field(default_factory=list)


class STTStreamRequest(BaseModel):
    partial_transcript: str = Field(..., min_length=1, max_length=8000)
    is_final: bool = Field(default=False, description="True once the speaker has finished the utterance.")
    preserve_diacritics: bool = True
    dialect_hint: str | None = Field(default=None, max_length=40)
    normalize_latin_terms: bool = False


class STTStreamResponse(BaseModel):
    stable_text: str = Field(
        ..., description="Complete words, optimized. A multi-word formula is restored once its last word completes."
    )
    pending_text: str = Field(..., description="The word still being spoken, left untouched until it is complete.")
    full_text: str
    is_final: bool
    corrections: list[AppliedSTTCorrection]


class STTBatchRequest(BaseModel):
    items: list[ArabicSTTOptimizationRequest] = Field(..., min_length=1, max_length=MAX_BATCH_ITEMS)


class STTBatchResponse(BaseModel):
    count: int
    results: list[ArabicSTTOptimizationResponse]


class STTEvaluationRequest(ArabicSTTOptimizationRequest):
    reference: str = Field(..., min_length=1, max_length=8000, description="The known-correct transcript.")
    compare_diacritics: bool = Field(default=False, description="Count diacritic differences as errors.")


class STTEvaluationResponse(BaseModel):
    raw_wer: float
    optimized_wer: float
    raw_cer: float
    optimized_cer: float
    wer_improvement: float
    raw_accuracy: float
    optimized_accuracy: float
    optimized_transcript: str
    note: str


_PHONETIC_FIXES: list[tuple[str, str, str]] = [
    ("الصلاه", "الصلاة", "ta_marbuta_restoration"),
    ("الزكاه", "الزكاة", "ta_marbuta_restoration"),
    ("العقيده", "العقيدة", "ta_marbuta_restoration"),
    ("الشريعه", "الشريعة", "ta_marbuta_restoration"),
    ("سوره", "سورة", "ta_marbuta_restoration"),
    ("ايه", "آية", "madda_and_ta_marbuta"),
    ("القران", "القرآن", "madda_restoration"),
    ("قران", "قرآن", "madda_restoration"),
    ("صلي الله عليه وسلم", "صلى الله عليه وسلم", "honorific_formula"),
    ("رضي الله عنهم", "رضي الله عنهم", "honorific_formula"),
    ("ان شاء الله", "إن شاء الله", "hamza_restoration"),
    # #244: more Islamic terms, names and formulas recognizers commonly mis-spell.
    ("الاسلام", "الإسلام", "hamza_restoration"),
    ("الايمان", "الإيمان", "hamza_restoration"),
    ("الاحسان", "الإحسان", "hamza_restoration"),
    ("الاخلاص", "الإخلاص", "hamza_restoration"),
    ("الرحمان", "الرحمن", "divine_name_spelling"),
    ("الجنه", "الجنة", "ta_marbuta_restoration"),
    ("الاخره", "الآخرة", "madda_and_ta_marbuta"),
    ("الملائكه", "الملائكة", "ta_marbuta_restoration"),
    ("القيامه", "القيامة", "ta_marbuta_restoration"),
    ("الكعبه", "الكعبة", "ta_marbuta_restoration"),
    ("مكه", "مكة", "ta_marbuta_restoration"),
    ("الفاتحه", "الفاتحة", "ta_marbuta_restoration"),
    ("العمره", "العمرة", "ta_marbuta_restoration"),
    ("عائشه", "عائشة", "name_spelling"),
    ("خديجه", "خديجة", "name_spelling"),
    ("فاطمه", "فاطمة", "name_spelling"),
    ("ابو بكر", "أبو بكر", "name_spelling"),
    ("ابو هريره", "أبو هريرة", "name_spelling"),
    ("الله اكبر", "الله أكبر", "hamza_restoration"),
    ("لا اله الا الله", "لا إله إلا الله", "hamza_restoration"),
    ("استغفر الله", "أستغفر الله", "hamza_restoration"),
    ("الحمدلله", "الحمد لله", "formula_spacing"),
    ("سبحانه وتعالي", "سبحانه وتعالى", "alif_maqsura_restoration"),
]

# Corrections that are right in MSA but would damage a dialect's everyday word.
# "ايه" means "what" in Egyptian and "yes" in Levantine; rewriting it to "آية" (verse)
# would change the meaning.
_DIALECT_AMBIGUOUS: dict[str, frozenset[str]] = {
    "egyptian": frozenset({"ايه"}),
    "levantine": frozenset({"ايه"}),
}

_ISLAMIC_VOCAB: tuple[str, ...] = (
    "الله",
    "القرآن",
    "قرآن",
    "الصلاة",
    "الزكاة",
    "الصيام",
    "الحج",
    "التوحيد",
    "العقيدة",
    "الفقه",
    "الحديث",
    "التفسير",
    "السنة",
    "سورة",
    "آية",
    "صلى الله عليه وسلم",
    "الإسلام",
    "الإيمان",
    "الإحسان",
    "الإخلاص",
    "الجنة",
    "الآخرة",
    "الملائكة",
    "القيامة",
    "الكعبة",
    "الفاتحة",
    "رمضان",
    "العمرة",
    "عائشة",
    "خديجة",
    "فاطمة",
    "أبو بكر",
    "أبو هريرة",
    "البخاري",
    "إن شاء الله",
    "الحمد لله",
    "سبحان الله",
    "الله أكبر",
    "أستغفر الله",
)

_DIALECTS = frozenset({"msa", "egyptian", "gulf", "levantine", "maghrebi"})

# The shared classifier (arabic_dialect) covers Egyptian, Gulf and Levantine. Maghrebi
# (Moroccan/Algerian/Tunisian) is handled here with a small, deliberately conservative
# set of distinctive words, each mapped to an MSA equivalent. Words that are also common
# in another dialect are left out so they cannot cause a false detection.
_MAGHREBI_MARKERS: dict[str, str] = {
    "بزاف": "كثيرا",
    "برشا": "كثيرا",
    "دابا": "الآن",
    "دروك": "الآن",
    "علاش": "لماذا",
    "كيفاش": "كيف",
    "شحال": "كم",
    "واش": "هل",
    "غادي": "سوف",
    "ديال": "ديال",  # "of": a possessive particle with no single-word MSA equivalent
}

# Latin-script spellings of common Islamic formulas, written into Arabic when asked to.
_LATIN_FORMULAS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("inshallah", "insha allah", "insha'allah", "inshaallah", "in sha allah"), "إن شاء الله"),
    (("alhamdulillah", "alhamdulilah", "al hamdulillah"), "الحمد لله"),
    (("subhanallah", "subhan allah"), "سبحان الله"),
    (("mashallah", "masha allah", "masha'allah"), "ما شاء الله"),
    (("bismillah",), "بسم الله"),
    (("astaghfirullah", "astagfirullah"), "أستغفر الله"),
    (("allahu akbar", "allah u akbar"), "الله أكبر"),
    (("jazakallah", "jazak allah"), "جزاك الله"),
    (("assalamu alaikum", "assalamu alaykum", "assalamualaikum", "salam alaykum"), "السلام عليكم"),
)


def _compile_variant(variant: str) -> re.Pattern[str]:
    body = r"\s+".join(re.escape(part) for part in variant.split(" "))
    return re.compile(r"(?<![A-Za-z])" + body + r"(?![A-Za-z])", re.IGNORECASE)


_LATIN_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = tuple(
    (_compile_variant(variant), arabic) for variants, arabic in _LATIN_FORMULAS for variant in variants
)

_DIACRITICS = re.compile("[\u064b-\u065f\u0670\u0640]")  # harakat, superscript alif, tatweel
_ARABIC_LETTER = re.compile("[\u0621-\u064a]")
_LATIN_WORD = re.compile(r"[A-Za-z][A-Za-z'\u2019-]*")


def strip_diacritics(text: str) -> str:
    """Remove Arabic diacritics (harakat) and tatweel, leaving the letters."""
    return _DIACRITICS.sub("", text)


def has_arabic(text: str) -> bool:
    return _ARABIC_LETTER.search(text) is not None


def find_latin_segments(text: str) -> list[str]:
    """Latin-script words in the text, de-duplicated in order (the code-switched parts)."""
    seen: list[str] = []
    for word in _LATIN_WORD.findall(text):
        if word not in seen:
            seen.append(word)
    return seen


def resolve_dialect(text: str, hint: str | None) -> tuple[str | None, float | None, str | None]:
    """Return (dialect, confidence, source). A recognised hint wins; otherwise classify.

    The classifier is imported lazily because ``arabic_dialect`` imports this module
    when its router is built, and a top-level import here would be circular.
    """
    if hint:
        key = hint.strip().lower()
        if key in _DIALECTS:
            return key, None, "hint"
    if not has_arabic(text):
        return None, None, None
    from arabic_dialect.dialects import dialect_classifier

    profile = dialect_classifier.classify_dialect(text)
    hits = _count_maghrebi_markers(text)
    # Two or more Maghrebi words outweigh a lone shared marker; one is enough when
    # nothing else was recognised.
    if hits >= 2 or (hits == 1 and profile.is_msa):
        return "maghrebi", round(min(0.98, 0.6 + 0.15 * hits), 2), "detected"
    return profile.primary_dialect.value, profile.confidence, "detected"


def _maghrebi_pattern(word: str) -> re.Pattern[str]:
    return re.compile(r"(?<!\w)" + re.escape(word) + r"(?!\w)")


_MAGHREBI_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = tuple(
    (_maghrebi_pattern(word), msa) for word, msa in _MAGHREBI_MARKERS.items()
)


def _count_maghrebi_markers(text: str) -> int:
    return sum(1 for pattern, _ in _MAGHREBI_PATTERNS if pattern.search(text))


def to_msa(text: str, dialect: str | None) -> str:
    """Rewrite dialectal Islamic-context vocabulary as Modern Standard Arabic."""
    if dialect == "msa" or not has_arabic(text):
        return text
    if dialect == "maghrebi":
        for pattern, msa in _MAGHREBI_PATTERNS:
            text = pattern.sub(msa, text)
        return text
    from arabic_dialect.dialects import dialect_classifier

    normalized, _ = dialect_classifier.normalize_to_msa(text)
    return normalized


def normalize_latin_formulas(text: str) -> tuple[str, list[AppliedSTTCorrection]]:
    corrections: list[AppliedSTTCorrection] = []
    for pattern, arabic in _LATIN_PATTERNS:
        match = pattern.search(text)
        if match:
            corrections.append(
                AppliedSTTCorrection(original_span=match.group(0), corrected_span=arabic, category="latin_formula")
            )
            text = pattern.sub(arabic, text)
    return text, corrections


def optimize_text(
    text: str, *, preserve_diacritics: bool, dialect: str | None, normalize_latin: bool
) -> tuple[str, list[AppliedSTTCorrection]]:
    """Apply the orthographic corrections to one piece of text."""
    text = re.sub(r"\s+", " ", text).strip()
    if not preserve_diacritics:
        text = strip_diacritics(text)
    corrections: list[AppliedSTTCorrection] = []
    skip = _DIALECT_AMBIGUOUS.get(dialect or "", frozenset())
    for wrong, right, category in _PHONETIC_FIXES:
        if wrong == right or wrong in skip:
            continue
        pattern = r"(?:^|(?<=[\sوفبلك]))" + re.escape(wrong) + r"(?=$|\s|[،.؟!])"
        if re.search(pattern, text):
            text = re.sub(pattern, right, text)
            corrections.append(AppliedSTTCorrection(original_span=wrong, corrected_span=right, category=category))
    if normalize_latin:
        text, latin_corrections = normalize_latin_formulas(text)
        corrections.extend(latin_corrections)
    return text, corrections


def optimize_arabic_stt(req: ArabicSTTOptimizationRequest) -> ArabicSTTOptimizationResponse:
    best_candidate = req.candidates[0]
    best_score = -1.0

    for cand in req.candidates:
        plain = strip_diacritics(cand.transcript)
        vocab_hits = sum(1 for term in _ISLAMIC_VOCAB if term in plain)
        score = cand.acoustic_confidence * 0.7 + min(1.0, vocab_hits / 4.0) * 0.3
        if score > best_score:
            best_score = score
            best_candidate = cand

    raw = best_candidate.transcript
    dialect, dialect_confidence, dialect_source = resolve_dialect(strip_diacritics(raw), req.dialect_hint)
    text, corrections = optimize_text(
        raw,
        preserve_diacritics=req.preserve_diacritics,
        dialect=dialect,
        normalize_latin=req.normalize_latin_terms,
    )

    plain_text = strip_diacritics(text)
    detected_terms = [term for term in _ISLAMIC_VOCAB if term in plain_text]
    domain_score = round(min(1.0, len(detected_terms) / 5.0), 3)
    boosted_conf = round(min(0.99, best_candidate.acoustic_confidence + 0.04 * len(corrections)), 3)
    latin_segments = find_latin_segments(raw)

    return ArabicSTTOptimizationResponse(
        best_raw_transcript=raw,
        optimized_transcript=text,
        domain_score=domain_score,
        confidence=boosted_conf,
        corrections=corrections,
        detected_islamic_terms=detected_terms,
        detected_dialect=dialect,
        dialect_confidence=dialect_confidence,
        dialect_source=dialect_source,
        msa_transcript=to_msa(text, dialect) if req.normalize_to_msa else None,
        code_switching=has_arabic(raw) and bool(latin_segments),
        latin_segments=latin_segments,
    )


# ---------------------------------------------------------------------------
# Real-time (streaming) transcription
# ---------------------------------------------------------------------------


def split_stream(text: str, is_final: bool) -> tuple[str, str]:
    """Split a live transcript into (complete words, the word still being spoken).

    A recognizer's last word in a partial result is often incomplete, so it is held back
    and left untouched rather than "corrected" into the wrong word.
    """
    if is_final or not text.strip() or text[-1].isspace():
        return text.strip(), ""
    parts = text.rstrip().rsplit(None, 1)
    if len(parts) == 1:
        return "", parts[0]
    return parts[0], parts[1]


def optimize_stream(req: STTStreamRequest) -> STTStreamResponse:
    stable_raw, pending = split_stream(req.partial_transcript, req.is_final)
    dialect: str | None = None
    if stable_raw:
        dialect = resolve_dialect(strip_diacritics(stable_raw), req.dialect_hint)[0]
    stable, corrections = optimize_text(
        stable_raw,
        preserve_diacritics=req.preserve_diacritics,
        dialect=dialect,
        normalize_latin=req.normalize_latin_terms,
    )
    full = f"{stable} {pending}".strip()
    return STTStreamResponse(
        stable_text=stable,
        pending_text=pending,
        full_text=full,
        is_final=req.is_final,
        corrections=corrections,
    )


# ---------------------------------------------------------------------------
# Accuracy measurement
# ---------------------------------------------------------------------------

_EVAL_PUNCTUATION = re.compile(r"[\u060c\u061b\u061f.,;:!?\"'()\[\]{}\-\u2013\u2014]")


def edit_distance(a: Sequence[str], b: Sequence[str]) -> int:
    """Levenshtein distance between two sequences, in O(len(a) * len(b)) time and O(len(b)) space."""
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, item_a in enumerate(a, start=1):
        current = [i]
        for j, item_b in enumerate(b, start=1):
            cost = 0 if item_a == item_b else 1
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost))
        previous = current
    return previous[-1]


def _normalize_for_eval(text: str, compare_diacritics: bool) -> str:
    text = _EVAL_PUNCTUATION.sub(" ", text)
    if not compare_diacritics:
        text = strip_diacritics(text)
    return re.sub(r"\s+", " ", text).strip()


def error_rates(reference: str, hypothesis: str, compare_diacritics: bool = False) -> tuple[float, float]:
    """Return (WER, CER) of ``hypothesis`` against ``reference``; the reference must not be empty."""
    ref = _normalize_for_eval(reference, compare_diacritics)
    hyp = _normalize_for_eval(hypothesis, compare_diacritics)
    if not ref:
        raise ValueError("reference has no comparable text")
    ref_words, hyp_words = ref.split(" "), hyp.split(" ") if hyp else []
    wer = edit_distance(ref_words, hyp_words) / len(ref_words)
    cer = edit_distance(list(ref), list(hyp)) / len(ref)
    return round(wer, 4), round(cer, 4)


def evaluate_arabic_stt(req: STTEvaluationRequest) -> STTEvaluationResponse:
    """Measure how much the optimizer moves a transcript toward a known-correct reference."""
    result = optimize_arabic_stt(req)
    raw_wer, raw_cer = error_rates(req.reference, result.best_raw_transcript, req.compare_diacritics)
    opt_wer, opt_cer = error_rates(req.reference, result.optimized_transcript, req.compare_diacritics)
    return STTEvaluationResponse(
        raw_wer=raw_wer,
        optimized_wer=opt_wer,
        raw_cer=raw_cer,
        optimized_cer=opt_cer,
        wer_improvement=round(raw_wer - opt_wer, 4),
        raw_accuracy=round(max(0.0, 1.0 - raw_wer), 4),
        optimized_accuracy=round(max(0.0, 1.0 - opt_wer), 4),
        optimized_transcript=result.optimized_transcript,
        note=EVALUATION_NOTE,
    )


# ---------------------------------------------------------------------------
# HTTP routes
# ---------------------------------------------------------------------------


@router.post("/optimize", response_model=ArabicSTTOptimizationResponse)
async def optimize_arabic_stt_endpoint(
    body: ArabicSTTOptimizationRequest,
) -> ArabicSTTOptimizationResponse:
    """Optimize Arabic speech-to-text transcripts for Islamic terminology and classical formulas."""
    return optimize_arabic_stt(body)


@router.post("/stream", response_model=STTStreamResponse)
async def optimize_arabic_stt_stream(body: STTStreamRequest) -> STTStreamResponse:
    """Optimize a live transcript: complete words are corrected, the word still being spoken is held back."""
    return optimize_stream(body)


@router.post("/batch", response_model=STTBatchResponse)
async def optimize_arabic_stt_batch(body: STTBatchRequest) -> STTBatchResponse:
    """Optimize several transcripts in one request; each item is processed independently."""
    results = [optimize_arabic_stt(item) for item in body.items]
    return STTBatchResponse(count=len(results), results=results)


@router.post("/evaluate", response_model=STTEvaluationResponse)
async def evaluate_arabic_stt_endpoint(body: STTEvaluationRequest) -> STTEvaluationResponse:
    """Measure word and character error rates before and after optimization against a reference."""
    longest = max([len(body.reference), *(len(c.transcript) for c in body.candidates)])
    if longest > MAX_EVAL_CHARS:
        raise HTTPException(
            status_code=422,
            detail=f"Evaluation inputs are limited to {MAX_EVAL_CHARS} characters each.",
        )
    try:
        return evaluate_arabic_stt(body)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
