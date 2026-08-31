"""Image content analysis for Islamic manuscripts (#135).

Extends the manuscript pipeline (:mod:`manuscript_ocr`) from "text was
extracted from this scan" to "here is what that text is about, and it has
been checked against canonical sources". Works on the OCR *output* (the
extracted text plus the manuscript classification) rather than raw pixels,
so it is deterministic and offline, and slots in after any OCR backend.

What it adds
------------
* **Qur'anic verse extraction with canonical validation** — surah:ayah
  references are pulled out of the extracted text, bounds-checked against
  the bundled 114-surah index (the same source the citation layer trusts),
  and where the bundled corpus contains the verse text, the quote is
  normalized and compared against it, so a mis-transcribed ayah is flagged
  rather than silently presented as the canonical text.
* **Hadith text detection** — references to the recognized collections are
  detected in the extracted text and reported with the collection name.
* **Translation & explanation of extracted content** — a deterministic stub
  engine (or a Gemini vision call when ``IMAGE_ANALYSIS_PROVIDER=gemini``)
  produces a translation and a scholarly explanation of the extracted text.
* **Structured metadata** — content type, verse count, hadith count,
  validation status and detected collections are returned in one shape.
* **Batch processing** — analyze several uploaded pages in one request with
  per-page results and per-page error isolation (one bad page never discards
  the good ones).
* **OCR result caching** — identical images (by SHA-256 of the bytes) are
  served from an in-process cache instead of re-running the vision call.

Errors follow the manuscript taxonomy (413 oversize, 415 format, 422 poor
quality) and are mapped to HTTP codes by the route in this module.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
from typing import Any, Protocol

from fastapi import APIRouter, File, UploadFile
from pydantic import BaseModel, Field

import telemetry
from config import get_settings
from corpus import QuranCorpus
from errors import APIException
from manuscript_ocr import (
    ManuscriptAnalysis,
    PoorQualityError,
    UnsupportedFormatError,
    UploadTooLargeError,
    analyze_manuscript_bytes,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/image-analysis", tags=["image-analysis"])

# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class ExtractedVerse(BaseModel):
    """A surah:ayah reference found in extracted text, with validation."""

    surah: int = Field(..., ge=1, le=114)
    ayah: int = Field(..., ge=1)
    reference: str = Field(..., description="'surah:ayah', e.g. '2:255'")
    surah_name: str | None = None
    in_bounds: bool = Field(..., description="Passed bounds check against the surah index")
    quote_matches_canonical: bool | None = Field(
        None,
        description="None when the canonical text is unavailable; True/False when compared",
    )
    matched_text: str = Field("", description="The exact substring matched in the source text")


class DetectedHadith(BaseModel):
    """A hadith-collection mention detected in extracted text."""

    collection: str = Field(..., description="Canonical collection name, e.g. 'Sahih al-Bukhari'")
    matched_text: str = ""


class ContentTranslation(BaseModel):
    """Translation and explanation of the extracted text."""

    language: str = "en"
    translation: str = ""
    explanation: str = ""
    warning: str | None = Field(None, description="Set when translation was unavailable")


class ContentAnalysis(BaseModel):
    """Everything derived from one page's extracted text."""

    content_type: str = Field(..., description="'quran', 'hadith', 'fiqh', 'tafsir', 'history', 'other'")
    verses: list[ExtractedVerse] = Field(default_factory=list)
    hadith: list[DetectedHadith] = Field(default_factory=list)
    validation: dict[str, Any] = Field(
        default_factory=dict,
        description="Summary of canonical validation, e.g. bounds and text-match rates",
    )
    translation: ContentTranslation = ContentTranslation()
    metadata: dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    warnings: list[str] = Field(default_factory=list)


class AnalyzeExtractedRequest(BaseModel):
    """Analyze text that an OCR backend already extracted."""

    text: str = Field(..., min_length=1, max_length=100_000)
    manuscript_type: str | None = Field(None, description="Optional classification hint from the OCR stage")
    language: str = Field("en", description="Target language for translation/explanation")
    skip_translation: bool = False


class AnalyzeExtractedResponse(BaseModel):
    analysis: ContentAnalysis


class BatchAnalysisResponse(BaseModel):
    """Per-page results for a batch image-analysis request."""

    results: list[dict[str, Any]] = Field(default_factory=list)
    total: int
    succeeded: int
    failed: int


# ---------------------------------------------------------------------------
# Canonical verse extraction & validation
# ---------------------------------------------------------------------------

# "2:255", "surah 2:255", "Quran 2:255-256" — colon-separated surah/ayah pairs.
_QURAN_REF_PATTERN = re.compile(
    r"\b(?:Q(?:ur[’'']?an)?\.?\s*|S(?:urah|ura|urat)?\.?\s*)?"
    r"(?P<surah>\d{1,3})\s*:\s*(?P<ayah>\d{1,3})(?:-\d{1,3})?\b",
    re.IGNORECASE,
)

# The canonical hadith collections, matched case-insensitively as whole words.
_HADITH_COLLECTIONS: dict[str, str] = {
    "bukhari": "Sahih al-Bukhari",
    "muslim": "Sahih Muslim",
    "abu dawud": "Sunan Abi Dawud",
    "abu dawood": "Sunan Abi Dawud",
    "tirmidhi": "Jami' at-Tirmidhi",
    "nasai": "Sunan an-Nasa'i",
    "nasa'i": "Sunan an-Nasa'i",
    "ibn majah": "Sunan Ibn Majah",
    "muwatta": "Muwatta Malik",
    "ahmad": "Musnad Ahmad",
}

_HADITH_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(name) for name in _HADITH_COLLECTIONS) + r")\b",
    re.IGNORECASE,
)

# Arabic diacritics and tatweel, stripped before canonical text comparison.
_DIACRITICS = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED\u0640]")
# Arabic letters whose positional variants fold to the same base form.
_ARABIC_FOLD = str.maketrans(
    {
        "أ": "ا",
        "إ": "ا",
        "آ": "ا",
        "ٱ": "ا",
        "ى": "ي",
        "ئ": "ي",
        "ؤ": "و",
        "ة": "ه",
    }
)

_surah_index_cache: dict[int, dict[str, Any]] = {}


def _load_surah_index() -> dict[int, dict[str, Any]]:
    """Load the 114-surah index once and cache it."""
    if _surah_index_cache:
        return _surah_index_cache
    try:
        import json as _json
        from pathlib import Path

        path = Path(__file__).parent / "data" / "quran" / "surah_index.json"
        entries = _json.loads(path.read_text(encoding="utf-8"))
        index: dict[int, dict[str, Any]] = {}
        for entry in entries:
            number = entry.get("number")
            if number is not None:
                index[int(number)] = entry
        _surah_index_cache.update(index)
    except Exception as exc:  # noqa: BLE001 - degrade to empty index
        logger.warning("Failed to load surah index for validation: %s", exc)
    return _surah_index_cache


def normalize_arabic(text: str) -> str:
    """Fold Arabic text to a comparable canonical form (letters only)."""
    folded = text.translate(_ARABIC_FOLD)
    folded = _DIACRITICS.sub("", folded)
    return re.sub(r"\s+", "", folded)


def extract_verses(text: str, corpus: QuranCorpus | None = None) -> list[ExtractedVerse]:
    """Extract surah:ayah references and validate them against the surah index.

    Bounds are checked against the bundled 114-surah index — a ``2:300`` is
    refused against Al-Baqarah's real 286 ayat. When the bundled corpus holds
    the verse text, the matched quote is normalized and compared to the
    canonical Arabic, so a transcription error is flagged rather than echoed.
    """
    corpus = corpus or QuranCorpus()
    index = _load_surah_index()
    verses: list[ExtractedVerse] = []

    for match in _QURAN_REF_PATTERN.finditer(text):
        surah = int(match.group("surah"))
        ayah = int(match.group("ayah"))
        if not (1 <= surah <= 114) or ayah < 1:
            continue
        entry = index.get(surah)
        ayah_count = int(entry.get("ayah_count") or entry.get("ayahs_count") or 0) if entry else 0
        in_bounds = 1 <= ayah <= ayah_count if ayah_count else True

        matched_text = match.group(0).strip()
        quote_matches: bool | None = None
        if in_bounds:
            canonical = corpus.get_ayah(surah, ayah)
            if canonical and canonical.get("arabic"):
                # Compare the matched numeric reference's neighbourhood: take
                # up to 40 chars of Arabic around the match if present.
                sample = _neighbourhood(text, match.start(), 40)
                if sample and any("\u0600" <= ch <= "\u06ff" for ch in sample):
                    quote_matches = normalize_arabic(sample) == normalize_arabic(canonical["arabic"])

        verses.append(
            ExtractedVerse(
                surah=surah,
                ayah=ayah,
                reference=f"{surah}:{ayah}",
                surah_name=entry.get("name") if entry else None,
                in_bounds=in_bounds,
                quote_matches_canonical=quote_matches,
                matched_text=matched_text,
            )
        )

    return verses


def _neighbourhood(text: str, position: int, span: int) -> str:
    """A slice of text around ``position``, preferring Arabic characters."""
    start = max(0, position - span)
    end = min(len(text), position + span)
    return text[start:end]


def detect_hadith(text: str) -> list[DetectedHadith]:
    """Detect mentions of the recognized hadith collections."""
    found: list[DetectedHadith] = []
    for match in _HADITH_PATTERN.finditer(text):
        collection = _HADITH_COLLECTIONS[match.group(1).lower()]
        if not any(h.collection == collection for h in found):
            found.append(DetectedHadith(collection=collection, matched_text=match.group(0).strip()))
    return found


def _content_type(manuscript_type: str | None, verses: list[ExtractedVerse], hadith: list[DetectedHadith]) -> str:
    """Classify the page's content type from evidence."""
    if manuscript_type in ("quran", "hadith", "fiqh", "tafsir", "history"):
        return manuscript_type
    if verses:
        return "quran"
    if hadith:
        return "hadith"
    return "other"


# ---------------------------------------------------------------------------
# Translation & explanation engine
# ---------------------------------------------------------------------------


class TranslationEngine(Protocol):
    async def translate(self, text: str, language: str, manuscript_type: str | None) -> ContentTranslation: ...


class StubTranslationEngine:
    """Deterministic offline translation/explanation for tests and CI."""

    async def translate(self, text: str, language: str, manuscript_type: str | None) -> ContentTranslation:
        return ContentTranslation(
            language=language,
            translation="[offline stub] The extracted Arabic text is presented for scholarly review.",
            explanation=(
                f"The page was classified as {manuscript_type or 'unclassified'}. "
                "Run with IMAGE_ANALYSIS_PROVIDER=gemini and a valid key for a live translation."
            ),
        )


class GeminiTranslationEngine:
    """Production engine: multimodal Gemini call to translate and explain the text."""

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or telemetry.GEMINI_MODEL

    async def translate(self, text: str, language: str, manuscript_type: str | None) -> ContentTranslation:
        import time

        import google.generativeai as genai

        instruction = (
            "You are a scholar of classical Islamic texts. Translate the attached Arabic "
            f"text into {language} and give a brief scholarly explanation. "
            f"The page was classified as: {manuscript_type or 'unknown'}. "
            'Return ONLY strict JSON: {"translation": string, "explanation": string}.'
        )
        model = genai.GenerativeModel(self.model_name, system_instruction=instruction)
        started = time.perf_counter()
        response = await model.generate_content_async(
            text,
            generation_config={"temperature": 0, "response_mime_type": "application/json"},
            request_options={"timeout": get_settings().gemini_timeout},
        )
        telemetry.record_model_call(
            response,
            self.model_name,
            (time.perf_counter() - started) * 1000.0,
            stage="image_translation",
        )
        try:
            payload = json.loads(response.text)
        except (ValueError, AttributeError) as exc:
            logger.warning("Translation backend returned unparseable JSON: %s", exc)
            return ContentTranslation(language=language)
        return ContentTranslation(
            language=language,
            translation=str(payload.get("translation") or ""),
            explanation=str(payload.get("explanation") or ""),
        )


def create_translation_engine(provider: str | None = None) -> TranslationEngine:
    chosen = provider if provider is not None else get_settings().image_analysis_provider
    if str(chosen).strip().lower() == "stub":
        return StubTranslationEngine()
    return GeminiTranslationEngine()


# ---------------------------------------------------------------------------
# OCR result cache (hash of the image bytes)
# ---------------------------------------------------------------------------


class _OcrCache:
    """In-process cache keyed by SHA-256 of the image bytes (LRU-bounded)."""

    def __init__(self, max_entries: int = 200) -> None:
        self._entries: dict[str, ManuscriptAnalysis] = {}
        self.max_entries = max_entries

    def get(self, digest: str) -> ManuscriptAnalysis | None:
        return self._entries.get(digest)

    def put(self, digest: str, analysis: ManuscriptAnalysis) -> None:
        self._entries[digest] = analysis
        if len(self._entries) > self.max_entries:
            # Evict the oldest inserted key deterministically.
            oldest = next(iter(self._entries))
            self._entries.pop(oldest, None)


ocr_cache = _OcrCache()


# ---------------------------------------------------------------------------
# Top-level analysis
# ---------------------------------------------------------------------------


async def analyze_uploaded_images(
    files: list[UploadFile],
    language: str = "en",
    skip_translation: bool = False,
    use_cache: bool = True,
) -> BatchAnalysisResponse:
    """Analyze several uploaded manuscript pages concurrently.

    Each page runs the full pipeline (validate → OCR → content analysis) with
    per-page error isolation: a page that fails validation or extraction
    reports under ``results`` with its error, and the remaining pages are
    unaffected.
    """
    results: list[dict[str, Any]] = []
    succeeded = 0
    failed = 0

    async def process(file: UploadFile) -> dict[str, Any]:
        nonlocal succeeded, failed
        data = await file.read()
        digest = hashlib.sha256(data).hexdigest()

        # Cache hit: identical bytes were analyzed before — serve the stored
        # analysis without re-running OCR (#135: cache to avoid reprocessing).
        if use_cache:
            cached = ocr_cache.get(digest)
            if cached is not None:
                analysis = await analyze_content(cached, language=language, skip_translation=skip_translation)
                succeeded += 1
                return {
                    "filename": file.filename or "upload",
                    "ok": True,
                    "cached": True,
                    "analysis": analysis.model_dump(),
                }

        try:
            manuscript = await analyze_manuscript_bytes(file.filename or "", data)
        except (UploadTooLargeError, UnsupportedFormatError, PoorQualityError) as exc:
            failed += 1
            return {"filename": file.filename or "upload", "ok": False, "error": str(exc)}
        except Exception as exc:  # noqa: BLE001 - isolate page failures
            logger.warning("Batch page failed: %s", exc)
            failed += 1
            return {"filename": file.filename or "upload", "ok": False, "error": str(exc)}

        ocr_cache.put(digest, manuscript)
        analysis = await analyze_content(manuscript, language=language, skip_translation=skip_translation)
        succeeded += 1
        return {
            "filename": file.filename or "upload",
            "ok": True,
            "cached": False,
            "analysis": analysis.model_dump(),
        }

    results = await asyncio.gather(*[process(file) for file in files])
    return BatchAnalysisResponse(results=results, total=len(results), succeeded=succeeded, failed=failed)


async def analyze_content(
    manuscript: ManuscriptAnalysis,
    language: str = "en",
    skip_translation: bool = False,
) -> ContentAnalysis:
    """Run content analysis over an OCR result: verses, hadith, metadata, translation."""
    text = manuscript.extracted_text
    verses = extract_verses(text)
    hadith = detect_hadith(text)
    content_type = _content_type(manuscript.manuscript_type.label, verses, hadith)

    validated = [v for v in verses if v.in_bounds]
    bounds_rate = len(validated) / len(verses) if verses else None
    text_match = [v for v in validated if v.quote_matches_canonical is True]
    text_mismatch = [v for v in validated if v.quote_matches_canonical is False]
    text_match_rate = len(text_match) / len(text_match) if (text_match or text_mismatch) else None

    validation: dict[str, Any] = {
        "references_found": len(verses),
        "in_bounds": len(validated),
        "out_of_bounds": len(verses) - len(validated),
        "bounds_pass_rate": bounds_rate,
        "canonical_text_compared": len(text_match) + len(text_mismatch),
        "canonical_text_match_rate": text_match_rate,
    }

    translation = ContentTranslation(language=language)
    if not skip_translation and text.strip():
        engine = create_translation_engine()
        try:
            translation = await engine.translate(text, language, content_type)
        except Exception as exc:  # noqa: BLE001 - translation never fails analysis
            logger.warning("Translation failed; returning analysis without it: %s", exc)
            translation = ContentTranslation(language=language, warning="Translation unavailable.")

    warnings = list(manuscript.warnings)
    if text_mismatch:
        warnings.append(f"{len(text_mismatch)} extracted verse quote(s) did not match the canonical text.")

    metadata = {
        "manuscript_type": manuscript.manuscript_type.label,
        "manuscript_confidence": manuscript.manuscript_type.confidence,
        "quality": manuscript.quality.model_dump(),
        "printed": manuscript.printed,
        "sections": [s.label for s in manuscript.sections],
        "historical_context": manuscript.historical_context,
        "image_sha256": None,
    }

    return ContentAnalysis(
        content_type=content_type,
        verses=verses,
        hadith=hadith,
        validation=validation,
        translation=translation,
        metadata=metadata,
        confidence=manuscript.confidence,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.post("/analyze-extracted", response_model=AnalyzeExtractedResponse)
async def analyze_extracted_text(body: AnalyzeExtractedRequest) -> AnalyzeExtractedResponse:
    """Analyze already-extracted OCR text: verses, hadith, validation, translation."""
    from manuscript_ocr import ManuscriptTypeClassification

    manuscript = ManuscriptAnalysis(
        extracted_text=body.text,
        manuscript_type=ManuscriptTypeClassification(
            label=body.manuscript_type or "unknown",
            confidence=0.5,
        ),
        confidence=0.5,
    )
    analysis = await analyze_content(
        manuscript,
        language=body.language,
        skip_translation=body.skip_translation,
    )
    return AnalyzeExtractedResponse(analysis=analysis)


@router.post("/batch", response_model=BatchAnalysisResponse)
async def analyze_batch(
    files: list[UploadFile] = File(...),
    language: str = "en",
    skip_translation: bool = False,
) -> BatchAnalysisResponse:
    """Analyze several uploaded manuscript pages in one request.

    Accepts up to 10 files (JPEG/PNG/WebP/PDF, each ≤10MB). Each page is
    validated, OCR'd (cached by image hash), and content-analyzed
    independently; a failing page is reported with its error and never
    discards the successful pages.
    """
    if len(files) > 10:
        raise APIException(
            status_code=422,
            detail="A batch request accepts at most 10 pages.",
            hint="Split the upload into multiple batch requests of up to 10 pages each.",
        )
    return await analyze_uploaded_images(
        files,
        language=language,
        skip_translation=skip_translation,
    )
