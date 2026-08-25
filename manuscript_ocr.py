"""Manuscript analysis pipeline (#233): upload validation, OCR, classification.

Accepts a scanned Islamic manuscript page (JPEG/PNG/PDF), validates it by
magic bytes rather than trusting the client-supplied content type, optionally
normalizes the image for the vision model, then extracts the Arabic text and
classifies the work. Two engines implement ``ManuscriptEngine``:

* ``GeminiManuscriptEngine`` — the production path; sends the image to the
  multimodal model with a structured-output prompt.
* ``StubManuscriptEngine`` — a deterministic offline fake keyed off marker
  bytes, so tests and CI (which have no secrets) exercise the full pipeline.

Provider selection is configuration-driven (``MANUSCRIPTS_PROVIDER``) and read
per request, so tests can flip it without restarting the app.

Errors are a small taxonomy mapped to HTTP codes by the route in ``main.py``:
413 over the size cap, 415 for unsupported or mismatched formats, 422 when the
extraction is too poor to be useful.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any, Protocol

from pydantic import BaseModel, Field

import telemetry
from config import get_settings
from feedback import RateLimiter, env_int

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration knobs
# ---------------------------------------------------------------------------

DEFAULT_MAX_UPLOAD_BYTES = 10 * 1024 * 1024

# Manuscript analysis is expensive (one vision call per upload) and bursty, so
# the per-client budget is tighter than chat's. Same sliding-window mechanism
# as /feedback — slowapi is wired up in main.py but no inline route uses it yet.
manuscript_rate_limiter = RateLimiter(
    max_calls=env_int("MANUSCRIPTS_RATE_LIMIT_MAX", 10),
    window_seconds=env_int("MANUSCRIPTS_RATE_LIMIT_WINDOW", 60),
)

MANUSCRIPT_TYPES: frozenset[str] = frozenset({"quran", "hadith", "fiqh", "tafsir", "history", "letter", "unknown"})


# ---------------------------------------------------------------------------
# Error taxonomy
# ---------------------------------------------------------------------------


class ManuscriptUploadError(Exception):
    """Base class for manuscript-pipeline failures with an HTTP mapping."""


class UploadTooLargeError(ManuscriptUploadError):
    """Upload exceeds the configured size cap → HTTP 413."""


class UnsupportedFormatError(ManuscriptUploadError):
    """Bytes are not a supported format, or clash with the filename → HTTP 415."""


class PoorQualityError(ManuscriptUploadError):
    """Nothing usable was extracted from the image → HTTP 422."""


# ---------------------------------------------------------------------------
# Upload validation: magic bytes first, extension must agree
# ---------------------------------------------------------------------------

_MAGIC_SIGNATURES: tuple[tuple[bytes, str], ...] = (
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"%PDF-", "application/pdf"),
)

_EXTENSION_MIME: dict[str, str] = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".pdf": "application/pdf",
}


@dataclass(frozen=True)
class ValidatedUpload:
    filename: str
    data: bytes
    mime: str


def sniff_format(data: bytes) -> str | None:
    """Detect the real format from leading magic bytes; None when unrecognized."""
    for signature, mime in _MAGIC_SIGNATURES:
        if data.startswith(signature):
            return mime
    return None


def validate_upload(filename: str, data: bytes, max_bytes: int = DEFAULT_MAX_UPLOAD_BYTES) -> ValidatedUpload:
    """Enforce the size cap and cross-check magic bytes against the filename.

    The content type a client declares is never trusted: the bytes decide what
    the file is, and the extension (when recognizable) must agree with them.
    """
    if len(data) > max_bytes:
        raise UploadTooLargeError(f"Upload is {len(data)} bytes; the maximum allowed is {max_bytes} bytes.")
    sniffed = sniff_format(data)
    if sniffed is None:
        raise UnsupportedFormatError("Unsupported file. Upload a JPEG image, PNG image, or PDF manuscript.")
    suffix = PurePosixPath(filename.replace("\\", "/")).suffix.lower()
    declared = _EXTENSION_MIME.get(suffix)
    if declared is None:
        raise UnsupportedFormatError(
            f"File extension {suffix or '(missing)'} is not supported; use .jpg, .jpeg, .png, or .pdf."
        )
    if declared != sniffed:
        raise UnsupportedFormatError(f"File content ({sniffed}) does not match its extension ({declared}).")
    return ValidatedUpload(filename=filename, data=data, mime=sniffed)


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class ManuscriptTypeClassification(BaseModel):
    label: str = "unknown"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence: list[str] = []


class QualityAssessment(BaseModel):
    legibility: float = Field(default=0.0, ge=0.0, le=1.0)
    completeness: float = Field(default=0.0, ge=0.0, le=1.0)


class ManuscriptSection(BaseModel):
    label: str
    text: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class ManuscriptAnalysis(BaseModel):
    extracted_text: str
    transliteration: str | None = None
    manuscript_type: ManuscriptTypeClassification
    analysis: str = ""
    historical_context: list[str] = []
    printed: bool = False
    quality: QualityAssessment = QualityAssessment()
    sections: list[ManuscriptSection] = []
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    warnings: list[str] = []


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------


@dataclass
class PreprocessedImage:
    data: bytes
    mime: str
    needs_ocr_backend: bool = False
    warnings: list[str] = field(default_factory=list)


class ImagePreprocessor(Protocol):
    def preprocess(self, data: bytes, mime: str) -> PreprocessedImage: ...


class PillowPreprocessor:
    """Light-touch normalization: grayscale, autocontrast, upscale low-res scans.

    Every failure degrades gracefully — the original bytes are returned with a
    warning rather than failing the request, because a mediocre image handed to
    the vision model beats a rejected upload.
    """

    # Scans below this width lose glyph detail for the vision model; doubling
    # is capped so a thumbnail cannot balloon memory.
    LOW_RESOLUTION_WIDTH = 800
    MAX_UPSCALED_WIDTH = 2400

    def preprocess(self, data: bytes, mime: str) -> PreprocessedImage:
        if mime == "application/pdf":
            # Page rasterization needs poppler/pdf2image; instead the PDF goes
            # to the vision backend untouched, which reads it natively.
            return PreprocessedImage(
                data=data,
                mime=mime,
                needs_ocr_backend=True,
                warnings=["PDF was sent to the vision backend without local rasterization."],
            )
        try:
            from PIL import Image, ImageOps

            with Image.open(io.BytesIO(data)) as image:
                normalized = ImageOps.autocontrast(image.convert("L"))
                warnings: list[str] = []
                scale = 1
                while (
                    normalized.width * scale < self.LOW_RESOLUTION_WIDTH
                    and normalized.width * scale * 2 <= self.MAX_UPSCALED_WIDTH
                ):
                    scale *= 2
                if scale > 1:
                    normalized = normalized.resize(
                        (normalized.width * scale, normalized.height * scale),
                        Image.Resampling.LANCZOS,
                    )
                    warnings.append(f"Low-resolution scan upscaled {scale}x before OCR.")
                buffer = io.BytesIO()
                normalized.save(buffer, format="PNG")
                return PreprocessedImage(data=buffer.getvalue(), mime="image/png", warnings=warnings)
        except Exception as exc:  # noqa: BLE001 - degradation beats rejection
            logger.warning("Manuscript preprocessing fell back to original bytes: %s", exc)
            return PreprocessedImage(
                data=data,
                mime=mime,
                warnings=[f"Preprocessing failed ({type(exc).__name__}); the original image was used."],
            )


pillow_preprocessor = PillowPreprocessor()


# ---------------------------------------------------------------------------
# Engines
# ---------------------------------------------------------------------------


class ManuscriptEngine(Protocol):
    async def analyze(self, image_bytes: bytes, mime: str) -> ManuscriptAnalysis: ...


_ANALYSIS_INSTRUCTION = (
    "You are a historian of Islamic manuscripts specializing in paleography and codicology. "
    "Read the attached manuscript image and return ONLY a strict JSON object with exactly these keys:\n"
    '{"extracted_text": string (the complete Arabic text verbatim, in Arabic script),\n'
    '"transliteration": string (Latin-script transliteration of the extracted text),\n'
    '"manuscript_type": {"label": one of "quran", "hadith", "fiqh", "tafsir", "history", "letter", '
    '"unknown", "confidence": number 0..1, "evidence": [short quotes supporting the classification]},\n'
    '"analysis": string (brief scholarly notes on script, layout, and content),\n'
    '"historical_context": [strings],\n'
    '"printed": boolean (true if mechanically printed rather than handwritten),\n'
    '"quality": {"legibility": number 0..1, "completeness": number 0..1},\n'
    '"sections": [{"label": string, "text": string, "confidence": number 0..1}],\n'
    '"confidence": number 0..1 (overall extraction confidence),\n'
    '"warnings": [strings]}'
)


def _clamp01(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return min(1.0, max(0.0, number))


def analysis_from_payload(raw: Any) -> ManuscriptAnalysis:
    """Coerce loose model JSON into the typed response; never trust the shape."""
    if not isinstance(raw, dict):
        raise PoorQualityError("The vision backend returned an unexpected response.")
    raw_type = raw.get("manuscript_type")
    if not isinstance(raw_type, dict):
        raw_type = {}
    evidence_raw = raw_type.get("evidence")
    if not isinstance(evidence_raw, list):
        evidence_raw = []
    quality_raw = raw.get("quality")
    if not isinstance(quality_raw, dict):
        quality_raw = {}
    sections_raw = raw.get("sections")
    if not isinstance(sections_raw, list):
        sections_raw = []
    history_raw = raw.get("historical_context")
    if not isinstance(history_raw, list):
        history_raw = []
    warnings_raw = raw.get("warnings")
    if not isinstance(warnings_raw, list):
        warnings_raw = []
    label = str(raw_type.get("label") or "unknown").strip().lower()
    if label not in MANUSCRIPT_TYPES:
        label = "unknown"
    transliteration = raw.get("transliteration")
    return ManuscriptAnalysis(
        extracted_text=str(raw.get("extracted_text") or ""),
        transliteration=str(transliteration) if transliteration else None,
        manuscript_type=ManuscriptTypeClassification(
            label=label,
            confidence=_clamp01(raw_type.get("confidence")),
            evidence=[str(quote) for quote in evidence_raw if str(quote).strip()],
        ),
        analysis=str(raw.get("analysis") or ""),
        historical_context=[str(note) for note in history_raw if str(note).strip()],
        printed=bool(raw.get("printed")),
        quality=QualityAssessment(
            legibility=_clamp01(quality_raw.get("legibility")),
            completeness=_clamp01(quality_raw.get("completeness")),
        ),
        sections=[
            ManuscriptSection(
                label=str(section.get("label") or "section"),
                text=str(section.get("text") or ""),
                confidence=_clamp01(section.get("confidence")),
            )
            for section in sections_raw
            if isinstance(section, dict)
        ],
        confidence=_clamp01(raw.get("confidence")),
        warnings=[str(warning) for warning in warnings_raw if str(warning).strip()],
    )


class GeminiManuscriptEngine:
    """Production engine: multimodal Gemini call with a structured-output prompt."""

    def __init__(self, model_name: str | None = None, timeout_seconds: int | None = None) -> None:
        self.model_name = model_name or telemetry.GEMINI_MODEL
        self.timeout_seconds = timeout_seconds

    async def analyze(self, image_bytes: bytes, mime: str) -> ManuscriptAnalysis:
        import google.generativeai as genai

        model = genai.GenerativeModel(self.model_name, system_instruction=_ANALYSIS_INSTRUCTION)
        started = time.perf_counter()
        response = await model.generate_content_async(
            [{"mime_type": mime, "data": image_bytes}],
            generation_config={"temperature": 0, "response_mime_type": "application/json"},
            request_options={"timeout": self.timeout_seconds or get_settings().gemini_timeout},
        )
        telemetry.record_model_call(
            response,
            self.model_name,
            (time.perf_counter() - started) * 1000.0,
            stage="manuscript_analysis",
        )
        return analysis_from_payload(_response_json(response))


def _response_json(response: Any) -> Any:
    """Extract and parse the JSON body, tolerating safety-blocked responses."""
    try:
        text = response.text
    except (ValueError, AttributeError):
        text = None
    if not text:
        raise PoorQualityError("The vision backend returned no analysis.")
    try:
        return json.loads(text)
    except ValueError as exc:
        logger.warning("Manuscript analysis returned invalid JSON: %s", exc)
        raise PoorQualityError("The vision backend returned an unparseable response.") from exc


class StubManuscriptEngine:
    """Deterministic offline engine driven by marker bytes in the upload.

    Markers (matched in the first 512 bytes):
      ``MSS-TYPE:<label>``  selects the classified manuscript type
      ``MSS-PRINTED``       classifies the page as printed
      ``MSS-LOW-QUALITY``   produces a near-empty, low-confidence extraction
    """

    TYPE_MARKER = re.compile(rb"MSS-TYPE:([a-z]+)")
    PRINTED_MARKER = b"MSS-PRINTED"
    LOW_QUALITY_MARKER = b"MSS-LOW-QUALITY"

    async def analyze(self, image_bytes: bytes, mime: str) -> ManuscriptAnalysis:
        window = image_bytes[:512]
        found = self.TYPE_MARKER.search(window)
        label = found.group(1).decode("ascii") if found else "quran"
        if label not in MANUSCRIPT_TYPES:
            label = "unknown"
        if self.LOW_QUALITY_MARKER in window:
            return ManuscriptAnalysis(
                extracted_text="",
                manuscript_type=ManuscriptTypeClassification(label="unknown", confidence=0.05),
                confidence=0.05,
                warnings=["Stub engine: low-quality marker detected."],
            )
        text = "بِسْمِ ٱللَّهِ ٱلرَّحْمَـٰنِ ٱلرَّحِيمِ"
        return ManuscriptAnalysis(
            extracted_text=text,
            transliteration="bismi llāhi r-raḥmāni r-raḥīm",
            manuscript_type=ManuscriptTypeClassification(
                label=label,
                confidence=0.92,
                evidence=["Opening invocation characteristic of the classified genre."],
            ),
            analysis=(
                f"Stub analysis: page identified as {label}, written in a clear naskh-style hand "
                "with vocalized script throughout."
            ),
            historical_context=[
                "Stub engine output; replace with MANUSCRIPTS_PROVIDER=gemini for real analysis.",
            ],
            printed=self.PRINTED_MARKER in window,
            quality=QualityAssessment(legibility=0.9, completeness=0.85),
            sections=[ManuscriptSection(label="opening", text=text, confidence=0.9)],
            confidence=0.9,
        )


def create_manuscript_engine(provider: str | None = None) -> ManuscriptEngine:
    """Resolve the configured provider; unknown names fall back to gemini."""
    chosen = provider if provider is not None else get_settings().manuscripts_provider
    chosen = str(chosen).strip().lower()
    if chosen == "stub":
        return StubManuscriptEngine()
    if chosen != "gemini":
        logger.warning("Unknown MANUSCRIPTS_PROVIDER %r; falling back to gemini.", chosen)
    return GeminiManuscriptEngine()


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


async def analyze_manuscript_bytes(filename: str, data: bytes) -> ManuscriptAnalysis:
    """Validate → preprocess → analyze → gate on quality. Raises the taxonomy above."""
    settings = get_settings()
    upload = validate_upload(filename, data, max_bytes=settings.manuscripts_max_upload_bytes)
    prepared = await asyncio.to_thread(pillow_preprocessor.preprocess, upload.data, upload.mime)
    engine = create_manuscript_engine(settings.manuscripts_provider)
    analysis = await engine.analyze(prepared.data, prepared.mime)
    if not analysis.extracted_text.strip():
        raise PoorQualityError("No readable Arabic text could be extracted from this manuscript.")
    if analysis.confidence < settings.manuscripts_min_confidence:
        raise PoorQualityError(
            f"Extraction confidence {analysis.confidence:.2f} is below the accepted "
            f"minimum of {settings.manuscripts_min_confidence:.2f}. Try a sharper photo."
        )
    analysis.warnings.extend(prepared.warnings)
    return analysis
