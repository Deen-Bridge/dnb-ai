"""Arabic calligraphy OCR and recognition (#234).

Recognizes text in stylized Arabic calligraphy AND classifies the script
style with artistic metadata. Calligraphy is a much harder OCR problem than
plain printed text: letters overlap into ligature clusters (thuluth, diwani),
ornaments invade the letter space, and historical scripts (kufi) omit dots —
so the pipeline reports calibrated confidence and explicit warnings instead
of silently returning a best guess.

Structure
---------
- ``CalligraphyStyleCatalog``: validated knowledge base of classical styles
  loaded from ``data/calligraphy_styles.json``, lookup by label or alias.
- ``CalligraphyEngine`` protocol with two implementations:
  - ``GeminiCalligraphyEngine``: vision model call following the house
    pattern (temperature=0, JSON mime type, request timeout), engineered to
    transcribe through overlapping ligatures, normalize orthography,
    classify style, flag decorative interference, and estimate period.
  - ``StubCalligraphyEngine``: deterministic offline fake driven by ASCII
    marker bytes embedded in the image payload (used by tests).
- ``calibrate()``: blends engine-reported region/style confidences and
  extraction completeness into one overall score plus warnings[].
- ``to_manuscript_payload()``: adapter for a future generic
  manuscript-analysis pipeline.

Offline seam: tests can monkeypatch ``GeminiCalligraphyEngine._call_gemini``
(the same approach memory.extraction uses for its Gemini seams).
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, Field

import telemetry

logger = logging.getLogger(__name__)

STYLE_KB_PATH = Path(__file__).resolve().parent / "data" / "calligraphy_styles.json"

# Fallback when no configured threshold is supplied (mirrors
# Settings.calligraphy_min_confidence).
DEFAULT_MIN_CONFIDENCE = 0.35

SUPPORTED_MIME_TYPES = ("image/jpeg", "image/png")

_JPEG_MAGIC = b"\xff\xd8\xff"
_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"

# Stub marker bytes (ASCII tokens embedded after a valid magic header).
_MARKER_STYLE = re.compile(rb"style=([a-z]+)")
_MARKER_LOWCONF = b"[lowconf]"
_MARKER_HEAVY = b"[heavy]"
_MARKER_NOLEGIBLE = b"[nolegible]"


# ---------------------------------------------------------------------------
# Output models
# ---------------------------------------------------------------------------


class StyleClassification(BaseModel):
    """Classified calligraphy style among known KB labels."""

    label: str
    confidence: float = Field(ge=0.0, le=1.0)
    alternates: list[str] = []


class PeriodEstimate(BaseModel):
    """Classical-vs-contemporary judgement plus a coarse era estimate."""

    era: str | None = None
    classical: bool | None = None


class RegionResult(BaseModel):
    """One recognized text region."""

    text: str
    confidence: float = Field(ge=0.0, le=1.0)
    bbox_hint: list[int] | None = None


class EngineMetadata(BaseModel):
    """Content-free provenance for one analysis run."""

    engine: str
    model_name: str | None = None
    latency_ms: float | None = None
    mime: str | None = None
    image_bytes: int | None = None


class CalligraphyAnalysis(BaseModel):
    """Full result of one calligraphy analysis."""

    extracted_text: str
    transcription_normalized: str | None = None
    style: StyleClassification
    period: PeriodEstimate = PeriodEstimate()
    decorations_detected: bool = False
    regions: list[RegionResult] = []
    warnings: list[str] = []
    legibility: float = Field(default=0.0, ge=0.0, le=1.0)
    overall_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    metadata: EngineMetadata


# ---------------------------------------------------------------------------
# Engine interface
# ---------------------------------------------------------------------------


class CalligraphyEngine(Protocol):
    """Any backend able to analyze one calligraphy image."""

    def analyze(self, image_bytes: bytes, mime: str) -> CalligraphyAnalysis: ...


# ---------------------------------------------------------------------------
# Style knowledge base
# ---------------------------------------------------------------------------


class CalligraphyStyle(BaseModel):
    label: str
    name: str
    arabic_name: str
    aliases: list[str] = []
    period_origin: str
    regional_traditions: list[str] = []
    visual_traits: list[str] = []
    decorativeness: float = Field(ge=0.0, le=1.0)
    common_use_cases: list[str] = []


_REQUIRED_FIELDS = (
    "label",
    "name",
    "arabic_name",
    "period_origin",
    "decorativeness",
)


def _normalize_key(value: str) -> str:
    """Fold punctuation variants ('ruqʿah', 'rüqah', "ruq'ah") onto one key."""
    lowered = value.strip().lower()
    replaced = (
        lowered.replace("ʿ", "")
        .replace("ʻ", "")
        .replace("ʼ", "")
        .replace("'", "")
        .replace("’", "")
        .replace("ü", "u")
        .replace("-", "")
        .replace("_", "")
        .replace(" ", "")
    )
    return re.sub(r"[^a-z]", "", replaced)


class CalligraphyStyleCatalog:
    """Loads and validates the bundled style KB; resolves labels and aliases."""

    def __init__(self, data_file: Path = STYLE_KB_PATH):
        self.data_file = data_file
        self.styles: dict[str, CalligraphyStyle] = {}
        self._by_alias: dict[str, str] = {}
        self._load()

    def _load(self) -> None:
        if not self.data_file.exists():
            logger.warning("Style KB missing at %s — style metadata disabled", self.data_file)
            return
        with open(self.data_file, encoding="utf-8") as f:
            payload = json.load(f)

        entries = payload.get("styles")
        if not isinstance(entries, list) or not entries:
            raise ValueError(f"{self.data_file}: 'styles' must be a non-empty list")

        seen: set[str] = set()
        for entry in entries:
            missing = [field for field in _REQUIRED_FIELDS if field not in entry]
            if missing:
                raise ValueError(f"{self.data_file}: style entry missing fields {missing}")
            try:
                style = CalligraphyStyle.model_validate(entry)
            except Exception as exc:
                raise ValueError(f"{self.data_file}: invalid style entry {entry.get('label')!r}: {exc}") from exc

            key = _normalize_key(style.label)
            if not key:
                raise ValueError(f"{self.data_file}: empty style label")
            if key in seen:
                raise ValueError(f"{self.data_file}: duplicate style label {style.label!r}")
            seen.add(key)

            self.styles[style.label] = style
            self._by_alias[key] = style.label
            for alias in style.aliases:
                alias_key = _normalize_key(alias)
                if alias_key:
                    self._by_alias.setdefault(alias_key, style.label)

    def lookup(self, label_or_alias: str) -> CalligraphyStyle | None:
        """Resolve a raw engine/user label or alias to a canonical style."""
        return self.styles.get(label_or_alias.strip().lower()) or self.styles.get(
            self._by_alias.get(_normalize_key(label_or_alias), "")
        )

    def labels(self) -> list[str]:
        return list(self.styles)

    def decorativeness_of(self, label: str) -> float:
        style = self.lookup(label)
        return style.decorativeness if style else 0.5


# Shared instance across the application
style_catalog = CalligraphyStyleCatalog()


# ---------------------------------------------------------------------------
# Confidence calibration
# ---------------------------------------------------------------------------


def calibrate(
    analysis: CalligraphyAnalysis,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    catalog: CalligraphyStyleCatalog = style_catalog,
) -> CalligraphyAnalysis:
    """Blend engine-reported signals into ``overall_confidence`` + ``warnings``.

    Weights: mean region confidence 60%, style classification 25%, extraction
    completeness 15% (how much of the raw extraction survived normalization).
    Mutates and returns *analysis* so engines can call it inline.
    """
    warnings: list[str] = []

    region_scores = [r.confidence for r in analysis.regions]
    region_mean = sum(region_scores) / len(region_scores) if region_scores else 0.0

    normalized_len = len((analysis.transcription_normalized or "").strip())
    extracted_len = len(analysis.extracted_text.strip())
    completeness = min(1.0, normalized_len / extracted_len) if extracted_len > 0 else 0.0

    overall = round(0.6 * region_mean + 0.25 * analysis.style.confidence + 0.15 * completeness, 4)
    overall = max(0.0, min(1.0, overall))

    for i, region in enumerate(analysis.regions, start=1):
        if region.confidence < min_confidence:
            preview = region.text.strip()[:30]
            warnings.append(f"Low-confidence region {i} ({region.confidence:.2f}): '{preview}'")

    if analysis.style.confidence < min_confidence:
        warnings.append(
            f"Style classification is uncertain ({analysis.style.confidence:.2f} for "
            f"'{analysis.style.label}') — alternates: {analysis.style.alternates or ['none']}"
        )

    decorativeness = catalog.decorativeness_of(analysis.style.label)
    if analysis.decorations_detected and (decorativeness >= 0.7 or overall < min_confidence):
        warnings.append(
            f"Heavy decoration interferes with reading '{analysis.style.label}' "
            f"(decorativeness {decorativeness:.2f}) — treat transcription as tentative"
        )

    if not analysis.extracted_text.strip():
        warnings.append("No legible calligraphy detected")

    analysis.overall_confidence = overall
    analysis.warnings = warnings
    return analysis


# ---------------------------------------------------------------------------
# Gemini vision engine
# ---------------------------------------------------------------------------

_ANALYSIS_INSTRUCTION = """You are an expert Arabic paleographer and calligrapher analyzing an image \
of Arabic calligraphy for Deen Bridge.

Perform all of the following tasks and return ONLY strict JSON (no markdown, no code fences):

1. TRANSCRIBE every readable word, even where letters overlap in dense ligatures or interlace with \
decorative elements. Reconstruct obscured letterforms from context before giving up.
2. Normalize orthography: restore omitted dots and hamzas expected by the style (e.g. kufi often \
omits i'jam dots), resolve stylized ligatures to standard forms, and supply modern standard spelling.
3. Classify the dominant calligraphy style. "style_label" MUST be exactly one of these labels: \
{labels}. Provide up to two plausible alternates in "style_alternates".
4. Report whether decorative ornaments (flourishes, knotwork, rosettes, illumination borders) \
interfere with reading any part of the text ("decorations_detected").
5. Estimate whether the hand is classical or contemporary ("classical": true/false) and give a \
coarse period estimate in "period_era" (e.g. "Ottoman, 16th century", "contemporary").
6. Give per-region results in reading order: each region's transcribed text, your confidence \
(0.0-1.0), and optionally a rough bbox_hint as [x, y, width, height] pixels.
7. Give an overall "legibility" score (0.0-1.0): how much of the visible text a human expert \
could read.

Return JSON with exactly these keys:
{{"regions": [{{"text": "...", "confidence": 0.0, "bbox_hint": null}}],
"extracted_text": "...", "transcription_normalized": "...",
"style_label": "{default_label}", "style_alternates": [], "style_confidence": 0.0,
"classical": true, "period_era": "...", "decorations_detected": false, "legibility": 0.0}}"""


def parse_engine_payload(payload: dict[str, Any]) -> CalligraphyAnalysis:
    """Build a partial analysis from an untrusted engine JSON dict.

    Unknown style labels are kept verbatim (the route layer decides whether to
    surface them); malformed numeric fields fall back to neutral values rather
    than failing the whole request.
    """

    def _clamp01(value: Any, default: float = 0.0) -> float:
        try:
            return max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            return default

    raw_regions = payload.get("regions")
    regions_raw: list[Any] = raw_regions if isinstance(raw_regions, list) else []
    regions: list[RegionResult] = []
    for item in regions_raw[:32]:
        if not isinstance(item, dict) or not str(item.get("text", "")).strip():
            continue
        bbox = item.get("bbox_hint")
        regions.append(
            RegionResult(
                text=str(item["text"]),
                confidence=_clamp01(item.get("confidence"), 0.5),
                bbox_hint=[int(v) for v in bbox] if isinstance(bbox, list) and len(bbox) == 4 else None,
            )
        )

    alternates_raw = payload.get("style_alternates")
    alternates = [str(a) for a in alternates_raw][:2] if isinstance(alternates_raw, list) else []

    classical = payload.get("classical")
    return CalligraphyAnalysis(
        extracted_text=str(payload.get("extracted_text") or ""),
        transcription_normalized=(payload.get("transcription_normalized") or None),
        style=StyleClassification(
            label=str(payload.get("style_label") or "unknown"),
            confidence=_clamp01(payload.get("style_confidence")),
            alternates=alternates,
        ),
        period=PeriodEstimate(
            era=(str(payload["period_era"]) if payload.get("period_era") else None),
            classical=bool(classical) if isinstance(classical, bool) else None,
        ),
        decorations_detected=bool(payload.get("decorations_detected")),
        regions=regions,
        legibility=_clamp01(payload.get("legibility")),
        metadata=EngineMetadata(engine="gemini"),
    )


class GeminiCalligraphyEngine:
    """Vision-model engine implementing :class:`CalligraphyEngine`."""

    def __init__(
        self,
        model_name: str = telemetry.GEMINI_MODEL,
        timeout: int = 30,
        min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    ):
        self.model_name = model_name
        self.timeout = timeout
        self.min_confidence = min_confidence

    def _build_prompt(self) -> str:
        return _ANALYSIS_INSTRUCTION.format(
            labels=", ".join(style_catalog.labels()),
            default_label="modern",
        )

    def _call_gemini(self, prompt: str, image_bytes: bytes, mime: str) -> str:
        """Raw SDK call. Offline seam: patch this method in tests."""
        import google.generativeai as genai

        model = genai.GenerativeModel(self.model_name, system_instruction=prompt)
        response = model.generate_content(
            [{"mime_type": mime, "data": image_bytes}],
            generation_config={"temperature": 0, "response_mime_type": "application/json"},
            request_options={"timeout": self.timeout},
        )
        telemetry.record_model_call(response, self.model_name, 0.0, stage="calligraphy_ocr")
        return response.text

    def analyze(self, image_bytes: bytes, mime: str) -> CalligraphyAnalysis:
        started = time.perf_counter()
        prompt = self._build_prompt()
        raw = self._call_gemini(prompt, image_bytes, mime)
        latency_ms = (time.perf_counter() - started) * 1000.0

        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Calligraphy engine returned invalid JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError("Calligraphy engine returned a non-object JSON payload")

        analysis = parse_engine_payload(payload)
        calibrate(analysis, self.min_confidence)
        analysis.metadata = EngineMetadata(
            engine="gemini",
            model_name=self.model_name,
            latency_ms=round(latency_ms, 2),
            mime=mime,
            image_bytes=len(image_bytes),
        )
        return analysis


# ---------------------------------------------------------------------------
# Deterministic offline stub (tests only)
# ---------------------------------------------------------------------------


class StubCalligraphyEngine:
    """Marker-driven fake so tests never touch the network.

    Markers are plain ASCII tokens inside the image payload:
      - ``style=<label>``  force the classified style (e.g. ``style=diwani``)
      - ``[lowconf]``      drop confidences below typical thresholds
      - ``[heavy]``        report heavy decorative interference
      - ``[nolegible]``    produce no extractable text (route returns 422)
    Without markers it deterministically returns the basmala in naskh.
    """

    DEFAULT_TEXT = "بسم الله الرحمن الرحيم"
    DEFAULT_NORMALIZED = "بسم الله الرحمن الرحيم"
    DEFAULT_ERA = "Ottoman, 18th century"

    def __init__(self, min_confidence: float = DEFAULT_MIN_CONFIDENCE):
        self.min_confidence = min_confidence

    def analyze(self, image_bytes: bytes, mime: str) -> CalligraphyAnalysis:
        lowconf = _MARKER_LOWCONF in image_bytes
        heavy = _MARKER_HEAVY in image_bytes
        nolegible = _MARKER_NOLEGIBLE in image_bytes
        style_match = _MARKER_STYLE.search(image_bytes)
        label = style_match.group(1).decode("ascii") if style_match else "naskh"

        region_conf = 0.2 if lowconf else 0.95
        style_conf = 0.2 if lowconf else 0.9
        text = "" if nolegible else self.DEFAULT_TEXT

        analysis = CalligraphyAnalysis(
            extracted_text=text,
            transcription_normalized=text or None,
            style=StyleClassification(
                label=label,
                confidence=style_conf,
                alternates=["thuluth"] if label == "diwani" else [],
            ),
            period=PeriodEstimate(era=self.DEFAULT_ERA, classical=True),
            decorations_detected=heavy or label == "diwani",
            regions=[RegionResult(text=t, confidence=region_conf) for t in ([text] if text else [])],
            legibility=0.2 if lowconf else 0.95,
            metadata=EngineMetadata(
                engine="stub",
                model_name="stub-calligraphy",
                mime=mime,
                image_bytes=len(image_bytes),
            ),
        )
        return calibrate(analysis, self.min_confidence)


# ---------------------------------------------------------------------------
# Image sniffing + manuscript-payload adapter
# ---------------------------------------------------------------------------


def sniff_image_mime(data: bytes) -> str | None:
    """Return the real MIME type from magic bytes, or None if unsupported."""
    if data.startswith(_JPEG_MAGIC):
        return "image/jpeg"
    if data.startswith(_PNG_MAGIC):
        return "image/png"
    return None


def to_manuscript_payload(analysis: CalligraphyAnalysis) -> dict[str, Any]:
    """Project a calligraphy analysis onto the generic manuscript-analysis shape.

    Field mapping (calligraphy → future manuscript pipeline):

    ================================  =====================================
    CalligraphyAnalysis field         Manuscript payload field
    ================================  =====================================
    extracted_text                    text.raw
    transcription_normalized          text.normalized
    regions[i].text                   regions[i].text
    regions[i].confidence             regions[i].confidence
    regions[i].bbox_hint              regions[i].bbox
    style.label                       script.family
    style.alternates                  script.variants
    style.confidence                  script.classification_confidence
    period.classical                  script.classical
    period.era                        script.period_estimate
    decorations_detected              quality.decorated
    legibility                        quality.legibility
    overall_confidence                quality.confidence
    warnings                          quality.warnings
    ================================  =====================================

    Provenance rides along under ``source`` (engine/model/byte size).
    """
    return {
        "text": {
            "raw": analysis.extracted_text,
            "normalized": analysis.transcription_normalized,
        },
        "regions": [{"text": r.text, "confidence": r.confidence, "bbox": r.bbox_hint} for r in analysis.regions],
        "script": {
            "family": analysis.style.label,
            "variants": analysis.style.alternates,
            "classification_confidence": analysis.style.confidence,
            "classical": analysis.period.classical,
            "period_estimate": analysis.period.era,
        },
        "quality": {
            "decorated": analysis.decorations_detected,
            "legibility": analysis.legibility,
            "confidence": analysis.overall_confidence,
            "warnings": analysis.warnings,
        },
        "source": {
            "engine": analysis.metadata.engine,
            "model_name": analysis.metadata.model_name,
            "image_mime": analysis.metadata.mime,
            "image_bytes": analysis.metadata.image_bytes,
        },
    }
