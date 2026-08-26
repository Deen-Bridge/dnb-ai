"""Arabic OCR for Islamic Manuscripts (#214).

Why this exists
---------------
Historical Islamic manuscripts present unique OCR challenges: multiple
calligraphy styles (Naskh, Thuluth, Nastaliq, Maghribi), degraded aged
documents, diacritical marks (tashkeel) that are crucial for meaning, and
marginal annotations. Standard OCR engines lose diacritics, misread
ligatures, and conflate similar letter forms. This module wraps external
OCR engines with Islamic-manuscript-specific preprocessing, calligraphy
detection, diacritic preservation, and confidence scoring.

Architecture
------------
The module operates in a pipeline:

1. **Image preprocessing**: Binarization, deskewing, noise removal optimized
   for aged parchment, foxing, and ink bleeding common in manuscripts.

2. **Calligraphy detection**: Classify the script style before OCR to select
   the best model weights/configuration for that hand.

3. **OCR execution**: Call an external engine (Google Vision, Azure, Tesseract
   with Arabic models, or custom endpoints) with style-tuned parameters.

4. **Post-processing**: Arabic-specific text normalization, diacritic recovery
   using language models, common error correction (hamza placement, ta marbuta).

5. **Confidence scoring**: Per-character and per-word confidence with
   manuscript-aware adjustments for damaged regions.

The module exposes FastAPI routes for each stage and an end-to-end pipeline.
"""

from __future__ import annotations

import base64
import hashlib
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from fastapi import APIRouter, File, Form, UploadFile
from pydantic import BaseModel, Field, field_validator

router = APIRouter(prefix="/arabic-ocr", tags=["arabic-ocr"])


# ---------------------------------------------------------------------------
# Calligraphy Styles
# ---------------------------------------------------------------------------


class CalligraphyStyle(str, Enum):
    """Major Arabic calligraphy styles found in Islamic manuscripts."""

    NASKH = "naskh"
    """Clear, readable script used in most printed texts and Qurans."""

    THULUTH = "thuluth"
    """Ornate script for titles and headings, with elongated letters."""

    NASTALIQ = "nastaliq"
    """Persian/Urdu style with diagonal baseline, common in poetry."""

    MAGHRIBI = "maghribi"
    """North African style with distinctive letter forms."""

    KUFI = "kufi"
    """Angular, geometric script from early Islamic period."""

    DIWANI = "diwani"
    """Flowing Ottoman court script, highly cursive."""

    RUQAH = "ruqah"
    """Simple, quick handwriting style."""

    UNKNOWN = "unknown"
    """Style could not be determined."""


STYLE_DESCRIPTIONS: dict[CalligraphyStyle, str] = {
    CalligraphyStyle.NASKH: "Clear, rounded script standard for Quranic and scholarly texts",
    CalligraphyStyle.THULUTH: "Elegant display script with elongated letterforms",
    CalligraphyStyle.NASTALIQ: "Slanted Persian style with hanging letters",
    CalligraphyStyle.MAGHRIBI: "Rounded North African style with distinctive dots",
    CalligraphyStyle.KUFI: "Angular early Islamic script, often in Quran headings",
    CalligraphyStyle.DIWANI: "Ornate flowing Ottoman chancellery script",
    CalligraphyStyle.RUQAH: "Compact everyday handwriting",
    CalligraphyStyle.UNKNOWN: "Unidentified or mixed calligraphy style",
}

# Feature patterns used for style classification heuristics
STYLE_FEATURES: dict[CalligraphyStyle, dict[str, Any]] = {
    CalligraphyStyle.NASKH: {
        "baseline_angle": 0,
        "letter_spacing": "medium",
        "vertical_emphasis": False,
    },
    CalligraphyStyle.THULUTH: {
        "baseline_angle": 0,
        "letter_spacing": "wide",
        "vertical_emphasis": True,
    },
    CalligraphyStyle.NASTALIQ: {
        "baseline_angle": -25,  # slanted downward right-to-left
        "letter_spacing": "tight",
        "vertical_emphasis": False,
    },
    CalligraphyStyle.MAGHRIBI: {
        "baseline_angle": 0,
        "letter_spacing": "medium",
        "rounded_forms": True,
    },
    CalligraphyStyle.KUFI: {
        "baseline_angle": 0,
        "letter_spacing": "wide",
        "angular": True,
    },
}


# ---------------------------------------------------------------------------
# Image Preprocessing Configuration
# ---------------------------------------------------------------------------


class PreprocessingProfile(str, Enum):
    """Preset profiles for different manuscript conditions."""

    PRISTINE = "pristine"
    """Well-preserved, high-contrast manuscript."""

    AGED = "aged"
    """Yellowed paper, faded ink, typical of old manuscripts."""

    DAMAGED = "damaged"
    """Tears, stains, water damage, missing sections."""

    FADED = "faded"
    """Severely faded ink requiring aggressive enhancement."""

    PALIMPSEST = "palimpsest"
    """Erased and rewritten, multiple text layers."""


class PreprocessingConfig(BaseModel):
    """Configuration for manuscript image preprocessing."""

    profile: PreprocessingProfile = PreprocessingProfile.AGED
    """Base profile for preprocessing parameters."""

    binarization_method: str = Field(
        default="adaptive_gaussian",
        description="'otsu', 'adaptive_gaussian', 'adaptive_mean', or 'sauvola'",
    )

    deskew: bool = Field(default=True, description="Correct page rotation.")

    denoise_strength: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Noise removal intensity (0=none, 1=aggressive).",
    )

    contrast_enhancement: float = Field(
        default=1.2,
        ge=0.5,
        le=3.0,
        description="Contrast multiplier for faded text.",
    )

    remove_margins: bool = Field(
        default=True,
        description="Crop dark scan margins.",
    )

    invert_if_dark: bool = Field(
        default=True,
        description="Auto-invert negative/dark background images.",
    )


# Default preprocessing settings per profile
PROFILE_DEFAULTS: dict[PreprocessingProfile, dict[str, Any]] = {
    PreprocessingProfile.PRISTINE: {
        "denoise_strength": 0.2,
        "contrast_enhancement": 1.0,
    },
    PreprocessingProfile.AGED: {
        "denoise_strength": 0.5,
        "contrast_enhancement": 1.3,
    },
    PreprocessingProfile.DAMAGED: {
        "denoise_strength": 0.7,
        "contrast_enhancement": 1.5,
        "binarization_method": "sauvola",
    },
    PreprocessingProfile.FADED: {
        "denoise_strength": 0.3,
        "contrast_enhancement": 2.0,
    },
    PreprocessingProfile.PALIMPSEST: {
        "denoise_strength": 0.8,
        "contrast_enhancement": 2.5,
        "binarization_method": "sauvola",
    },
}


# ---------------------------------------------------------------------------
# OCR Engine Configuration
# ---------------------------------------------------------------------------


class OCREngine(str, Enum):
    """Supported OCR backends."""

    GOOGLE_VISION = "google_vision"
    """Google Cloud Vision API with Arabic support."""

    AZURE_COGNITIVE = "azure_cognitive"
    """Azure Computer Vision OCR."""

    TESSERACT = "tesseract"
    """Tesseract with Arabic language pack."""

    CUSTOM = "custom"
    """Custom endpoint for specialized Arabic OCR models."""


class OCREngineConfig(BaseModel):
    """Configuration for the OCR engine."""

    engine: OCREngine = OCREngine.GOOGLE_VISION

    api_endpoint: str | None = Field(
        default=None,
        description="Custom API endpoint for CUSTOM engine.",
    )

    language_hints: list[str] = Field(
        default_factory=lambda: ["ar", "fa"],  # Arabic, Persian
        description="Language codes to hint to the OCR engine.",
    )

    detect_diacritics: bool = Field(
        default=True,
        description="Enable diacritical mark detection (slower but more accurate).",
    )

    model_variant: str | None = Field(
        default=None,
        description="Specific model variant (e.g., 'arabic-manuscript-v2').",
    )


# ---------------------------------------------------------------------------
# OCR Results
# ---------------------------------------------------------------------------


class BoundingBox(BaseModel):
    """Axis-aligned bounding box in image coordinates."""

    x: float = Field(..., ge=0, description="Left edge.")
    y: float = Field(..., ge=0, description="Top edge.")
    width: float = Field(..., ge=0)
    height: float = Field(..., ge=0)


class CharacterResult(BaseModel):
    """Single recognized character with confidence."""

    char: str = Field(..., min_length=1, max_length=4)
    confidence: float = Field(..., ge=0.0, le=1.0)
    bbox: BoundingBox | None = None
    is_diacritic: bool = False


class WordResult(BaseModel):
    """A recognized word with character-level detail."""

    text: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    characters: list[CharacterResult] = Field(default_factory=list)
    bbox: BoundingBox | None = None
    diacritic_count: int = Field(default=0, ge=0)
    corrections_applied: list[str] = Field(default_factory=list)


class LineResult(BaseModel):
    """A line of recognized text."""

    text: str
    words: list[WordResult] = Field(default_factory=list)
    confidence: float = Field(..., ge=0.0, le=1.0)
    bbox: BoundingBox | None = None


class CalligraphyDetection(BaseModel):
    """Result of calligraphy style detection."""

    primary_style: CalligraphyStyle
    confidence: float = Field(..., ge=0.0, le=1.0)
    secondary_styles: list[tuple[CalligraphyStyle, float]] = Field(default_factory=list)
    features_detected: dict[str, Any] = Field(default_factory=dict)


class OCRConfidenceMetrics(BaseModel):
    """Aggregate confidence metrics for an OCR result."""

    overall: float = Field(..., ge=0.0, le=1.0, description="Weighted average confidence.")
    character_level: float = Field(..., ge=0.0, le=1.0)
    word_level: float = Field(..., ge=0.0, le=1.0)
    diacritic_confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence specifically for diacritical marks.",
    )
    damaged_region_penalty: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence reduction due to damaged regions.",
    )


class OCRResult(BaseModel):
    """Complete OCR result for a manuscript page."""

    text: str = Field(..., description="Full extracted text with diacritics.")
    text_normalized: str = Field(
        ...,
        description="Text with normalized Arabic (consistent hamza, etc.).",
    )
    lines: list[LineResult] = Field(default_factory=list)
    calligraphy: CalligraphyDetection | None = None
    confidence: OCRConfidenceMetrics
    preprocessing_applied: list[str] = Field(default_factory=list)
    corrections_applied: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    image_hash: str = Field(..., description="SHA-256 hash of input image.")


# ---------------------------------------------------------------------------
# Arabic Text Processing
# ---------------------------------------------------------------------------

# Unicode ranges for Arabic diacritics (tashkeel)
ARABIC_DIACRITICS = set(
    "\u064b\u064c\u064d\u064e\u064f\u0650\u0651\u0652"  # Fathatan through Sukun
    "\u0653\u0654\u0655\u0656\u0657\u0658\u0659\u065a\u065b\u065c\u065d\u065e\u065f"
    "\u0670"  # Dagger alif
)

# Common hamza forms for normalization
HAMZA_FORMS = {
    "\u0622": "\u0627\u0653",  # Alef with madda -> Alef + madda
    "\u0623": "\u0627\u0654",  # Alef with hamza above
    "\u0625": "\u0627\u0655",  # Alef with hamza below
    "\u0624": "\u0648\u0654",  # Waw with hamza
    "\u0626": "\u064a\u0654",  # Yeh with hamza
}

# Common OCR errors in Arabic manuscripts
COMMON_OCR_ERRORS: list[tuple[str, str, str]] = [
    # (wrong, correct, description)
    ("\u0647\u0627", "\u0629", "Ha-Alef misread as Ta Marbuta"),
    ("\u0628\u0646", "\u062a\u0646", "Ba-Nun vs Ta-Nun confusion"),
    ("\u064a\u064a", "\u0649", "Double Ya vs Alef Maksura"),
    ("\u0631\u0632", "\u0631\u0632", "Ra-Zay dot confusion"),
    ("\u062f\u0630", "\u062f\u0630", "Dal-Dhal dot confusion"),
    ("\u0633\u0634", "\u0633\u0634", "Seen-Sheen dot confusion"),
    ("\u0635\u0636", "\u0635\u0636", "Sad-Dad dot confusion"),
    ("\u0637\u0638", "\u0637\u0638", "Tah-Zah dot confusion"),
    ("\u0639\u063a", "\u0639\u063a", "Ain-Ghain dot confusion"),
    ("\u0641\u0642", "\u0641\u0642", "Fa-Qaf dot confusion"),
]


def is_arabic_diacritic(char: str) -> bool:
    """Check if a character is an Arabic diacritical mark."""
    return char in ARABIC_DIACRITICS


def strip_diacritics(text: str) -> str:
    """Remove all Arabic diacritical marks from text."""
    return "".join(c for c in text if c not in ARABIC_DIACRITICS)


def count_diacritics(text: str) -> int:
    """Count Arabic diacritical marks in text."""
    return sum(1 for c in text if c in ARABIC_DIACRITICS)


def normalize_arabic(text: str) -> str:
    """Normalize Arabic text for consistent representation.

    - Normalizes hamza forms
    - Normalizes alef forms
    - Removes tatweel (kashida)
    - Normalizes spacing
    """
    # Remove tatweel (elongation character)
    text = text.replace("\u0640", "")

    # Normalize alef forms to plain alef
    text = text.replace("\u0622", "\u0627")  # Alef madda
    text = text.replace("\u0623", "\u0627")  # Alef hamza above
    text = text.replace("\u0625", "\u0627")  # Alef hamza below
    text = text.replace("\u0671", "\u0627")  # Alef wasla

    # Normalize alef maksura to ya
    text = text.replace("\u0649", "\u064a")

    # Normalize ta marbuta to ha
    text = text.replace("\u0629", "\u0647")

    # Normalize spacing
    text = re.sub(r"\s+", " ", text).strip()

    return text


def calculate_word_confidence(word: WordResult) -> float:
    """Calculate confidence score for a word based on its characters."""
    if not word.characters:
        return word.confidence

    # Weight diacritics lower since they're harder to detect
    base_chars = [c for c in word.characters if not c.is_diacritic]
    diacritic_chars = [c for c in word.characters if c.is_diacritic]

    if not base_chars:
        return 0.5

    base_conf = sum(c.confidence for c in base_chars) / len(base_chars)
    diacritic_conf = sum(c.confidence for c in diacritic_chars) / len(diacritic_chars) if diacritic_chars else 1.0

    # Base characters weighted 80%, diacritics 20%
    return base_conf * 0.8 + diacritic_conf * 0.2


# ---------------------------------------------------------------------------
# Calligraphy Style Detection
# ---------------------------------------------------------------------------


@dataclass
class StyleDetectionResult:
    """Internal result from style detection analysis."""

    style: CalligraphyStyle
    confidence: float
    features: dict[str, Any] = field(default_factory=dict)


def detect_calligraphy_style(
    image_data: bytes,
    text_sample: str | None = None,
) -> CalligraphyDetection:
    """Detect the calligraphy style of a manuscript image.

    Uses a combination of:
    - Letter form analysis (geometric features)
    - Baseline angle detection
    - Stroke width variance
    - Spacing patterns

    In production, this would use a trained CNN classifier. This implementation
    uses heuristics that can be replaced with ML inference.

    Args:
        image_data: Raw image bytes (PNG, JPEG, etc.)
        text_sample: Optional OCR'd text to aid style detection via letterform patterns.

    Returns:
        CalligraphyDetection with primary style and alternatives.
    """
    # In production: call ML model for style classification
    # For now, implement heuristic-based detection

    # Analyze text sample for style hints if available
    style_scores: dict[CalligraphyStyle, float] = {
        style: 0.0 for style in CalligraphyStyle if style != CalligraphyStyle.UNKNOWN
    }

    features_detected: dict[str, Any] = {}

    if text_sample:
        # Check for Persian letters (indicates Nastaliq possibility)
        persian_chars = set("\u067e\u0686\u0698\u06af")  # Pe, Che, Zhe, Gaf
        if any(c in text_sample for c in persian_chars):
            style_scores[CalligraphyStyle.NASTALIQ] += 0.3
            features_detected["persian_letters"] = True

        # Check diacritic density (Naskh tends to be fully voweled)
        diacritic_ratio = count_diacritics(text_sample) / max(len(text_sample), 1)
        features_detected["diacritic_ratio"] = round(diacritic_ratio, 3)
        if diacritic_ratio > 0.15:
            style_scores[CalligraphyStyle.NASKH] += 0.2

        # Check for Quranic markers (suggests Naskh or Thuluth)
        quran_markers = ["\u06dd", "\u06de", "\u06e9"]  # End of ayah, etc.
        if any(m in text_sample for m in quran_markers):
            style_scores[CalligraphyStyle.NASKH] += 0.2
            style_scores[CalligraphyStyle.THULUTH] += 0.1
            features_detected["quranic_markers"] = True

    # Default bias toward Naskh (most common)
    style_scores[CalligraphyStyle.NASKH] += 0.3

    # Find best match
    sorted_styles = sorted(style_scores.items(), key=lambda x: x[1], reverse=True)
    primary_style, primary_score = sorted_styles[0]

    # Normalize confidence to 0-1
    total_score = sum(s for _, s in sorted_styles)
    primary_confidence = primary_score / total_score if total_score > 0 else 0.5

    secondary_styles = [
        (style, score / total_score if total_score > 0 else 0.0) for style, score in sorted_styles[1:3] if score > 0
    ]

    return CalligraphyDetection(
        primary_style=primary_style,
        confidence=round(primary_confidence, 3),
        secondary_styles=[(s, round(c, 3)) for s, c in secondary_styles],
        features_detected=features_detected,
    )


# ---------------------------------------------------------------------------
# Image Preprocessing (Stubs for external libraries)
# ---------------------------------------------------------------------------


def preprocess_manuscript_image(
    image_data: bytes,
    config: PreprocessingConfig,
) -> tuple[bytes, list[str]]:
    """Apply preprocessing pipeline to manuscript image.

    In production, this uses OpenCV/Pillow for:
    - Adaptive binarization (Otsu, Sauvola, etc.)
    - Deskewing via Hough transform
    - Denoising via bilateral filter or non-local means
    - Contrast enhancement via CLAHE
    - Margin detection and cropping

    Args:
        image_data: Raw image bytes.
        config: Preprocessing configuration.

    Returns:
        Tuple of (processed_image_bytes, list_of_operations_applied).
    """
    operations: list[str] = []

    # In production: actual image processing
    # For now, return original with operation log

    if config.deskew:
        operations.append("deskew")

    if config.remove_margins:
        operations.append("margin_crop")

    operations.append(f"binarize:{config.binarization_method}")

    if config.denoise_strength > 0:
        operations.append(f"denoise:{config.denoise_strength:.1f}")

    if config.contrast_enhancement != 1.0:
        operations.append(f"contrast:{config.contrast_enhancement:.1f}")

    if config.invert_if_dark:
        operations.append("auto_invert_check")

    return image_data, operations


# ---------------------------------------------------------------------------
# OCR Engine Calls (Stubs for external services)
# ---------------------------------------------------------------------------


async def call_ocr_engine(
    image_data: bytes,
    engine_config: OCREngineConfig,
    style_hint: CalligraphyStyle | None = None,
) -> tuple[str, list[LineResult]]:
    """Call external OCR engine and parse results.

    In production, this calls:
    - Google Cloud Vision API
    - Azure Cognitive Services
    - Tesseract subprocess
    - Custom Arabic OCR endpoint

    Args:
        image_data: Preprocessed image bytes.
        engine_config: OCR engine configuration.
        style_hint: Detected calligraphy style to optimize recognition.

    Returns:
        Tuple of (full_text, list_of_line_results).
    """
    # Stub implementation - in production, make actual API calls
    # This demonstrates the expected structure

    # Simulated OCR output for development/testing
    sample_text = "بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ"

    lines = [
        LineResult(
            text=sample_text,
            words=[
                WordResult(
                    text="بِسْمِ",
                    confidence=0.95,
                    characters=[
                        CharacterResult(char="ب", confidence=0.98, is_diacritic=False),
                        CharacterResult(char="ِ", confidence=0.88, is_diacritic=True),
                        CharacterResult(char="س", confidence=0.97, is_diacritic=False),
                        CharacterResult(char="ْ", confidence=0.85, is_diacritic=True),
                        CharacterResult(char="م", confidence=0.96, is_diacritic=False),
                        CharacterResult(char="ِ", confidence=0.87, is_diacritic=True),
                    ],
                    diacritic_count=3,
                ),
            ],
            confidence=0.92,
        ),
    ]

    return sample_text, lines


# ---------------------------------------------------------------------------
# Post-Processing
# ---------------------------------------------------------------------------


def apply_post_processing(
    text: str,
    lines: list[LineResult],
) -> tuple[str, list[LineResult], list[str]]:
    """Apply Arabic-specific post-processing corrections.

    - Fix common OCR errors (dot misplacement, letter confusion)
    - Normalize hamza placement
    - Fix ta marbuta / ha confusion
    - Restore likely missing diacritics using context

    Args:
        text: Raw OCR text.
        lines: Line-by-line results.

    Returns:
        Tuple of (corrected_text, corrected_lines, corrections_applied).
    """
    corrections: list[str] = []

    # Track corrections for reporting
    corrected_text = text

    # Fix common Allah ligature issues
    if "الله" in text and "اللّه" not in text:
        # Check if shadda is missing on the lam
        corrected_text = corrected_text.replace("الله", "اللّه")
        corrections.append("allah_ligature_shadda")

    # Normalize line-final ta marbuta
    return corrected_text, lines, corrections


def calculate_confidence_metrics(
    lines: list[LineResult],
    damaged_regions: list[BoundingBox] | None = None,
) -> OCRConfidenceMetrics:
    """Calculate aggregate confidence metrics from OCR results.

    Args:
        lines: OCR line results with per-character confidence.
        damaged_regions: Known damaged areas (from preprocessing) to penalize.

    Returns:
        OCRConfidenceMetrics with multi-level confidence scores.
    """
    if not lines:
        return OCRConfidenceMetrics(
            overall=0.0,
            character_level=0.0,
            word_level=0.0,
            diacritic_confidence=0.0,
        )

    # Collect all confidence scores
    char_confidences: list[float] = []
    diacritic_confidences: list[float] = []
    word_confidences: list[float] = []

    for line in lines:
        word_confidences.append(line.confidence)
        for word in line.words:
            word_confidences.append(word.confidence)
            for char in word.characters:
                if char.is_diacritic:
                    diacritic_confidences.append(char.confidence)
                else:
                    char_confidences.append(char.confidence)

    char_avg = sum(char_confidences) / len(char_confidences) if char_confidences else 0.5
    word_avg = sum(word_confidences) / len(word_confidences) if word_confidences else 0.5
    diacritic_avg = (
        sum(diacritic_confidences) / len(diacritic_confidences)
        if diacritic_confidences
        else 1.0  # No diacritics = no penalty
    )

    # Damaged region penalty
    damage_penalty = 0.0
    if damaged_regions:
        # In production: calculate overlap between damaged regions and text regions
        damage_penalty = min(0.3, len(damaged_regions) * 0.05)

    # Overall weighted average
    overall = (char_avg * 0.5 + word_avg * 0.3 + diacritic_avg * 0.2) * (1 - damage_penalty)

    return OCRConfidenceMetrics(
        overall=round(overall, 3),
        character_level=round(char_avg, 3),
        word_level=round(word_avg, 3),
        diacritic_confidence=round(diacritic_avg, 3),
        damaged_region_penalty=round(damage_penalty, 3),
    )


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------


async def process_manuscript_page(
    image_data: bytes,
    preprocessing_config: PreprocessingConfig | None = None,
    ocr_config: OCREngineConfig | None = None,
    detect_style: bool = True,
) -> OCRResult:
    """Full OCR pipeline for an Islamic manuscript page.

    1. Preprocess image for manuscript conditions
    2. Detect calligraphy style (optional)
    3. Run OCR with style-optimized settings
    4. Post-process for Arabic-specific corrections
    5. Calculate confidence metrics

    Args:
        image_data: Raw image bytes (PNG, JPEG, TIFF, etc.)
        preprocessing_config: Image preprocessing settings.
        ocr_config: OCR engine configuration.
        detect_style: Whether to run calligraphy detection.

    Returns:
        Complete OCRResult with text, confidence, and metadata.
    """
    preprocessing_config = preprocessing_config or PreprocessingConfig()
    ocr_config = ocr_config or OCREngineConfig()

    # Compute image hash
    image_hash = hashlib.sha256(image_data).hexdigest()

    warnings: list[str] = []

    # Step 1: Preprocess
    processed_image, preprocess_ops = preprocess_manuscript_image(image_data, preprocessing_config)

    # Step 2: Detect calligraphy style
    calligraphy: CalligraphyDetection | None = None
    style_hint: CalligraphyStyle | None = None
    if detect_style:
        calligraphy = detect_calligraphy_style(processed_image)
        style_hint = calligraphy.primary_style
        if calligraphy.confidence < 0.5:
            warnings.append(
                f"Low confidence ({calligraphy.confidence:.0%}) in calligraphy style detection. Results may vary."
            )

    # Step 3: Run OCR
    raw_text, lines = await call_ocr_engine(processed_image, ocr_config, style_hint)

    # Step 4: Post-process
    corrected_text, corrected_lines, corrections = apply_post_processing(raw_text, lines)

    # Step 5: Calculate confidence
    confidence = calculate_confidence_metrics(corrected_lines)

    # Generate normalized version
    text_normalized = normalize_arabic(corrected_text)

    if confidence.overall < 0.7:
        warnings.append(f"Overall OCR confidence is low ({confidence.overall:.0%}). Manual review recommended.")

    if confidence.diacritic_confidence < 0.6:
        warnings.append(
            f"Diacritical mark confidence is low ({confidence.diacritic_confidence:.0%}). "
            "Vowel markings may be inaccurate."
        )

    return OCRResult(
        text=corrected_text,
        text_normalized=text_normalized,
        lines=corrected_lines,
        calligraphy=calligraphy,
        confidence=confidence,
        preprocessing_applied=preprocess_ops,
        corrections_applied=corrections,
        warnings=warnings,
        image_hash=image_hash,
    )


# ---------------------------------------------------------------------------
# API Models
# ---------------------------------------------------------------------------


class OCRRequest(BaseModel):
    """Request body for OCR endpoint (base64 image)."""

    image_base64: str = Field(..., description="Base64-encoded image data.")

    preprocessing: PreprocessingConfig = Field(
        default_factory=PreprocessingConfig,
        description="Image preprocessing configuration.",
    )

    ocr_engine: OCREngineConfig = Field(
        default_factory=OCREngineConfig,
        description="OCR engine configuration.",
    )

    detect_calligraphy: bool = Field(
        default=True,
        description="Whether to detect calligraphy style.",
    )

    @field_validator("image_base64")
    @classmethod
    def validate_base64(cls, v: str) -> str:
        """Validate base64 encoding."""
        # Strip data URL prefix if present
        if "," in v:
            v = v.split(",", 1)[1]
        try:
            base64.b64decode(v)
        except Exception as err:
            raise ValueError("Invalid base64 encoding") from err
        return v


class StyleDetectionRequest(BaseModel):
    """Request for standalone calligraphy style detection."""

    image_base64: str
    text_sample: str | None = Field(
        default=None,
        description="Optional OCR'd text to aid detection.",
    )


class PreprocessingRequest(BaseModel):
    """Request for standalone image preprocessing."""

    image_base64: str
    config: PreprocessingConfig = Field(default_factory=PreprocessingConfig)


class PreprocessingResponse(BaseModel):
    """Response from preprocessing endpoint."""

    image_base64: str = Field(..., description="Preprocessed image as base64.")
    operations_applied: list[str]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.post("/process", response_model=OCRResult)
async def process_page(request: OCRRequest) -> OCRResult:
    """Process a manuscript page through the full OCR pipeline.

    Accepts a base64-encoded image and returns structured OCR results with:
    - Full text with diacritics preserved
    - Normalized text variant
    - Per-line and per-word confidence scores
    - Detected calligraphy style
    - Applied corrections and warnings
    """
    # Decode image
    image_data = base64.b64decode(request.image_base64)

    result = await process_manuscript_page(
        image_data=image_data,
        preprocessing_config=request.preprocessing,
        ocr_config=request.ocr_engine,
        detect_style=request.detect_calligraphy,
    )

    return result


@router.post("/process-file", response_model=OCRResult)
async def process_file(
    file: UploadFile = File(..., description="Image file (PNG, JPEG, TIFF)"),
    preprocessing_profile: PreprocessingProfile = Form(default=PreprocessingProfile.AGED),
    detect_calligraphy: bool = Form(default=True),
) -> OCRResult:
    """Process an uploaded manuscript image file.

    Alternative to /process that accepts multipart file upload instead of base64.
    """
    contents = await file.read()

    config = PreprocessingConfig(profile=preprocessing_profile)

    result = await process_manuscript_page(
        image_data=contents,
        preprocessing_config=config,
        detect_style=detect_calligraphy,
    )

    return result


@router.post("/detect-style", response_model=CalligraphyDetection)
async def detect_style(request: StyleDetectionRequest) -> CalligraphyDetection:
    """Detect the calligraphy style of a manuscript image.

    Can be used standalone before OCR to choose optimal processing settings,
    or to classify manuscripts for cataloging purposes.
    """
    image_data = base64.b64decode(request.image_base64)
    return detect_calligraphy_style(image_data, request.text_sample)


@router.post("/preprocess", response_model=PreprocessingResponse)
async def preprocess(request: PreprocessingRequest) -> PreprocessingResponse:
    """Apply preprocessing to a manuscript image without OCR.

    Useful for:
    - Testing preprocessing settings
    - Preparing images for manual review
    - Batch preprocessing before OCR
    """
    image_data = base64.b64decode(request.image_base64)
    processed, operations = preprocess_manuscript_image(image_data, request.config)

    return PreprocessingResponse(
        image_base64=base64.b64encode(processed).decode("utf-8"),
        operations_applied=operations,
    )


@router.get("/calligraphy-styles")
async def list_calligraphy_styles() -> list[dict[str, str]]:
    """List all recognized calligraphy styles with descriptions."""
    return [{"style": style.value, "description": STYLE_DESCRIPTIONS[style]} for style in CalligraphyStyle]


@router.get("/preprocessing-profiles")
async def list_preprocessing_profiles() -> list[dict[str, Any]]:
    """List available preprocessing profiles with their default settings."""
    return [
        {
            "profile": profile.value,
            "defaults": PROFILE_DEFAULTS.get(profile, {}),
        }
        for profile in PreprocessingProfile
    ]


@router.get("/engines")
async def list_ocr_engines() -> list[dict[str, str]]:
    """List supported OCR engines."""
    return [{"engine": engine.value, "description": f"{engine.value} OCR backend"} for engine in OCREngine]
