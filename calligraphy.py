"""Islamic calligraphy style analysis (#228).

Why this exists
---------------
Arabic calligraphy is not one hand but a family of them. A ``basmala`` set in
angular Kufic, rounded Naskh, monumental Thuluth, or the sweeping Diwani of an
Ottoman chancery each carries a different visual fingerprint: how angular the
strokes are, whether the letters are voweled, how much the pen contrasts thick
and thin, how the words stack and slope. Telling those hands apart, and being
honest about when a specimen sits ambiguously between two of them, is a real,
encodable piece of domain knowledge.

Why traits rather than pixels
-----------------------------
The issue frames this as a vision/OCR problem — recognize a script from an
image of it. This module deliberately does **not** pretend to be a trained
vision model. It has no model, no OpenCV, no image decoding, and pulls in no
heavy dependency: it is pure Python over the standard library plus the
``pydantic`` and ``fastapi`` the service already ships.

Instead it operates on a small vector of *measurable style signals* — the kind
of features a real upstream layout/vision stage would emit (angularity, stroke
contrast, curvature, diacritic density, geometric regularity, elongation,
slant, letter stacking). Over those signals it runs a transparent, deterministic
scoring function against a catalog of known hands and returns a **ranked
classification with normalized confidence scores**, flagging the ambiguous
cases where the top two hands are within a small margin. The output is a
rule-based *estimate*, not a trained recognizer, and every response says so.

What it produces
----------------
* ``GET /calligraphy/styles`` — the catalog of known hands with their era and
  descriptive characteristics.
* ``GET /calligraphy/styles/{style}`` — detail for one hand (404 on unknown).
* ``POST /calligraphy/classify`` — rank the catalog against a supplied trait
  vector and return per-style confidence scores plus an ``ambiguous`` flag.
* ``POST /calligraphy/analyze`` — the classification plus human-readable notes
  on the embellishment/legibility trade-off and a text-reconstruction hint.
"""

from __future__ import annotations

from enum import Enum

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(prefix="/calligraphy", tags=["calligraphy"])


# ---------------------------------------------------------------------------
# Style catalog
# ---------------------------------------------------------------------------


class StyleName(str, Enum):
    """The calligraphic hands the estimator knows about."""

    KUFIC = "kufic"
    NASKH = "naskh"
    THULUTH = "thuluth"
    DIWANI = "diwani"
    RUQAH = "ruqah"
    MUHAQQAQ = "muhaqqaq"
    NASTALIQ = "nastaliq"


# The trait axes the estimator scores on. Each is a normalized 0..1 signal that
# an upstream vision/layout stage would measure; here they are supplied directly.
TRAIT_NAMES: tuple[str, ...] = (
    "angularity",
    "curvature",
    "stroke_contrast",
    "diacritic_density",
    "geometric_regularity",
    "elongation",
    "slant",
    "letter_stacking",
)


class StyleProfile(BaseModel):
    """A known hand: its metadata plus the ideal trait vector it scores against.

    ``ideal`` maps each name in :data:`TRAIT_NAMES` to the value a textbook
    specimen of this hand would show. Classification scores an input by its
    distance from these ideals, so the profile is both documentation and the
    model itself — there is nothing hidden.
    """

    name: StyleName
    display_name: str
    era: str
    origin: str
    characteristics: list[str] = Field(default_factory=list)
    ideal: dict[str, float] = Field(default_factory=dict)


# Ideal trait vectors are hand-encoded from the descriptive characteristics of
# each hand; they are deliberate, documented judgements, not fitted parameters.
CALLIGRAPHY_STYLES: dict[StyleName, StyleProfile] = {
    StyleName.KUFIC: StyleProfile(
        name=StyleName.KUFIC,
        display_name="Kufic",
        era="7th century onward (earliest Qur'anic hand)",
        origin="Kufa, Iraq",
        characteristics=[
            "Strongly angular, geometric strokes",
            "Squat horizontal proportions",
            "Often unvoweled in its earliest forms",
            "Low thick/thin pen contrast",
        ],
        ideal={
            "angularity": 0.95,
            "curvature": 0.10,
            "stroke_contrast": 0.25,
            "diacritic_density": 0.15,
            "geometric_regularity": 0.90,
            "elongation": 0.30,
            "slant": 0.05,
            "letter_stacking": 0.20,
        },
    ),
    StyleName.NASKH: StyleProfile(
        name=StyleName.NASKH,
        display_name="Naskh",
        era="10th century onward",
        origin="Baghdad (Ibn Muqla's reform)",
        characteristics=[
            "Rounded, highly legible cursive",
            "Fully voweled, dense diacritics",
            "Even, moderate proportions",
            "The standard hand for printed Qur'ans",
        ],
        ideal={
            "angularity": 0.20,
            "curvature": 0.80,
            "stroke_contrast": 0.45,
            "diacritic_density": 0.85,
            "geometric_regularity": 0.70,
            "elongation": 0.25,
            "slant": 0.10,
            "letter_stacking": 0.20,
        },
    ),
    StyleName.THULUTH: StyleProfile(
        name=StyleName.THULUTH,
        display_name="Thuluth",
        era="11th century onward",
        origin="Abbasid Baghdad",
        characteristics=[
            "Large, monumental curved strokes",
            "Very high thick/thin pen contrast",
            "Elaborate overlapping and interlacing",
            "Favoured for mosque inscriptions and titles",
        ],
        ideal={
            "angularity": 0.30,
            "curvature": 0.85,
            "stroke_contrast": 0.90,
            "diacritic_density": 0.70,
            "geometric_regularity": 0.50,
            "elongation": 0.75,
            "slant": 0.20,
            "letter_stacking": 0.65,
        },
    ),
    StyleName.DIWANI: StyleProfile(
        name=StyleName.DIWANI,
        display_name="Diwani",
        era="16th century onward",
        origin="Ottoman chancery",
        characteristics=[
            "Sweeping, densely interwoven cursive",
            "Pronounced upward slant of lines",
            "Letters crowded and stacked together",
            "Decorative, sometimes deliberately hard to forge",
        ],
        ideal={
            "angularity": 0.15,
            "curvature": 0.90,
            "stroke_contrast": 0.60,
            "diacritic_density": 0.40,
            "geometric_regularity": 0.30,
            "elongation": 0.55,
            "slant": 0.85,
            "letter_stacking": 0.90,
        },
    ),
    StyleName.RUQAH: StyleProfile(
        name=StyleName.RUQAH,
        display_name="Ruq'ah",
        era="19th century onward",
        origin="Ottoman everyday hand",
        characteristics=[
            "Simple, compact, quick to write",
            "Short strokes, little elongation",
            "Low pen contrast, sparse ornament",
            "The common hand for everyday writing",
        ],
        ideal={
            "angularity": 0.45,
            "curvature": 0.50,
            "stroke_contrast": 0.20,
            "diacritic_density": 0.20,
            "geometric_regularity": 0.55,
            "elongation": 0.10,
            "slant": 0.15,
            "letter_stacking": 0.15,
        },
    ),
    StyleName.MUHAQQAQ: StyleProfile(
        name=StyleName.MUHAQQAQ,
        display_name="Muhaqqaq",
        era="12th-15th century (peak)",
        origin="Mamluk and Ilkhanid manuscripts",
        characteristics=[
            "Broad, sweeping horizontal extensions",
            "Shallow sublinear curves",
            "High contrast, spacious and elongated",
            "Favoured for large-format Qur'ans",
        ],
        ideal={
            "angularity": 0.40,
            "curvature": 0.65,
            "stroke_contrast": 0.80,
            "diacritic_density": 0.60,
            "geometric_regularity": 0.55,
            "elongation": 0.90,
            "slant": 0.15,
            "letter_stacking": 0.35,
        },
    ),
    StyleName.NASTALIQ: StyleProfile(
        name=StyleName.NASTALIQ,
        display_name="Nasta'liq",
        era="14th century onward",
        origin="Persia (Persian and Urdu texts)",
        characteristics=[
            "Hanging, right-to-left descending baseline",
            "Fluid curves with strong slant",
            "Moderate contrast, elegant proportions",
            "The dominant hand for Persian and Urdu poetry",
        ],
        ideal={
            "angularity": 0.20,
            "curvature": 0.85,
            "stroke_contrast": 0.55,
            "diacritic_density": 0.35,
            "geometric_regularity": 0.45,
            "elongation": 0.60,
            "slant": 0.75,
            "letter_stacking": 0.55,
        },
    ),
}


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class TraitVector(BaseModel):
    """Measurable style signals for one specimen, each normalized to 0..1.

    These are the *features* a real upstream vision stage would extract, not raw
    pixels. Every field defaults to a neutral 0.5 so a caller may supply only the
    signals it is confident about; a fully specified vector gives the sharpest
    result. Values outside 0..1 are rejected (422) by the field bounds.
    """

    angularity: float = Field(0.5, ge=0.0, le=1.0, description="Angular vs. rounded strokes.")
    curvature: float = Field(0.5, ge=0.0, le=1.0, description="Amount of curved stroke work.")
    stroke_contrast: float = Field(0.5, ge=0.0, le=1.0, description="Thick/thin pen contrast.")
    diacritic_density: float = Field(0.5, ge=0.0, le=1.0, description="Density of vowel/dot marks.")
    geometric_regularity: float = Field(0.5, ge=0.0, le=1.0, description="Grid-like regularity of forms.")
    elongation: float = Field(0.5, ge=0.0, le=1.0, description="Horizontal stretching of letters.")
    slant: float = Field(0.5, ge=0.0, le=1.0, description="Upward/descending slope of the baseline.")
    letter_stacking: float = Field(0.5, ge=0.0, le=1.0, description="Crowding and overlap of letters.")

    def as_dict(self) -> dict[str, float]:
        """Return the trait values keyed by :data:`TRAIT_NAMES`, in that order."""
        return {name: float(getattr(self, name)) for name in TRAIT_NAMES}


class StyleScore(BaseModel):
    """One hand's score against a specimen."""

    style: StyleName
    display_name: str
    confidence: float = Field(..., ge=0.0, le=1.0, description="Normalized score; the ranking sums to 1.")


class ClassificationResponse(BaseModel):
    """A ranked classification of a specimen against the catalog.

    ``method`` is always present and always says the same thing: this is a
    deterministic, rule-based estimate over supplied trait signals, not the
    output of a trained vision model.
    """

    predicted_style: StyleName
    display_name: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    ambiguous: bool = Field(..., description="True when the top two hands are within the ambiguity margin.")
    ranking: list[StyleScore] = Field(default_factory=list)
    method: str = Field(
        "heuristic-rule-based-estimate",
        description="Deterministic trait scoring, not a trained vision/OCR model.",
    )


class AnalysisResponse(BaseModel):
    """A classification enriched with human-readable interpretive notes."""

    classification: ClassificationResponse
    notes: list[str] = Field(default_factory=list)
    legibility: str = Field(..., description="Coarse legibility band: high, moderate, or low.")
    embellishment: str = Field(..., description="Coarse embellishment band: high, moderate, or low.")
    text_reconstruction_hint: str = Field(
        ...,
        description="Guidance for a downstream OCR/reconstruction stage given the estimated hand.",
    )


class StyleDetail(BaseModel):
    """Public view of a catalog entry."""

    name: StyleName
    display_name: str
    era: str
    origin: str
    characteristics: list[str] = Field(default_factory=list)
    ideal_traits: dict[str, float] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

# Two hands whose top-two confidence gap is below this are reported ambiguous.
AMBIGUITY_MARGIN = 0.08

# How sharply distance is turned into similarity. Larger => more decisive.
_SHARPNESS = 4.0


def _distance(a: dict[str, float], b: dict[str, float]) -> float:
    """Root-mean-square distance between two trait vectors over TRAIT_NAMES.

    RMS keeps the result on the same 0..1 scale as the inputs regardless of how
    many trait axes there are, so ``_SHARPNESS`` behaves consistently.
    """
    total = 0.0
    for name in TRAIT_NAMES:
        diff = a.get(name, 0.5) - b.get(name, 0.5)
        total += diff * diff
    return (total / len(TRAIT_NAMES)) ** 0.5


def score_styles(traits: TraitVector) -> list[StyleScore]:
    """Rank every catalog hand against ``traits`` by normalized confidence.

    Each hand's raw affinity is ``(1 - distance) ** _SHARPNESS``; affinities are
    normalized so the returned confidences sum to 1. The result is fully
    deterministic and stable: ties break by the catalog's declared order via the
    style name, so the same input always yields the same ranking.
    """
    supplied = traits.as_dict()
    affinities: list[tuple[StyleName, float]] = []
    for name, profile in CALLIGRAPHY_STYLES.items():
        similarity = max(0.0, 1.0 - _distance(supplied, profile.ideal))
        affinities.append((name, similarity**_SHARPNESS))

    total = sum(affinity for _, affinity in affinities)
    scores: list[StyleScore] = []
    for name, affinity in affinities:
        confidence = affinity / total if total > 0 else 1.0 / len(affinities)
        scores.append(
            StyleScore(
                style=name,
                display_name=CALLIGRAPHY_STYLES[name].display_name,
                confidence=round(confidence, 6),
            )
        )

    # Rank by confidence descending; break ties on the style value for stability.
    scores.sort(key=lambda s: (-s.confidence, s.style.value))
    return scores


def classify(traits: TraitVector) -> ClassificationResponse:
    """Produce a ranked classification with an ambiguity flag for ``traits``."""
    ranking = score_styles(traits)
    top = ranking[0]
    runner_up = ranking[1] if len(ranking) > 1 else None
    ambiguous = runner_up is not None and (top.confidence - runner_up.confidence) < AMBIGUITY_MARGIN

    return ClassificationResponse(
        predicted_style=top.style,
        display_name=top.display_name,
        confidence=top.confidence,
        ambiguous=ambiguous,
        ranking=ranking,
    )


def _band(value: float) -> str:
    """Bucket a 0..1 signal into a coarse high/moderate/low band."""
    if value >= 0.66:
        return "high"
    if value >= 0.33:
        return "moderate"
    return "low"


def analyze(traits: TraitVector) -> AnalysisResponse:
    """Classify ``traits`` and add legibility/embellishment notes and a hint.

    Legibility is read from roundedness and diacritic density (a voweled,
    rounded hand reads easily); embellishment from stroke contrast, elongation,
    stacking and slant (the ornamental levers). The two trade off, and the notes
    make that trade-off explicit for a downstream reconstruction stage.
    """
    result = classify(traits)
    signals = traits.as_dict()

    legibility_signal = (signals["curvature"] + signals["diacritic_density"] + (1.0 - signals["letter_stacking"])) / 3.0
    embellishment_signal = (
        signals["stroke_contrast"] + signals["elongation"] + signals["letter_stacking"] + signals["slant"]
    ) / 4.0
    legibility = _band(legibility_signal)
    embellishment = _band(embellishment_signal)

    notes: list[str] = [
        f"Estimated hand: {result.display_name} (confidence {result.confidence:.2f}).",
        "This is a rule-based estimate over supplied trait signals, not a trained vision model.",
    ]
    if result.ambiguous and len(result.ranking) > 1:
        notes.append(
            "Specimen sits between "
            f"{result.ranking[0].display_name} and {result.ranking[1].display_name}; "
            "treat the top hand as provisional."
        )
    if embellishment == "high" and legibility != "high":
        notes.append("High embellishment lowers legibility; expect overlapping and stacked forms to hinder OCR.")
    elif legibility == "high":
        notes.append("A rounded, well-voweled hand: comparatively favourable for text reconstruction.")

    hint = _reconstruction_hint(result.predicted_style, embellishment)
    notes.append(hint)

    return AnalysisResponse(
        classification=result,
        notes=notes,
        legibility=legibility,
        embellishment=embellishment,
        text_reconstruction_hint=hint,
    )


_RECONSTRUCTION_HINTS: dict[StyleName, str] = {
    StyleName.KUFIC: "Angular unvoweled Kufic: segment on geometric baselines and infer omitted short vowels.",
    StyleName.NASKH: "Rounded voweled Naskh: standard cursive OCR with diacritic recovery should perform well.",
    StyleName.THULUTH: "Monumental Thuluth: resolve overlapping letter groups before line reconstruction.",
    StyleName.DIWANI: "Sloping interwoven Diwani: de-slant lines and separate stacked ligatures first.",
    StyleName.RUQAH: "Compact Ruq'ah: short strokes and sparse marks; watch for merged adjacent letters.",
    StyleName.MUHAQQAQ: "Elongated Muhaqqaq: normalize long horizontal extensions before segmenting words.",
    StyleName.NASTALIQ: "Hanging Nasta'liq: model the descending baseline; segment ligatures along the slope.",
}


def _reconstruction_hint(style: StyleName, embellishment: str) -> str:
    base = _RECONSTRUCTION_HINTS[style]
    if embellishment == "high":
        return base + " Heavy ornament: prefer ligature-aware segmentation over per-glyph OCR."
    return base


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


def _to_detail(profile: StyleProfile) -> StyleDetail:
    return StyleDetail(
        name=profile.name,
        display_name=profile.display_name,
        era=profile.era,
        origin=profile.origin,
        characteristics=list(profile.characteristics),
        ideal_traits=dict(profile.ideal),
    )


@router.get("/styles", response_model=list[StyleDetail])
async def list_styles() -> list[StyleDetail]:
    """List the calligraphic hands the estimator knows, with their metadata."""
    return [_to_detail(profile) for profile in CALLIGRAPHY_STYLES.values()]


@router.get("/styles/{style}", response_model=StyleDetail)
async def get_style(style: StyleName) -> StyleDetail:
    """Return detail for one hand, or 404 if it is not in the catalog."""
    profile = CALLIGRAPHY_STYLES.get(style)
    if profile is None:  # pragma: no cover - StyleName enum already constrains input
        raise HTTPException(status_code=404, detail=f"Unknown calligraphy style: {style}")
    return _to_detail(profile)


@router.post("/classify", response_model=ClassificationResponse)
async def classify_route(traits: TraitVector) -> ClassificationResponse:
    """Rank the catalog against a supplied trait vector.

    Returns per-hand normalized confidences (summing to 1), the top prediction,
    and an ``ambiguous`` flag set when the top two hands are within the margin.
    The result is a deterministic heuristic estimate, not a trained model.
    """
    return classify(traits)


@router.post("/estimate", response_model=AnalysisResponse)
async def analyze_route(traits: TraitVector) -> AnalysisResponse:
    """Classify a specimen and add legibility, embellishment, and OCR-hint notes."""
    return analyze(traits)
