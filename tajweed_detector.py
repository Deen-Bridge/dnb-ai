"""Tajweed error detection and educational feedback system.

A rule-based engine that analyses structured phonetic/feature input and
reports which Tajweed rules are violated, with severity scoring, per-rule
breakdowns, and educational feedback matched to a learner's difficulty level.

No audio processing is performed — the caller supplies a list of dictionaries
describing the letter sequence, diacritics, vowel markers, and pause markers
for each position in the recitation.

Knowledge encoding
------------------
Each rule is a ``TajweedRule`` dataclass whose ``check`` method receives a
``RecitationContext`` snapshot (letter, diacritics, neighbours, pause info) at
every position and returns a list of ``TajweedError`` values.  The rule
definitions, makharij letter classifications, and madd durations are loaded
from ``data/tajweed_rules.json``; the JSON is the single source of truth for
all classification tables while the ``check`` functions implement the
procedural logic that operates on those tables.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from fastapi import APIRouter
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/tajweed", tags=["tajweed"])

# ---------------------------------------------------------------------------
# Load knowledge base
# ---------------------------------------------------------------------------

_DATA_PATH = Path(__file__).resolve().parent / "data" / "tajweed_rules.json"


def _load_rules_data() -> dict[str, Any]:
    with open(_DATA_PATH, encoding="utf-8") as fh:
        return json.load(fh)


_RULES_DATA: dict[str, Any] = _load_rules_data()

# ---------------------------------------------------------------------------
# Internal data model
# ---------------------------------------------------------------------------

LETTER_KEY = "letter"
DIACRITICS_KEY = "diacritics"       # set of str, e.g. {"fatha", "sukun", "shadda"}
VOWEL_KEY = "vowel"                 # str: "fatha", "kasra", "damma", "sukun", "tanwin_*"
PAUSE_KEY = "pause"                 # bool — explicit pause / waqf marker
POSITION_KEY = "position"           # int — 0-based index in recitation


@dataclass
class RecitationContext:
    """Snapshot of the recitation at a particular position."""

    index: int
    letter: str
    diacritics: frozenset[str]
    vowel: str | None
    is_pause: bool
    prev_letter: str | None
    prev_vowel: str | None
    prev_diacritics: frozenset[str]
    next_letter: str | None
    next_vowel: str | None
    next_diacritics: frozenset[str]
    is_word_start: bool
    is_word_end: bool
    is_recitation_end: bool
    raw: dict[str, Any]


# ---------------------------------------------------------------------------
# Error and feedback types
# ---------------------------------------------------------------------------


@dataclass
class TajweedError:
    """A single detected violation."""

    rule_id: str
    rule_name: str
    position: int
    severity: float
    detail: str
    explanation: str
    exercise: str


@dataclass
class FeedbackItem:
    """Educational feedback for one or more related errors."""

    rule_id: str
    rule_name: str
    severity: float
    explanation: str
    exercises: list[str]
    occurrences: int


@dataclass
class RuleBreakdown:
    rule_id: str
    rule_name: str
    error_count: int
    total_weight: float


@dataclass
class TajweedAnalysis:
    """Full output of a recitation analysis."""

    errors: list[TajweedError]
    score: float
    total_positions: int
    breakdown: list[RuleBreakdown]
    level: str


# ---------------------------------------------------------------------------
# Rule definitions
# ---------------------------------------------------------------------------

AQ_LETTERS: frozenset[str] = frozenset(
    "ءأإؤئابتثجحخدذرزسشصضطظعغفقكلمنهويىآ"
    + "".join(chr(c) for c in range(0x0627, 0x064B))  # basic Arabic range
)

GHUNNAH_LETTERS: frozenset[str] = frozenset(_RULES_DATA["makharij"]["ghunnah_letters"])
IDGHAAM_NO_GH: frozenset[str] = frozenset(_RULES_DATA["makharij"]["idghaam_letters_no_ghunnah"])
IDGHAAM_WITH_GH: frozenset[str] = frozenset(_RULES_DATA["makharij"]["idghaam_letters_with_ghunnah"])
IKHFA_LETTERS: frozenset[str] = frozenset(_RULES_DATA["makharij"]["ikhfa_letters"])
QALQALAH_LETTERS: frozenset[str] = frozenset(_RULES_DATA["makharij"]["qalqalah_letters"])
LONG_VOWEL_LETTERS: frozenset[str] = frozenset("اويىآ")  # alif, waw, ya, alif-madda

MADD_TABII_DUR = _RULES_DATA["madd_durations"]["madd_al_tabi_i"]["duration_harakat"]
MADD_MUTASIL_DUR = _RULES_DATA["madd_durations"]["madd_al_mutasil"]["duration_harakat"]
MADD_MUNFASIL_DUR = _RULES_DATA["madd_durations"]["madd_al_munfasil"]["duration_harakat"]
MADD_AARID_DUR = _RULES_DATA["madd_durations"]["madd_al_aarid_lissukun"]["duration_harakat"]


def _build_context(seq: list[dict[str, Any]], idx: int) -> RecitationContext:
    """Build a RecitationContext for position *idx*."""
    pos = seq[idx]
    diacs = frozenset(pos.get(DIACRITICS_KEY, []))
    prev_pos = seq[idx - 1] if idx > 0 else None
    next_pos = seq[idx + 1] if idx < len(seq) - 1 else None

    # Detect word boundaries via a "word" key or assume contiguous
    is_word_start = pos.get("word_start", False) or idx == 0
    is_word_end = pos.get("word_end", False) or idx == len(seq) - 1

    return RecitationContext(
        index=idx,
        letter=pos.get(LETTER_KEY, ""),
        diacritics=diacs,
        vowel=pos.get(VOWEL_KEY),
        is_pause=pos.get(PAUSE_KEY, False),
        prev_letter=prev_pos.get(LETTER_KEY) if prev_pos else None,
        prev_vowel=prev_pos.get(VOWEL_KEY) if prev_pos else None,
        prev_diacritics=frozenset(prev_pos.get(DIACRITICS_KEY, [])) if prev_pos else frozenset(),
        next_letter=next_pos.get(LETTER_KEY) if next_pos else None,
        next_vowel=next_pos.get(VOWEL_KEY) if next_pos else None,
        next_diacritics=frozenset(next_pos.get(DIACRITICS_KEY, [])) if next_pos else frozenset(),
        is_word_start=is_word_start,
        is_word_end=is_word_end,
        is_recitation_end=idx == len(seq) - 1,
        raw=pos,
    )


# --- Individual rule check functions -------------------------------------


def _check_ghunnah(ctx: RecitationContext) -> list[TajweedError]:
    """Detect ghunnah violations: nun/mim with shadda or after sukun."""
    errors: list[TajweedError] = []
    letter = ctx.letter
    if letter not in GHUNNAH_LETTERS:
        return errors

    has_shadda = "shadda" in ctx.diacritics
    has_sukun = ctx.vowel == "sukun" or "sukun" in ctx.diacritics
    prev_has_sukun = ctx.prev_vowel == "sukun" or "sukun" in ctx.prev_diacritics

    requires_ghunnah = has_shadda or (has_sukun and prev_has_sukun)
    if requires_ghunnah:
        # Check: the user should have indicated ghunnah holding
        # We flag when there is no "ghunnah_held" marker in the raw data
        ghunnah_held = ctx.raw.get("ghunnah_held", False)
        if not ghunnah_held:
            rules = _RULES_DATA["rules"]["ghunnah"]
            errors.append(
                TajweedError(
                    rule_id="ghunnah",
                    rule_name=rules["name"],
                    position=ctx.index,
                    severity=rules["severity_weight"],
                    detail=f"Ghunnah required for '{letter}' but nasalization was not held",
                    explanation=rules["explanation_template"].format(
                        position=ctx.index,
                    ),
                    exercise=rules["exercises"][0],
                )
            )
    return errors


def _check_idghaam(ctx: RecitationContext) -> list[TajweedError]:
    """Detect idghaam violations: nun sakinah not merged with following letter."""
    errors: list[TajweedError] = []
    if ctx.letter != "ن" or ctx.vowel != "sukun":
        return errors
    if not ctx.next_letter:
        return errors
    next_let = ctx.next_letter

    is_no_gh_merge = next_let in IDGHAAM_NO_GH
    is_with_gh_merge = next_let in IDGHAAM_WITH_GH
    is_ikhfa = next_let in IKHFA_LETTERS

    if not (is_no_gh_merge or is_with_gh_merge or is_ikhfa):
        return errors

    merged = ctx.raw.get("merged", False)
    if not merged:
        rules = _RULES_DATA["rules"]["idghaam"]
        errors.append(
            TajweedError(
                rule_id="idghaam",
                rule_name=rules["name"],
                position=ctx.index,
                severity=rules["severity_weight"],
                detail=f"نْ before '{next_let}' should merge (idghaam)",
                explanation=rules["explanation_template"].format(
                    target_letter=next_let,
                    with_without="with" if is_with_gh_merge else "without",
                    position=ctx.index,
                ),
                exercise=rules["exercises"][0],
            )
        )
    return errors


def _check_ikhfa(ctx: RecitationContext) -> list[TajweedError]:
    """Detect ikhfa violations: nun sakinah before ikhfa letters not concealed."""
    errors: list[TajweedError] = []
    if ctx.letter != "ن" or ctx.vowel != "sukun":
        return errors
    if not ctx.next_letter:
        return errors
    if ctx.next_letter not in IKHFA_LETTERS:
        return errors

    concealed = ctx.raw.get("concealed", False)
    if not concealed:
        rules = _RULES_DATA["rules"]["ikhfa"]
        errors.append(
            TajweedError(
                rule_id="ikhfa",
                rule_name=rules["name"],
                position=ctx.index,
                severity=rules["severity_weight"],
                detail=f"نْ before '{ctx.next_letter}' should be concealed (ikhfa)",
                explanation=rules["explanation_template"].format(
                    target_letter=ctx.next_letter,
                    position=ctx.index,
                ),
                exercise=rules["exercises"][0],
            )
        )
    return errors


def _check_iqlab(ctx: RecitationContext) -> list[TajweedError]:
    """Detect iqlab violations: nun sakinah before ب not converted to mim."""
    errors: list[TajweedError] = []
    if ctx.letter != "ن" or ctx.vowel != "sukun":
        return errors
    if ctx.next_letter != "ب":
        return errors

    converted = ctx.raw.get("iqlab_applied", False)
    if not converted:
        rules = _RULES_DATA["rules"]["iqlab"]
        errors.append(
            TajweedError(
                rule_id="iqlab",
                rule_name=rules["name"],
                position=ctx.index,
                severity=rules["severity_weight"],
                detail="نْ before ب should be converted to م (iqlab)",
                explanation=rules["explanation_template"].format(position=ctx.index),
                exercise=rules["exercises"][0],
            )
        )
    return errors


def _check_qalqalah(ctx: RecitationContext) -> list[TajweedError]:
    """Detect qalqalah violations: b/j/d/t/q with sukun missing echo."""
    errors: list[TajweedError] = []
    if ctx.letter not in QALQALAH_LETTERS:
        return errors
    if ctx.vowel != "sukun" and "sukun" not in ctx.diacritics:
        return errors

    echo_produced = ctx.raw.get("qalqalah_echo", False)
    if not echo_produced:
        rules = _RULES_DATA["rules"]["qalqalah"]
        errors.append(
            TajweedError(
                rule_id="qalqalah",
                rule_name=rules["name"],
                position=ctx.index,
                severity=rules["severity_weight"],
                detail=f"Qalqalah letter '{ctx.letter}' with sukun requires echo",
                explanation=rules["explanation_template"].format(
                    letter=ctx.letter,
                    position=ctx.index,
                ),
                exercise=rules["exercises"][0],
            )
        )
    return errors


def _check_madd(ctx: RecitationContext) -> list[TajweedError]:
    """Detect elongation violations for long vowels (madd)."""
    errors: list[TajweedError] = []
    if ctx.letter not in LONG_VOWEL_LETTERS:
        return errors

    expected = ctx.raw.get("expected_madd_duration", None)
    actual = ctx.raw.get("actual_madd_duration", None)
    madd_type = ctx.raw.get("madd_type", None)

    if expected is None or actual is None:
        return errors

    if actual < expected:
        rule_data = _RULES_DATA["rules"].get(madd_type, _RULES_DATA["rules"]["madd_al_tabi_i"])
        errors.append(
            TajweedError(
                rule_id=madd_type or "madd_al_tabi_i",
                rule_name=rule_data["name"],
                position=ctx.index,
                severity=rule_data["severity_weight"],
                detail=f"Madd duration {actual} harakat < required {expected} harakat ({madd_type or 'unknown'})",
                explanation=rule_data["explanation_template"].format(
                    vowel=ctx.letter,
                    actual=actual,
                    position=ctx.index,
                ),
                exercise=rule_data["exercises"][0],
            )
        )
    return errors


def _check_waqf_ibtida(ctx: RecitationContext) -> list[TajweedError]:
    """Detect waqf/ibtida violations at pause points."""
    errors: list[TajweedError] = []
    if not ctx.is_pause:
        return errors

    waqf_correct = ctx.raw.get("waqf_correct", True)
    waqf_type = ctx.raw.get("waqf_type", "optional")
    waqf_detail = ctx.raw.get("waqf_detail", "")

    if not waqf_correct:
        rules = _RULES_DATA["rules"]["waqf_ibtida"]
        errors.append(
            TajweedError(
                rule_id="waqf_ibtida",
                rule_name=rules["name"],
                position=ctx.index,
                severity=rules["severity_weight"],
                detail=f"Waqf ({waqf_type}) rule violated: {waqf_detail}",
                explanation=rules["explanation_template"].format(
                    waqf_type=waqf_type,
                    detail=waqf_detail,
                    position=ctx.index,
                ),
                exercise=rules["exercises"][0],
            )
        )
    return errors


# ---------------------------------------------------------------------------
# All check functions
# ---------------------------------------------------------------------------

ALL_CHECKS: list[Callable[[RecitationContext], list[TajweedError]]] = [
    _check_ghunnah,
    _check_idghaam,
    _check_ikhfa,
    _check_iqlab,
    _check_qalqalah,
    _check_madd,
    _check_waqf_ibtida,
]

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def analyze_recitation(
    phonetic_sequence: list[dict[str, Any]],
    level: str = "advanced",
) -> TajweedAnalysis:
    """Analyse a structured phonetic sequence for Tajweed violations.

    Parameters
    ----------
    phonetic_sequence : list[dict]
        Each dict describes one position with keys: ``letter``, ``diacritics``
        (list[str]), ``vowel`` (str), ``pause`` (bool), plus optional feature
        flags (``merged``, ``concealed``, ``iqlab_applied``,
        ``qalqalah_echo``, ``ghunnah_held``, ``expected_madd_duration``,
        ``actual_madd_duration``, ``madd_type``, ``waqf_correct``,
        ``waqf_type``, ``waqf_detail``, ``word_start``, ``word_end``).
    level : str
        Difficulty level filtering: ``beginner``, ``intermediate``, or
        ``advanced``.  Only rules up to this level are checked.

    Returns
    -------
    TajweedAnalysis
    """
    active_rules: frozenset[str] = frozenset(_RULES_DATA["levels"].get(level, _RULES_DATA["levels"]["advanced"]))
    all_errors: list[TajweedError] = []

    for idx in range(len(phonetic_sequence)):
        ctx = _build_context(phonetic_sequence, idx)
        for check_fn in ALL_CHECKS:
            for err in check_fn(ctx):
                if err.rule_id in active_rules:
                    all_errors.append(err)

    score = _compute_score(all_errors, len(phonetic_sequence))
    breakdown = _build_breakdown(all_errors)

    return TajweedAnalysis(
        errors=all_errors,
        score=score,
        total_positions=len(phonetic_sequence),
        breakdown=breakdown,
        level=level,
    )


def _compute_score(errors: list[TajweedError], total_positions: int) -> float:
    """Compute a 0–1 score (1 = perfect)."""
    if total_positions == 0:
        return 1.0
    penalty = sum(e.severity for e in errors)
    penalty = min(penalty, 1.0)
    return round(1.0 - penalty, 4)


def _build_breakdown(errors: list[TajweedError]) -> list[RuleBreakdown]:
    """Aggregate errors per rule."""
    rule_map: dict[str, dict[str, Any]] = {}
    for err in errors:
        entry = rule_map.setdefault(
            err.rule_id,
            {"name": err.rule_name, "count": 0, "weight": 0.0},
        )
        entry["count"] += 1
        entry["weight"] += err.severity
    return [
        RuleBreakdown(
            rule_id=rid,
            rule_name=info["name"],
            error_count=info["count"],
            total_weight=round(info["weight"], 4),
        )
        for rid, info in sorted(rule_map.items(), key=lambda x: x[1]["weight"], reverse=True)
    ]


def generate_feedback(
    errors: list[TajweedError],
    level: str = "advanced",
) -> list[FeedbackItem]:
    """Generate educational feedback items from detected errors.

    Groups errors by rule, picks an explanation and up to 2 exercises per rule.
    Filters to the given difficulty level.
    """
    active_rules: frozenset[str] = frozenset(_RULES_DATA["levels"].get(level, _RULES_DATA["levels"]["advanced"]))
    grouped: dict[str, list[TajweedError]] = {}
    for err in errors:
        if err.rule_id in active_rules:
            grouped.setdefault(err.rule_id, []).append(err)

    items: list[FeedbackItem] = []
    for rule_id, errs in grouped.items():
        rule_data = _RULES_DATA["rules"].get(rule_id, {})
        exercises = rule_data.get("exercises", errs[0].exercise)
        if isinstance(exercises, str):
            exercises = [exercises]
        items.append(
            FeedbackItem(
                rule_id=rule_id,
                rule_name=errs[0].rule_name,
                severity=round(sum(e.severity for e in errs), 4),
                explanation=errs[0].explanation,
                exercises=exercises[:2],
                occurrences=len(errs),
            )
        )
    items.sort(key=lambda f: f.severity, reverse=True)
    return items


def get_all_rules() -> list[dict[str, Any]]:
    """Return the full list of rule definitions from the knowledge base."""
    rules = _RULES_DATA.get("rules", {})
    return [
        {
            "rule_id": rid,
            "name": info["name"],
            "description": info["description"],
            "severity_weight": info["severity_weight"],
            "level": info["level"],
            "exercises": info["exercises"],
        }
        for rid, info in rules.items()
    ]


# ---------------------------------------------------------------------------
# Pydantic request / response models
# ---------------------------------------------------------------------------


class PhoneticPosition(BaseModel):
    letter: str = Field(..., description="Arabic letter at this position")
    diacritics: list[str] = Field(default_factory=list, description="Active diacritics, e.g. ['fatha', 'sukun']")
    vowel: str | None = Field(default=None, description="Primary vowel: fatha, kasra, damma, sukun, tanwin_*")
    pause: bool = Field(default=False, description="Explicit waqf/pause marker")
    word_start: bool = Field(default=False, description="True if this position starts a new word")
    word_end: bool = Field(default=False, description="True if this position ends a word")
    ghunnah_held: bool = Field(default=False, description="True if ghunnah nasalization was held for 2+ harakat")
    merged: bool = Field(default=False, description="True if idghaam merge was performed")
    concealed: bool = Field(default=False, description="True if ikhfa concealment was performed")
    iqlab_applied: bool = Field(default=False, description="True if iqlab conversion was performed")
    qalqalah_echo: bool = Field(default=False, description="True if qalqalah echo was produced")
    expected_madd_duration: int | None = Field(default=None, description="Expected madd duration in harakat")
    actual_madd_duration: int | None = Field(default=None, description="Actual madd duration in harakat")
    madd_type: str | None = Field(default=None, description="Type of madd rule applicable")
    waqf_correct: bool = Field(default=True, description="True if waqf was applied correctly")
    waqf_type: str = Field(default="optional", description="Type of waqf: mandatory, optional, prohibited")
    waqf_detail: str = Field(default="", description="Details about waqf violation")


class AnalyzeRequest(BaseModel):
    phonetic_sequence: list[PhoneticPosition] = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Structured phonetic sequence of the recitation to analyse.",
    )
    level: str = Field(
        default="advanced",
        description="Difficulty level: beginner, intermediate, or advanced.",
    )


class TajweedErrorModel(BaseModel):
    rule_id: str
    rule_name: str
    position: int
    severity: float
    detail: str
    explanation: str
    exercise: str


class RuleBreakdownModel(BaseModel):
    rule_id: str
    rule_name: str
    error_count: int
    total_weight: float


class TajweedAnalysisModel(BaseModel):
    errors: list[TajweedErrorModel]
    score: float
    total_positions: int
    breakdown: list[RuleBreakdownModel]
    level: str


class FeedbackItemModel(BaseModel):
    rule_id: str
    rule_name: str
    severity: float
    explanation: str
    exercises: list[str]
    occurrences: int


class FeedbackRequest(BaseModel):
    errors: list[TajweedErrorModel] = Field(
        ...,
        description="Errors from a previous analysis to generate feedback for.",
    )
    level: str = Field(default="advanced", description="Difficulty level for feedback filtering.")


class RuleListItem(BaseModel):
    rule_id: str
    name: str
    description: str
    severity_weight: float
    level: str
    exercises: list[str]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.post("/analyze", response_model=TajweedAnalysisModel)
async def analyze_tajweed(request: AnalyzeRequest) -> TajweedAnalysisModel:
    """Analyse a phonetic sequence for Tajweed rule violations.

    Returns detected errors, a score (0–1), per-rule breakdown, and the
    difficulty level used for filtering.
    """
    seq = [pos.model_dump() for pos in request.phonetic_sequence]
    analysis = analyze_recitation(seq, level=request.level)
    return TajweedAnalysisModel(
        errors=[TajweedErrorModel(**vars(e)) for e in analysis.errors],
        score=analysis.score,
        total_positions=analysis.total_positions,
        breakdown=[RuleBreakdownModel(**vars(b)) for b in analysis.breakdown],
        level=analysis.level,
    )


@router.get("/rules", response_model=list[RuleListItem])
async def list_tajweed_rules() -> list[RuleListItem]:
    """Return all registered Tajweed rules with descriptions and exercises."""
    return [RuleListItem(**r) for r in get_all_rules()]


@router.post("/feedback", response_model=list[FeedbackItemModel])
async def tajweed_feedback(request: FeedbackRequest) -> list[FeedbackItemModel]:
    """Generate educational feedback items from a list of detected errors.

    Errors are grouped by rule and matched with explanations and corrective
    exercises appropriate for the specified difficulty level.
    """
    errors = [TajweedError(**vars(e)) for e in request.errors]
    items = generate_feedback(errors, level=request.level)
    return [FeedbackItemModel(**vars(item)) for item in items]
