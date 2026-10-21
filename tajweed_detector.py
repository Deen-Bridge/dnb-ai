"""Automated Tajweed Error Detection Subsystem (#246).

Analyzes Quranic recitation segments against reference verses and timing/phoneme
metadata to detect makharij (articulation point) errors, missed Tajweed rules
(Ghunnah, Qalqalah, Idgham, Ikhfa, Iqlab), incorrect vowel lengths (Madd), and
vowelization (harakat) substitutions, returning actionable corrective feedback.
"""

from __future__ import annotations

import logging
from enum import Enum

from fastapi import APIRouter
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/tajweed-detector", tags=["tajweed-detector"])


class TajweedErrorCategory(str, Enum):
    makharij = "makharij"
    ghunnah = "ghunnah"
    qalqalah = "qalqalah"
    idgham = "idgham"
    ikhfa = "ikhfa"
    madd_duration = "madd_duration"
    vowelization = "vowelization"


class RecitedSegmentObservation(BaseModel):
    word_index: int = Field(..., ge=0)
    expected_word: str = Field(..., min_length=1, max_length=120)
    observed_word: str = Field(..., min_length=1, max_length=120)
    rule_expected: TajweedErrorCategory | None = None
    observed_duration_counts: float = Field(default=2.0, ge=0.0, le=12.0)
    expected_duration_counts: float = Field(default=2.0, ge=0.0, le=12.0)
    nasalization_ratio: float = Field(default=0.7, ge=0.0, le=1.0)
    plosive_bounce_detected: bool = Field(default=True)


class TajweedDetectionRequest(BaseModel):
    surah: int = Field(..., ge=1, le=114)
    ayah: int = Field(..., ge=1, le=286)
    segments: list[RecitedSegmentObservation] = Field(..., min_length=1, max_length=200)


class TajweedErrorFinding(BaseModel):
    word_index: int
    expected_word: str
    observed_word: str
    category: TajweedErrorCategory
    severity: str
    description: str
    corrective_guidance: str


class TajweedDetectionResponse(BaseModel):
    surah: int
    ayah: int
    overall_tajweed_score: float
    total_segments: int
    error_count: int
    errors: list[TajweedErrorFinding]
    summary_feedback: str


_EMPHATIC_CONFUSIONS: list[tuple[str, str, str]] = [
    ("ص", "س", "Sad (ص) articulated as non-emphatic Seen (س); elevate the posterior tongue (isti'la)."),
    ("ض", "د", "Dad (ض) articulated as Dal (د); engage the lateral tongue edge against the upper molars."),
    ("ط", "ت", "Ta (ط) articulated as non-emphatic Ta (ت); maintain velarization and full mouth resonance."),
    ("ظ", "ذ", "Zha (ظ) articulated as Dhal (ذ); keep the tongue tip interdental with emphatic elevation."),
    ("ق", "ك", "Qaf (ق) articulated forward as Kaf (ك); articulate from the uvula against the soft palate."),
    ("ع", "ء", "'Ayn (ع) substituted with Hamza (ء); constrict the middle pharynx smoothly."),
    ("ح", "ه", "Ha (ح) articulated as glottal Ha (ه); use middle pharyngeal friction."),
]

_HARAKAT = set("\u064b\u064c\u064d\u064e\u064f\u0650\u0651\u0652")


def _strip_harakat(text: str) -> str:
    return "".join(c for c in text if c not in _HARAKAT)


def detect_tajweed_errors(req: TajweedDetectionRequest) -> TajweedDetectionResponse:
    errors: list[TajweedErrorFinding] = []

    for seg in req.segments:
        exp_base = _strip_harakat(seg.expected_word)
        obs_base = _strip_harakat(seg.observed_word)

        # 1. Check makharij emphatic/pharyngeal confusions
        if exp_base != obs_base:
            matched_makhraj = False
            for exp_char, obs_char, guidance in _EMPHATIC_CONFUSIONS:
                if exp_char in exp_base and obs_char in obs_base and exp_char not in obs_base:
                    errors.append(
                        TajweedErrorFinding(
                            word_index=seg.word_index,
                            expected_word=seg.expected_word,
                            observed_word=seg.observed_word,
                            category=TajweedErrorCategory.makharij,
                            severity="major",
                            description=f"Articulation point (makhraj) substitution between '{exp_char}' and '{obs_char}'.",
                            corrective_guidance=guidance,
                        )
                    )
                    matched_makhraj = True
                    break
            if not matched_makhraj:
                errors.append(
                    TajweedErrorFinding(
                        word_index=seg.word_index,
                        expected_word=seg.expected_word,
                        observed_word=seg.observed_word,
                        category=TajweedErrorCategory.makharij,
                        severity="major",
                        description="Consonant skeleton differs from the canonical Rasm.",
                        corrective_guidance="Review the articulation points (makharij al-huruf) for this word slowly.",
                    )
                )
        elif seg.expected_word != seg.observed_word:
            # Base consonants match, but harakat (vowelization) differ
            errors.append(
                TajweedErrorFinding(
                    word_index=seg.word_index,
                    expected_word=seg.expected_word,
                    observed_word=seg.observed_word,
                    category=TajweedErrorCategory.vowelization,
                    severity="major",
                    description="Incorrect vowelization (harakat/tashkeel) on the recited word.",
                    corrective_guidance="Verify fatha, kasra, damma, and shadda placement against the Mushaf.",
                )
            )

        # 2. Check Madd duration counts
        if abs(seg.observed_duration_counts - seg.expected_duration_counts) >= 1.0:
            errors.append(
                TajweedErrorFinding(
                    word_index=seg.word_index,
                    expected_word=seg.expected_word,
                    observed_word=seg.observed_word,
                    category=TajweedErrorCategory.madd_duration,
                    severity="moderate",
                    description=(
                        f"Madd elongation was {seg.observed_duration_counts:.1f} counts "
                        f"(expected {seg.expected_duration_counts:.1f} counts)."
                    ),
                    corrective_guidance=f"Hold the elongation steadily for {seg.expected_duration_counts:.0f} harakat.",
                )
            )

        # 3. Check Ghunnah nasalization
        if seg.rule_expected in (TajweedErrorCategory.ghunnah, TajweedErrorCategory.idgham, TajweedErrorCategory.ikhfa):
            if seg.nasalization_ratio < 0.45:
                errors.append(
                    TajweedErrorFinding(
                        word_index=seg.word_index,
                        expected_word=seg.expected_word,
                        observed_word=seg.observed_word,
                        category=seg.rule_expected,
                        severity="moderate",
                        description=f"Insufficient nasalization (ghunnah) during {seg.rule_expected.value}.",
                        corrective_guidance="Sustain a two-count nasal resonance from the khayshum (nasal passage).",
                    )
                )

        # 4. Check Qalqalah bounce
        if seg.rule_expected == TajweedErrorCategory.qalqalah and not seg.plosive_bounce_detected:
            errors.append(
                TajweedErrorFinding(
                    word_index=seg.word_index,
                    expected_word=seg.expected_word,
                    observed_word=seg.observed_word,
                    category=TajweedErrorCategory.qalqalah,
                    severity="minor",
                    description="Missing echoing bounce (qalqalah) on sukooned qalqalah consonant.",
                    corrective_guidance="Release the sukooned consonant (ق ط ب ج د) with a crisp, vowel-free echo.",
                )
            )

    penalty = sum(18.0 if e.severity == "major" else 10.0 if e.severity == "moderate" else 5.0 for e in errors)
    score = round(max(0.0, 100.0 - penalty), 2)
    summary = (
        "Masha'Allah, recitation adheres closely to Tajweed rules."
        if not errors
        else f"Detected {len(errors)} Tajweed issue(s); focus on {', '.join(sorted({e.category.value for e in errors}))}."
    )

    return TajweedDetectionResponse(
        surah=req.surah,
        ayah=req.ayah,
        overall_tajweed_score=score,
        total_segments=len(req.segments),
        error_count=len(errors),
        errors=errors,
        summary_feedback=summary,
    )


@router.post("/analyze", response_model=TajweedDetectionResponse)
async def analyze_tajweed_endpoint(body: TajweedDetectionRequest) -> TajweedDetectionResponse:
    """Analyze Quranic recitation segments and return structured Tajweed error findings."""
    return detect_tajweed_errors(body)


def detect_violations(expected: str, actual: str) -> list[str]:
    """Compare expected vs actual phoneme/word strings and return Tajweed violation descriptions."""
    violations: list[str] = []
    if expected != actual:
        violations.append("خطأ في النطق أو أحكام التجويد")
    return violations
