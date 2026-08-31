"""Hallucination detection and benchmarking for Islamic AI responses.

Detects fabricated citations, incorrect attributions, unsupported claims,
temporal confusion, and factual inconsistencies in AI-generated Islamic content.

Components:
- HallucinationTaxonomy: classification system for hallucination types
- SeverityScorer: rates hallucinations from minor inaccuracy to critical fabrication
- FabricatedCitationDetector: identifies made-up Quran/Hadith references
- MisattributionDetector: catches wrong scholar attributions
- FactualConsistencyChecker: cross-checks against authoritative sources
- TemporalAccuracyChecker: validates historical dates and chronology
- BenchmarkRunner: orchestrates adversarial tests and produces metrics
"""

from __future__ import annotations

import json
import logging
import re
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from corpus import corpus
from verifier import (
    VerificationStatus,
    extract_and_verify_all,
)

logger = logging.getLogger(__name__)

BENCHMARK_DATA_PATH = Path(__file__).parent / "data" / "hallucination_benchmark.json"


# ---------------------------------------------------------------------------
# Hallucination Taxonomy
# ---------------------------------------------------------------------------


class HallucinationType(str, Enum):
    """Classification of hallucination categories."""

    FABRICATED_CITATION = "fabricated_citation"
    MISATTRIBUTION = "misattribution"
    UNSUPPORTED_CLAIM = "unsupported_claim"
    FACTUAL_INCONSISTENCY = "factual_inconsistency"
    TEMPORAL_CONFUSION = "temporal_confusion"
    SCHOLAR_POSITION_ERROR = "scholar_position_error"
    FABRICATED_VERSE = "fabricated_verse"
    FABRICATED_HADITH = "fabricated_hadith"


class HallucinationSeverity(str, Enum):
    """Severity levels for detected hallucinations."""

    MINOR = "minor"  # Small inaccuracy, low impact
    MODERATE = "moderate"  # Notable error, could mislead
    MAJOR = "major"  # Significant error in religious content
    CRITICAL = "critical"  # Fabricated verse/hadith or dangerous claim

    @property
    def weight(self) -> float:
        return {
            HallucinationSeverity.MINOR: 0.25,
            HallucinationSeverity.MODERATE: 0.50,
            HallucinationSeverity.MAJOR: 0.75,
            HallucinationSeverity.CRITICAL: 1.0,
        }[self]


# ---------------------------------------------------------------------------
# Detection Models
# ---------------------------------------------------------------------------


class HallucinationFlag(BaseModel):
    """A single detected hallucination instance."""

    hallucination_type: HallucinationType
    severity: HallucinationSeverity
    description: str
    evidence: str
    source_text: str = ""
    expected_fact: str | None = None
    confidence: float = Field(0.0, ge=0.0, le=1.0)


class HallucinationScanResult(BaseModel):
    """Result of scanning a response for hallucinations."""

    flags: list[HallucinationFlag] = []
    hallucination_detected: bool = False
    hallucination_count: int = 0
    max_severity: HallucinationSeverity | None = None
    severity_score: float = Field(0.0, ge=0.0, le=1.0)
    category_breakdown: dict[str, int] = {}


class BenchmarkExample(BaseModel):
    """A single benchmark test case."""

    id: str
    category: HallucinationType
    severity: HallucinationSeverity
    prompt: str
    response: str
    expected_hallucinations: list[str] = []
    ground_truth: str = ""
    notes: str = ""


class BenchmarkResult(BaseModel):
    """Aggregate benchmark metrics."""

    total_examples: int = 0
    detection_rate: float = 0.0
    false_positive_rate: float = 0.0
    false_negative_rate: float = 0.0
    critical_detection_rate: float = 0.0
    fabricated_verse_detection_rate: float = 0.0
    misattribution_detection_rate: float = 0.0
    category_detection_rates: dict[str, float] = {}
    severity_distribution: dict[str, int] = {}
    average_severity_score: float = 0.0
    pass_criteria: dict[str, bool] = {}


# ---------------------------------------------------------------------------
# Known Scholar Positions (for misattribution detection)
# ---------------------------------------------------------------------------

KNOWN_SCHOLAR_POSITIONS: dict[str, dict[str, Any]] = {
    "imam_abu_hanifa": {
        "known_positions": [
            "wiping over leather socks is permissible",
            "intention for wudu is recommended not obligatory",
            "qunut in fajr is not practiced",
            "loud bismillah in prayer is not practiced",
        ],
        "era": "699-767 CE",
        "school": "Hanafi",
    },
    "imam_malik": {
        "known_positions": [
            "practice of people of madinah is authoritative",
            "raising hands only at opening takbir",
            "arms at sides during prayer",
        ],
        "era": "711-795 CE",
        "school": "Maliki",
    },
    "imam_shafi": {
        "known_positions": [
            "bismillah is part of al-fatiha",
            "qunut in fajr prayer is sunnah",
            "raising hands at ruku and rising from ruku",
        ],
        "era": "767-820 CE",
        "school": "Shafi'i",
    },
    "imam_ahmad": {
        "known_positions": [
            "strict adherence to hadith over qiyas",
            "created quran controversy stance",
            "perseverance during the mihna",
        ],
        "era": "780-855 CE",
        "school": "Hanbali",
    },
    "ibn_taymiyyah": {
        "known_positions": [
            "tawassul through dead is not permissible",
            "triple talaq counts as one",
            "traveled fatwa on visiting graves",
        ],
        "era": "1263-1328 CE",
        "school": "Hanbali",
    },
    "imam_ghazali": {
        "known_positions": [
            "tasawwuf is integral to islam",
            "ihya ulum al-din as comprehensive work",
            "critique of philosophers in tahafut",
        ],
        "era": "1058-1111 CE",
        "school": "Shafi'i",
    },
}

# Historical events with verified dates for temporal checking
HISTORICAL_EVENTS: dict[str, dict[str, Any]] = {
    "hijra": {"year_ce": 622, "description": "Migration from Mecca to Medina"},
    "badr": {"year_ce": 624, "year_ah": 2, "description": "Battle of Badr"},
    "uhud": {"year_ce": 625, "year_ah": 3, "description": "Battle of Uhud"},
    "khandaq": {"year_ce": 627, "year_ah": 5, "description": "Battle of the Trench"},
    "hudaybiyyah": {"year_ce": 628, "year_ah": 6, "description": "Treaty of Hudaybiyyah"},
    "conquest_mecca": {"year_ce": 630, "year_ah": 8, "description": "Conquest of Mecca"},
    "farewell_pilgrimage": {"year_ce": 632, "year_ah": 10, "description": "Farewell Pilgrimage"},
    "prophet_death": {"year_ce": 632, "year_ah": 11, "description": "Death of Prophet Muhammad ﷺ"},
    "abu_bakr_caliphate": {"year_ce": 632, "description": "Abu Bakr becomes Caliph"},
    "umar_caliphate": {"year_ce": 634, "description": "Umar ibn al-Khattab becomes Caliph"},
    "uthman_caliphate": {"year_ce": 644, "description": "Uthman ibn Affan becomes Caliph"},
    "ali_caliphate": {"year_ce": 656, "description": "Ali ibn Abi Talib becomes Caliph"},
    "karbala": {"year_ce": 680, "year_ah": 61, "description": "Battle of Karbala"},
}

# Known fabricated/weak hadith patterns
FABRICATED_HADITH_PATTERNS: list[dict[str, str]] = [
    {
        "pattern": r"seek\s+knowledge\s+even\s+(if\s+)?(in|unto)\s+china",
        "status": "fabricated",
        "note": "Often cited but classified as fabricated by hadith scholars",
    },
    {
        "pattern": r"love\s+of\s+(one'?s\s+)?country\s+is\s+(part\s+of|from)\s+faith",
        "status": "fabricated",
        "note": "No authentic chain; classified as fabricated by Al-Albani",
    },
    {
        "pattern": r"difference\s+of\s+(my\s+)?ummah\s+is\s+a\s+(mercy|rahma)",
        "status": "weak",
        "note": "Has no authentic chain according to major hadith scholars",
    },
    {
        "pattern": r"paradise\s+(lies\s+)?(is\s+)?under\s+the\s+feet\s+of\s+(your\s+)?mother",
        "status": "weak_chain",
        "note": "Meaning is supported by Quran but specific wording chain is weak",
    },
]

# Surah count for validating Quran references
TOTAL_SURAHS = 114


# ---------------------------------------------------------------------------
# Detection Functions
# ---------------------------------------------------------------------------


def detect_fabricated_citations(text: str) -> list[HallucinationFlag]:
    """Detect fabricated Quran and Hadith citations in text."""
    flags: list[HallucinationFlag] = []

    # Use the existing verifier to check citations
    verifications = extract_and_verify_all(text)
    for v in verifications:
        if v.get("status") == VerificationStatus.MISMATCH:
            source = v.get("source", "unknown")
            if source == "quran":
                surah = v.get("surah", 0)
                ayah = v.get("ayah", 0)
                # Check if it's a completely fabricated verse vs misquote
                if surah < 1 or surah > TOTAL_SURAHS:
                    sev = HallucinationSeverity.CRITICAL
                    htype = HallucinationType.FABRICATED_VERSE
                    desc = f"Fabricated Quran reference: Surah {surah} does not exist."
                else:
                    max_ayahs = corpus.get_ayah_count(surah)
                    if max_ayahs and ayah > max_ayahs:
                        sev = HallucinationSeverity.CRITICAL
                        htype = HallucinationType.FABRICATED_VERSE
                        desc = f"Fabricated verse: Surah {surah}:{ayah} does not exist (max {max_ayahs} ayahs)."
                    else:
                        sev = HallucinationSeverity.MAJOR
                        htype = HallucinationType.FABRICATED_CITATION
                        desc = f"Misquoted Quran {surah}:{ayah}: text does not match corpus."

                flags.append(
                    HallucinationFlag(
                        hallucination_type=htype,
                        severity=sev,
                        description=desc,
                        evidence=v.get("reason", ""),
                        expected_fact=v.get("correct_text", ""),
                        confidence=0.95,
                    )
                )
            elif source == "hadith":
                flags.append(
                    HallucinationFlag(
                        hallucination_type=HallucinationType.FABRICATED_CITATION,
                        severity=HallucinationSeverity.MAJOR,
                        description=f"Unverifiable hadith citation: {v.get('collection', '')} {v.get('number', '')}",
                        evidence=v.get("reason", ""),
                        confidence=0.7,
                    )
                )

    # Check for known fabricated hadith text patterns
    for entry in FABRICATED_HADITH_PATTERNS:
        if re.search(entry["pattern"], text, re.IGNORECASE):
            flags.append(
                HallucinationFlag(
                    hallucination_type=HallucinationType.FABRICATED_HADITH,
                    severity=HallucinationSeverity.MAJOR,
                    description=f"Known {entry['status']} hadith detected.",
                    evidence=entry["note"],
                    source_text=re.search(entry["pattern"], text, re.IGNORECASE).group(0),  # type: ignore[union-attr]
                    confidence=0.9,
                )
            )

    # Detect impossible Quran references in text (e.g. "Surah 115")
    impossible_refs = re.finditer(r"(?:surah|quran|qur'an)\s+(\d+)", text, re.IGNORECASE)
    for match in impossible_refs:
        surah_num = int(match.group(1))
        if surah_num > TOTAL_SURAHS or surah_num < 1:
            flags.append(
                HallucinationFlag(
                    hallucination_type=HallucinationType.FABRICATED_VERSE,
                    severity=HallucinationSeverity.CRITICAL,
                    description=f"Reference to non-existent Surah {surah_num} (Quran has {TOTAL_SURAHS} surahs).",
                    evidence=match.group(0),
                    confidence=1.0,
                )
            )

    return flags


def detect_misattributions(text: str) -> list[HallucinationFlag]:
    """Detect incorrect attributions of positions to Islamic scholars."""
    flags: list[HallucinationFlag] = []

    for scholar_key, info in KNOWN_SCHOLAR_POSITIONS.items():
        scholar_name = scholar_key.replace("_", " ").title()
        era_val = info.get("era", "")
        school_val = info.get("school", "")
        era = era_val if isinstance(era_val, str) else ""
        school = school_val if isinstance(school_val, str) else ""

        # Check for anachronistic attributions
        if era:
            era_match = re.search(re.escape(scholar_name), text, re.IGNORECASE)
            if era_match:
                # Check if there's a date near the scholar name that's wrong
                context_start = max(0, era_match.start() - 200)
                context_end = min(len(text), era_match.end() + 200)
                context = text[context_start:context_end]

                # Look for CE dates
                date_matches = re.finditer(r"(\d{3,4})\s*(?:CE|AD|AH)", context, re.IGNORECASE)
                for dm in date_matches:
                    year = int(dm.group(1))
                    if "CE" in dm.group(0).upper() or "AD" in dm.group(0).upper():
                        era_parts = era.split("-")
                        if len(era_parts) == 2:
                            birth = int(re.sub(r"\D", "", era_parts[0]))
                            death = int(re.sub(r"\D", "", era_parts[1]))
                            if year < birth - 50 or year > death + 50:
                                flags.append(
                                    HallucinationFlag(
                                        hallucination_type=HallucinationType.TEMPORAL_CONFUSION,
                                        severity=HallucinationSeverity.MODERATE,
                                        description=(
                                            f"Possible anachronism: {scholar_name} lived {era}, "
                                            f"but text references year {year} CE near their name."
                                        ),
                                        evidence=context.strip(),
                                        expected_fact=f"{scholar_name} lived {era}",
                                        confidence=0.7,
                                    )
                                )

        # Check for wrong school attributions
        if school:
            wrong_school_pattern = re.compile(
                rf"{re.escape(scholar_name)}.*?(?:founder|imam)\s+of\s+(?:the\s+)?(\w+)\s+(?:school|madhhab)",
                re.IGNORECASE | re.DOTALL,
            )
            wsm = wrong_school_pattern.search(text)
            if wsm:
                attributed_school = wsm.group(1).strip().lower()
                if attributed_school != school.lower() and not school.lower().startswith(attributed_school):
                    flags.append(
                        HallucinationFlag(
                            hallucination_type=HallucinationType.MISATTRIBUTION,
                            severity=HallucinationSeverity.MAJOR,
                            description=(
                                f"Wrong school attribution: {scholar_name} is associated with "
                                f"the {school} school, but text says '{attributed_school}'."
                            ),
                            evidence=wsm.group(0),
                            expected_fact=f"{scholar_name} is associated with the {school} school.",
                            confidence=0.9,
                        )
                    )

    return flags


def detect_unsupported_claims(text: str) -> list[HallucinationFlag]:
    """Detect claims presented as Islamic rulings without proper sourcing."""
    flags: list[HallucinationFlag] = []

    # Pattern: absolute claims about Islamic rulings without citations
    absolute_claim_patterns = [
        (
            r"islam\s+(categorically|absolutely|unequivocally)\s+(forbids?|prohibits?|allows?|permits?)",
            "Absolute categorical claim about Islamic ruling without nuance",
        ),
        (r"all\s+scholars?\s+(?:unanimously\s+)?(?:agree|consensus)\s+that", "Claim of unanimous scholarly consensus"),
        (
            r"the\s+quran\s+(clearly|explicitly)\s+(states?|says?|commands?)\s+that.*(?![\[\(]\d)",
            "Claim about Quran content without verse reference",
        ),
        (
            r"the\s+prophet\s+(?:ﷺ\s+)?said\s*[:\"](?!.*(?:bukhari|muslim|tirmidhi|abu\s+dawud|nasa'i|ibn\s+majah))",
            "Hadith quotation without collection reference",
        ),
    ]

    for pattern, description in absolute_claim_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            flags.append(
                HallucinationFlag(
                    hallucination_type=HallucinationType.UNSUPPORTED_CLAIM,
                    severity=HallucinationSeverity.MODERATE,
                    description=description,
                    evidence=match.group(0),
                    confidence=0.6,
                )
            )

    return flags


def detect_temporal_errors(text: str) -> list[HallucinationFlag]:
    """Detect incorrect dates and chronological errors for Islamic historical events."""
    flags: list[HallucinationFlag] = []

    for _event_key, event_info in HISTORICAL_EVENTS.items():
        event_desc = event_info["description"]
        expected_year = event_info["year_ce"]

        # Build patterns for the event
        event_terms = event_desc.lower().split()
        # Try matching event description near a year
        event_pattern = re.compile(
            rf"{'.*'.join(re.escape(t) for t in event_terms[:3])}.*?(\d{{3,4}})\s*(?:CE|AD|AH)?",
            re.IGNORECASE | re.DOTALL,
        )
        for match in event_pattern.finditer(text):
            mentioned_year = int(match.group(1))
            # Allow ±2 year tolerance for CE dates
            if "AH" in match.group(0).upper():
                expected_ah = event_info.get("year_ah")
                if expected_ah and abs(mentioned_year - expected_ah) > 2:
                    flags.append(
                        HallucinationFlag(
                            hallucination_type=HallucinationType.TEMPORAL_CONFUSION,
                            severity=HallucinationSeverity.MODERATE,
                            description=f"Incorrect date for {event_desc}: {mentioned_year} AH (expected ~{expected_ah} AH).",
                            evidence=match.group(0),
                            expected_fact=f"{event_desc} occurred in {expected_ah} AH ({expected_year} CE).",
                            confidence=0.85,
                        )
                    )
            elif abs(mentioned_year - expected_year) > 5:
                flags.append(
                    HallucinationFlag(
                        hallucination_type=HallucinationType.TEMPORAL_CONFUSION,
                        severity=HallucinationSeverity.MODERATE,
                        description=f"Incorrect date for {event_desc}: {mentioned_year} CE (expected ~{expected_year} CE).",
                        evidence=match.group(0),
                        expected_fact=f"{event_desc} occurred in {expected_year} CE.",
                        confidence=0.85,
                    )
                )

    return flags


def detect_scholar_position_errors(text: str) -> list[HallucinationFlag]:
    """Detect incorrect representation of known scholarly positions."""
    flags: list[HallucinationFlag] = []

    # Cross-school confusion patterns
    cross_school_errors = [
        {
            "pattern": r"hanafi\s+(?:school|madhhab|scholars?)\s+(?:holds?|says?|believes?|teaches?).*?qunut\s+in\s+fajr",
            "error": "Hanafi school does not practice qunut in fajr; this is a Shafi'i practice.",
            "severity": HallucinationSeverity.MAJOR,
        },
        {
            "pattern": r"shafi'?i\s+(?:school|madhhab)\s+(?:says?|holds?|believes?).*?arms?\s+at\s+(?:the\s+)?sides?",
            "error": "Placing arms at sides is a Maliki practice; Shafi'i school places right hand on left.",
            "severity": HallucinationSeverity.MAJOR,
        },
        {
            "pattern": r"ibn\s+taymiyyah\s+(?:supported|approved|endorsed|permitted).*?tawassul\s+through.*?dead",
            "error": "Ibn Taymiyyah explicitly opposed tawassul through deceased persons.",
            "severity": HallucinationSeverity.MAJOR,
        },
    ]

    for entry in cross_school_errors:
        if re.search(entry["pattern"], text, re.IGNORECASE):
            flags.append(
                HallucinationFlag(
                    hallucination_type=HallucinationType.SCHOLAR_POSITION_ERROR,
                    severity=entry["severity"],
                    description=entry["error"],
                    evidence=re.search(entry["pattern"], text, re.IGNORECASE).group(0),  # type: ignore[union-attr]
                    confidence=0.85,
                )
            )

    return flags


# ---------------------------------------------------------------------------
# Unified Scanner
# ---------------------------------------------------------------------------


def scan_for_hallucinations(text: str) -> HallucinationScanResult:
    """Run all hallucination detectors on the given text and return unified results."""
    all_flags: list[HallucinationFlag] = []

    all_flags.extend(detect_fabricated_citations(text))
    all_flags.extend(detect_misattributions(text))
    all_flags.extend(detect_unsupported_claims(text))
    all_flags.extend(detect_temporal_errors(text))
    all_flags.extend(detect_scholar_position_errors(text))

    # Deduplicate flags by description
    seen_descriptions: set[str] = set()
    unique_flags: list[HallucinationFlag] = []
    for flag in all_flags:
        if flag.description not in seen_descriptions:
            seen_descriptions.add(flag.description)
            unique_flags.append(flag)

    # Compute category breakdown
    category_breakdown: dict[str, int] = {}
    for flag in unique_flags:
        cat = flag.hallucination_type.value
        category_breakdown[cat] = category_breakdown.get(cat, 0) + 1

    # Compute severity score (weighted average)
    severity_score = 0.0
    if unique_flags:
        severity_score = sum(f.severity.weight * f.confidence for f in unique_flags) / len(unique_flags)

    # Find max severity
    severity_order = [
        HallucinationSeverity.CRITICAL,
        HallucinationSeverity.MAJOR,
        HallucinationSeverity.MODERATE,
        HallucinationSeverity.MINOR,
    ]
    max_severity = None
    for s in severity_order:
        if any(f.severity == s for f in unique_flags):
            max_severity = s
            break

    return HallucinationScanResult(
        flags=unique_flags,
        hallucination_detected=len(unique_flags) > 0,
        hallucination_count=len(unique_flags),
        max_severity=max_severity,
        severity_score=round(min(1.0, severity_score), 4),
        category_breakdown=category_breakdown,
    )


# ---------------------------------------------------------------------------
# Benchmark Runner
# ---------------------------------------------------------------------------


def load_benchmark_dataset(path: Path | None = None) -> list[BenchmarkExample]:
    """Load the adversarial benchmark dataset from JSON."""
    data_path = path or BENCHMARK_DATA_PATH
    if not data_path.exists():
        logger.warning("Benchmark dataset not found at %s", data_path)
        return []
    with open(data_path, encoding="utf-8") as f:
        raw = json.load(f)
    return [BenchmarkExample(**item) for item in raw.get("examples", [])]


def run_benchmark(examples: list[BenchmarkExample] | None = None) -> BenchmarkResult:
    """Run the hallucination detection benchmark and compute metrics."""
    if examples is None:
        examples = load_benchmark_dataset()

    if not examples:
        return BenchmarkResult()

    total = len(examples)
    true_positives = 0
    false_negatives = 0
    category_tp: dict[str, int] = {}
    category_total: dict[str, int] = {}
    severity_dist: dict[str, int] = {}
    severity_scores: list[float] = []

    critical_total = 0
    critical_detected = 0
    fabricated_verse_total = 0
    fabricated_verse_detected = 0
    misattribution_total = 0
    misattribution_detected = 0

    for example in examples:
        cat = example.category.value
        sev = example.severity.value
        category_total[cat] = category_total.get(cat, 0) + 1
        severity_dist[sev] = severity_dist.get(sev, 0) + 1

        if example.severity == HallucinationSeverity.CRITICAL:
            critical_total += 1
        if example.category == HallucinationType.FABRICATED_VERSE:
            fabricated_verse_total += 1
        if example.category == HallucinationType.MISATTRIBUTION:
            misattribution_total += 1

        result = scan_for_hallucinations(example.response)
        severity_scores.append(result.severity_score)

        if result.hallucination_detected:
            true_positives += 1
            category_tp[cat] = category_tp.get(cat, 0) + 1

            if example.severity == HallucinationSeverity.CRITICAL:
                critical_detected += 1
            if example.category == HallucinationType.FABRICATED_VERSE:
                fabricated_verse_detected += 1
            if example.category == HallucinationType.MISATTRIBUTION:
                misattribution_detected += 1
        else:
            false_negatives += 1

    detection_rate = true_positives / total if total else 0.0
    fn_rate = false_negatives / total if total else 0.0

    category_rates = {}
    for cat, count in category_total.items():
        tp = category_tp.get(cat, 0)
        category_rates[cat] = tp / count if count else 0.0

    crit_rate = critical_detected / critical_total if critical_total else 1.0
    fv_rate = fabricated_verse_detected / fabricated_verse_total if fabricated_verse_total else 1.0
    ma_rate = misattribution_detected / misattribution_total if misattribution_total else 1.0
    avg_severity = sum(severity_scores) / len(severity_scores) if severity_scores else 0.0

    # Success criteria checks
    pass_criteria = {
        "critical_detection_>95%": crit_rate > 0.95,
        "fabricated_verse_detection_100%": fv_rate == 1.0,
        "misattribution_detection_>99%": ma_rate > 0.99,
        "overall_detection_>85%": detection_rate > 0.85,
    }

    return BenchmarkResult(
        total_examples=total,
        detection_rate=round(detection_rate, 4),
        false_positive_rate=0.0,  # computed separately with clean examples
        false_negative_rate=round(fn_rate, 4),
        critical_detection_rate=round(crit_rate, 4),
        fabricated_verse_detection_rate=round(fv_rate, 4),
        misattribution_detection_rate=round(ma_rate, 4),
        category_detection_rates={k: round(v, 4) for k, v in category_rates.items()},
        severity_distribution=severity_dist,
        average_severity_score=round(avg_severity, 4),
        pass_criteria=pass_criteria,
    )
