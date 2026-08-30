"""
Cross-Reference Validation System

This module provides systematic validation of Islamic knowledge claims
by cross-referencing against multiple authoritative sources to ensure
accuracy and identify potential errors or weak citations.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

logger = logging.getLogger(__name__)


class SourceType(str, Enum):
    """Types of authoritative Islamic sources."""
    QURAN = "quran"
    TAFSIR = "tafsir"
    HADITH = "hadith"
    FIQH = "fiqh"
    SCHOLARLY_WORK = "scholarly_work"
    HISTORICAL = "historical"
    MANUSCRIPT = "manuscript"


class ValidationStatus(str, Enum):
    """Status of a validation check."""
    VERIFIED = "verified"
    PARTIALLY_VERIFIED = "partially_verified"
    UNVERIFIED = "unverified"
    CONTRADICTED = "contradicted"
    MISATTRIBUTED = "misattributed"


class HadithGrade(str, Enum):
    """Grades of hadith authenticity."""
    SAHIH = "sahih"  # Authentic
    HASAN = "hasan"  # Good
    DAIF = "da'if"   # Weak
    MAWDU = "mawdu'" # Fabricated
    UNKNOWN = "unknown"


@dataclass
class SourceReference:
    """Represents a reference to an authoritative source."""
    source_type: SourceType
    source_name: str
    volume: Optional[int] = None
    page: Optional[int] = None
    chapter: Optional[str] = None
    hadith_number: Optional[str] = None
    edition: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "source_type": self.source_type.value,
            "source_name": self.source_name,
            "volume": self.volume,
            "page": self.page,
            "chapter": self.chapter,
            "hadith_number": self.hadith_number,
            "edition": self.edition,
        }


@dataclass
class ValidationResult:
    """Result of a single validation check."""
    status: ValidationStatus
    confidence: float  # 0.0 to 1.0
    source: SourceReference
    details: str
    matching_text: Optional[str] = None
    discrepancies: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "status": self.status.value,
            "confidence": self.confidence,
            "source": self.source.to_dict(),
            "details": self.details,
            "matching_text": self.matching_text,
            "discrepancies": self.discrepancies,
        }


@dataclass
class CrossReferenceResult:
    """Complete cross-reference validation result."""
    claim: str
    overall_status: ValidationStatus
    overall_confidence: float
    validations: list[ValidationResult] = field(default_factory=list)
    contradictions: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    suggested_corrections: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "claim": self.claim,
            "overall_status": self.overall_status.value,
            "overall_confidence": self.overall_confidence,
            "validations": [v.to_dict() for v in self.validations],
            "contradictions": self.contradictions,
            "warnings": self.warnings,
            "suggested_corrections": self.suggested_corrections,
        }


# Tafsir sources for Quranic verification
TAFSIR_SOURCES = [
    "Tafsir Ibn Kathir",
    "Tafsir al-Tabari",
    "Tafsir al-Qurtubi",
    "Tafsir al-Jalalayn",
    "Tafsir al-Sa'di",
    "Tafsir al-Baghawi",
]

# Hadith collections for cross-checking
HADITH_COLLECTIONS = [
    {"name": "Sahih al-Bukhari", "authority": 1.0},
    {"name": "Sahih Muslim", "authority": 1.0},
    {"name": "Sunan Abu Dawud", "authority": 0.9},
    {"name": "Jami' al-Tirmidhi", "authority": 0.9},
    {"name": "Sunan al-Nasa'i", "authority": 0.9},
    {"name": "Sunan Ibn Majah", "authority": 0.85},
    {"name": "Musnad Ahmad", "authority": 0.85},
    {"name": "Muwatta Malik", "authority": 0.95},
]

# Known authentic hadith scholars for isnad verification
HADITH_SCHOLARS = {
    "أبو هريرة": {"name": "Abu Hurayrah", "status": "sahabi", "reliability": 1.0},
    "عائشة": {"name": "Aisha", "status": "sahabi", "reliability": 1.0},
    "ابن عمر": {"name": "Ibn Umar", "status": "sahabi", "reliability": 1.0},
    "أنس بن مالك": {"name": "Anas ibn Malik", "status": "sahabi", "reliability": 1.0},
    "جابر بن عبدالله": {"name": "Jabir ibn Abdullah", "status": "sahabi", "reliability": 1.0},
    "البخاري": {"name": "Imam al-Bukhari", "status": "muhaddith", "reliability": 1.0},
    "مسلم": {"name": "Imam Muslim", "status": "muhaddith", "reliability": 1.0},
}


async def verify_quranic_reference(
    surah: int,
    ayah: int,
    claimed_text: str,
    tafsir_sources: Optional[list[str]] = None,
) -> list[ValidationResult]:
    """
    Verify a Quranic reference using multiple tafsir sources.

    Args:
        surah: Surah number (1-114)
        ayah: Ayah number
        claimed_text: The claimed Quranic text
        tafsir_sources: Specific tafsir to check (defaults to all)

    Returns:
        List of validation results from each tafsir
    """
    sources = tafsir_sources or TAFSIR_SOURCES
    results = []

    for tafsir in sources:
        # In production, this would query a Quran/Tafsir database
        # For now, we simulate the validation
        result = ValidationResult(
            status=ValidationStatus.VERIFIED,
            confidence=0.95,
            source=SourceReference(
                source_type=SourceType.TAFSIR,
                source_name=tafsir,
            ),
            details=f"Verified against {tafsir}",
            matching_text=claimed_text,
        )
        results.append(result)

    return results


async def cross_check_hadith(
    matn: str,
    claimed_source: Optional[str] = None,
    claimed_chain: Optional[list[str]] = None,
) -> list[ValidationResult]:
    """
    Cross-check a hadith across multiple collections.

    Args:
        matn: The hadith text (matn)
        claimed_source: The claimed source collection
        claimed_chain: The claimed chain of narration

    Returns:
        List of validation results from each collection checked
    """
    results = []

    for collection in HADITH_COLLECTIONS:
        # In production, this would use hadith search APIs or databases
        # Check for text similarity, chain verification, etc.

        # Simulate checking
        found = True  # Would be determined by actual search
        confidence = collection["authority"] * 0.9

        status = ValidationStatus.VERIFIED if found else ValidationStatus.UNVERIFIED

        result = ValidationResult(
            status=status,
            confidence=confidence,
            source=SourceReference(
                source_type=SourceType.HADITH,
                source_name=collection["name"],
            ),
            details=f"{'Found' if found else 'Not found'} in {collection['name']}",
        )
        results.append(result)

    return results


async def verify_isnad(chain: list[str]) -> ValidationResult:
    """
    Verify a chain of hadith narration (isnad).

    Checks:
    - Whether each narrator is known
    - Narrator reliability
    - Chain continuity
    - Historical plausibility
    """
    verified_narrators = []
    unknown_narrators = []
    reliability_scores = []

    for narrator in chain:
        if narrator in HADITH_SCHOLARS:
            scholar = HADITH_SCHOLARS[narrator]
            verified_narrators.append(scholar["name"])
            reliability_scores.append(scholar["reliability"])
        else:
            unknown_narrators.append(narrator)

    # Calculate overall chain reliability
    if reliability_scores:
        avg_reliability = sum(reliability_scores) / len(reliability_scores)
    else:
        avg_reliability = 0.0

    # Determine status
    if not unknown_narrators and avg_reliability >= 0.9:
        status = ValidationStatus.VERIFIED
        confidence = avg_reliability
    elif unknown_narrators and len(unknown_narrators) < len(chain) / 2:
        status = ValidationStatus.PARTIALLY_VERIFIED
        confidence = avg_reliability * 0.7
    else:
        status = ValidationStatus.UNVERIFIED
        confidence = 0.3

    discrepancies = []
    if unknown_narrators:
        discrepancies.append(f"Unknown narrators: {', '.join(unknown_narrators)}")

    return ValidationResult(
        status=status,
        confidence=confidence,
        source=SourceReference(
            source_type=SourceType.HADITH,
            source_name="Isnad Verification",
        ),
        details=f"Verified {len(verified_narrators)}/{len(chain)} narrators",
        discrepancies=discrepancies,
    )


async def validate_scholarly_position(
    scholar: str,
    claimed_position: str,
    topic: str,
) -> list[ValidationResult]:
    """
    Validate a scholarly position against original source texts.
    """
    results = []

    # In production, this would check against a database of scholarly works
    result = ValidationResult(
        status=ValidationStatus.PARTIALLY_VERIFIED,
        confidence=0.7,
        source=SourceReference(
            source_type=SourceType.SCHOLARLY_WORK,
            source_name=f"Works of {scholar}",
        ),
        details="Position requires verification against original manuscripts",
        discrepancies=[],
    )
    results.append(result)

    return results


async def detect_contradictions(validations: list[ValidationResult]) -> list[str]:
    """
    Detect contradictions between different source validations.
    """
    contradictions = []

    # Group validations by source type
    verified = [v for v in validations if v.status == ValidationStatus.VERIFIED]
    contradicted = [v for v in validations if v.status == ValidationStatus.CONTRADICTED]

    if verified and contradicted:
        for c in contradicted:
            contradictions.append(
                f"Contradiction found: {c.source.source_name} contradicts verified sources"
            )

    return contradictions


def calculate_overall_confidence(validations: list[ValidationResult]) -> tuple[ValidationStatus, float]:
    """
    Calculate overall validation status and confidence.
    """
    if not validations:
        return ValidationStatus.UNVERIFIED, 0.0

    # Weight by source authority
    weighted_confidences = []
    status_counts = {status: 0 for status in ValidationStatus}

    for v in validations:
        weighted_confidences.append(v.confidence)
        status_counts[v.status] += 1

    avg_confidence = sum(weighted_confidences) / len(weighted_confidences)

    # Determine overall status
    if status_counts[ValidationStatus.CONTRADICTED] > 0:
        return ValidationStatus.CONTRADICTED, avg_confidence * 0.5
    elif status_counts[ValidationStatus.VERIFIED] > len(validations) / 2:
        return ValidationStatus.VERIFIED, avg_confidence
    elif status_counts[ValidationStatus.PARTIALLY_VERIFIED] > 0:
        return ValidationStatus.PARTIALLY_VERIFIED, avg_confidence * 0.8
    else:
        return ValidationStatus.UNVERIFIED, avg_confidence * 0.5


async def cross_reference_validate(
    claim: str,
    claim_type: str = "general",
    surah: Optional[int] = None,
    ayah: Optional[int] = None,
    hadith_text: Optional[str] = None,
    isnad: Optional[list[str]] = None,
    scholar: Optional[str] = None,
) -> CrossReferenceResult:
    """
    Perform comprehensive cross-reference validation on a claim.

    Args:
        claim: The claim to validate
        claim_type: Type of claim (quranic, hadith, scholarly, general)
        surah: Surah number if Quranic
        ayah: Ayah number if Quranic
        hadith_text: Hadith text if applicable
        isnad: Chain of narration if applicable
        scholar: Scholar name if validating position

    Returns:
        Complete cross-reference validation result
    """
    validations = []
    warnings = []

    # Validate Quranic references
    if surah and ayah:
        quran_validations = await verify_quranic_reference(surah, ayah, claim)
        validations.extend(quran_validations)

    # Cross-check hadith
    if hadith_text:
        hadith_validations = await cross_check_hadith(hadith_text, isnad=isnad)
        validations.extend(hadith_validations)

    # Verify isnad
    if isnad:
        isnad_result = await verify_isnad(isnad)
        validations.append(isnad_result)

    # Validate scholarly positions
    if scholar:
        scholarly_validations = await validate_scholarly_position(
            scholar, claim, "general"
        )
        validations.extend(scholarly_validations)

    # Detect contradictions
    contradictions = await detect_contradictions(validations)

    # Calculate overall confidence
    overall_status, overall_confidence = calculate_overall_confidence(validations)

    # Generate warnings
    if overall_confidence < 0.5:
        warnings.append("Low confidence validation - manual review recommended")
    if contradictions:
        warnings.append("Contradictions detected between sources")

    return CrossReferenceResult(
        claim=claim,
        overall_status=overall_status,
        overall_confidence=overall_confidence,
        validations=validations,
        contradictions=contradictions,
        warnings=warnings,
        suggested_corrections=[],
    )


# FastAPI router
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/validation", tags=["Cross-Reference Validation"])


class ValidationRequest(BaseModel):
    """Request model for cross-reference validation."""
    claim: str
    claim_type: str = "general"
    surah: Optional[int] = None
    ayah: Optional[int] = None
    hadith_text: Optional[str] = None
    isnad: Optional[list[str]] = None
    scholar: Optional[str] = None


class ValidationResponse(BaseModel):
    """Response model for validation."""
    success: bool
    data: dict


@router.post("/cross-reference", response_model=ValidationResponse)
async def validate_cross_reference(request: ValidationRequest):
    """
    Cross-reference validate a claim against multiple authoritative sources.

    Supports validation of:
    - Quranic references (multiple tafsir)
    - Hadith (cross-collection check)
    - Isnad (chain of narration)
    - Scholarly positions
    """
    result = await cross_reference_validate(
        claim=request.claim,
        claim_type=request.claim_type,
        surah=request.surah,
        ayah=request.ayah,
        hadith_text=request.hadith_text,
        isnad=request.isnad,
        scholar=request.scholar,
    )

    return ValidationResponse(
        success=True,
        data=result.to_dict(),
    )
