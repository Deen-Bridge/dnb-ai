"""Data models and enumerations for cross-session factual consistency.

Tracks factual claims, ruling classifications, scholarly attributions,
contradiction severity, and reconciliation strategies.
"""

from __future__ import annotations

import time
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ClaimType(str, Enum):
    """Category of factual assertion."""

    RULING = "ruling"  # Fiqh legal judgment (halal, haram, etc.)
    ATTRIBUTION = "attribution"  # Scholar/madhhab attribution
    INTERPRETATION = "interpretation"  # Exegesis / theological explanation
    EVIDENCE = "evidence"  # Proof citation (Quran/Hadith)
    NUMERICAL = "numerical"  # Numbers/quantities (rates, counts, thresholds)
    CORE_PRINCIPLE = "core_principle"  # Fundamental Aqeedah / Ijma assertion
    FACTUAL = "factual"  # General Islamic factual claim


class RulingType(str, Enum):
    """Fiqh status / normative ruling."""

    FARD = "fard"  # Obligatory (definitive)
    WAJIB = "wajib"  # Obligatory (probabilistic)
    MUSTAHABB = "mustahabb"  # Recommended / Sunnah
    MUBAH = "mubah"  # Permissible / Neutral
    MAKRUH = "makruh"  # Disliked
    HARAM = "haram"  # Prohibited / Forbidden
    HALAL = "halal"  # Permissible (broad)
    VALID = "valid"  # Sahih (contract / act)
    INVALID = "invalid"  # Batil / Fasid (nullified)
    NULLIFIED = "nullified"  # Breaks wudu / fasting / etc.
    NOT_NULLIFIED = "not_nullified"  # Does not break wudu / fasting
    CONDITIONAL = "conditional"  # Depends on conditions


class ContradictionSeverity(str, Enum):
    """Impact severity of detected inconsistency."""

    CRITICAL = "critical"  # Direct clash on core aqeedah or clear Ijma
    HIGH = "high"  # 180-degree ruling contradiction without explanation
    MEDIUM = "medium"  # Unexplained attribution swap or evidence mismatch
    LOW = "low"  # Minor stylistic or wording drift
    NONE = "none"  # Fully consistent or valid legitimate variation


class ContradictionCategory(str, Enum):
    """Nature of the consistency violation."""

    RULING_CONTRADICTION = "ruling_contradiction"
    ATTRIBUTION_MISMATCH = "attribution_mismatch"
    EVIDENCE_CONFLICT = "evidence_conflict"
    INTERPRETATION_DRIFT = "interpretation_drift"
    CORE_POSITION_DIVERGENCE = "core_position_divergence"
    NUMERICAL_DISCREPANCY = "numerical_discrepancy"


class ConsistencyAction(str, Enum):
    """Enforcement policy action."""

    ALLOW = "allow"  # Pass through unchanged
    RECONCILE = "reconcile"  # Append/inject scholarly reconciliation explanation
    WARN = "warn"  # Append advisory warning note
    BLOCK = "block"  # Reject delivery / trigger corrective fallback


class ConsensusLevel(str, Enum):
    """Degree of classical scholarly consensus."""

    IJMA = "ijma"  # Unanimous consensus
    JUMHUR = "jumhur"  # Majority opinion
    IKHTILAF = "ikhtilaf"  # Recognized difference of opinion among schools


class FactualClaim(BaseModel):
    """An atomic factual assertion extracted from a response turn."""

    claim_id: str
    text: str
    normalized_text: str = ""
    topic: str
    entity: str | None = None  # Key subject (e.g. "wudu_touching_spouse", "zakat_rate")
    claim_type: ClaimType = ClaimType.FACTUAL
    ruling: RulingType | None = None
    polarity: bool = True  # True = affirmative/permissible, False = negative/impermissible
    condition: str | None = None  # e.g. "traveler", "sick", "forgetfulness"
    attribution: str | None = None  # e.g. "Imam Abu Hanifa", "Hanafi school"
    madhhab: str | None = None  # hanafi, maliki, shafii, hanbali
    citations: list[str] = Field(default_factory=list)  # e.g. ["Quran 2:255", "Bukhari:1"]
    session_id: str | None = None
    user_id: str | None = None
    chat_id: str | None = None
    timestamp: float = Field(default_factory=time.time)
    confidence: float = 1.0

    model_config = {"extra": "forbid"}


class Contradiction(BaseModel):
    """Structured record of an identified contradiction or divergence."""

    historical_claim: FactualClaim | None = None
    candidate_claim: FactualClaim
    severity: ContradictionSeverity
    category: ContradictionCategory
    description: str
    is_legitimate_variation: bool = False
    legitimate_reason: str | None = None  # e.g. "madhhab_difference", "conditional_context"
    reconciliation_text: str | None = None
    confidence: float = 1.0

    model_config = {"extra": "forbid"}


class ConsistencyCheckResult(BaseModel):
    """Full outcome of a consistency evaluation against historical sessions."""

    is_consistent: bool
    contradiction_found: bool
    highest_severity: ContradictionSeverity = ContradictionSeverity.NONE
    contradictions: list[Contradiction] = Field(default_factory=list)
    action: ConsistencyAction = ConsistencyAction.ALLOW
    original_text: str
    final_text: str
    reconciliation_notes: list[str] = Field(default_factory=list)
    claims_evaluated: int = 0
    historical_claims_matched: int = 0
    latency_ms: float = 0.0

    model_config = {"extra": "forbid"}


class CorePosition(BaseModel):
    """Canonical ground-truth position on an invariant or agreed-upon topic."""

    topic: str
    title: str
    consensus_level: ConsensusLevel
    orthodox_position: str
    valid_perspectives: list[str] = Field(default_factory=list)  # Valid school positions
    prohibited_contradictions: list[str] = Field(default_factory=list)  # Heretical / false claims
    citations: list[str] = Field(default_factory=list)

    model_config = {"extra": "forbid"}


class AttributionRecord(BaseModel):
    """Standardized scholar / madhhab position mapping."""

    scholar: str
    madhhab: str | None = None
    topic: str
    position_summary: str
    primary_source: str | None = None

    model_config = {"extra": "forbid"}


class ConsistencyAlert(BaseModel):
    """System alert generated on significant consistency violations."""

    alert_id: str
    severity: ContradictionSeverity
    category: ContradictionCategory
    chat_id: str | None = None
    user_id: str | None = None
    summary: str
    details: dict[str, Any] = Field(default_factory=dict)
    timestamp: float = Field(default_factory=time.time)

    model_config = {"extra": "forbid"}


class CoherenceMetrics(BaseModel):
    """Aggregate statistics for knowledge base consistency and temporal drift."""

    total_claims_indexed: int = 0
    total_checks_performed: int = 0
    total_contradictions_detected: int = 0
    legitimate_variations_reconciled: int = 0
    violations_blocked: int = 0
    consistency_rate: float = 1.0
    attribution_reliability_rate: float = 1.0
    cross_session_drift_score: float = 0.0  # 0.0 = zero drift, 1.0 = heavy drift
    active_alerts_count: int = 0

    model_config = {"extra": "forbid"}
