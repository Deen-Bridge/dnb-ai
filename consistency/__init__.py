"""Factual consistency subsystem for cross-session coherence.

Detects and resolves contradictions, preserves attribution fidelity,
differentiates legitimate ikhtilaf from errors, and monitors knowledge coherence.
"""

from __future__ import annotations

from consistency.alerts import ConsistencyAlertManager, get_alert_manager
from consistency.core_anchors import (
    CORE_POSITIONS,
    IKHTILAF_MAP,
    get_core_position,
    get_ikhtilaf_entry,
    is_core_principle_violation,
)
from consistency.detector import detect_contradictions
from consistency.enforcer import ConsistencyEnforcer, get_consistency_enforcer
from consistency.extractor import extract_claims, normalize_text_for_claim
from consistency.models import (
    ClaimType,
    CoherenceMetrics,
    ConsensusLevel,
    ConsistencyAction,
    ConsistencyAlert,
    ConsistencyCheckResult,
    Contradiction,
    ContradictionCategory,
    ContradictionSeverity,
    CorePosition,
    FactualClaim,
    RulingType,
)
from consistency.reconciliation import (
    format_reconciliation_note,
    reconcile_response_text,
)
from consistency.router import router as consistency_router
from consistency.store import (
    ClaimStore,
    InMemoryClaimStore,
    RedisClaimStore,
    create_claim_store,
)

__all__ = [
    "CORE_POSITIONS",
    "IKHTILAF_MAP",
    "ClaimStore",
    "ClaimType",
    "CoherenceMetrics",
    "ConsensusLevel",
    "ConsistencyAction",
    "ConsistencyAlert",
    "ConsistencyAlertManager",
    "ConsistencyCheckResult",
    "ConsistencyEnforcer",
    "Contradiction",
    "ContradictionCategory",
    "ContradictionSeverity",
    "CorePosition",
    "FactualClaim",
    "InMemoryClaimStore",
    "RedisClaimStore",
    "RulingType",
    "consistency_router",
    "create_claim_store",
    "detect_contradictions",
    "extract_claims",
    "format_reconciliation_note",
    "get_alert_manager",
    "get_consistency_enforcer",
    "get_core_position",
    "get_ikhtilaf_entry",
    "is_core_principle_violation",
    "normalize_text_for_claim",
    "reconcile_response_text",
]
