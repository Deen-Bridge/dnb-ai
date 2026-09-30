"""Consistency enforcement pipeline and policy orchestrator.

Integrates claim extraction, historical retrieval, contradiction detection,
reconciliation generation, and policy enforcement into a single pipeline.
"""

from __future__ import annotations

import logging
import os
import time

from consistency.alerts import ConsistencyAlertManager, get_alert_manager
from consistency.detector import detect_contradictions
from consistency.extractor import extract_claims
from consistency.models import (
    ConsistencyAction,
    ConsistencyCheckResult,
    ContradictionSeverity,
    FactualClaim,
)
from consistency.reconciliation import reconcile_response_text
from consistency.store import ClaimStore, create_claim_store

logger = logging.getLogger(__name__)

CONSISTENCY_ENABLED = os.getenv("CONSISTENCY_ENABLED", "true").lower() not in {"0", "false", "off"}
CONSISTENCY_STRICT_BLOCK = os.getenv("CONSISTENCY_STRICT_BLOCK", "true").lower() not in {"0", "false", "off"}


class ConsistencyEnforcer:
    """Orchestrates factual consistency verification and enforcement across turns and sessions."""

    def __init__(
        self,
        store: ClaimStore | None = None,
        alert_manager: ConsistencyAlertManager | None = None,
    ) -> None:
        self.store = store or create_claim_store()
        self.alert_manager = alert_manager or get_alert_manager()

    async def evaluate_response(
        self,
        response_text: str,
        prompt: str = "",
        chat_id: str | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
        madhhab: str | None = None,
    ) -> ConsistencyCheckResult:
        """Evaluate a candidate model response for cross-session consistency."""
        start_time = time.perf_counter()

        if not CONSISTENCY_ENABLED or not response_text or not response_text.strip():
            return ConsistencyCheckResult(
                is_consistent=True,
                contradiction_found=False,
                highest_severity=ContradictionSeverity.NONE,
                action=ConsistencyAction.ALLOW,
                original_text=response_text,
                final_text=response_text,
                latency_ms=0.0,
            )

        # 1. Extract atomic factual claims from response
        candidate_claims = extract_claims(
            response_text,
            chat_id=chat_id,
            user_id=user_id,
            session_id=session_id,
        )

        if not candidate_claims:
            latency = (time.perf_counter() - start_time) * 1000.0
            return ConsistencyCheckResult(
                is_consistent=True,
                contradiction_found=False,
                highest_severity=ContradictionSeverity.NONE,
                action=ConsistencyAction.ALLOW,
                original_text=response_text,
                final_text=response_text,
                claims_evaluated=0,
                latency_ms=round(latency, 2),
            )

        # 2. Retrieve relevant historical claims across sessions
        historical_claims: list[FactualClaim] = []
        seen_claim_ids: set[str] = set()

        for c in candidate_claims:
            # Query by entity, topic, and user
            claims = await self.store.find_relevant_claims(
                query=c.text,
                topic=c.topic,
                entity=c.entity,
                user_id=user_id,
                limit=15,
            )
            for hc in claims:
                if hc.claim_id not in seen_claim_ids:
                    seen_claim_ids.add(hc.claim_id)
                    historical_claims.append(hc)

        # 3. Detect contradictions against historical claims & core anchors
        contradictions = detect_contradictions(candidate_claims, historical_claims)

        # 4. Determine highest severity & policy action
        highest_severity = ContradictionSeverity.NONE
        action = ConsistencyAction.ALLOW
        final_text = response_text
        reconciliation_notes: list[str] = []

        if contradictions:
            # Calculate highest severity
            severity_order = {
                ContradictionSeverity.CRITICAL: 4,
                ContradictionSeverity.HIGH: 3,
                ContradictionSeverity.MEDIUM: 2,
                ContradictionSeverity.LOW: 1,
                ContradictionSeverity.NONE: 0,
            }
            highest_severity = max(contradictions, key=lambda c: severity_order[c.severity]).severity

            has_critical = any(c.severity == ContradictionSeverity.CRITICAL for c in contradictions)
            has_high_unlegit = any(
                c.severity == ContradictionSeverity.HIGH and not c.is_legitimate_variation for c in contradictions
            )
            has_legitimate_variation = any(c.is_legitimate_variation for c in contradictions)

            if has_critical and CONSISTENCY_STRICT_BLOCK:
                action = ConsistencyAction.BLOCK
                final_text = (
                    "⚠️ **Authenticity & Consistency Notice**: This response was withheld because it diverged "
                    "from established orthodox Islamic consensus or core principles. Please consult verified "
                    "classical sources and qualified scholars."
                )
            elif has_high_unlegit and CONSISTENCY_STRICT_BLOCK:
                action = ConsistencyAction.WARN
                final_text = (
                    f"{response_text.rstrip()}\n\n"
                    "⚠️ **Consistency Advisory**: This answer shows divergence with prior established guidance on this topic. "
                    "Please verify with a qualified scholar or authenticated reference texts."
                )
            elif has_legitimate_variation:
                action = ConsistencyAction.RECONCILE
                final_text = reconcile_response_text(response_text, contradictions)
            elif highest_severity in (ContradictionSeverity.MEDIUM, ContradictionSeverity.LOW):
                action = ConsistencyAction.WARN
                final_text = (
                    f"{response_text.rstrip()}\n\n"
                    "⚠️ *Note: Minor attribution or contextual variations exist regarding this ruling across sources.*"
                )

        # 5. Record evaluation in alert manager
        self.alert_manager.record_evaluation(
            contradictions=contradictions,
            action_taken=action.value,
            chat_id=chat_id,
            user_id=user_id,
        )

        latency = (time.perf_counter() - start_time) * 1000.0

        return ConsistencyCheckResult(
            is_consistent=not bool(contradictions) or all(c.is_legitimate_variation for c in contradictions),
            contradiction_found=bool(contradictions),
            highest_severity=highest_severity,
            contradictions=contradictions,
            action=action,
            original_text=response_text,
            final_text=final_text,
            reconciliation_notes=reconciliation_notes,
            claims_evaluated=len(candidate_claims),
            historical_claims_matched=len(historical_claims),
            latency_ms=round(latency, 2),
        )

    async def index_claims(
        self,
        text: str,
        chat_id: str | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
    ) -> list[FactualClaim]:
        """Extract and persist verified claims into the long-term claim store."""
        if not text or not text.strip():
            return []

        claims = extract_claims(text, chat_id=chat_id, user_id=user_id, session_id=session_id)
        if claims:
            await self.store.save_claims(claims)
            logger.debug("Indexed %d factual claims for user %s (chat %s)", len(claims), user_id, chat_id)
        return claims


_global_enforcer: ConsistencyEnforcer | None = None


def get_consistency_enforcer(store: ClaimStore | None = None) -> ConsistencyEnforcer:
    global _global_enforcer
    if _global_enforcer is None or store is not None:
        _global_enforcer = ConsistencyEnforcer(store=store)
    return _global_enforcer
