"""Alerting and knowledge base coherence monitoring.

Tracks consistency violations, logs alerts, and compiles coherence and drift metrics.
"""

from __future__ import annotations

import logging
import uuid

from consistency.models import (
    CoherenceMetrics,
    ConsistencyAlert,
    Contradiction,
    ContradictionSeverity,
)

logger = logging.getLogger(__name__)

MAX_ALERT_HISTORY = 1000


class ConsistencyAlertManager:
    """Tracks contradiction events and knowledge base coherence over time."""

    def __init__(self, max_alerts: int = MAX_ALERT_HISTORY) -> None:
        self.max_alerts = max_alerts
        self._alerts: list[ConsistencyAlert] = []
        self._total_checks = 0
        self._total_contradictions = 0
        self._legitimate_reconciliations = 0
        self._violations_blocked = 0
        self._attributions_checked = 0
        self._attributions_mismatched = 0

    def record_evaluation(
        self,
        contradictions: list[Contradiction],
        action_taken: str,
        chat_id: str | None = None,
        user_id: str | None = None,
    ) -> None:
        self._total_checks += 1

        for c in contradictions:
            self._total_contradictions += 1
            if c.is_legitimate_variation:
                self._legitimate_reconciliations += 1
            elif c.severity in (ContradictionSeverity.CRITICAL, ContradictionSeverity.HIGH):
                if action_taken == "block":
                    self._violations_blocked += 1

            if c.category.value == "attribution_mismatch":
                self._attributions_mismatched += 1
            self._attributions_checked += 1

            # Emit alert for non-trivial inconsistencies
            if not c.is_legitimate_variation and c.severity != ContradictionSeverity.LOW:
                alert = ConsistencyAlert(
                    alert_id=str(uuid.uuid4()),
                    severity=c.severity,
                    category=c.category,
                    chat_id=chat_id,
                    user_id=user_id,
                    summary=c.description,
                    details={
                        "candidate_claim": c.candidate_claim.model_dump(),
                        "historical_claim": c.historical_claim.model_dump() if c.historical_claim else None,
                        "action_taken": action_taken,
                    },
                )
                self._alerts.append(alert)
                if len(self._alerts) > self.max_alerts:
                    self._alerts.pop(0)

                logger.warning(
                    "Consistency Alert [%s]: %s (chat: %s, user: %s)",
                    c.severity.value.upper(),
                    c.description,
                    chat_id,
                    user_id,
                )

    def get_alerts(self, limit: int = 50, severity: str | None = None) -> list[ConsistencyAlert]:
        """Retrieve recent consistency alerts, optionally filtered by severity."""
        results = self._alerts
        if severity:
            results = [a for a in results if a.severity.value == severity.lower()]
        return list(reversed(results))[:limit]

    def clear_alerts(self) -> None:
        self._alerts.clear()

    def get_metrics(self, total_claims_in_store: int = 0) -> CoherenceMetrics:
        """Compute aggregate coherence and drift metrics."""
        consistency_rate = 1.0 - (self._total_contradictions - self._legitimate_reconciliations) / max(
            1, self._total_checks
        )
        consistency_rate = max(0.0, min(1.0, round(consistency_rate, 4)))

        attr_reliability = 1.0 - (self._attributions_mismatched / max(1, self._attributions_checked))
        attr_reliability = max(0.0, min(1.0, round(attr_reliability, 4)))

        drift_score = 1.0 - consistency_rate

        return CoherenceMetrics(
            total_claims_indexed=total_claims_in_store,
            total_checks_performed=self._total_checks,
            total_contradictions_detected=self._total_contradictions,
            legitimate_variations_reconciled=self._legitimate_reconciliations,
            violations_blocked=self._violations_blocked,
            consistency_rate=consistency_rate,
            attribution_reliability_rate=attr_reliability,
            cross_session_drift_score=round(drift_score, 4),
            active_alerts_count=len(self._alerts),
        )


_alert_manager: ConsistencyAlertManager | None = None


def get_alert_manager() -> ConsistencyAlertManager:
    global _alert_manager
    if _alert_manager is None:
        _alert_manager = ConsistencyAlertManager()
    return _alert_manager
