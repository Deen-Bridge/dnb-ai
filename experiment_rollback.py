"""Safe Experimentation, Canary Deployment, and Automated Rollback System (#269).

Provides:
- Feature flag system for gradual percentage-based rollouts and user whitelisting
- A/B testing across models, prompt templates, and runtime configurations
- Automated rollback triggers based on error rate, latency (P95), and consecutive failures
- Versioned configuration history with instant manual and automated rollback
- Seamless fallback to control variant during rollback to preserve user experience
- Complete audit trail of all experiment lifecycle transitions and canary adjustments
"""

from __future__ import annotations

import copy
import datetime
import hashlib
import logging
import uuid
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums & Statuses
# ---------------------------------------------------------------------------


class ExperimentStatus(StrEnum):
    DRAFT = "draft"
    ACTIVE = "active"
    PAUSED = "paused"
    ROLLED_BACK = "rolled_back"
    COMPLETED = "completed"


class ExperimentTargetType(StrEnum):
    PROMPT = "prompt"
    MODEL = "model"
    FEATURE_FLAG = "feature_flag"
    CONFIG = "config"


class AuditAction(StrEnum):
    CREATE = "create"
    UPDATE_ROLLOUT = "update_rollout"
    UPDATE_CONFIG = "update_config"
    MANUAL_ROLLBACK = "manual_rollback"
    AUTO_ROLLBACK = "auto_rollback"
    KILL = "kill"
    RESUME = "resume"
    FLAG_TOGGLE = "flag_toggle"


# ---------------------------------------------------------------------------
# Feature Flag System
# ---------------------------------------------------------------------------


class FeatureFlag(BaseModel):
    """A feature flag supporting gradual rollouts and user/role targeting."""

    flag_id: str = Field(..., min_length=1, max_length=128)
    name: str = Field(..., min_length=1, max_length=200)
    description: str | None = None
    enabled: bool = False
    rollout_percentage: float = Field(0.0, ge=0.0, le=100.0)
    allowed_users: list[str] = Field(default_factory=list)
    allowed_roles: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=lambda: datetime.datetime.now(datetime.UTC).isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.datetime.now(datetime.UTC).isoformat())

    def is_enabled_for(
        self,
        user_or_session_id: str | None = None,
        role: str | None = None,
    ) -> bool:
        """Evaluate if the flag is enabled for the specific user/session."""
        if not self.enabled:
            return False

        if role and role in self.allowed_roles:
            return True

        if user_or_session_id and user_or_session_id in self.allowed_users:
            return True

        if self.rollout_percentage >= 100.0:
            return True

        if self.rollout_percentage <= 0.0:
            return False

        if not user_or_session_id:
            return False

        # Deterministic hash for consistent user assignment
        hash_val = int(hashlib.sha256(f"{self.flag_id}:{user_or_session_id}".encode()).hexdigest(), 16) % 10000 / 100.0
        return hash_val < self.rollout_percentage


# ---------------------------------------------------------------------------
# Experiment Models & Triggers
# ---------------------------------------------------------------------------


class RollbackTriggerConfig(BaseModel):
    """Conditions that trigger an immediate automatic rollback to the control variant."""

    max_error_rate: float = Field(0.05, ge=0.0, le=1.0, description="Max tolerable error rate (e.g. 5%)")
    max_p95_latency_ms: float = Field(2500.0, ge=50.0, le=60000.0, description="Max acceptable P95 latency in ms")
    max_consecutive_errors: int = Field(5, ge=1, le=100, description="Consecutive errors to trigger rollback")
    min_sample_size: int = Field(10, ge=1, le=1000, description="Minimum turns evaluated before auto-triggering")
    auto_rollback_enabled: bool = True


class ExperimentVariant(BaseModel):
    """A variant inside an A/B or canary experiment."""

    name: str = Field(..., min_length=1, max_length=64)
    model: str | None = None
    template_name: str | None = None
    template_version: str | None = None
    config_overrides: dict[str, Any] = Field(default_factory=dict)
    weight: float = Field(1.0, ge=0.0)
    is_control: bool = False


class VariantMetrics(BaseModel):
    """Telemetry metrics accumulated for a single variant."""

    request_count: int = 0
    success_count: int = 0
    error_count: int = 0
    consecutive_errors: int = 0
    total_latency_ms: float = 0.0
    latencies: list[float] = Field(default_factory=list)

    def record(self, latency_ms: float, is_error: bool) -> None:
        self.request_count += 1
        self.total_latency_ms += latency_ms
        # Keep sliding reservoir of last 500 latencies
        if len(self.latencies) >= 500:
            self.latencies.pop(0)
        self.latencies.append(latency_ms)

        if is_error:
            self.error_count += 1
            self.consecutive_errors += 1
        else:
            self.success_count += 1
            self.consecutive_errors = 0

    @property
    def error_rate(self) -> float:
        if self.request_count == 0:
            return 0.0
        return round(self.error_count / self.request_count, 4)

    @property
    def avg_latency_ms(self) -> float:
        if self.request_count == 0:
            return 0.0
        return round(self.total_latency_ms / self.request_count, 2)

    @property
    def p95_latency_ms(self) -> float:
        if not self.latencies:
            return 0.0
        sorted_l = sorted(self.latencies)
        idx = int(len(sorted_l) * 0.95)
        idx = min(idx, len(sorted_l) - 1)
        return round(sorted_l[idx], 2)


class ExperimentAuditEntry(BaseModel):
    """Audit record capturing an experiment state change or rollback."""

    entry_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    experiment_id: str
    version: int
    timestamp: str = Field(default_factory=lambda: datetime.datetime.now(datetime.UTC).isoformat())
    author: str
    action: AuditAction
    reason: str
    diff: dict[str, Any] = Field(default_factory=dict)
    snapshot: dict[str, Any] = Field(default_factory=dict)


class ExperimentConfigVersion(BaseModel):
    """Historical version snapshot of an experiment configuration."""

    version: int
    timestamp: str
    status: ExperimentStatus
    canary_percentage: float
    control: ExperimentVariant
    variants: list[ExperimentVariant]
    triggers: RollbackTriggerConfig
    kill_switch: bool


class ExperimentState(BaseModel):
    """Runtime state, version history, and metrics for a registered experiment."""

    experiment_id: str
    name: str
    target_type: ExperimentTargetType = ExperimentTargetType.PROMPT
    status: ExperimentStatus = ExperimentStatus.ACTIVE
    canary_percentage: float = 0.0  # 0 to 100
    control: ExperimentVariant
    variants: list[ExperimentVariant] = Field(default_factory=list)
    triggers: RollbackTriggerConfig = Field(default_factory=RollbackTriggerConfig)
    kill_switch: bool = False
    current_version: int = 1
    version_history: list[ExperimentConfigVersion] = Field(default_factory=list)
    metrics: dict[str, VariantMetrics] = Field(default_factory=dict)
    created_at: str = Field(default_factory=lambda: datetime.datetime.now(datetime.UTC).isoformat())
    last_rollback_reason: str | None = None


# ---------------------------------------------------------------------------
# Manager Implementation
# ---------------------------------------------------------------------------


class ExperimentRollbackManager:
    """Orchestrates feature flags, experiments, canary deployments, and automated rollbacks."""

    def __init__(self) -> None:
        self._experiments: dict[str, ExperimentState] = {}
        self._flags: dict[str, FeatureFlag] = {}
        self._audit_log: list[ExperimentAuditEntry] = []

    # --- Feature Flags ---

    def register_flag(self, flag: FeatureFlag) -> None:
        self._flags[flag.flag_id] = flag

    def get_flag(self, flag_id: str) -> FeatureFlag | None:
        return self._flags.get(flag_id)

    def list_flags(self) -> list[FeatureFlag]:
        return list(self._flags.values())

    def update_flag(
        self,
        flag_id: str,
        enabled: bool | None = None,
        rollout_percentage: float | None = None,
        allowed_users: list[str] | None = None,
        author: str = "admin",
        reason: str = "Feature flag update",
    ) -> FeatureFlag:
        flag = self._flags.get(flag_id)
        if not flag:
            raise KeyError(f"Feature flag '{flag_id}' not found")

        old_state = flag.model_dump()
        if enabled is not None:
            flag.enabled = enabled
        if rollout_percentage is not None:
            flag.rollout_percentage = max(0.0, min(100.0, rollout_percentage))
        if allowed_users is not None:
            flag.allowed_users = allowed_users
        flag.updated_at = datetime.datetime.now(datetime.UTC).isoformat()

        self._audit_log.append(
            ExperimentAuditEntry(
                experiment_id=flag_id,
                version=1,
                author=author,
                action=AuditAction.FLAG_TOGGLE,
                reason=reason,
                diff={"old": old_state, "new": flag.model_dump()},
                snapshot=flag.model_dump(),
            )
        )
        return flag

    # --- Experiment Registration ---

    def create_experiment(
        self,
        experiment_id: str,
        name: str,
        control: ExperimentVariant,
        variants: list[ExperimentVariant] | None = None,
        target_type: ExperimentTargetType = ExperimentTargetType.PROMPT,
        canary_percentage: float = 0.0,
        triggers: RollbackTriggerConfig | None = None,
        author: str = "admin",
        reason: str = "Initial experiment creation",
    ) -> ExperimentState:
        """Register an experiment and snapshot initial version."""
        control.is_control = True
        var_list = variants or []
        trigger_cfg = triggers or RollbackTriggerConfig()

        state = ExperimentState(
            experiment_id=experiment_id,
            name=name,
            target_type=target_type,
            status=ExperimentStatus.ACTIVE,
            canary_percentage=canary_percentage,
            control=control,
            variants=var_list,
            triggers=trigger_cfg,
            kill_switch=False,
            current_version=1,
            metrics={control.name: VariantMetrics()},
        )
        for v in var_list:
            state.metrics[v.name] = VariantMetrics()

        initial_snapshot = ExperimentConfigVersion(
            version=1,
            timestamp=datetime.datetime.now(datetime.UTC).isoformat(),
            status=state.status,
            canary_percentage=canary_percentage,
            control=copy.deepcopy(control),
            variants=copy.deepcopy(var_list),
            triggers=copy.deepcopy(trigger_cfg),
            kill_switch=False,
        )
        state.version_history.append(initial_snapshot)
        self._experiments[experiment_id] = state

        self._audit_log.append(
            ExperimentAuditEntry(
                experiment_id=experiment_id,
                version=1,
                author=author,
                action=AuditAction.CREATE,
                reason=reason,
                diff={"status": "created"},
                snapshot=state.model_dump(),
            )
        )
        return state

    def get_experiment(self, experiment_id: str) -> ExperimentState | None:
        return self._experiments.get(experiment_id)

    def list_experiments(self) -> list[ExperimentState]:
        return list(self._experiments.values())

    # --- Variant Assignment & Fallback ---

    def assign_variant(
        self,
        experiment_id: str,
        session_id: str,
    ) -> tuple[ExperimentVariant, bool]:
        """Deterministically assign session to variant.

        Returns (chosen_variant, was_fallback_to_control).
        Preserves user experience by safely returning control variant when
        experiment is killed, paused, or rolled back.
        """
        exp = self._experiments.get(experiment_id)
        if not exp:
            raise KeyError(f"Experiment '{experiment_id}' not found")

        # Fallback to control if rolled back, paused, or killed
        if exp.kill_switch or exp.status in (ExperimentStatus.ROLLED_BACK, ExperimentStatus.PAUSED):
            return exp.control, True

        if not exp.variants:
            return exp.control, False

        # If canary percentage is configured, check if session lands in canary pool
        if exp.canary_percentage > 0.0:
            canary_hash = (
                int(hashlib.sha256(f"{experiment_id}:canary:{session_id}".encode()).hexdigest(), 16) % 10000 / 100.0
            )
            if canary_hash >= exp.canary_percentage:
                # Outside canary percentage -> control variant
                return exp.control, False

        # Weighted selection among all variants (including control)
        all_variants = [exp.control] + exp.variants
        total_weight = sum(v.weight for v in all_variants)
        if total_weight <= 0:
            return exp.control, False

        hash_val = int(hashlib.sha256(f"{experiment_id}:{session_id}".encode()).hexdigest(), 16) % 10000 / 10000.0
        pick = hash_val * total_weight
        cumulative = 0.0
        chosen = exp.control
        for v in all_variants:
            cumulative += v.weight
            if pick <= cumulative:
                chosen = v
                break

        return chosen, False

    # --- Telemetry & Automated Rollback ---

    def record_turn_result(
        self,
        experiment_id: str,
        variant_name: str,
        latency_ms: float,
        is_error: bool = False,
        error_message: str | None = None,
    ) -> bool:
        """Record turn latency and error outcome.

        Evaluates automated rollback triggers. If breached, automatically rolls back
        the experiment and returns True. Otherwise returns False.
        """
        exp = self._experiments.get(experiment_id)
        if not exp:
            return False

        if variant_name not in exp.metrics:
            exp.metrics[variant_name] = VariantMetrics()

        metric = exp.metrics[variant_name]
        metric.record(latency_ms, is_error)

        # Automated rollback checks on candidate variants (do not trigger auto-rollback on control)
        if variant_name != exp.control.name and exp.triggers.auto_rollback_enabled:
            if exp.status == ExperimentStatus.ACTIVE and metric.request_count >= exp.triggers.min_sample_size:
                breach_reason: str | None = None

                # 1. Error rate check
                if metric.error_rate > exp.triggers.max_error_rate:
                    breach_reason = (
                        f"Error rate {metric.error_rate * 100:.1f}% exceeded trigger threshold "
                        f"{exp.triggers.max_error_rate * 100:.1f}%"
                    )

                # 2. P95 latency check
                elif metric.p95_latency_ms > exp.triggers.max_p95_latency_ms:
                    breach_reason = (
                        f"P95 latency {metric.p95_latency_ms:.1f}ms exceeded trigger threshold "
                        f"{exp.triggers.max_p95_latency_ms:.1f}ms"
                    )

                # 3. Consecutive errors check
                elif metric.consecutive_errors >= exp.triggers.max_consecutive_errors:
                    breach_reason = (
                        f"Consecutive errors ({metric.consecutive_errors}) reached trigger limit "
                        f"({exp.triggers.max_consecutive_errors})"
                    )

                if breach_reason:
                    logger.critical(
                        "AUTOMATED ROLLBACK ACTIVATED for experiment '%s': %s",
                        experiment_id,
                        breach_reason,
                    )
                    self.trigger_automated_rollback(experiment_id, breach_reason)
                    return True

        return False

    def trigger_automated_rollback(self, experiment_id: str, breach_reason: str) -> None:
        """Automate rollback when telemetry triggers are breached."""
        exp = self._experiments.get(experiment_id)
        if not exp:
            return

        exp.status = ExperimentStatus.ROLLED_BACK
        exp.kill_switch = True
        exp.last_rollback_reason = breach_reason
        exp.current_version += 1

        snapshot = ExperimentConfigVersion(
            version=exp.current_version,
            timestamp=datetime.datetime.now(datetime.UTC).isoformat(),
            status=exp.status,
            canary_percentage=0.0,
            control=copy.deepcopy(exp.control),
            variants=copy.deepcopy(exp.variants),
            triggers=copy.deepcopy(exp.triggers),
            kill_switch=True,
        )
        exp.version_history.append(snapshot)

        self._audit_log.append(
            ExperimentAuditEntry(
                experiment_id=experiment_id,
                version=exp.current_version,
                author="system:auto-monitor",
                action=AuditAction.AUTO_ROLLBACK,
                reason=breach_reason,
                diff={"status": "rolled_back", "kill_switch": True},
                snapshot=exp.model_dump(),
            )
        )

    # --- Canary Rollout & Version Rollbacks ---

    def set_canary_rollout(
        self,
        experiment_id: str,
        percentage: float,
        author: str = "admin",
        reason: str = "Canary percentage adjustment",
    ) -> ExperimentState:
        """Adjust the canary rollout percentage (0 to 100%)."""
        exp = self._experiments.get(experiment_id)
        if not exp:
            raise KeyError(f"Experiment '{experiment_id}' not found")

        old_pct = exp.canary_percentage
        exp.canary_percentage = max(0.0, min(100.0, percentage))
        exp.current_version += 1

        snapshot = ExperimentConfigVersion(
            version=exp.current_version,
            timestamp=datetime.datetime.now(datetime.UTC).isoformat(),
            status=exp.status,
            canary_percentage=exp.canary_percentage,
            control=copy.deepcopy(exp.control),
            variants=copy.deepcopy(exp.variants),
            triggers=copy.deepcopy(exp.triggers),
            kill_switch=exp.kill_switch,
        )
        exp.version_history.append(snapshot)

        self._audit_log.append(
            ExperimentAuditEntry(
                experiment_id=experiment_id,
                version=exp.current_version,
                author=author,
                action=AuditAction.UPDATE_ROLLOUT,
                reason=reason,
                diff={"canary_percentage": {"old": old_pct, "new": exp.canary_percentage}},
                snapshot=exp.model_dump(),
            )
        )
        return exp

    def rollback(
        self,
        experiment_id: str,
        to_version: int | None = None,
        author: str = "admin",
        reason: str = "Manual experiment rollback",
    ) -> ExperimentState:
        """Roll back experiment to previous stable version or specific historical snapshot."""
        exp = self._experiments.get(experiment_id)
        if not exp:
            raise KeyError(f"Experiment '{experiment_id}' not found")

        target_snapshot: ExperimentConfigVersion | None = None
        if to_version is not None:
            for s in exp.version_history:
                if s.version == to_version:
                    target_snapshot = s
                    break
            if not target_snapshot:
                raise ValueError(f"Version {to_version} not found in experiment history")
        else:
            if len(exp.version_history) < 2:
                # If only 1 version exists, fallback by killing variants and defaulting to control
                exp.kill_switch = True
                exp.status = ExperimentStatus.ROLLED_BACK
                exp.last_rollback_reason = reason
                return exp
            target_snapshot = exp.version_history[-2]

        exp.control = copy.deepcopy(target_snapshot.control)
        exp.variants = copy.deepcopy(target_snapshot.variants)
        exp.canary_percentage = target_snapshot.canary_percentage
        exp.triggers = copy.deepcopy(target_snapshot.triggers)
        exp.kill_switch = target_snapshot.kill_switch
        exp.status = target_snapshot.status
        exp.last_rollback_reason = reason
        exp.current_version += 1

        new_snapshot = ExperimentConfigVersion(
            version=exp.current_version,
            timestamp=datetime.datetime.now(datetime.UTC).isoformat(),
            status=exp.status,
            canary_percentage=exp.canary_percentage,
            control=copy.deepcopy(exp.control),
            variants=copy.deepcopy(exp.variants),
            triggers=copy.deepcopy(exp.triggers),
            kill_switch=exp.kill_switch,
        )
        exp.version_history.append(new_snapshot)

        self._audit_log.append(
            ExperimentAuditEntry(
                experiment_id=experiment_id,
                version=exp.current_version,
                author=author,
                action=AuditAction.MANUAL_ROLLBACK,
                reason=reason,
                diff={"restored_from_version": target_snapshot.version},
                snapshot=exp.model_dump(),
            )
        )
        return exp

    def kill_experiment(
        self,
        experiment_id: str,
        author: str = "admin",
        reason: str = "Emergency kill switch engaged",
    ) -> ExperimentState:
        """Immediately kill-switch an experiment, routing 100% of traffic to control."""
        exp = self._experiments.get(experiment_id)
        if not exp:
            raise KeyError(f"Experiment '{experiment_id}' not found")

        exp.kill_switch = True
        exp.status = ExperimentStatus.PAUSED
        exp.current_version += 1

        self._audit_log.append(
            ExperimentAuditEntry(
                experiment_id=experiment_id,
                version=exp.current_version,
                author=author,
                action=AuditAction.KILL,
                reason=reason,
                diff={"kill_switch": True, "status": "paused"},
                snapshot=exp.model_dump(),
            )
        )
        return exp

    def resume_experiment(
        self,
        experiment_id: str,
        author: str = "admin",
        reason: str = "Resumed experiment",
    ) -> ExperimentState:
        """Resume a killed or paused experiment."""
        exp = self._experiments.get(experiment_id)
        if not exp:
            raise KeyError(f"Experiment '{experiment_id}' not found")

        exp.kill_switch = False
        exp.status = ExperimentStatus.ACTIVE
        exp.current_version += 1

        self._audit_log.append(
            ExperimentAuditEntry(
                experiment_id=experiment_id,
                version=exp.current_version,
                author=author,
                action=AuditAction.RESUME,
                reason=reason,
                diff={"kill_switch": False, "status": "active"},
                snapshot=exp.model_dump(),
            )
        )
        return exp

    def get_audit_trail(
        self,
        experiment_id: str | None = None,
        limit: int = 50,
    ) -> list[ExperimentAuditEntry]:
        """Fetch audit log entries, newest first."""
        records = list(self._audit_log)
        if experiment_id:
            records = [r for r in records if r.experiment_id == experiment_id]
        return sorted(records, key=lambda r: r.timestamp, reverse=True)[:limit]


# ---------------------------------------------------------------------------
# Global Singleton
# ---------------------------------------------------------------------------

_rollback_manager: ExperimentRollbackManager | None = None


def get_rollback_manager() -> ExperimentRollbackManager:
    """Access global ExperimentRollbackManager instance."""
    global _rollback_manager
    if _rollback_manager is None:
        _rollback_manager = ExperimentRollbackManager()
    return _rollback_manager


def reset_rollback_manager() -> ExperimentRollbackManager:
    """Reset global manager instance (for test isolation)."""
    global _rollback_manager
    _rollback_manager = ExperimentRollbackManager()
    return _rollback_manager
