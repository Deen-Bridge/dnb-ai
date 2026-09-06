"""Runtime Confidence Threshold Tuning and Administration System (#270).

Enables dynamic adjustment of confidence thresholds for retrieval and response
generation at runtime without requiring redeployment or server restarts.

Features:
- Dynamic threshold configuration with validation safeguards against extreme values
- Per-Islamic-topic category thresholds (fiqh, hadith, aqidah, tafsir, general)
- Source-specific confidence and retrieval requirements
- A/B testing framework for comparing candidate threshold variants
- Telemetry tracking of threshold changes on response quality and distribution
- Automated tuning recommendations based on empirical performance
- Gradual stepwise threshold adjustments to avoid sudden production changes
- Comprehensive audit trail with before/after snapshots and instant rollback
"""

from __future__ import annotations

import copy
import datetime
import hashlib
import logging
import uuid
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants & Safeguard Boundaries
# ---------------------------------------------------------------------------

MIN_ALLOWED_THRESHOLD = 0.05
MAX_ALLOWED_THRESHOLD = 0.95
MIN_THRESHOLD_SPREAD = 0.05
MAX_HIGH_STAKES_PENALTY = 0.60
MAX_GRADUAL_STEP_DELTA = 0.20


class AuditAction(StrEnum):
    INIT = "init"
    UPDATE = "update"
    GRADUAL_STEP = "gradual_step"
    ROLLBACK = "rollback"
    AB_START = "ab_start"
    AB_STOP = "ab_stop"
    RESET = "reset"


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class TopicThresholdConfig(BaseModel):
    """Custom confidence thresholds for a specific Islamic knowledge category."""

    low_threshold: float = Field(..., ge=MIN_ALLOWED_THRESHOLD, le=MAX_ALLOWED_THRESHOLD)
    high_threshold: float = Field(..., ge=MIN_ALLOWED_THRESHOLD, le=MAX_ALLOWED_THRESHOLD)
    scholar_queue_threshold: float | None = Field(default=None, ge=MIN_ALLOWED_THRESHOLD, le=MAX_ALLOWED_THRESHOLD)
    high_stakes_penalty: float | None = Field(default=None, ge=0.0, le=MAX_HIGH_STAKES_PENALTY)
    description: str | None = None

    @field_validator("high_threshold")
    @classmethod
    def validate_spread(cls, v: float, info: Any) -> float:
        low = info.data.get("low_threshold")
        if low is not None and v < low + MIN_THRESHOLD_SPREAD:
            raise ValueError(
                f"high_threshold ({v}) must exceed low_threshold ({low}) by at least {MIN_THRESHOLD_SPREAD}"
            )
        return v


class SourceThresholdConfig(BaseModel):
    """Confidence and relevance requirements for specific Islamic sources."""

    min_retrieval_score: float = Field(0.30, ge=0.0, le=1.0)
    min_confidence: float = Field(0.40, ge=0.0, le=1.0)
    is_authoritative: bool = True
    description: str | None = None


class ConfidenceThresholdConfig(BaseModel):
    """Global and domain-specific runtime confidence threshold configuration."""

    low_threshold: float = Field(default=0.40, ge=MIN_ALLOWED_THRESHOLD, le=MAX_ALLOWED_THRESHOLD)
    high_threshold: float = Field(default=0.70, ge=MIN_ALLOWED_THRESHOLD, le=MAX_ALLOWED_THRESHOLD)
    scholar_queue_threshold: float = Field(default=0.40, ge=MIN_ALLOWED_THRESHOLD, le=MAX_ALLOWED_THRESHOLD)
    high_stakes_penalty: float = Field(default=0.15, ge=0.0, le=MAX_HIGH_STAKES_PENALTY)
    no_signal_prior: float = Field(default=0.55, ge=0.0, le=1.0)
    unverified_ceiling: float = Field(default=0.65, ge=0.0, le=1.0)
    retrieval_min_score: float = Field(default=0.25, ge=0.0, le=1.0)

    topics: dict[str, TopicThresholdConfig] = Field(default_factory=dict)
    sources: dict[str, SourceThresholdConfig] = Field(default_factory=dict)

    @field_validator("high_threshold")
    @classmethod
    def validate_global_spread(cls, v: float, info: Any) -> float:
        low = info.data.get("low_threshold")
        if low is not None and v < low + MIN_THRESHOLD_SPREAD:
            raise ValueError(
                f"high_threshold ({v}) must exceed low_threshold ({low}) by at least {MIN_THRESHOLD_SPREAD}"
            )
        return v


def default_threshold_config() -> ConfidenceThresholdConfig:
    """Build the production default configuration with tailored Islamic topics and sources."""
    return ConfidenceThresholdConfig(
        low_threshold=0.40,
        high_threshold=0.70,
        scholar_queue_threshold=0.40,
        high_stakes_penalty=0.15,
        no_signal_prior=0.55,
        unverified_ceiling=0.65,
        retrieval_min_score=0.25,
        topics={
            "fiqh": TopicThresholdConfig(
                low_threshold=0.50,
                high_threshold=0.75,
                scholar_queue_threshold=0.50,
                high_stakes_penalty=0.20,
                description="Jurisprudence requires high confidence and lower tolerance for ambiguity",
            ),
            "aqidah": TopicThresholdConfig(
                low_threshold=0.55,
                high_threshold=0.80,
                scholar_queue_threshold=0.55,
                high_stakes_penalty=0.25,
                description="Core theology and creed requires strict certainty",
            ),
            "hadith": TopicThresholdConfig(
                low_threshold=0.45,
                high_threshold=0.70,
                scholar_queue_threshold=0.45,
                high_stakes_penalty=0.15,
                description="Prophetic narrations requiring authentic isnad and verification",
            ),
            "tafsir": TopicThresholdConfig(
                low_threshold=0.40,
                high_threshold=0.70,
                scholar_queue_threshold=0.40,
                high_stakes_penalty=0.10,
                description="Quranic exegesis accommodating scholarly commentary",
            ),
            "general": TopicThresholdConfig(
                low_threshold=0.35,
                high_threshold=0.65,
                scholar_queue_threshold=0.35,
                high_stakes_penalty=0.05,
                description="General educational queries with moderate thresholds",
            ),
        },
        sources={
            "quran": SourceThresholdConfig(
                min_retrieval_score=0.35,
                min_confidence=0.60,
                is_authoritative=True,
                description="The primary holy scripture",
            ),
            "sahih_bukhari": SourceThresholdConfig(
                min_retrieval_score=0.30,
                min_confidence=0.50,
                is_authoritative=True,
                description="Canonical Sahih Hadith collection",
            ),
            "sahih_muslim": SourceThresholdConfig(
                min_retrieval_score=0.30,
                min_confidence=0.50,
                is_authoritative=True,
                description="Canonical Sahih Hadith collection",
            ),
            "classical_tafsir": SourceThresholdConfig(
                min_retrieval_score=0.25,
                min_confidence=0.45,
                is_authoritative=True,
                description="Recognized historical exegesis (Ibn Kathir, Tabari, Qurtubi)",
            ),
            "web": SourceThresholdConfig(
                min_retrieval_score=0.45,
                min_confidence=0.55,
                is_authoritative=False,
                description="General external content requiring higher verification threshold",
            ),
        },
    )


class PerformanceMetrics(BaseModel):
    """Execution distribution and user feedback metrics for a configuration."""

    total_evaluations: int = 0
    abstain_count: int = 0
    uncertain_count: int = 0
    confident_count: int = 0
    scholar_queue_count: int = 0
    positive_feedback: int = 0
    negative_feedback: int = 0

    def record_evaluation(
        self,
        band: str,
        queued: bool,
        feedback: bool | None = None,
    ) -> None:
        self.total_evaluations += 1
        if band == "abstain":
            self.abstain_count += 1
        elif band == "uncertain":
            self.uncertain_count += 1
        elif band == "confident":
            self.confident_count += 1

        if queued:
            self.scholar_queue_count += 1

        if feedback is True:
            self.positive_feedback += 1
        elif feedback is False:
            self.negative_feedback += 1

    @property
    def abstain_rate(self) -> float:
        return round(self.abstain_count / max(1, self.total_evaluations), 4)

    @property
    def uncertain_rate(self) -> float:
        return round(self.uncertain_count / max(1, self.total_evaluations), 4)

    @property
    def confident_rate(self) -> float:
        return round(self.confident_count / max(1, self.total_evaluations), 4)

    @property
    def satisfaction_rate(self) -> float:
        total_feedback = self.positive_feedback + self.negative_feedback
        if total_feedback == 0:
            return 1.0
        return round(self.positive_feedback / total_feedback, 4)


class ThresholdAuditEntry(BaseModel):
    """Audit record capturing a threshold change, justification, and diff."""

    entry_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    version: int
    timestamp: str = Field(default_factory=lambda: datetime.datetime.now(datetime.UTC).isoformat())
    author: str
    action: AuditAction
    reason: str
    changes: dict[str, Any] = Field(default_factory=dict)
    snapshot: dict[str, Any] = Field(default_factory=dict)


class ABExperimentConfig(BaseModel):
    """Active A/B testing experiment comparing two threshold policies."""

    experiment_id: str
    name: str
    traffic_split: float = Field(0.50, ge=0.01, le=0.99)
    active: bool = True
    variant_a: ConfidenceThresholdConfig
    variant_b: ConfidenceThresholdConfig
    created_at: str = Field(default_factory=lambda: datetime.datetime.now(datetime.UTC).isoformat())
    metrics_a: PerformanceMetrics = Field(default_factory=PerformanceMetrics)
    metrics_b: PerformanceMetrics = Field(default_factory=PerformanceMetrics)


# ---------------------------------------------------------------------------
# Manager Implementation
# ---------------------------------------------------------------------------


class ConfidenceTuningManager:
    """Manages runtime confidence thresholds, safeguards, A/B experiments, and audit history."""

    def __init__(self, initial_config: ConfidenceThresholdConfig | None = None) -> None:
        self._current_config = initial_config or default_threshold_config()
        self._version = 1
        self._history: list[ThresholdAuditEntry] = []
        self._metrics = PerformanceMetrics()
        self._topic_metrics: dict[str, PerformanceMetrics] = {}
        self._ab_experiment: ABExperimentConfig | None = None

        # Log initial entry
        self._history.append(
            ThresholdAuditEntry(
                version=self._version,
                author="system",
                action=AuditAction.INIT,
                reason="Initial baseline configuration",
                changes={"status": "initialized"},
                snapshot=self._current_config.model_dump(),
            )
        )

    @property
    def version(self) -> int:
        return self._version

    @property
    def current_config(self) -> ConfidenceThresholdConfig:
        return self._current_config

    @property
    def metrics(self) -> PerformanceMetrics:
        return self._metrics

    @property
    def topic_metrics(self) -> dict[str, PerformanceMetrics]:
        return self._topic_metrics

    @property
    def ab_experiment(self) -> ABExperimentConfig | None:
        return self._ab_experiment

    def get_audit_trail(
        self,
        limit: int = 50,
        action: AuditAction | None = None,
        author: str | None = None,
    ) -> list[ThresholdAuditEntry]:
        """Query audit history filtered by action or author, newest first."""
        results = list(self._history)
        if action:
            results = [e for e in results if e.action == action]
        if author:
            results = [e for e in results if e.author.lower() == author.lower()]
        return sorted(results, key=lambda e: e.version, reverse=True)[:limit]

    def _determine_variant_config(
        self,
        request_key: str | None = None,
    ) -> tuple[str | None, ConfidenceThresholdConfig]:
        """Determine whether request routes to Variant A or B during an active experiment."""
        if not self._ab_experiment or not self._ab_experiment.active:
            return None, self._current_config

        if not request_key:
            # Deterministic pseudo-random split
            request_key = str(uuid.uuid4())

        # Hash request key to 0..1 range
        hash_val = int(hashlib.md5(request_key.encode("utf-8")).hexdigest(), 16) % 10000 / 10000.0
        if hash_val < self._ab_experiment.traffic_split:
            return "variant_b", self._ab_experiment.variant_b
        return "variant_a", self._ab_experiment.variant_a

    def get_effective_thresholds(
        self,
        topic: str | None = None,
        source: str | None = None,
        request_key: str | None = None,
    ) -> dict[str, float]:
        """Compute the active thresholds applying topic, source, and experiment overrides."""
        _, config = self._determine_variant_config(request_key)

        low = config.low_threshold
        high = config.high_threshold
        scholar_queue = config.scholar_queue_threshold
        high_stakes_penalty = config.high_stakes_penalty
        retrieval_min_score = config.retrieval_min_score

        # Apply topic specialization if configured
        if topic and topic.lower() in config.topics:
            t_cfg = config.topics[topic.lower()]
            low = t_cfg.low_threshold
            high = t_cfg.high_threshold
            if t_cfg.scholar_queue_threshold is not None:
                scholar_queue = t_cfg.scholar_queue_threshold
            if t_cfg.high_stakes_penalty is not None:
                high_stakes_penalty = t_cfg.high_stakes_penalty

        # Apply source specialization if configured
        if source and source.lower() in config.sources:
            s_cfg = config.sources[source.lower()]
            retrieval_min_score = s_cfg.min_retrieval_score
            # If source requires higher minimum confidence, elevate low threshold
            if s_cfg.min_confidence > low:
                low = s_cfg.min_confidence
                if high < low + MIN_THRESHOLD_SPREAD:
                    high = min(MAX_ALLOWED_THRESHOLD, low + MIN_THRESHOLD_SPREAD)

        return {
            "low": round(low, 4),
            "high": round(high, 4),
            "scholar_queue": round(scholar_queue, 4),
            "high_stakes_penalty": round(high_stakes_penalty, 4),
            "no_signal_prior": round(config.no_signal_prior, 4),
            "unverified_ceiling": round(config.unverified_ceiling, 4),
            "retrieval_min_score": round(retrieval_min_score, 4),
        }

    def resolve_band(
        self,
        score: float,
        topic: str | None = None,
        source: str | None = None,
        request_key: str | None = None,
    ) -> str:
        """Resolve score to 'abstain', 'uncertain', or 'confident' based on runtime policy."""
        thresholds = self.get_effective_thresholds(topic=topic, source=source, request_key=request_key)
        if score < thresholds["low"]:
            return "abstain"
        if score < thresholds["high"]:
            return "uncertain"
        return "confident"

    def should_queue_scholar(
        self,
        score: float,
        is_religious: bool,
        topic: str | None = None,
        request_key: str | None = None,
    ) -> bool:
        """Check if an answer should be sent to the scholar queue under runtime policy."""
        if not is_religious:
            return False
        thresholds = self.get_effective_thresholds(topic=topic, request_key=request_key)
        return score < thresholds["scholar_queue"]

    def record_turn(
        self,
        score: float,
        band: str,
        is_religious: bool,
        topic: str | None = None,
        feedback: bool | None = None,
        request_key: str | None = None,
    ) -> None:
        """Record telemetry for a completed turn to evaluate policy quality."""
        queued = self.should_queue_scholar(score, is_religious, topic=topic, request_key=request_key)
        self._metrics.record_evaluation(band, queued, feedback)

        if topic:
            if topic not in self._topic_metrics:
                self._topic_metrics[topic] = PerformanceMetrics()
            self._topic_metrics[topic].record_evaluation(band, queued, feedback)

        # Record in active A/B experiment if present
        if self._ab_experiment and self._ab_experiment.active:
            variant_name, _ = self._determine_variant_config(request_key)
            if variant_name == "variant_b":
                self._ab_experiment.metrics_b.record_evaluation(band, queued, feedback)
            else:
                self._ab_experiment.metrics_a.record_evaluation(band, queued, feedback)

    def update_config(
        self,
        new_config: ConfidenceThresholdConfig | dict[str, Any],
        author: str = "admin",
        reason: str = "Runtime threshold update",
    ) -> ThresholdAuditEntry:
        """Atomically update threshold policy, validating constraints and generating audit diff."""
        if isinstance(new_config, dict):
            validated = ConfidenceThresholdConfig(**new_config)
        else:
            validated = new_config

        # Check safeguard deltas
        delta_low = abs(validated.low_threshold - self._current_config.low_threshold)
        delta_high = abs(validated.high_threshold - self._current_config.high_threshold)
        if delta_low > MAX_GRADUAL_STEP_DELTA or delta_high > MAX_GRADUAL_STEP_DELTA:
            logger.warning(
                "Large threshold step detected (low_delta=%.2f, high_delta=%.2f). Ensure this is intentional.",
                delta_low,
                delta_high,
            )

        old_snapshot = self._current_config.model_dump()
        new_snapshot = validated.model_dump()

        diff: dict[str, Any] = {}
        for key in ("low_threshold", "high_threshold", "scholar_queue_threshold", "high_stakes_penalty"):
            if old_snapshot[key] != new_snapshot[key]:
                diff[key] = {"old": old_snapshot[key], "new": new_snapshot[key]}

        self._version += 1
        self._current_config = validated

        entry = ThresholdAuditEntry(
            version=self._version,
            author=author,
            action=AuditAction.UPDATE,
            reason=reason,
            changes=diff,
            snapshot=new_snapshot,
        )
        self._history.append(entry)
        return entry

    def gradual_adjust(
        self,
        target_low: float,
        target_high: float,
        steps: int = 3,
        author: str = "admin",
        reason: str = "Gradual threshold adjustment",
    ) -> list[ThresholdAuditEntry]:
        """Perform gradual stepwise adjustment to converge towards target thresholds."""
        if steps < 1:
            raise ValueError("Steps must be at least 1")
        if target_low < MIN_ALLOWED_THRESHOLD or target_low > MAX_ALLOWED_THRESHOLD:
            raise ValueError(f"target_low must be between {MIN_ALLOWED_THRESHOLD} and {MAX_ALLOWED_THRESHOLD}")
        if target_high < target_low + MIN_THRESHOLD_SPREAD:
            raise ValueError(f"target_high must be >= target_low + {MIN_THRESHOLD_SPREAD}")

        entries: list[ThresholdAuditEntry] = []
        cur_low = self._current_config.low_threshold
        cur_high = self._current_config.high_threshold

        for step_idx in range(1, steps + 1):
            fraction = step_idx / float(steps)
            step_low = round(cur_low + (target_low - cur_low) * fraction, 4)
            step_high = round(cur_high + (target_high - cur_high) * fraction, 4)

            # Ensure safeguards on intermediate values
            if step_high < step_low + MIN_THRESHOLD_SPREAD:
                step_high = min(MAX_ALLOWED_THRESHOLD, step_low + MIN_THRESHOLD_SPREAD)

            new_cfg = copy.deepcopy(self._current_config)
            new_cfg.low_threshold = step_low
            new_cfg.high_threshold = step_high

            entry = self.update_config(
                new_cfg,
                author=author,
                reason=f"{reason} (step {step_idx}/{steps})",
            )
            entry.action = AuditAction.GRADUAL_STEP
            entries.append(entry)

        return entries

    def rollback(
        self,
        to_version: int | None = None,
        author: str = "admin",
        reason: str = "Rollback to previous threshold configuration",
    ) -> ThresholdAuditEntry:
        """Roll back to the previous or a specific historic configuration version."""
        if len(self._history) < 2 and to_version is None:
            raise ValueError("Cannot rollback: no previous configuration exists in history")

        target_entry: ThresholdAuditEntry | None = None
        if to_version is not None:
            for entry in self._history:
                if entry.version == to_version:
                    target_entry = entry
                    break
            if not target_entry:
                raise ValueError(f"Configuration version {to_version} not found in audit trail")
        else:
            # Revert to version immediately preceding current
            target_entry = self._history[-2]

        restored_config = ConfidenceThresholdConfig(**target_entry.snapshot)
        self._version += 1
        self._current_config = restored_config

        audit = ThresholdAuditEntry(
            version=self._version,
            author=author,
            action=AuditAction.ROLLBACK,
            reason=f"{reason} (restored version {target_entry.version})",
            changes={"restored_from_version": target_entry.version},
            snapshot=restored_config.model_dump(),
        )
        self._history.append(audit)
        return audit

    def start_ab_experiment(
        self,
        experiment_id: str,
        name: str,
        variant_b_config: ConfidenceThresholdConfig | dict[str, Any],
        traffic_split: float = 0.50,
        author: str = "admin",
    ) -> ABExperimentConfig:
        """Launch an A/B test comparing current baseline (Variant A) against a candidate (Variant B)."""
        if isinstance(variant_b_config, dict):
            b_validated = ConfidenceThresholdConfig(**variant_b_config)
        else:
            b_validated = variant_b_config

        self._ab_experiment = ABExperimentConfig(
            experiment_id=experiment_id,
            name=name,
            traffic_split=traffic_split,
            active=True,
            variant_a=copy.deepcopy(self._current_config),
            variant_b=b_validated,
        )

        self._version += 1
        self._history.append(
            ThresholdAuditEntry(
                version=self._version,
                author=author,
                action=AuditAction.AB_START,
                reason=f"Started A/B experiment: {name} ({experiment_id})",
                changes={"experiment_id": experiment_id, "split": traffic_split},
                snapshot=self._current_config.model_dump(),
            )
        )
        return self._ab_experiment

    def stop_ab_experiment(
        self,
        promote_variant_b: bool = False,
        author: str = "admin",
    ) -> None:
        """Halt active A/B experiment, optionally promoting candidate Variant B as default."""
        if not self._ab_experiment:
            return

        exp_name = self._ab_experiment.name
        exp_b = self._ab_experiment.variant_b
        self._ab_experiment.active = False
        self._ab_experiment = None

        if promote_variant_b:
            self.update_config(
                exp_b,
                author=author,
                reason=f"Promoted Variant B from concluded experiment: {exp_name}",
            )
        else:
            self._version += 1
            self._history.append(
                ThresholdAuditEntry(
                    version=self._version,
                    author=author,
                    action=AuditAction.AB_STOP,
                    reason=f"Stopped A/B experiment: {exp_name} without promotion",
                    changes={"experiment_name": exp_name},
                    snapshot=self._current_config.model_dump(),
                )
            )

    def generate_recommendations(self) -> list[dict[str, Any]]:
        """Generate empirical threshold tuning recommendations based on collected performance data."""
        recs: list[dict[str, Any]] = []
        metrics = self._metrics

        if metrics.total_evaluations < 5:
            recs.append(
                {
                    "type": "insufficient_data",
                    "message": f"Only {metrics.total_evaluations} turn(s) recorded. Gather more traffic for recommendations.",
                    "action": "none",
                }
            )
            return recs

        # 1. High Abstain Rate Recommendation
        if metrics.abstain_rate > 0.35:
            recs.append(
                {
                    "type": "high_abstain_rate",
                    "severity": "medium",
                    "message": (
                        f"Abstain rate is high ({metrics.abstain_rate * 100:.1f}%). "
                        f"Consider lowering low_threshold from {self._current_config.low_threshold:.2f} "
                        f"to {max(MIN_ALLOWED_THRESHOLD, self._current_config.low_threshold - 0.05):.2f} "
                        "to provide answers to more users while retaining safety."
                    ),
                    "action": "lower_low_threshold",
                    "suggested_value": max(MIN_ALLOWED_THRESHOLD, self._current_config.low_threshold - 0.05),
                }
            )

        # 2. Low User Satisfaction on High Confidence
        if metrics.satisfaction_rate < 0.70 and metrics.confident_count > 0:
            recs.append(
                {
                    "type": "low_satisfaction_quality",
                    "severity": "high",
                    "message": (
                        f"User satisfaction is low ({metrics.satisfaction_rate * 100:.1f}%). "
                        f"Consider raising high_threshold from {self._current_config.high_threshold:.2f} "
                        f"to {min(MAX_ALLOWED_THRESHOLD, self._current_config.high_threshold + 0.05):.2f} "
                        "to ensure only thoroughly corroborated responses bypass uncertainty notes."
                    ),
                    "action": "raise_high_threshold",
                    "suggested_value": min(MAX_ALLOWED_THRESHOLD, self._current_config.high_threshold + 0.05),
                }
            )

        # 3. High Scholar Queue Load
        if metrics.scholar_queue_count > 0 and (metrics.scholar_queue_count / metrics.total_evaluations) > 0.40:
            recs.append(
                {
                    "type": "heavy_scholar_queue",
                    "severity": "low",
                    "message": (
                        "Scholar queue intake exceeds 40% of queries. Verify if scholar_queue_threshold "
                        f"({self._current_config.scholar_queue_threshold:.2f}) should be tightened to focus on core rulings."
                    ),
                    "action": "adjust_scholar_threshold",
                }
            )

        # 4. A/B Experiment Comparison
        if self._ab_experiment and self._ab_experiment.active:
            ma = self._ab_experiment.metrics_a
            mb = self._ab_experiment.metrics_b
            if ma.total_evaluations >= 10 and mb.total_evaluations >= 10:
                sat_diff = mb.satisfaction_rate - ma.satisfaction_rate
                if sat_diff > 0.05:
                    recs.append(
                        {
                            "type": "ab_test_winner",
                            "severity": "high",
                            "message": (
                                f"Variant B demonstrates higher user satisfaction (+{sat_diff * 100:.1f}%) "
                                "over Variant A. Recommend concluding experiment and promoting Variant B."
                            ),
                            "action": "promote_variant_b",
                        }
                    )

        if not recs:
            recs.append(
                {
                    "type": "optimal",
                    "message": "Current confidence thresholds are operating within target equilibrium parameters.",
                    "action": "none",
                }
            )

        return recs


# ---------------------------------------------------------------------------
# Global Singleton
# ---------------------------------------------------------------------------

_tuning_manager: ConfidenceTuningManager | None = None


def get_tuning_manager() -> ConfidenceTuningManager:
    """Return the global runtime confidence tuning manager instance."""
    global _tuning_manager
    if _tuning_manager is None:
        _tuning_manager = ConfidenceTuningManager()
    return _tuning_manager


def reset_tuning_manager(config: ConfidenceThresholdConfig | None = None) -> ConfidenceTuningManager:
    """Reset the singleton instance (primarily used for unit test isolation)."""
    global _tuning_manager
    _tuning_manager = ConfidenceTuningManager(config)
    return _tuning_manager
