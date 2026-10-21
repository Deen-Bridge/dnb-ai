"""FastAPI Router for Runtime Confidence Threshold Tuning & Administration (#270).

Exposes administrative REST API endpoints to:
- Inspect and dynamically update confidence thresholds at runtime
- Configure per-topic Islamic category thresholds (fiqh, hadith, aqidah, tafsir)
- Configure source-specific confidence & retrieval score requirements
- Execute gradual stepwise adjustments to prevent traffic disruption
- Review audit trail records with versioned diffs and full snapshots
- Roll back to previous threshold states instantly
- Orchestrate A/B testing experiments and monitor performance metrics
- Generate empirical tuning recommendations based on user feedback and distribution
"""

from __future__ import annotations

import logging
import os
import secrets
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

from confidence_tuning import (
    AuditAction,
    ConfidenceThresholdConfig,
    SourceThresholdConfig,
    TopicThresholdConfig,
    get_tuning_manager,
)
from errors import APIException

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin/confidence", tags=["confidence-admin"])

_admin_header = APIKeyHeader(name="X-Admin-Token", auto_error=False)


async def verify_admin_auth(token: str | None = Depends(_admin_header)) -> None:
    """Validate X-Admin-Token header against ADMIN_TOKEN environment variable."""
    admin_token = os.getenv("ADMIN_TOKEN", "")
    if not admin_token:
        raise APIException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ADMIN_TOKEN is not configured on this server.",
            hint="Set the ADMIN_TOKEN environment variable in server configuration to enable admin management routes.",
        )
    if not token or not secrets.compare_digest(token.encode("utf-8"), admin_token.encode("utf-8")):
        raise APIException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid or missing admin token.",
            hint="Include the configured admin secret in the 'X-Admin-Token' request header.",
        )


# ---------------------------------------------------------------------------
# Request Models
# ---------------------------------------------------------------------------


class UpdateConfigRequest(BaseModel):
    config: ConfidenceThresholdConfig
    reason: str = Field("Runtime configuration update", min_length=1, max_length=500)
    author: str = Field("admin", min_length=1, max_length=100)


class GradualAdjustRequest(BaseModel):
    target_low: float = Field(..., ge=0.05, le=0.95)
    target_high: float = Field(..., ge=0.05, le=0.95)
    steps: int = Field(3, ge=1, le=10)
    reason: str = Field("Gradual threshold adjustment", min_length=1, max_length=500)
    author: str = Field("admin", min_length=1, max_length=100)


class RollbackRequest(BaseModel):
    to_version: int | None = Field(default=None, ge=1)
    reason: str = Field("Rollback to previous configuration", min_length=1, max_length=500)
    author: str = Field("admin", min_length=1, max_length=100)


class StartABTestRequest(BaseModel):
    experiment_id: str = Field(..., min_length=1, max_length=128)
    name: str = Field(..., min_length=1, max_length=200)
    traffic_split: float = Field(0.50, ge=0.01, le=0.99)
    variant_b: ConfidenceThresholdConfig
    author: str = Field("admin", min_length=1, max_length=100)


class StopABTestRequest(BaseModel):
    promote_variant_b: bool = False
    author: str = Field("admin", min_length=1, max_length=100)


class TurnFeedbackRequest(BaseModel):
    score: float = Field(..., ge=0.0, le=1.0)
    band: str = Field(..., pattern="^(abstain|uncertain|confident)$")
    is_religious: bool = False
    topic: str | None = None
    feedback: bool | None = None
    request_key: str | None = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/config", dependencies=[Depends(verify_admin_auth)])
async def get_current_configuration() -> dict[str, Any]:
    """Retrieve active runtime confidence configuration, version, and overrides."""
    mgr = get_tuning_manager()
    return {
        "version": mgr.version,
        "config": mgr.current_config.model_dump(),
        "ab_experiment_active": bool(mgr.ab_experiment and mgr.ab_experiment.active),
    }


@router.put("/config", dependencies=[Depends(verify_admin_auth)])
async def update_configuration(body: UpdateConfigRequest) -> dict[str, Any]:
    """Dynamically update runtime confidence thresholds with audit logging."""
    mgr = get_tuning_manager()
    try:
        entry = mgr.update_config(body.config, author=body.author, reason=body.reason)
        return {
            "status": "success",
            "version": mgr.version,
            "audit_entry": entry.model_dump(),
        }
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e


@router.put("/topics/{topic}", dependencies=[Depends(verify_admin_auth)])
async def update_topic_threshold(
    topic: str,
    body: TopicThresholdConfig,
    author: str = "admin",
) -> dict[str, Any]:
    """Set custom confidence thresholds for a specific Islamic knowledge topic."""
    mgr = get_tuning_manager()
    current_cfg = mgr.current_config.model_copy(deep=True)
    current_cfg.topics[topic.lower()] = body
    entry = mgr.update_config(
        current_cfg,
        author=author,
        reason=f"Updated topic threshold for '{topic}'",
    )
    return {
        "status": "success",
        "topic": topic.lower(),
        "version": mgr.version,
        "audit_entry": entry.model_dump(),
    }


@router.delete("/topics/{topic}", dependencies=[Depends(verify_admin_auth)])
async def remove_topic_threshold(topic: str, author: str = "admin") -> dict[str, Any]:
    """Remove custom confidence threshold for an Islamic topic, reverting to global default."""
    mgr = get_tuning_manager()
    current_cfg = mgr.current_config.model_copy(deep=True)
    if topic.lower() not in current_cfg.topics:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Topic '{topic}' does not have a custom configuration",
        )
    del current_cfg.topics[topic.lower()]
    entry = mgr.update_config(
        current_cfg,
        author=author,
        reason=f"Removed topic override for '{topic}'",
    )
    return {
        "status": "success",
        "removed_topic": topic.lower(),
        "version": mgr.version,
        "audit_entry": entry.model_dump(),
    }


@router.put("/sources/{source}", dependencies=[Depends(verify_admin_auth)])
async def update_source_threshold(
    source: str,
    body: SourceThresholdConfig,
    author: str = "admin",
) -> dict[str, Any]:
    """Set source-specific minimum confidence and retrieval score requirements."""
    mgr = get_tuning_manager()
    current_cfg = mgr.current_config.model_copy(deep=True)
    current_cfg.sources[source.lower()] = body
    entry = mgr.update_config(
        current_cfg,
        author=author,
        reason=f"Updated source requirements for '{source}'",
    )
    return {
        "status": "success",
        "source": source.lower(),
        "version": mgr.version,
        "audit_entry": entry.model_dump(),
    }


@router.delete("/sources/{source}", dependencies=[Depends(verify_admin_auth)])
async def remove_source_threshold(source: str, author: str = "admin") -> dict[str, Any]:
    """Remove source-specific requirement override."""
    mgr = get_tuning_manager()
    current_cfg = mgr.current_config.model_copy(deep=True)
    if source.lower() not in current_cfg.sources:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Source '{source}' does not have a custom configuration",
        )
    del current_cfg.sources[source.lower()]
    entry = mgr.update_config(
        current_cfg,
        author=author,
        reason=f"Removed source override for '{source}'",
    )
    return {
        "status": "success",
        "removed_source": source.lower(),
        "version": mgr.version,
        "audit_entry": entry.model_dump(),
    }


@router.post("/gradual-adjust", dependencies=[Depends(verify_admin_auth)])
async def gradual_adjust(body: GradualAdjustRequest) -> dict[str, Any]:
    """Stepwise adjustment towards target thresholds to prevent jarring production shifts."""
    mgr = get_tuning_manager()
    try:
        entries = mgr.gradual_adjust(
            target_low=body.target_low,
            target_high=body.target_high,
            steps=body.steps,
            author=body.author,
            reason=body.reason,
        )
        return {
            "status": "success",
            "steps_applied": len(entries),
            "final_version": mgr.version,
            "entries": [e.model_dump() for e in entries],
        }
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e


@router.post("/rollback", dependencies=[Depends(verify_admin_auth)])
async def rollback_configuration(body: RollbackRequest) -> dict[str, Any]:
    """Roll back to a previous or specific historic threshold configuration."""
    mgr = get_tuning_manager()
    try:
        entry = mgr.rollback(to_version=body.to_version, author=body.author, reason=body.reason)
        return {
            "status": "success",
            "restored_version": mgr.version,
            "audit_entry": entry.model_dump(),
        }
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e


@router.get("/audit-trail", dependencies=[Depends(verify_admin_auth)])
async def get_audit_trail(
    limit: int = Query(50, ge=1, le=500),
    action: AuditAction | None = Query(None),
    author: str | None = Query(None),
) -> dict[str, Any]:
    """Search and filter the threshold audit trail."""
    mgr = get_tuning_manager()
    entries = mgr.get_audit_trail(limit=limit, action=action, author=author)
    return {
        "count": len(entries),
        "audit_trail": [e.model_dump() for e in entries],
    }


@router.post("/ab-test/start", dependencies=[Depends(verify_admin_auth)])
async def start_ab_experiment(body: StartABTestRequest) -> dict[str, Any]:
    """Start an A/B test comparing baseline Variant A against candidate Variant B."""
    mgr = get_tuning_manager()
    try:
        exp = mgr.start_ab_experiment(
            experiment_id=body.experiment_id,
            name=body.name,
            variant_b_config=body.variant_b,
            traffic_split=body.traffic_split,
            author=body.author,
        )
        return {"status": "success", "experiment": exp.model_dump()}
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e


@router.post("/ab-test/stop", dependencies=[Depends(verify_admin_auth)])
async def stop_ab_experiment(body: StopABTestRequest) -> dict[str, Any]:
    """Halt active A/B test, optionally promoting Variant B to active configuration."""
    mgr = get_tuning_manager()
    if not mgr.ab_experiment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No active A/B experiment found to stop",
        )
    mgr.stop_ab_experiment(promote_variant_b=body.promote_variant_b, author=body.author)
    return {
        "status": "success",
        "promoted_variant_b": body.promote_variant_b,
        "current_version": mgr.version,
    }


@router.get("/ab-test", dependencies=[Depends(verify_admin_auth)])
async def get_ab_test_status() -> dict[str, Any]:
    """Inspect active A/B experiment configuration and comparative performance metrics."""
    mgr = get_tuning_manager()
    if not mgr.ab_experiment:
        return {"active": False, "experiment": None}
    return {"active": True, "experiment": mgr.ab_experiment.model_dump()}


@router.get("/metrics", dependencies=[Depends(verify_admin_auth)])
async def get_performance_metrics() -> dict[str, Any]:
    """Get confidence evaluation metrics, distributions, and feedback rates."""
    mgr = get_tuning_manager()
    m = mgr.metrics
    return {
        "global": {
            "total_evaluations": m.total_evaluations,
            "abstain_count": m.abstain_count,
            "uncertain_count": m.uncertain_count,
            "confident_count": m.confident_count,
            "scholar_queue_count": m.scholar_queue_count,
            "positive_feedback": m.positive_feedback,
            "negative_feedback": m.negative_feedback,
            "abstain_rate": m.abstain_rate,
            "uncertain_rate": m.uncertain_rate,
            "confident_rate": m.confident_rate,
            "satisfaction_rate": m.satisfaction_rate,
        },
        "topics": {
            topic: {
                "total": tm.total_evaluations,
                "abstain_rate": tm.abstain_rate,
                "confident_rate": tm.confident_rate,
                "satisfaction_rate": tm.satisfaction_rate,
            }
            for topic, tm in mgr.topic_metrics.items()
        },
    }


@router.get("/recommendations", dependencies=[Depends(verify_admin_auth)])
async def get_tuning_recommendations() -> dict[str, Any]:
    """Retrieve data-driven recommendations for threshold tuning."""
    mgr = get_tuning_manager()
    return {"recommendations": mgr.generate_recommendations()}


@router.post("/feedback", dependencies=[Depends(verify_admin_auth)])
async def record_turn_feedback(body: TurnFeedbackRequest) -> dict[str, str]:
    """Record response evaluation telemetry and user feedback for tuning data."""
    mgr = get_tuning_manager()
    mgr.record_turn(
        score=body.score,
        band=body.band,
        is_religious=body.is_religious,
        topic=body.topic,
        feedback=body.feedback,
        request_key=body.request_key,
    )
    return {"status": "recorded"}
