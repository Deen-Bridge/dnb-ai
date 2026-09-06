"""FastAPI Router for Experiment Rollback, Canary Deployments, and Feature Flags (#269).

Administrative endpoints for:
- Creating and managing A/B and canary experiments across models, prompts, and configs
- Configuring percentage-based canary rollouts
- Monitoring real-time latency and error rates
- Automated and manual rollback triggers
- Managing feature flags with gradual percentage rollouts and user whitelisting
- Auditing experiment lifecycle state changes
"""

from __future__ import annotations

import logging
import os
import secrets
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

from errors import APIException
from experiment_rollback import (
    ExperimentTargetType,
    ExperimentVariant,
    FeatureFlag,
    RollbackTriggerConfig,
    get_rollback_manager,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin", tags=["experiment-rollback-admin"])

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


class CreateExperimentRequest(BaseModel):
    experiment_id: str = Field(..., min_length=1, max_length=128)
    name: str = Field(..., min_length=1, max_length=200)
    target_type: ExperimentTargetType = ExperimentTargetType.PROMPT
    control: ExperimentVariant
    variants: list[ExperimentVariant] = Field(default_factory=list)
    canary_percentage: float = Field(0.0, ge=0.0, le=100.0)
    triggers: RollbackTriggerConfig | None = None
    author: str = Field("admin", min_length=1, max_length=100)
    reason: str = Field("Initial experiment creation", min_length=1, max_length=500)


class CanaryRolloutRequest(BaseModel):
    percentage: float = Field(..., ge=0.0, le=100.0)
    author: str = Field("admin", min_length=1, max_length=100)
    reason: str = Field("Canary rollout percentage update", min_length=1, max_length=500)


class RollbackExperimentRequest(BaseModel):
    to_version: int | None = Field(default=None, ge=1)
    author: str = Field("admin", min_length=1, max_length=100)
    reason: str = Field("Manual rollback", min_length=1, max_length=500)


class RecordTurnRequest(BaseModel):
    variant_name: str
    latency_ms: float = Field(..., ge=0.0)
    is_error: bool = False
    error_message: str | None = None


class UpdateFeatureFlagRequest(BaseModel):
    enabled: bool | None = None
    rollout_percentage: float | None = Field(default=None, ge=0.0, le=100.0)
    allowed_users: list[str] | None = None
    author: str = Field("admin", min_length=1, max_length=100)
    reason: str = Field("Feature flag update", min_length=1, max_length=500)


# ---------------------------------------------------------------------------
# Experiment Endpoints
# ---------------------------------------------------------------------------


@router.get("/experiments", dependencies=[Depends(verify_admin_auth)])
async def list_experiments() -> dict[str, Any]:
    """List all registered experiments and their health/status."""
    mgr = get_rollback_manager()
    return {"experiments": [e.model_dump() for e in mgr.list_experiments()]}


@router.post("/experiments", dependencies=[Depends(verify_admin_auth)])
async def create_experiment(body: CreateExperimentRequest) -> dict[str, Any]:
    """Create a new experiment with canary rollout and automated rollback triggers."""
    mgr = get_rollback_manager()
    try:
        exp = mgr.create_experiment(
            experiment_id=body.experiment_id,
            name=body.name,
            target_type=body.target_type,
            control=body.control,
            variants=body.variants,
            canary_percentage=body.canary_percentage,
            triggers=body.triggers,
            author=body.author,
            reason=body.reason,
        )
        return {"status": "success", "experiment": exp.model_dump()}
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e


@router.get("/experiments/{experiment_id}", dependencies=[Depends(verify_admin_auth)])
async def get_experiment_details(experiment_id: str) -> dict[str, Any]:
    """Get detailed status, metrics, and version snapshots for an experiment."""
    mgr = get_rollback_manager()
    exp = mgr.get_experiment(experiment_id)
    if not exp:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Experiment '{experiment_id}' not found",
        )
    return {"experiment": exp.model_dump()}


@router.post("/experiments/{experiment_id}/canary", dependencies=[Depends(verify_admin_auth)])
async def update_canary_rollout(
    experiment_id: str,
    body: CanaryRolloutRequest,
) -> dict[str, Any]:
    """Adjust the canary deployment rollout percentage (0 to 100%)."""
    mgr = get_rollback_manager()
    try:
        exp = mgr.set_canary_rollout(
            experiment_id=experiment_id,
            percentage=body.percentage,
            author=body.author,
            reason=body.reason,
        )
        return {"status": "success", "experiment": exp.model_dump()}
    except KeyError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e)) from e


@router.post("/experiments/{experiment_id}/rollback", dependencies=[Depends(verify_admin_auth)])
async def rollback_experiment(
    experiment_id: str,
    body: RollbackExperimentRequest,
) -> dict[str, Any]:
    """Roll back experiment to previous stable version or specific historical version."""
    mgr = get_rollback_manager()
    try:
        exp = mgr.rollback(
            experiment_id=experiment_id,
            to_version=body.to_version,
            author=body.author,
            reason=body.reason,
        )
        return {"status": "success", "experiment": exp.model_dump()}
    except (KeyError, ValueError) as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e


@router.post("/experiments/{experiment_id}/kill", dependencies=[Depends(verify_admin_auth)])
async def kill_experiment(
    experiment_id: str,
    author: str = "admin",
    reason: str = "Emergency kill switch engaged",
) -> dict[str, Any]:
    """Immediately kill-switch experiment to control variant."""
    mgr = get_rollback_manager()
    try:
        exp = mgr.kill_experiment(experiment_id=experiment_id, author=author, reason=reason)
        return {"status": "success", "kill_switch": exp.kill_switch, "experiment_id": experiment_id}
    except KeyError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e)) from e


@router.post("/experiments/{experiment_id}/resume", dependencies=[Depends(verify_admin_auth)])
async def resume_experiment(
    experiment_id: str,
    author: str = "admin",
    reason: str = "Resumed experiment",
) -> dict[str, Any]:
    """Resume a killed or paused experiment."""
    mgr = get_rollback_manager()
    try:
        exp = mgr.resume_experiment(experiment_id=experiment_id, author=author, reason=reason)
        return {"status": "success", "kill_switch": exp.kill_switch, "experiment_id": experiment_id}
    except KeyError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e)) from e


@router.post("/experiments/{experiment_id}/record-turn", dependencies=[Depends(verify_admin_auth)])
async def record_experiment_turn(
    experiment_id: str,
    body: RecordTurnRequest,
) -> dict[str, Any]:
    """Record turn metrics and check for automated rollback triggers."""
    mgr = get_rollback_manager()
    triggered = mgr.record_turn_result(
        experiment_id=experiment_id,
        variant_name=body.variant_name,
        latency_ms=body.latency_ms,
        is_error=body.is_error,
        error_message=body.error_message,
    )
    return {
        "status": "recorded",
        "automated_rollback_triggered": triggered,
    }


@router.get("/experiments/{experiment_id}/metrics", dependencies=[Depends(verify_admin_auth)])
async def get_experiment_metrics(experiment_id: str) -> dict[str, Any]:
    """Get metrics breakdown for all variants inside an experiment."""
    mgr = get_rollback_manager()
    exp = mgr.get_experiment(experiment_id)
    if not exp:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Experiment '{experiment_id}' not found")

    metrics_out = {}
    for vname, m in exp.metrics.items():
        metrics_out[vname] = {
            "request_count": m.request_count,
            "success_count": m.success_count,
            "error_count": m.error_count,
            "error_rate": m.error_rate,
            "consecutive_errors": m.consecutive_errors,
            "avg_latency_ms": m.avg_latency_ms,
            "p95_latency_ms": m.p95_latency_ms,
        }
    return {"experiment_id": experiment_id, "metrics": metrics_out}


@router.get("/experiments-audit-trail", dependencies=[Depends(verify_admin_auth)])
async def get_experiments_audit_trail(
    experiment_id: str | None = Query(None),
    limit: int = Query(50, ge=1, le=500),
) -> dict[str, Any]:
    """Fetch audit history for experiments and rollbacks."""
    mgr = get_rollback_manager()
    entries = mgr.get_audit_trail(experiment_id=experiment_id, limit=limit)
    return {"count": len(entries), "audit_trail": [e.model_dump() for e in entries]}


# ---------------------------------------------------------------------------
# Feature Flag Endpoints
# ---------------------------------------------------------------------------


@router.get("/feature-flags", dependencies=[Depends(verify_admin_auth)])
async def list_feature_flags() -> dict[str, Any]:
    """List all registered feature flags."""
    mgr = get_rollback_manager()
    return {"flags": [f.model_dump() for f in mgr.list_flags()]}


@router.post("/feature-flags", dependencies=[Depends(verify_admin_auth)])
async def create_feature_flag(body: FeatureFlag) -> dict[str, Any]:
    """Register a new feature flag."""
    mgr = get_rollback_manager()
    mgr.register_flag(body)
    return {"status": "success", "flag": body.model_dump()}


@router.put("/feature-flags/{flag_id}", dependencies=[Depends(verify_admin_auth)])
async def update_feature_flag(
    flag_id: str,
    body: UpdateFeatureFlagRequest,
) -> dict[str, Any]:
    """Update feature flag status, rollout percentage, or whitelisted users."""
    mgr = get_rollback_manager()
    try:
        flag = mgr.update_flag(
            flag_id=flag_id,
            enabled=body.enabled,
            rollout_percentage=body.rollout_percentage,
            allowed_users=body.allowed_users,
            author=body.author,
            reason=body.reason,
        )
        return {"status": "success", "flag": flag.model_dump()}
    except KeyError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e)) from e


@router.get("/feature-flags/{flag_id}/evaluate")
async def evaluate_feature_flag(
    flag_id: str,
    session_id: str | None = Query(None),
    role: str | None = Query(None),
) -> dict[str, Any]:
    """Public/Client check to determine if a feature flag is active for a session."""
    mgr = get_rollback_manager()
    flag = mgr.get_flag(flag_id)
    if not flag:
        return {"flag_id": flag_id, "is_enabled": False, "reason": "not_found"}

    enabled = flag.is_enabled_for(user_or_session_id=session_id, role=role)
    return {
        "flag_id": flag_id,
        "is_enabled": enabled,
        "rollout_percentage": flag.rollout_percentage,
    }
