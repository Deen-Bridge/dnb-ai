"""FastAPI router for cross-session factual consistency.

Exposes endpoints to test consistency, query factual claims, inspect alerts,
and monitor knowledge base coherence.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Query
from pydantic import BaseModel, Field

from consistency.alerts import get_alert_manager
from consistency.core_anchors import CORE_POSITIONS, IKHTILAF_MAP
from consistency.enforcer import get_consistency_enforcer
from consistency.models import (
    CoherenceMetrics,
    ConsistencyCheckResult,
)

router = APIRouter(prefix="/consistency", tags=["consistency"])


class CheckConsistencyRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=15000, description="Response text to evaluate")
    prompt: str = Field("", max_length=5000, description="User question prompt")
    chat_id: str | None = Field(None, max_length=128)
    user_id: str | None = Field(None, max_length=128)
    session_id: str | None = Field(None, max_length=128)
    madhhab: str | None = Field(None, max_length=50)


class IndexClaimsRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=15000)
    chat_id: str | None = Field(None, max_length=128)
    user_id: str | None = Field(None, max_length=128)
    session_id: str | None = Field(None, max_length=128)


@router.post("/check", response_model=ConsistencyCheckResult)
async def check_consistency(body: CheckConsistencyRequest) -> ConsistencyCheckResult:
    """Evaluate response text against past session claims and core knowledge anchors."""
    enforcer = get_consistency_enforcer()
    return await enforcer.evaluate_response(
        response_text=body.text,
        prompt=body.prompt,
        chat_id=body.chat_id,
        user_id=body.user_id,
        session_id=body.session_id,
        madhhab=body.madhhab,
    )


@router.post("/index")
async def index_claims_endpoint(body: IndexClaimsRequest) -> dict[str, Any]:
    """Manually extract and persist factual claims into the store."""
    enforcer = get_consistency_enforcer()
    claims = await enforcer.index_claims(
        text=body.text,
        chat_id=body.chat_id,
        user_id=body.user_id,
        session_id=body.session_id,
    )
    return {
        "status": "ok",
        "claims_indexed": len(claims),
        "claims": [c.model_dump() for c in claims],
    }


@router.get("/claims")
async def list_claims(
    topic: str | None = Query(None, description="Filter by topic"),
    entity: str | None = Query(None, description="Filter by entity"),
    user_id: str | None = Query(None, description="Filter by user ID"),
    session_id: str | None = Query(None, description="Filter by session ID"),
    limit: int = Query(50, ge=1, le=500),
) -> dict[str, Any]:
    """List indexed factual claims with flexible filters."""
    enforcer = get_consistency_enforcer()
    store = enforcer.store

    if user_id:
        claims = await store.get_claims_by_user(user_id, limit=limit)
    elif session_id:
        claims = await store.get_claims_by_session(session_id)
    elif entity:
        claims = await store.get_claims_by_entity(entity, limit=limit)
    elif topic:
        claims = await store.get_claims_by_topic(topic, limit=limit)
    else:
        claims = await store.get_all_claims(limit=limit)

    return {
        "total": len(claims),
        "claims": [c.model_dump() for c in claims],
    }


@router.get("/alerts")
async def list_alerts(
    severity: str | None = Query(None, description="Filter by severity: critical, high, medium, low"),
    limit: int = Query(50, ge=1, le=200),
) -> dict[str, Any]:
    """Retrieve recent consistency alerts."""
    manager = get_alert_manager()
    alerts = manager.get_alerts(limit=limit, severity=severity)
    return {
        "total": len(alerts),
        "alerts": [a.model_dump() for a in alerts],
    }


@router.get("/coherence", response_model=CoherenceMetrics)
async def get_coherence_metrics() -> CoherenceMetrics:
    """Get aggregate knowledge base coherence and drift metrics."""
    enforcer = get_consistency_enforcer()
    total_claims = await enforcer.store.count_claims()
    return get_alert_manager().get_metrics(total_claims_in_store=total_claims)


@router.get("/core-positions")
async def get_core_knowledge_anchors() -> dict[str, Any]:
    """Inspect the canonical core knowledge anchors and ikhtilaf maps."""
    return {
        "core_positions": {k: v.model_dump() for k, v in CORE_POSITIONS.items()},
        "ikhtilaf_map": IKHTILAF_MAP,
    }


@router.delete("/claims/{user_id}")
async def delete_user_claims_endpoint(user_id: str) -> dict[str, Any]:
    """Delete all indexed factual claims associated with a user."""
    enforcer = get_consistency_enforcer()
    deleted = await enforcer.store.delete_user_claims(user_id)
    return {
        "status": "ok",
        "user_id": user_id,
        "deleted_count": deleted,
    }
