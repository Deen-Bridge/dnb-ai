"""Scholar-review endpoints — the human end of the abstention loop.

A low-confidence religious answer is queued by ``main.py``; this router is where
a qualified reviewer sees the queue and records a verdict. An approved or
corrected answer is then fed back into the knowledge base so the same question
is answered better next time.

Access
------
These endpoints expose real user questions and answers awaiting vetting, so they
are **closed by default**: without ``SCHOLAR_REVIEW_TOKEN`` set they return 503
rather than serving the queue to anyone who finds the route. When it is set,
every request must present it as ``X-Review-Token``.

Feeding approved answers back
-----------------------------
Two existing sinks, no third pipeline:

1. The semantic cache (#27) — a scholar-vetted answer is exactly what a cache
   should be replaying.
2. A JSONL export at ``REVIEW_EXPORT_PATH`` in an eval-case shape, for the eval
   set (#16) and the feedback loop (#43). When those land with a fixed schema,
   ``export_reviewed_item`` is the one function to adapt.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import secrets
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Header, Query
from pydantic import BaseModel, Field, model_validator

from errors import APIException
from review_store import (
    AlreadyReviewedError,
    ReviewItem,
    Verdict,
    get_review_store,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["scholar-review"])

SCHOLAR_REVIEW_TOKEN = os.getenv("SCHOLAR_REVIEW_TOKEN", "")
REVIEW_EXPORT_PATH = os.getenv("REVIEW_EXPORT_PATH", "data/review/reviewed.jsonl")


def require_reviewer(token: str | None) -> None:
    """Authorize a reviewer request, or raise.

    Closed by default: an unset token disables the endpoints entirely rather
    than leaving the queue readable by anyone who guesses the path.
    """
    if not SCHOLAR_REVIEW_TOKEN:
        raise APIException(
            status_code=503,
            detail="Scholar review is not configured. Set SCHOLAR_REVIEW_TOKEN to enable the reviewer endpoints.",
            hint="Configure the SCHOLAR_REVIEW_TOKEN environment variable in server configuration to enable access to the scholar review queue.",
        )
    # Constant-time comparison: a timing-distinguishable check on a shared
    # secret is worth avoiding even on a low-traffic endpoint.
    if not token or not secrets.compare_digest(token, SCHOLAR_REVIEW_TOKEN):
        raise APIException(
            status_code=401,
            detail="A valid X-Review-Token header is required.",
            hint="Provide the configured review token in the 'X-Review-Token' HTTP header (e.g., 'X-Review-Token: <token>').",
        )


# ---------------------------------------------------------------------------
# Feedback into the knowledge base
# ---------------------------------------------------------------------------


def export_reviewed_item(item: ReviewItem, export_path: str | None = None) -> bool:
    """Append a vetted answer to the eval/feedback export. Returns True if written.

    Rejected answers are exported too, with ``verdict: "reject"`` — a wrong
    answer a scholar caught is one of the most valuable eval cases there is.
    Corrections carry the reviewer's answer as the expected one.
    """
    path = Path(export_path or REVIEW_EXPORT_PATH)
    record: dict[str, Any] = {
        "id": item.id,
        "question": item.question,
        "answer": item.final_answer or item.answer,
        "original_answer": item.answer,
        "verdict": item.verdict.value if item.verdict else None,
        "status": item.status.value,
        "confidence": item.confidence,
        "signals": item.signals,
        "reviewer": item.reviewer,
        "reviewed_at": item.reviewed_at,
        "source": "scholar_review",
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return True
    except OSError as exc:
        logger.warning("Could not write review export to %s: %s", path, exc)
        return False


async def enqueue_for_review(
    question: str,
    answer: str,
    score: float,
    band: str,
    signals: dict[str, float] | None = None,
    chat_id: str | None = None,
) -> ReviewItem:
    """Persist a low-confidence religious answer for a scholar to vet.

    The *original* answer is stored, not the abstention message the user saw —
    the reviewer needs to judge what the model actually produced.
    """
    item = ReviewItem(
        question=question,
        answer=answer,
        confidence=score,
        band=band,
        signals=signals or {},
        chat_id=chat_id,
    )
    return await get_review_store().add(item)


def cache_reviewed_answer(item: ReviewItem) -> bool:
    """Put a scholar-approved answer into the semantic cache (#27).

    Best-effort: embedding needs a live API, and a cache write must never be the
    reason a reviewer's verdict fails to record.
    """
    answer = item.final_answer
    if not answer:
        return False
    try:
        from semantic_cache import (
            SEMANTIC_CACHE_ENABLED,
            embed_text,
            get_cache,
            normalize_text,
        )

        if not SEMANTIC_CACHE_ENABLED:
            return False
        embedding = embed_text(normalize_text(item.question))
        get_cache().put(embedding, answer, item.chat_id or item.id, [])
        return True
    except Exception as exc:  # noqa: BLE001 - never fail a verdict over a cache write
        logger.warning("Could not cache reviewed answer %s: %s", item.id, exc)
        return False


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class VerdictRequest(BaseModel):
    verdict: Verdict = Field(..., description="approve, correct, or reject")
    corrected_answer: str | None = Field(None, description="Required when the verdict is 'correct'")
    reviewer: str | None = Field(None, description="Reviewer's name or identifier")
    note: str | None = Field(None, description="Optional note for the record")

    @model_validator(mode="after")
    def correction_requires_an_answer(self) -> VerdictRequest:
        if self.verdict is Verdict.CORRECT and not (self.corrected_answer or "").strip():
            raise ValueError("corrected_answer is required when verdict is 'correct'")
        if self.verdict is not Verdict.CORRECT and self.corrected_answer:
            raise ValueError("corrected_answer is only accepted when verdict is 'correct'")
        return self


class VerdictResponse(BaseModel):
    item: ReviewItem
    cached: bool = Field(..., description="Answer was written to the semantic cache")
    exported: bool = Field(..., description="Answer was appended to the eval export")


class PendingResponse(BaseModel):
    count: int
    items: list[ReviewItem]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/review/pending", response_model=PendingResponse)
async def list_pending(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    x_review_token: str | None = Header(None),
) -> PendingResponse:
    """Answers awaiting a scholar's verdict, longest-waiting first."""
    require_reviewer(x_review_token)
    items = await get_review_store().list_pending(limit=limit, offset=offset)
    return PendingResponse(count=len(items), items=items)


@router.get("/review/reviewed", response_model=PendingResponse)
async def list_reviewed(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    x_review_token: str | None = Header(None),
) -> PendingResponse:
    """Answers that already carry a verdict, most recently decided first."""
    require_reviewer(x_review_token)
    items = await get_review_store().list_reviewed(limit=limit, offset=offset)
    return PendingResponse(count=len(items), items=items)


@router.get("/review/stats")
async def review_stats(x_review_token: str | None = Header(None)) -> dict[str, Any]:
    """Queue depth and whether the queue is actually durable."""
    require_reviewer(x_review_token)
    return await get_review_store().stats()


@router.get("/review/{item_id}", response_model=ReviewItem)
async def get_item(item_id: str, x_review_token: str | None = Header(None)) -> ReviewItem:
    require_reviewer(x_review_token)
    item = await get_review_store().get(item_id)
    if item is None:
        raise APIException(
            status_code=404,
            detail=f"No review item {item_id}.",
            hint="Verify the item_id UUID. Use 'GET /review/pending' to list all currently queued items awaiting review.",
        )
    return item


@router.post("/review/{item_id}/verdict", response_model=VerdictResponse)
async def record_verdict(
    item_id: str,
    request: VerdictRequest,
    x_review_token: str | None = Header(None),
) -> VerdictResponse:
    """Record a scholar's verdict and feed a vetted answer back into the system."""
    require_reviewer(x_review_token)
    store = get_review_store()

    # The store claims the pending-to-reviewed transition atomically, so two
    # concurrent verdicts cannot both win; a lost verdict would defeat the
    # point of having a scholar decide.
    try:
        item = await store.record_verdict(
            item_id,
            request.verdict,
            corrected_answer=request.corrected_answer,
            reviewer=request.reviewer,
            reviewer_note=request.note,
        )
    except AlreadyReviewedError as exc:
        raise APIException(
            status_code=409,
            detail=f"Item {item_id} was already reviewed ({exc.item.status.value}); it cannot be decided twice.",
            hint="This review item has already received a verdict and is finalized. Use 'GET /review/reviewed' to view decided items.",
        ) from exc
    if item is None:
        raise APIException(
            status_code=404,
            detail=f"No review item {item_id}.",
            hint="Verify the item_id exists in the review queue. Use 'GET /review/pending' to check pending items.",
        )

    # Both sinks block — one on disk, one on the embedding API — so they run
    # off the event loop rather than stalling every other in-flight request.
    cached = await asyncio.to_thread(cache_reviewed_answer, item)
    exported = await asyncio.to_thread(export_reviewed_item, item)
    return VerdictResponse(item=item, cached=cached, exported=exported)
