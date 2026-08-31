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

# ---------------------------------------------------------------------------
# LLM-as-judge evaluation framework
# ---------------------------------------------------------------------------

LLM_JUDGE_ENABLED = os.getenv("LLM_JUDGE_ENABLED", "true").lower() == "true"
LLM_JUDGE_MODELS = [
    m.strip()
    for m in os.getenv("LLM_JUDGE_MODELS", "gpt-4o,claude-3-5-sonnet,command-r-plus").split(",")
    if m.strip()
]
LLM_JUDGE_TEMPERATURE = float(os.getenv("LLM_JUDGE_TEMPERATURE", "0"))
LLM_JUDGE_MAX_CONSENSUS_DELTA = float(os.getenv("LLM_JUDGE_MAX_CONSENSUS_DELTA", "0.5"))

JUDGE_DIMENSIONS: tuple[str, ...] = (
    "accuracy",
    "completeness",
    "appropriateness",
    "citation_quality",
    "tone",
    "theological_correctness",
)

JUDGE_RUBRIC: dict[str, str] = {
    "accuracy": "factual consistency with Qur'an, Sunnah, and orthodox Islamic sources",
    "completeness": "covers the question's essential points without material omission",
    "appropriateness": "suitable for the questioner's context, sensitivity, and intent",
    "citation_quality": "citations are specific, verifiable, and correctly attributed",
    "tone": "respectful, compassionate, and scholarly register",
    "theological_correctness": "alignment with mainstream Aqidah and fiqh methodology",
}

class LLMJudgeScore(BaseModel):
    dimension: str
    score: float = Field(..., ge=0, le=5)
    rationale: str = ""

class LLMJudgeEvaluation(BaseModel):
    item_id: str | None = None
    question: str
    answer: str
    reference_answer: str | None = None
    model: str
    scores: dict[str, float]
    overall: float = Field(..., ge=0, le=5)
    verdict: str
    reasoning: str
    duration_ms: int = 0

class LLMJudgeRequest(BaseModel):
    question: str = Field(..., min_length=1)
    answer: str = Field(..., min_length=1)
    reference_answer: str | None = None
    item_id: str | None = None
    models: list[str] | None = None

class LLMJudgeResponse(BaseModel):
    evaluations: list[LLMJudgeEvaluation]
    aggregate: dict[str, Any]

def build_judge_prompt(question: str, answer: str, reference_answer: str | None = None) -> str:
    rubric_lines = "\n".join(f"- {dim}: {desc}" for dim, desc in JUDGE_RUBRIC.items())
    reference = reference_answer or "(no reference answer provided)"
    return f"""You are a senior Islamic scholar and rigorous LLM-as-judge evaluator.
Evaluate the candidate answer to the Islamic question below.

Question:
{question}

Candidate answer:
{answer}

Reference answer:
{reference}

Rubric:
{rubric_lines}

First reason step by step about accuracy, completeness, appropriateness,
citation quality, tone, and theological correctness. Then return ONLY JSON:
{{
  "reasoning": "your chain-of-thought reasoning",
  "scores": {{"accuracy": 0.0, "completeness": 0.0, "appropriateness": 0.0,
             "citation_quality": 0.0, "tone": 0.0, "theological_correctness": 0.0}},
  "overall": 0.0,
  "verdict": "approve|correct|reject"
}}
Scores are 0-5. Prefer 'approve' for a strong, complete, correct answer;
'correct' when important points are missing or a citation is weak; 'reject'
for materially wrong or misleading content."""

def _extract_json_object(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:].strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        try:
            parsed = json.loads(text[start : end + 1])
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
    raise ValueError("LLM judge response did not contain a JSON object")

async def call_judge_model(model: str, prompt: str) -> str:
    api_url = os.getenv("LLM_JUDGE_API_URL")
    if api_url:
        return await _call_openai_compatible(model, prompt, api_url)
    if model.startswith("gpt-") or model.startswith("o1") or model.startswith("o3"):
        return await _call_openai(model, prompt)
    if model.startswith("claude"):
        return await _call_anthropic(model, prompt)
    raise RuntimeError(f"No LLM judge provider configured for model {model!r}")

def _post_json_sync(url: str, headers: dict[str, str], payload: dict[str, Any]) -> dict[str, Any]:
    import urllib.request

    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.loads(response.read().decode("utf-8"))

async def _call_openai_compatible(model: str, prompt: str, api_url: str) -> str:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": LLM_JUDGE_TEMPERATURE,
    }
    headers = {"Authorization": f"Bearer {os.getenv('LLM_JUDGE_API_KEY', '')}"}
    data = await asyncio.to_thread(_post_json_sync, api_url, headers, payload)
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(f"Unexpected judge response shape: {data}") from exc

async def _call_openai(model: str, prompt: str) -> str:
    import openai

    client = openai.AsyncOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_API_BASE") or None,
    )
    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=LLM_JUDGE_TEMPERATURE,
    )
    return response.choices[0].message.content or ""

async def _call_anthropic(model: str, prompt: str) -> str:
    import anthropic

    client = anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    response = await client.messages.create(
        model=model,
        max_tokens=2048,
        temperature=LLM_JUDGE_TEMPERATURE,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(block.text for block in response.content if block.type == "text")

async def run_single_judge(
    question: str,
    answer: str,
    model: str,
    reference_answer: str | None = None,
    item_id: str | None = None,
) -> LLMJudgeEvaluation:
    import time

    prompt = build_judge_prompt(question, answer, reference_answer)
    start = time.monotonic()
    raw = await call_judge_model(model, prompt)
    data = _extract_json_object(raw)
    scores = {dim: float(data.get("scores", {}).get(dim, 0.0)) for dim in JUDGE_DIMENSIONS}
    reasoning = str(data.get("reasoning", ""))
    overall = float(data.get("overall", sum(scores.values()) / len(scores) if scores else 0.0))
    verdict = str(data.get("verdict", "reject"))
    duration_ms = int((time.monotonic() - start) * 1000)
    return LLMJudgeEvaluation(
        item_id=item_id,
        question=question,
        answer=answer,
        reference_answer=reference_answer,
        model=model,
        scores=scores,
        overall=overall,
        verdict=verdict,
        reasoning=reasoning,
        duration_ms=duration_ms,
    )

async def run_judge_ensemble(request: LLMJudgeRequest) -> LLMJudgeResponse:
    models = request.models or LLM_JUDGE_MODELS
    evaluations = await asyncio.gather(
        *(
            run_single_judge(
                question=request.question,
                answer=request.answer,
                reference_answer=request.reference_answer,
                item_id=request.item_id,
                model=model,
            )
            for model in models
        )
    )
    dimension_scores = {dim: [] for dim in JUDGE_DIMENSIONS}
    for evaluation in evaluations:
        for dim in JUDGE_DIMENSIONS:
            dimension_scores[dim].append(evaluation.scores.get(dim, 0.0))
    overall_scores = [evaluation.overall for evaluation in evaluations]
    disagreement = {
        dim: (max(values) - min(values))
        for dim, values in dimension_scores.items()
        if values
    }
    aggregate: dict[str, Any] = {
        "mean_scores": {dim: sum(values) / len(values) for dim, values in dimension_scores.items() if values},
        "mean_overall": sum(overall_scores) / len(overall_scores) if overall_scores else 0.0,
        "consensus": max(overall_scores) - min(overall_scores) if overall_scores else 0.0,
        "within_consensus_delta": (max(overall_scores) - min(overall_scores) <= LLM_JUDGE_MAX_CONSENSUS_DELTA)
        if overall_scores
        else False,
        "disagreement": disagreement,
    }
    return LLMJudgeResponse(evaluations=evaluations, aggregate=aggregate)

@router.get("/review/judge/models")
async def judge_models(x_review_token: str | None = Header(None)) -> dict[str, Any]:
    require_reviewer(x_review_token)
    return {
        "enabled": LLM_JUDGE_ENABLED,
        "models": LLM_JUDGE_MODELS,
        "max_consensus_delta": LLM_JUDGE_MAX_CONSENSUS_DELTA,
    }

@router.post("/review/judge", response_model=LLMJudgeResponse)
async def judge_response(
    request: LLMJudgeRequest,
    x_review_token: str | None = Header(None),
) -> LLMJudgeResponse:
    """Run the LLM-judge ensemble on a candidate answer."""
    require_reviewer(x_review_token)
    if not LLM_JUDGE_ENABLED:
        raise APIException(
            status_code=503,
            detail="LLM judge is disabled.",
            hint="Set LLM_JUDGE_ENABLED=true to enable automatic evaluation.",
        )
    return await run_judge_ensemble(request)



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
