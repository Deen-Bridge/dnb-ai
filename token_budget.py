"""Per-Request Token Budgeting & Salience-Driven Context Compression (#92).

Enforces a deterministic token ceiling across system scaffold, retrieved
context (tafsir, zakat, purchases), user memory, and conversation history,
assigns a per-intent max_output_tokens cap, compresses low-salience or
duplicate retrieved passages before prompting, and records pre-call budget vs
post-call actual token telemetry.
"""

from __future__ import annotations

import logging
import re
from enum import Enum

from fastapi import APIRouter
from pydantic import BaseModel, Field

from history import estimate_tokens

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/token-budget", tags=["token-budget"])


class PromptIntent(str, Enum):
    factual = "factual"
    fiqh = "fiqh"
    tafsir = "tafsir"
    zakat = "zakat"
    general = "general"


INTENT_OUTPUT_BUDGETS: dict[PromptIntent, int] = {
    PromptIntent.factual: 512,
    PromptIntent.zakat: 1024,
    PromptIntent.general: 1024,
    PromptIntent.fiqh: 1536,
    PromptIntent.tafsir: 2048,
}

DEFAULT_INPUT_CEILING = 6000


class RetrievedBlockInput(BaseModel):
    block_id: str = Field(..., min_length=1, max_length=120)
    source_type: str = Field(default="tafsir", max_length=60)
    text: str = Field(..., min_length=1, max_length=50000)
    salience: float = Field(default=0.5, ge=0.0, le=1.0)


class HistoryTurnInput(BaseModel):
    role: str = Field(default="user", max_length=32)
    content: str = Field(..., min_length=1, max_length=20000)


class TokenBudgetRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=8000)
    intent: PromptIntent = PromptIntent.general
    input_token_ceiling: int = Field(default=DEFAULT_INPUT_CEILING, ge=256, le=32000)
    system_scaffold: str = Field(
        default="You are Deen Bridge AI, an Islamic knowledge assistant.",
        max_length=20000,
    )
    user_memory: str = Field(default="", max_length=10000)
    retrieved_blocks: list[RetrievedBlockInput] = Field(default_factory=list, max_length=100)
    history: list[HistoryTurnInput] = Field(default_factory=list, max_length=200)


class TokenBudgetAllocation(BaseModel):
    intent: PromptIntent
    input_token_ceiling: int
    max_output_tokens: int
    raw_input_tokens: int
    budgeted_input_tokens: int
    tokens_saved: int
    reduction_percent: float
    kept_block_ids: list[str]
    dropped_block_ids: list[str]
    kept_history_turns: int
    dropped_history_turns: int
    assembled_prompt: str


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def compress_passage_to_query(text: str, query: str, max_tokens: int) -> str:
    """Salience-filter sentences in ``text`` against ``query`` terms under ``max_tokens``."""
    cleaned = _normalize_whitespace(text)
    if not cleaned:
        return ""
    if estimate_tokens(cleaned) <= max_tokens:
        return cleaned

    query_terms = {w.lower() for w in re.findall(r"\w{3,}", query)}
    sentences = [s.strip() for s in re.split(r"(?<=[.!؟?])\s+", cleaned) if s.strip()]
    if not sentences:
        return cleaned[: max(4, max_tokens * 4)]

    scored: list[tuple[float, int, str]] = []
    for idx, sent in enumerate(sentences):
        words = {w.lower() for w in re.findall(r"\w{3,}", sent)}
        overlap = len(words & query_terms)
        score = overlap * 2.0 + (1.0 if idx == 0 else 0.0)
        scored.append((score, idx, sent))

    scored.sort(key=lambda item: (-item[0], item[1]))
    selected: list[tuple[int, str]] = []
    for _, idx, sent in scored:
        trial = sorted(selected + [(idx, sent)], key=lambda item: item[0])
        joined = " ".join(s for _, s in trial)
        if estimate_tokens(joined) <= max_tokens:
            selected = trial

    if not selected:
        return sentences[0][: max(4, max_tokens * 4)]

    return " ".join(s for _, s in selected)


def enforce_token_budget(req: TokenBudgetRequest) -> TokenBudgetAllocation:
    max_out = INTENT_OUTPUT_BUDGETS[req.intent]
    raw_parts = (
        [req.system_scaffold, req.user_memory, req.query]
        + [b.text for b in req.retrieved_blocks]
        + [h.content for h in req.history]
    )
    raw_tokens = sum(estimate_tokens(p) for p in raw_parts if p)

    scaffold_text = _normalize_whitespace(req.system_scaffold)
    query_text = _normalize_whitespace(req.query)
    memory_text = compress_passage_to_query(req.user_memory, req.query, max(16, req.input_token_ceiling // 10))

    base_tokens = (
        estimate_tokens(scaffold_text)
        + estimate_tokens(query_text)
        + (estimate_tokens(memory_text) if memory_text else 0)
    )
    remaining = max(0, req.input_token_ceiling - base_tokens - 16)

    seen_texts: set[str] = set()
    sorted_blocks = sorted(req.retrieved_blocks, key=lambda b: -b.salience)
    kept_blocks: list[tuple[str, str]] = []
    dropped_ids: list[str] = []

    retrieval_budget = max(32, int(remaining * 0.65))
    used_retrieval = 0
    for block in sorted_blocks:
        norm = _normalize_whitespace(block.text)
        if norm in seen_texts:
            dropped_ids.append(block.block_id)
            continue
        seen_texts.add(norm)
        avail = retrieval_budget - used_retrieval
        if avail < 12:
            dropped_ids.append(block.block_id)
            continue
        compressed = compress_passage_to_query(norm, req.query, avail)
        tok = estimate_tokens(compressed) if compressed else 0
        if compressed and used_retrieval + tok <= retrieval_budget:
            kept_blocks.append((block.block_id, compressed))
            used_retrieval += tok
        else:
            dropped_ids.append(block.block_id)

    history_budget = max(0, remaining - used_retrieval)
    kept_history: list[HistoryTurnInput] = []
    used_history = 0
    for turn in reversed(req.history):
        tok = estimate_tokens(turn.content)
        if used_history + tok <= history_budget:
            kept_history.append(turn)
            used_history += tok
    kept_history.reverse()
    dropped_history = len(req.history) - len(kept_history)

    sections = [scaffold_text]
    if memory_text:
        sections.append(f"[USER MEMORY]\n{memory_text}")
    if kept_blocks:
        sections.append("[RETRIEVED CONTEXT]\n" + "\n".join(text for _, text in kept_blocks))
    if kept_history:
        sections.append("[HISTORY]\n" + "\n".join(f"{t.role}: {t.content}" for t in kept_history))
    sections.append(f"[QUERY]\n{query_text}")

    assembled = "\n\n".join(sections)
    while estimate_tokens(assembled) > req.input_token_ceiling and len(assembled) > 32:
        assembled = assembled[: int(len(assembled) * 0.9)]

    budgeted_tokens = estimate_tokens(assembled)
    saved = max(0, raw_tokens - budgeted_tokens)
    reduction_pct = round((saved / raw_tokens) * 100.0, 2) if raw_tokens > 0 else 0.0

    return TokenBudgetAllocation(
        intent=req.intent,
        input_token_ceiling=req.input_token_ceiling,
        max_output_tokens=max_out,
        raw_input_tokens=raw_tokens,
        budgeted_input_tokens=budgeted_tokens,
        tokens_saved=saved,
        reduction_percent=reduction_pct,
        kept_block_ids=[bid for bid, _ in kept_blocks],
        dropped_block_ids=dropped_ids,
        kept_history_turns=len(kept_history),
        dropped_history_turns=dropped_history,
        assembled_prompt=assembled,
    )


@router.post("/allocate", response_model=TokenBudgetAllocation)
async def allocate_token_budget_endpoint(body: TokenBudgetRequest) -> TokenBudgetAllocation:
    """Enforce per-request input token ceiling, salience compression, and per-intent output cap."""
    return enforce_token_budget(body)
