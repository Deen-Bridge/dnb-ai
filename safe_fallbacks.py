"""Explicit, typed safe fallbacks when model providers fail or retrieval evidence is missing (#530)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

FallbackStatus = Literal[
    "provider_timeout",
    "provider_error",
    "no_evidence",
    "partial_evidence",
    "grounded_answer",
]


@dataclass(frozen=True)
class SafeAIResponse:
    status: FallbackStatus
    retryable: bool
    answer: str | None
    user_guidance: str
    evidence_count: int


def build_safe_ai_response(
    status: FallbackStatus,
    answer: str | None = None,
    evidence_count: int = 0,
) -> SafeAIResponse:
    """Return a deterministic, credential-safe response without inventing answers when evidence is absent."""
    if status == "provider_timeout":
        return SafeAIResponse(
            status="provider_timeout",
            retryable=True,
            answer=None,
            user_guidance="The AI provider timed out before a verified answer could be generated. Please retry shortly.",
            evidence_count=evidence_count,
        )
    if status == "provider_error":
        return SafeAIResponse(
            status="provider_error",
            retryable=True,
            answer=None,
            user_guidance="The upstream AI service is temporarily unavailable. No unverified answer was generated.",
            evidence_count=evidence_count,
        )
    if status == "no_evidence" or evidence_count <= 0:
        return SafeAIResponse(
            status="no_evidence",
            retryable=False,
            answer=None,
            user_guidance="No verified primary evidence was found for this query, so the assistant declined to guess.",
            evidence_count=0,
        )
    if status == "partial_evidence":
        return SafeAIResponse(
            status="partial_evidence",
            retryable=False,
            answer=answer,
            user_guidance="Partial sourced evidence was found; please review the cited references for scope limitations.",
            evidence_count=evidence_count,
        )
    return SafeAIResponse(
        status="grounded_answer",
        retryable=False,
        answer=answer,
        user_guidance="Grounded in verified corpus references.",
        evidence_count=evidence_count,
    )
