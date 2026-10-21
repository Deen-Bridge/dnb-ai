"""Tests for explicit safe fallbacks on timeout, provider error, and missing evidence (#530)."""

from __future__ import annotations

from safe_fallbacks import build_safe_ai_response


def test_timeout_and_provider_error_are_retryable_and_never_fabricate_answers() -> None:
    t = build_safe_ai_response("provider_timeout")
    assert t.retryable is True
    assert t.answer is None

    e = build_safe_ai_response("provider_error")
    assert e.retryable is True
    assert e.answer is None


def test_no_evidence_and_partial_evidence_states_are_distinct() -> None:
    none_res = build_safe_ai_response("no_evidence", answer="should_be_discarded", evidence_count=0)
    assert none_res.status == "no_evidence"
    assert none_res.answer is None
    assert none_res.retryable is False

    partial_res = build_safe_ai_response("partial_evidence", answer="Sourced summary", evidence_count=1)
    assert partial_res.status == "partial_evidence"
    assert partial_res.answer == "Sourced summary"
