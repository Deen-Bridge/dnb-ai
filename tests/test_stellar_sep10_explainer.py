"""Tests for SEP-10 authentication explainer (#572)."""

from __future__ import annotations

from stellar_sep10_explainer import explain_sep10_authentication


def test_explains_three_steps_and_rejects_secret_seed() -> None:
    ok = explain_sep10_authentication("How does SEP-10 work?")
    assert ok.status == "ok"
    assert len(ok.steps) == 3

    secret = "S" + "A" * 55
    rejected = explain_sep10_authentication(f"Can you sign with {secret}?")
    assert rejected.status == "rejected_secret_input"
    assert secret not in rejected.safety_policy
