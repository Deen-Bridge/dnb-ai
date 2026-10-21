"""Tests for Stellar claimable balance lifecycle explainer (#563)."""

from __future__ import annotations

from datetime import UTC, datetime

from stellar_claimable_balance_explainer import explain_claimable_balance


def test_explains_unconditional_and_utc_time_predicates() -> None:
    now = datetime(2026, 10, 21, 8, 0, 0, tzinfo=UTC)
    rec = {
        "id": "00000000" + "a" * 64,
        "sponsor": "G" + "A" * 55,
        "asset": "USDC:" + "G" + "B" * 55,
        "amount": "100.0000000",
        "claimants": [
            {"destination": "G" + "C" * 55, "predicate": {"unconditional": True}},
            {
                "destination": "G" + "D" * 55,
                "predicate": {"abs_before": "2026-10-20T00:00:00Z"},
            },
        ],
    }
    res = explain_claimable_balance(rec, now_utc=now)
    assert res.amount == "100.0000000"
    assert res.claimants[0].is_currently_claimable is True
    assert res.claimants[1].predicate_type == "before_utc"
    assert res.claimants[1].is_currently_claimable is False
