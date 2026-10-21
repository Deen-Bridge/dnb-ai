"""Tests for Soroban invocation stages guide (#567)."""

from __future__ import annotations

from soroban_invocation_stages import get_soroban_invocation_guide


def test_stages_separate_read_only_simulation_from_wallet_signing_and_submission() -> None:
    guide = get_soroban_invocation_guide()
    stages = guide["stages"]
    assert len(stages) == 5
    by_name = {s["stage_name"]: s for s in stages}
    assert by_name["simulate"]["is_read_only"] is True
    assert by_name["simulate"]["ai_service_permitted"] is True
    assert by_name["sign"]["ai_service_permitted"] is False
    assert by_name["submit"]["ai_service_permitted"] is False
    assert by_name["submit"]["mutates_ledger"] is True
