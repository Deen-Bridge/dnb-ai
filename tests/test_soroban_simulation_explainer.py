"""Tests for Soroban simulation explainer (#565)."""

from __future__ import annotations

import pytest

from soroban_simulation_explainer import (
    TrustedSorobanSimulationOutput,
    explain_soroban_simulation,
)


def test_explains_success_failure_and_restore_without_submitting() -> None:
    ok = explain_soroban_simulation(
        TrustedSorobanSimulationOutput(
            network="testnet",
            is_success=True,
            cpu_instructions=150000,
            memory_bytes=4096,
            min_resource_fee_stroops=5000,
        )
    )
    assert ok.simulation_outcome == "succeeded"
    assert ok.was_submitted is False

    restore = explain_soroban_simulation(
        TrustedSorobanSimulationOutput(
            network="public",
            is_success=False,
            cpu_instructions=50000,
            memory_bytes=1024,
            min_resource_fee_stroops=1000,
            restore_required=True,
        )
    )
    assert restore.simulation_outcome == "restore_needed"

    with pytest.raises(TypeError):
        explain_soroban_simulation({"raw": "untrusted"})  # type: ignore[arg-type]
