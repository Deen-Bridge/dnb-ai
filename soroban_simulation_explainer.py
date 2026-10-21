"""Explain Soroban simulation results without submitting transactions (#565)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class TrustedSorobanSimulationOutput:
    """Typed simulation payload produced only by the trusted provider adapter."""

    network: str
    is_success: bool
    cpu_instructions: int
    memory_bytes: int
    min_resource_fee_stroops: int
    error_code: str | None = None
    restore_required: bool = False


@dataclass(frozen=True)
class SorobanSimulationExplanation:
    """Safe, read-only user guidance derived from a Soroban simulation."""

    network: str
    simulation_outcome: str  # "succeeded", "restore_needed", "failed"
    cpu_instructions: int
    memory_bytes: int
    min_resource_fee_stroops: int
    guidance: str
    was_submitted: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def explain_soroban_simulation(
    sim: TrustedSorobanSimulationOutput,
) -> SorobanSimulationExplanation:
    """Translate trusted Soroban simulation output into bounded read-only guidance."""
    if not isinstance(sim, TrustedSorobanSimulationOutput):
        raise TypeError("Simulation input must be a TrustedSorobanSimulationOutput instance")

    if sim.restore_required:
        return SorobanSimulationExplanation(
            network=sim.network,
            simulation_outcome="restore_needed",
            cpu_instructions=sim.cpu_instructions,
            memory_bytes=sim.memory_bytes,
            min_resource_fee_stroops=sim.min_resource_fee_stroops,
            guidance=(
                "Simulation indicates archived contract state must be restored before "
                "invocation. No transaction was signed or submitted."
            ),
            was_submitted=False,
        )

    if not sim.is_success:
        code = sim.error_code or "SIMULATION_ERROR"
        return SorobanSimulationExplanation(
            network=sim.network,
            simulation_outcome="failed",
            cpu_instructions=sim.cpu_instructions,
            memory_bytes=sim.memory_bytes,
            min_resource_fee_stroops=sim.min_resource_fee_stroops,
            guidance=(
                f"Soroban simulation failed with code '{code}'. The transaction was not submitted to the network."
            ),
            was_submitted=False,
        )

    return SorobanSimulationExplanation(
        network=sim.network,
        simulation_outcome="succeeded",
        cpu_instructions=sim.cpu_instructions,
        memory_bytes=sim.memory_bytes,
        min_resource_fee_stroops=sim.min_resource_fee_stroops,
        guidance=(
            "Simulation succeeded in read-only mode. Simulation does not submit or "
            "guarantee on-chain execution if ledger state changes."
        ),
        was_submitted=False,
    )
