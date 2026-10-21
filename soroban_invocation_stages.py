"""Document Soroban invocation stages and simulation-versus-submission (#567)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class SorobanInvocationStage:
    """Single stage in the Soroban contract invocation lifecycle."""

    stage_order: int
    stage_name: str
    actor: str  # "client_or_ai_read_only", "user_wallet", "stellar_network"
    is_read_only: bool
    mutates_ledger: bool
    ai_service_permitted: bool
    description: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


SOROBAN_LIFECYCLE_STAGES: tuple[SorobanInvocationStage, ...] = (
    SorobanInvocationStage(
        stage_order=1,
        stage_name="prepare",
        actor="client_or_ai_read_only",
        is_read_only=True,
        mutates_ledger=False,
        ai_service_permitted=True,
        description="Construct unsigned contract invocation parameters in memory.",
    ),
    SorobanInvocationStage(
        stage_order=2,
        stage_name="simulate",
        actor="client_or_ai_read_only",
        is_read_only=True,
        mutates_ledger=False,
        ai_service_permitted=True,
        description="Run read-only Soroban RPC simulateTransaction to estimate footprint and fees.",
    ),
    SorobanInvocationStage(
        stage_order=3,
        stage_name="sign",
        actor="user_wallet",
        is_read_only=False,
        mutates_ledger=False,
        ai_service_permitted=False,
        description="User wallet authorizes and signs the assembled transaction with private keys.",
    ),
    SorobanInvocationStage(
        stage_order=4,
        stage_name="submit",
        actor="user_wallet",
        is_read_only=False,
        mutates_ledger=True,
        ai_service_permitted=False,
        description="Submit the signed transaction to Soroban RPC / Stellar network.",
    ),
    SorobanInvocationStage(
        stage_order=5,
        stage_name="confirm",
        actor="client_or_ai_read_only",
        is_read_only=True,
        mutates_ledger=False,
        ai_service_permitted=True,
        description="Poll read-only transaction status until ledger inclusion is confirmed.",
    ),
)


def get_soroban_invocation_guide() -> dict[str, Any]:
    """Return the complete Soroban lifecycle guide stopping before wallet signing."""
    return {
        "ai_boundary_policy": (
            "The AI service may explain and inspect read-only simulation or confirmation "
            "data, but never signs or submits transactions on behalf of a user."
        ),
        "stages": [s.to_dict() for s in SOROBAN_LIFECYCLE_STAGES],
    }
