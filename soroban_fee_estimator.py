"""Explain Soroban resource fees using timestamped RPC estimates (#571)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class SorobanFeeEstimateExplanation:
    """Timestamped, provider-backed Soroban fee estimate explanation."""

    status: str  # "estimated" or "unavailable"
    network: str
    is_estimate: bool
    observed_at: str | None
    base_inclusion_fee_stroops: int | None
    resource_fee_stroops: int | None
    total_estimated_fee_stroops: int | None
    explanation: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def explain_soroban_resource_fee(
    *,
    network: str,
    rpc_estimate: dict[str, Any] | None,
) -> SorobanFeeEstimateExplanation:
    """Format a timestamped Soroban RPC fee estimate without inventing fixed fees."""
    net = network.strip().lower()
    if net not in {"testnet", "public"}:
        raise ValueError("network must be 'testnet' or 'public'")

    if rpc_estimate is None or "resource_fee_stroops" not in rpc_estimate:
        return SorobanFeeEstimateExplanation(
            status="unavailable",
            network=net,
            is_estimate=True,
            observed_at=None,
            base_inclusion_fee_stroops=None,
            resource_fee_stroops=None,
            total_estimated_fee_stroops=None,
            explanation=(
                f"Soroban RPC fee estimate is currently unavailable for {net}; the assistant "
                "does not invent fixed Soroban resource fees."
            ),
        )

    base_fee = int(rpc_estimate.get("base_inclusion_fee_stroops", 100))
    res_fee = int(rpc_estimate["resource_fee_stroops"])
    observed_at = str(rpc_estimate.get("observed_at", "")).strip()
    if not observed_at:
        raise ValueError("rpc_estimate requires a non-empty observed_at timestamp")

    total = base_fee + res_fee
    return SorobanFeeEstimateExplanation(
        status="estimated",
        network=net,
        is_estimate=True,
        observed_at=observed_at,
        base_inclusion_fee_stroops=base_fee,
        resource_fee_stroops=res_fee,
        total_estimated_fee_stroops=total,
        explanation=(
            f"Estimated total Soroban fee on {net} is {total} stroops ({base_fee} inclusion "
            f"+ {res_fee} resource fee) based on RPC simulation observed at {observed_at}."
        ),
    )
