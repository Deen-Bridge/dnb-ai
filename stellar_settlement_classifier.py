"""Distinguish payment submission from Stellar ledger confirmation (#559)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class StellarSettlementEvaluation:
    """Explicit settlement state derived strictly from ledger evidence."""

    network: str
    settlement_state: str  # "unknown", "pending", "confirmed", "failed"
    is_settled: bool
    ledger: int | None
    is_provider_stale: bool
    explanation: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def classify_payment_settlement(
    *,
    network: str,
    has_frontend_callback: bool = False,
    submitted_tx_hash: str | None = None,
    ledger_lookup_status: str | None = None,
    ledger_tx_successful: bool | None = None,
    ledger_sequence: int | None = None,
    provider_age_seconds: int = 0,
    max_provider_age_seconds: int = 60,
) -> StellarSettlementEvaluation:
    """Classify payment state using only verified ledger inclusion as proof of settlement."""
    net = network.strip().lower()
    if net not in {"testnet", "public"}:
        raise ValueError("network must be 'testnet' or 'public'")

    is_stale = provider_age_seconds > max_provider_age_seconds
    if is_stale:
        return StellarSettlementEvaluation(
            network=net,
            settlement_state="unknown",
            is_settled=False,
            ledger=None,
            is_provider_stale=True,
            explanation=(
                "Provider response is stale; cannot confirm ledger settlement without fresh "
                f"evidence from the {net} network."
            ),
        )

    if (
        ledger_lookup_status == "found"
        and ledger_sequence is not None
        and ledger_sequence > 0
    ):
        if ledger_tx_successful is True:
            return StellarSettlementEvaluation(
                network=net,
                settlement_state="confirmed",
                is_settled=True,
                ledger=ledger_sequence,
                is_provider_stale=False,
                explanation=(
                    f"Transaction is confirmed in {net} ledger {ledger_sequence} with successful status."
                ),
            )
        if ledger_tx_successful is False:
            return StellarSettlementEvaluation(
                network=net,
                settlement_state="failed",
                is_settled=False,
                ledger=ledger_sequence,
                is_provider_stale=False,
                explanation=(
                    f"Transaction was included in {net} ledger {ledger_sequence} but failed execution."
                ),
            )

    if ledger_lookup_status == "pending" or (
        (has_frontend_callback or submitted_tx_hash) and ledger_lookup_status is None
    ):
        return StellarSettlementEvaluation(
            network=net,
            settlement_state="pending",
            is_settled=False,
            ledger=None,
            is_provider_stale=False,
            explanation=(
                "Transaction submission or wallet callback was recorded, but ledger inclusion "
                f"on {net} is still pending verification."
            ),
        )

    return StellarSettlementEvaluation(
        network=net,
        settlement_state="unknown",
        is_settled=False,
        ledger=None,
        is_provider_stale=False,
        explanation=(
            f"No confirmed ledger record found on {net}; a transaction hash or callback alone "
            "does not prove settlement."
        ),
    )
