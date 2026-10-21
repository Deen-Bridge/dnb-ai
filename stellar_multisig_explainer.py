"""Explain Stellar account multisignature thresholds without signing (#564)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class PublicSignerWeight:
    """Public signer key and weight on a Stellar account."""

    key: str
    weight: int


@dataclass(frozen=True)
class MultisigThresholdExplanation:
    """Read-only summary of account thresholds and signer weights."""

    account_id: str
    low_threshold: int
    med_threshold: int
    high_threshold: int
    total_active_weight: int
    master_key_disabled: bool
    signers: tuple[PublicSignerWeight, ...]
    non_custodial_notice: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def explain_account_multisig(account_data: dict[str, Any]) -> MultisigThresholdExplanation:
    """Summarize thresholds and signer weights strictly from public account data."""
    acct_id = str(account_data.get("id", "")).strip()
    if not acct_id.startswith("G"):
        raise ValueError("Valid public account G-address is required")

    thresh = account_data.get("thresholds", {})
    low = int(thresh.get("low_threshold", 0))
    med = int(thresh.get("med_threshold", 0))
    high = int(thresh.get("high_threshold", 0))

    signers: list[PublicSignerWeight] = []
    master_weight = 0
    for s in account_data.get("signers", []):
        key = str(s.get("key", "")).strip()
        if key.startswith("S"):
            raise ValueError("Secret keys are forbidden in multisig explanations")
        w = int(s.get("weight", 0))
        if key == acct_id:
            master_weight = w
        signers.append(PublicSignerWeight(key=key, weight=w))

    total_weight = sum(s.weight for s in signers if s.weight > 0)
    return MultisigThresholdExplanation(
        account_id=acct_id,
        low_threshold=low,
        med_threshold=med,
        high_threshold=high,
        total_active_weight=total_weight,
        master_key_disabled=(master_weight == 0),
        signers=tuple(signers),
        non_custodial_notice=(
            "This explanation is read-only; the assistant never collects secret keys "
            "or signs multisignature transactions."
        ),
    )
