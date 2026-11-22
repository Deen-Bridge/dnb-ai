"""Fail-closed checks for read-only Stellar account state."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from stellar_network_boundary import validate_stellar_network_context


@dataclass(frozen=True)
class StellarContractState:
    """Sanitized verdict for whether a Stellar account result is trustworthy."""

    ok: bool
    network: str
    ledger_sequence: int | None
    freshness_age_seconds: int | None
    reason: str | None = None


def validate_contract_request(
    *,
    network: str,
    configured_network: str,
    asset_code: str,
    asset_issuer: str,
    expected_issuer: str,
    public_key_valid: bool,
) -> str | None:
    """Validate network, asset identity, and public-key evidence before I/O."""
    boundary = validate_stellar_network_context(
        requested_network=network,
        configured_network=configured_network,
    )
    if not boundary.ok:
        return boundary.error_code
    if not public_key_valid:
        return "INVALID_PUBLIC_KEY_EVIDENCE"
    if asset_code.strip().upper() != "USDC" or asset_issuer.strip() != expected_issuer:
        return "UNVERIFIED_ASSET_ISSUER"
    return None


def evaluate_provider_freshness(
    provider_state: dict[str, Any],
    *,
    now: datetime,
    max_age_seconds: int = 30,
) -> StellarContractState:
    """Reject absent, malformed, future-dated, or stale Horizon ledger evidence."""
    network = str(provider_state.get("network", "")).strip().lower()
    if network not in {"testnet", "public"}:
        return StellarContractState(False, network, None, None, "INVALID_PROVIDER_NETWORK")
    if max_age_seconds <= 0:
        raise ValueError("max_age_seconds must be positive")
    if now.tzinfo is None:
        return StellarContractState(False, network, None, None, "INVALID_CHECK_TIME")
    try:
        ledger = int(provider_state["history_latest_ledger"])
        closed_at = datetime.fromisoformat(
            str(provider_state["history_latest_ledger_closed_at"]).replace("Z", "+00:00")
        )
    except (KeyError, TypeError, ValueError):
        return StellarContractState(False, network, None, None, "MISSING_LEDGER_EVIDENCE")
    if isinstance(provider_state["history_latest_ledger"], bool) or ledger <= 0 or closed_at.tzinfo is None:
        return StellarContractState(False, network, ledger, None, "INVALID_LEDGER_EVIDENCE")
    age = int((now.astimezone(UTC) - closed_at.astimezone(UTC)).total_seconds())
    if age < 0:
        return StellarContractState(False, network, ledger, age, "FUTURE_LEDGER_EVIDENCE")
    if age > max_age_seconds:
        return StellarContractState(False, network, ledger, age, "STALE_LEDGER_EVIDENCE")
    return StellarContractState(True, network, ledger, age)
