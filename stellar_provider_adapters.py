"""Typed Horizon and Soroban provider adapter interfaces (#574)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class ProviderMetadata:
    """Normalized metadata accompanying every provider read response."""

    provider_kind: str  # "horizon" or "soroban_rpc"
    network: str
    observed_at: str
    ledger_sequence: int | None


@dataclass(frozen=True)
class NormalizedProviderResult:
    """Transport-agnostic result returned by Horizon or Soroban adapters."""

    ok: bool
    metadata: ProviderMetadata
    payload: dict[str, Any] | None
    error_code: str | None = None
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class HorizonReadAdapter(Protocol):
    """Read-only Horizon adapter interface."""

    def get_account(self, *, network: str, account_id: str) -> NormalizedProviderResult: ...

    def get_transaction(self, *, network: str, tx_hash: str) -> NormalizedProviderResult: ...


class SorobanRpcReadAdapter(Protocol):
    """Read-only Soroban RPC adapter interface."""

    def simulate_invocation(self, *, network: str, contract_id: str, method: str) -> NormalizedProviderResult: ...


def normalize_provider_exception(
    *,
    provider_kind: str,
    network: str,
    observed_at: str,
    exc: Exception,
) -> NormalizedProviderResult:
    """Normalize transport exceptions into typed internal provider results."""
    meta = ProviderMetadata(
        provider_kind=provider_kind,
        network=network,
        observed_at=observed_at,
        ledger_sequence=None,
    )
    if isinstance(exc, TimeoutError):
        return NormalizedProviderResult(
            ok=False,
            metadata=meta,
            payload=None,
            error_code="PROVIDER_TIMEOUT",
            error_message="Read-only provider request timed out.",
        )
    return NormalizedProviderResult(
        ok=False,
        metadata=meta,
        payload=None,
        error_code="PROVIDER_TRANSPORT_ERROR",
        error_message="External Stellar provider returned an unexpected error.",
    )
