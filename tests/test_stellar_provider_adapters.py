"""Tests for Horizon and Soroban provider adapter interfaces (#574)."""

from __future__ import annotations

from stellar_provider_adapters import (
    NormalizedProviderResult,
    ProviderMetadata,
    normalize_provider_exception,
)


def test_normalizes_timeout_and_transport_exceptions() -> None:
    to_res = normalize_provider_exception(
        provider_kind="horizon",
        network="public",
        observed_at="2026-10-21T08:15:00Z",
        exc=TimeoutError("deadline exceeded"),
    )
    assert to_res.ok is False
    assert to_res.error_code == "PROVIDER_TIMEOUT"

    ok_res = NormalizedProviderResult(
        ok=True,
        metadata=ProviderMetadata(
            provider_kind="soroban_rpc",
            network="testnet",
            observed_at="2026-10-21T08:15:00Z",
            ledger_sequence=777,
        ),
        payload={"cpu_insns": 1000},
    )
    assert ok_res.to_dict()["metadata"]["provider_kind"] == "soroban_rpc"
