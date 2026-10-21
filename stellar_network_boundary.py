"""Require explicit network context on every Stellar lookup (#556)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

NAMED_STELLAR_NETWORKS: frozenset[str] = frozenset({"testnet", "public"})


@dataclass(frozen=True)
class StellarLookupEnvelope:
    """Normalized Stellar tool response preserving explicit network context."""

    ok: bool
    network: str | None
    data: dict[str, Any] | None
    error_code: str | None = None
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_stellar_network_context(
    *,
    requested_network: str | None,
    configured_network: str | None = None,
) -> StellarLookupEnvelope:
    """Validate explicit network selection with no implicit mainnet fallback."""
    if requested_network is None or not requested_network.strip():
        return StellarLookupEnvelope(
            ok=False,
            network=None,
            data=None,
            error_code="MISSING_NETWORK_CONTEXT",
            error_message="Explicit Stellar network ('testnet' or 'public') is required.",
        )

    req = requested_network.strip().lower()
    if req not in NAMED_STELLAR_NETWORKS:
        return StellarLookupEnvelope(
            ok=False,
            network=req,
            data=None,
            error_code="UNSUPPORTED_NETWORK_CONTEXT",
            error_message=f"Unsupported Stellar network '{req}'. Allowed: testnet, public.",
        )

    if configured_network is not None and configured_network.strip():
        cfg = configured_network.strip().lower()
        if cfg != req:
            return StellarLookupEnvelope(
                ok=False,
                network=req,
                data=None,
                error_code="CONTRADICTORY_NETWORK_CONTEXT",
                error_message=(f"Request network '{req}' contradicts configured network '{cfg}'."),
            )

    return StellarLookupEnvelope(
        ok=True,
        network=req,
        data={"network": req},
    )
