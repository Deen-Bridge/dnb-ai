"""Explain Stellar path-payment routes and quoted price impact (#569)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class RouteHopAsset:
    """Single asset hop along a Stellar path-payment route."""

    hop_index: int
    asset_code: str
    asset_issuer: str | None


@dataclass(frozen=True)
class PathPaymentRouteExplanation:
    """Read-only explanation of a verified Stellar path-payment quote."""

    network: str
    source_amount: str
    destination_amount: str
    quoted_at: str
    route_hops: tuple[RouteHopAsset, ...]
    is_quote_only: bool
    slippage_guidance: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def explain_path_payment_quote(
    *,
    network: str,
    source_asset: dict[str, str | None],
    intermediate_hops: list[dict[str, str | None]],
    destination_asset: dict[str, str | None],
    source_amount: str,
    destination_amount: str,
    quoted_at: str,
) -> PathPaymentRouteExplanation:
    """Explain a path-payment route in hop order with quote-vs-settlement disclaimers."""
    all_assets = [source_asset, *intermediate_hops, destination_asset]
    hops: list[RouteHopAsset] = []
    for idx, item in enumerate(all_assets):
        code = str(item.get("asset_code") or "XLM").strip().upper()
        issuer = str(item["asset_issuer"]).strip() if item.get("asset_issuer") and code != "XLM" else None
        hops.append(RouteHopAsset(hop_index=idx, asset_code=code, asset_issuer=issuer))

    return PathPaymentRouteExplanation(
        network=network.strip().lower(),
        source_amount=source_amount.strip(),
        destination_amount=destination_amount.strip(),
        quoted_at=quoted_at.strip(),
        route_hops=tuple(hops),
        is_quote_only=True,
        slippage_guidance=(
            "This path-payment route is an indicative quote observed at quoted_at, not a "
            "submitted or settled payment. On-chain execution requires setting dest_min "
            "or send_max bounds to protect against order-book slippage."
        ),
    )
