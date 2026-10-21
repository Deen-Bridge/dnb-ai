"""Warn about lookalike USDC assets and verify issuer identity (#555)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class VerifiedAssetAssessment:
    """Result of checking an issued Stellar asset against a reviewed network registry."""

    network: str
    asset_code: str
    asset_issuer: str
    verification_status: str  # "canonical", "lookalike_warning", "unknown_asset"
    is_canonical: bool
    warning_message: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ReviewedAssetIssuerRegistry:
    """Reviewed per-network registry of canonical Stellar asset issuers."""

    def __init__(self, registry: dict[str, dict[str, str]]) -> None:
        self._registry: dict[str, dict[str, str]] = {
            net.lower(): {code.upper(): issuer.strip() for code, issuer in assets.items()}
            for net, assets in registry.items()
        }

    def assess_issued_asset(
        self, *, network: str, asset_code: str, asset_issuer: str
    ) -> VerifiedAssetAssessment:
        net = network.strip().lower()
        code = asset_code.strip().upper()
        issuer = asset_issuer.strip()
        if not net or not code or not issuer.startswith("G") or len(issuer) != 56:
            raise ValueError(
                "Valid network, asset_code, and 56-char G-address issuer are required"
            )

        net_map = self._registry.get(net, {})
        canonical_issuer = net_map.get(code)
        if canonical_issuer is None:
            return VerifiedAssetAssessment(
                network=net,
                asset_code=code,
                asset_issuer=issuer,
                verification_status="unknown_asset",
                is_canonical=False,
                warning_message=(
                    f"Asset {code}:{issuer} is not registered in the reviewed {net} issuer registry."
                ),
            )

        if issuer == canonical_issuer:
            return VerifiedAssetAssessment(
                network=net,
                asset_code=code,
                asset_issuer=issuer,
                verification_status="canonical",
                is_canonical=True,
                warning_message=None,
            )

        return VerifiedAssetAssessment(
            network=net,
            asset_code=code,
            asset_issuer=issuer,
            verification_status="lookalike_warning",
            is_canonical=False,
            warning_message=(
                f"Lookalike asset warning: {code} issued by {issuer} does NOT match the "
                f"reviewed canonical {code} issuer on {net} ({canonical_issuer})."
            ),
        )
