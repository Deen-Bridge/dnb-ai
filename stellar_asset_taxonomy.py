"""Separate anchor off-chain representations from native and issued Stellar assets (#573)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class StellarAssetClassification:
    """Terminology-safe classification of native, issued, and anchor-mediated assets."""

    category: str  # "native_xlm", "issued_stellar_asset", "anchor_offchain_representation"
    stellar_asset_code: str | None
    stellar_asset_issuer: str | None
    anchor_name: str | None
    external_currency: str | None
    is_settled_on_stellar_ledger: bool
    terminology_guidance: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def classify_asset_representation(
    *,
    category: str,
    stellar_asset_code: str | None = None,
    stellar_asset_issuer: str | None = None,
    anchor_name: str | None = None,
    external_currency: str | None = None,
    has_verified_onchain_trustline_balance: bool = False,
) -> StellarAssetClassification:
    """Prevent conflating off-chain anchor balances with on-chain Stellar assets."""
    cat = category.strip().lower()
    if cat == "native":
        return StellarAssetClassification(
            category="native_xlm",
            stellar_asset_code="XLM",
            stellar_asset_issuer=None,
            anchor_name=None,
            external_currency=None,
            is_settled_on_stellar_ledger=True,
            terminology_guidance="Native XLM is held directly in a Stellar account entry.",
        )

    if cat == "issued":
        if not stellar_asset_code or not stellar_asset_issuer:
            raise ValueError("Issued Stellar assets require both code and issuer")
        return StellarAssetClassification(
            category="issued_stellar_asset",
            stellar_asset_code=stellar_asset_code.strip().upper(),
            stellar_asset_issuer=stellar_asset_issuer.strip(),
            anchor_name=anchor_name.strip() if anchor_name else None,
            external_currency=None,
            is_settled_on_stellar_ledger=True,
            terminology_guidance=(
                f"On-chain issued asset {stellar_asset_code.strip().upper()} is identified "
                f"by its Stellar issuer {stellar_asset_issuer.strip()}."
            ),
        )

    if cat == "anchor":
        if not anchor_name or not external_currency:
            raise ValueError("Anchor representations require anchor_name and external_currency")
        return StellarAssetClassification(
            category="anchor_offchain_representation",
            stellar_asset_code=stellar_asset_code.strip().upper() if stellar_asset_code else None,
            stellar_asset_issuer=stellar_asset_issuer.strip() if stellar_asset_issuer else None,
            anchor_name=anchor_name.strip(),
            external_currency=external_currency.strip().upper(),
            is_settled_on_stellar_ledger=has_verified_onchain_trustline_balance,
            terminology_guidance=(
                f"Off-chain {external_currency.strip().upper()} held or processed by anchor "
                f"'{anchor_name.strip()}' is NOT an on-chain Stellar balance unless settled "
                "into a verified Stellar trustline."
            ),
        )

    raise ValueError(f"Unsupported asset category: {category}")
