"""Explain trustline requirements for issued Stellar assets (#554)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

OFFICIAL_TRUSTLINE_DOC_URL = (
    "https://developers.stellar.org/docs/learn/fundamentals/"
    "stellar-data-structures/accounts#trustlines"
)


@dataclass(frozen=True)
class TrustlineExplanation:
    """Grounded explanation of Stellar trustline state and reserve rules."""

    network: str
    asset_type: str  # "native" or "issued"
    asset_code: str
    asset_issuer: str | None
    scenario: str
    can_receive_payment: bool
    subentry_reserve_increase: int
    explanation: str
    network_dependent_facts: tuple[str, ...]
    official_source_url: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def explain_trustline_requirement(
    *,
    network: str,
    asset_code: str,
    asset_issuer: str | None,
    has_trustline: bool = False,
    issuer_auth_required: bool = False,
    is_trustline_authorized: bool = True,
) -> TrustlineExplanation:
    """Build a read-only, non-custodial explanation of trustline behavior."""
    net = network.strip().lower()
    if net not in {"testnet", "public"}:
        raise ValueError("network must be 'testnet' or 'public'")

    code = asset_code.strip().upper()
    if code == "XLM" and not asset_issuer:
        return TrustlineExplanation(
            network=net,
            asset_type="native",
            asset_code="XLM",
            asset_issuer=None,
            scenario="native_asset",
            can_receive_payment=True,
            subentry_reserve_increase=0,
            explanation=(
                "Native XLM does not require a trustline on Stellar; any funded account "
                "meeting the base reserve can hold and receive XLM."
            ),
            network_dependent_facts=(f"Base reserve and account existence on {net} network",),
            official_source_url=OFFICIAL_TRUSTLINE_DOC_URL,
        )

    if not asset_issuer or not asset_issuer.strip().startswith("G"):
        raise ValueError("Issued assets require a valid public issuer G-address")
    issuer = asset_issuer.strip()

    net_facts = (
        f"Issuer account {issuer} configuration on {net}",
        f"Recipient account trustline subentry and base reserve on {net}",
    )

    if not has_trustline:
        return TrustlineExplanation(
            network=net,
            asset_type="issued",
            asset_code=code,
            asset_issuer=issuer,
            scenario="missing_trustline",
            can_receive_payment=False,
            subentry_reserve_increase=1,
            explanation=(
                f"Payments of issued asset {code} ({issuer}) fail with op_no_trust when the "
                "recipient account has not established a trustline. Each trustline adds 1 "
                "subentry to the account's minimum XLM reserve requirement."
            ),
            network_dependent_facts=net_facts,
            official_source_url=OFFICIAL_TRUSTLINE_DOC_URL,
        )

    if issuer_auth_required and not is_trustline_authorized:
        return TrustlineExplanation(
            network=net,
            asset_type="issued",
            asset_code=code,
            asset_issuer=issuer,
            scenario="auth_required_unapproved",
            can_receive_payment=False,
            subentry_reserve_increase=1,
            explanation=(
                f"The recipient account has a trustline for {code} ({issuer}), but the issuer "
                "enforces AUTH_REQUIRED_FLAG and has not yet authorized this trustline, so "
                "incoming payments fail with op_not_authorized."
            ),
            network_dependent_facts=net_facts,
            official_source_url=OFFICIAL_TRUSTLINE_DOC_URL,
        )

    return TrustlineExplanation(
        network=net,
        asset_type="issued",
        asset_code=code,
        asset_issuer=issuer,
        scenario="existing_trustline",
        can_receive_payment=True,
        subentry_reserve_increase=1,
        explanation=(
            f"The account holds an active, authorized trustline for {code} ({issuer}) on {net} "
            "and can receive payments up to its configured trustline limit."
        ),
        network_dependent_facts=net_facts,
        official_source_url=OFFICIAL_TRUSTLINE_DOC_URL,
    )
