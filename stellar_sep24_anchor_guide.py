"""Document SEP-24 anchor deposit and withdrawal flows (#570)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

OFFICIAL_SEP24_DOC_URL = "https://github.com/stellar/stellar-protocol/blob/master/ecosystem/sep-0024.md"


@dataclass(frozen=True)
class Sep24AnchorFlowGuide:
    """Grounded explanation of SEP-24 anchor-mediated deposit and withdrawal flows."""

    flow_type: str  # "deposit" or "withdrawal"
    anchor_domain: str
    asset_code: str
    is_third_party_provider: bool
    steps: tuple[str, ...]
    official_sep_url: str
    security_notice: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def explain_sep24_anchor_flow(*, flow_type: str, anchor_domain: str, asset_code: str) -> Sep24AnchorFlowGuide:
    """Explain SEP-24 interactive flows clearly separating the anchor from the Stellar network."""
    kind = flow_type.strip().lower()
    if kind not in {"deposit", "withdrawal"}:
        raise ValueError("flow_type must be 'deposit' or 'withdrawal'")
    domain = anchor_domain.strip().lower()
    code = asset_code.strip().upper()
    if not domain or not code:
        raise ValueError("anchor_domain and asset_code must be non-empty")

    if kind == "deposit":
        steps = (
            f"1. Authenticate wallet ownership with {domain} via SEP-10 challenge signing.",
            f"2. Complete {domain}'s interactive SEP-24 web form (KYC / off-chain payment details).",
            "3. Poll the anchor's /transaction endpoint until off-chain funds clear.",
            f"4. Verify on-chain Stellar ledger settlement of {code} to the user's trustline.",
        )
    else:
        steps = (
            f"1. Authenticate wallet ownership with {domain} via SEP-10 challenge signing.",
            f"2. Complete {domain}'s interactive SEP-24 withdrawal form for destination details.",
            f"3. User wallet signs and sends the on-chain {code} Stellar payment to the anchor's account.",
            "4. Poll the anchor's /transaction endpoint until off-chain payout completes.",
        )

    return Sep24AnchorFlowGuide(
        flow_type=kind,
        anchor_domain=domain,
        asset_code=code,
        is_third_party_provider=True,
        steps=steps,
        official_sep_url=OFFICIAL_SEP24_DOC_URL,
        security_notice=(
            f"Anchor '{domain}' is an independent third-party service provider, separate "
            "from the Stellar protocol and this AI assistant. Never share secret keys."
        ),
    )
