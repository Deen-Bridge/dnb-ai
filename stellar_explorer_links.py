"""Generate Stellar explorer links from verified network and transaction data (#561)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

REVIEWED_EXPLORER_TEMPLATES: dict[str, dict[str, str]] = {
    "stellar_expert": {
        "public": "https://stellar.expert/explorer/public/tx/{tx_hash}",
        "testnet": "https://stellar.expert/explorer/testnet/tx/{tx_hash}",
    },
    "stellarchain": {
        "public": "https://stellarchain.io/transactions/{tx_hash}",
        "testnet": "https://testnet.stellarchain.io/transactions/{tx_hash}",
    },
}


@dataclass(frozen=True)
class VerifiedExplorerLinkResult:
    """Safe, allowlist-backed explorer URL result."""

    ok: bool
    network: str | None
    explorer: str | None
    url: str | None
    error_code: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _is_valid_hex64(value: str) -> bool:
    if len(value) != 64:
        return False
    hex_chars = set("0123456789abcdef")
    return all(ch in hex_chars for ch in value)


def build_verified_explorer_tx_link(
    *,
    network: str,
    tx_hash: str,
    explorer: str = "stellar_expert",
) -> VerifiedExplorerLinkResult:
    """Build an explorer transaction URL strictly from allowlisted templates."""
    net = network.strip().lower()
    exp = explorer.strip().lower()
    if exp not in REVIEWED_EXPLORER_TEMPLATES:
        return VerifiedExplorerLinkResult(
            ok=False,
            network=net,
            explorer=exp,
            url=None,
            error_code="DISALLOWED_EXPLORER_HOST",
        )
    net_map = REVIEWED_EXPLORER_TEMPLATES[exp]
    if net not in net_map:
        return VerifiedExplorerLinkResult(
            ok=False,
            network=net,
            explorer=exp,
            url=None,
            error_code="UNSUPPORTED_NETWORK",
        )
    normalized_hash = tx_hash.strip().lower()
    if not _is_valid_hex64(normalized_hash):
        return VerifiedExplorerLinkResult(
            ok=False,
            network=net,
            explorer=exp,
            url=None,
            error_code="MALFORMED_TX_HASH",
        )
    url = net_map[net].format(tx_hash=normalized_hash)
    return VerifiedExplorerLinkResult(
        ok=True,
        network=net,
        explorer=exp,
        url=url,
    )
