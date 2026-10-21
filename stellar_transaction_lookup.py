"""Read-only Stellar transaction lookup by hash (#558)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable

SUPPORTED_NETWORKS: frozenset[str] = frozenset({"testnet", "public"})


@dataclass(frozen=True)
class StellarTransactionLookupResult:
    """Evidence-based read-only transaction lookup response."""

    lookup_outcome: str
    network: str
    tx_hash: str | None
    tx_status: str | None
    ledger: int | None
    created_at: str | None
    fee_charged: str | None
    memo_type: str | None
    error_code: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def is_valid_tx_hash(tx_hash: str) -> bool:
    """Validate 64-character hexadecimal Stellar transaction hash."""
    cleaned = tx_hash.strip()
    if len(cleaned) != 64:
        return False
    hex_chars = set("0123456789abcdefABCDEF")
    return all(ch in hex_chars for ch in cleaned)


def lookup_stellar_transaction(
    *,
    tx_hash: str,
    network: str,
    provider_fetcher: Callable[[str, str], dict[str, Any] | None],
) -> StellarTransactionLookupResult:
    """Look up a Stellar transaction by hash without inferring failure on not-found."""
    net = network.strip().lower()
    if net not in SUPPORTED_NETWORKS:
        return StellarTransactionLookupResult(
            lookup_outcome="invalid_input",
            network=net,
            tx_hash=None,
            tx_status=None,
            ledger=None,
            created_at=None,
            fee_charged=None,
            memo_type=None,
            error_code="UNSUPPORTED_NETWORK",
        )

    cleaned_hash = tx_hash.strip().lower()
    if not is_valid_tx_hash(cleaned_hash):
        return StellarTransactionLookupResult(
            lookup_outcome="invalid_input",
            network=net,
            tx_hash=None,
            tx_status=None,
            ledger=None,
            created_at=None,
            fee_charged=None,
            memo_type=None,
            error_code="MALFORMED_TX_HASH",
        )

    try:
        raw = provider_fetcher(cleaned_hash, net)
    except TimeoutError:
        return StellarTransactionLookupResult(
            lookup_outcome="timeout",
            network=net,
            tx_hash=cleaned_hash,
            tx_status="unknown",
            ledger=None,
            created_at=None,
            fee_charged=None,
            memo_type=None,
            error_code="PROVIDER_TIMEOUT",
        )
    except Exception:
        return StellarTransactionLookupResult(
            lookup_outcome="provider_error",
            network=net,
            tx_hash=cleaned_hash,
            tx_status="unknown",
            ledger=None,
            created_at=None,
            fee_charged=None,
            memo_type=None,
            error_code="PROVIDER_ERROR",
        )

    if raw is None:
        return StellarTransactionLookupResult(
            lookup_outcome="unknown_not_found",
            network=net,
            tx_hash=cleaned_hash,
            tx_status="unknown",
            ledger=None,
            created_at=None,
            fee_charged=None,
            memo_type=None,
            error_code=None,
        )

    succeeded = raw.get("successful")
    tx_status = (
        "successful"
        if succeeded is True
        else ("failed" if succeeded is False else "unknown")
    )
    ledger = (
        int(raw["ledger"]) if "ledger" in raw and raw["ledger"] is not None else None
    )
    created_at = str(raw["created_at"]) if raw.get("created_at") else None
    fee_charged = (
        str(raw["fee_charged"]) if raw.get("fee_charged") is not None else None
    )
    memo_type = str(raw["memo_type"]) if raw.get("memo_type") else None

    return StellarTransactionLookupResult(
        lookup_outcome="found",
        network=net,
        tx_hash=cleaned_hash,
        tx_status=tx_status,
        ledger=ledger,
        created_at=created_at,
        fee_charged=fee_charged,
        memo_type=memo_type,
    )
