"""Read-only Horizon account summary tool (#553)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable

SUPPORTED_NETWORKS: frozenset[str] = frozenset({"testnet", "public"})


@dataclass(frozen=True)
class TrustlineSummary:
    """Summary of a non-native asset trustline."""

    asset_code: str
    asset_issuer: str
    balance: str
    limit: str
    is_authorized: bool


@dataclass(frozen=True)
class HorizonAccountSummaryResult:
    """Read-only summary of a public Stellar account."""

    status: str  # "ok", "not_found", "invalid_input", "provider_error"
    network: str
    account_id: str | None
    native_balance: str | None
    sequence: str | None
    thresholds: dict[str, int] | None
    trustlines: tuple[TrustlineSummary, ...]
    last_modified_ledger: int | None
    last_modified_time: str | None
    error_code: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def is_valid_public_account_id(value: str) -> bool:
    """Validate that value is a 56-character Stellar G... public key and never a secret seed."""
    cleaned = value.strip()
    if len(cleaned) != 56 or not cleaned.startswith("G"):
        return False
    base32_alphabet = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ234567")
    return all(ch in base32_alphabet for ch in cleaned)


def summarize_horizon_account(
    *,
    account_id: str,
    network: str,
    provider_fetcher: Callable[[str, str], dict[str, Any] | None],
) -> HorizonAccountSummaryResult:
    """Fetch and summarize public Horizon account state without accepting secrets."""
    net = network.strip().lower()
    if net not in SUPPORTED_NETWORKS:
        return HorizonAccountSummaryResult(
            status="invalid_input",
            network=net,
            account_id=None,
            native_balance=None,
            sequence=None,
            thresholds=None,
            trustlines=(),
            last_modified_ledger=None,
            last_modified_time=None,
            error_code="UNSUPPORTED_NETWORK",
        )
    cleaned_acct = account_id.strip()
    if cleaned_acct.startswith("S"):
        return HorizonAccountSummaryResult(
            status="invalid_input",
            network=net,
            account_id=None,
            native_balance=None,
            sequence=None,
            thresholds=None,
            trustlines=(),
            last_modified_ledger=None,
            last_modified_time=None,
            error_code="SECRET_KEY_FORBIDDEN",
        )
    if not is_valid_public_account_id(cleaned_acct):
        return HorizonAccountSummaryResult(
            status="invalid_input",
            network=net,
            account_id=None,
            native_balance=None,
            sequence=None,
            thresholds=None,
            trustlines=(),
            last_modified_ledger=None,
            last_modified_time=None,
            error_code="MALFORMED_PUBLIC_ACCOUNT",
        )

    try:
        raw = provider_fetcher(cleaned_acct, net)
    except Exception:
        return HorizonAccountSummaryResult(
            status="provider_error",
            network=net,
            account_id=cleaned_acct,
            native_balance=None,
            sequence=None,
            thresholds=None,
            trustlines=(),
            last_modified_ledger=None,
            last_modified_time=None,
            error_code="HORIZON_PROVIDER_ERROR",
        )

    if raw is None:
        return HorizonAccountSummaryResult(
            status="not_found",
            network=net,
            account_id=cleaned_acct,
            native_balance=None,
            sequence=None,
            thresholds=None,
            trustlines=(),
            last_modified_ledger=None,
            last_modified_time=None,
            error_code="ACCOUNT_NOT_FOUND",
        )

    native_balance = "0.0000000"
    trustlines: list[TrustlineSummary] = []
    for bal in raw.get("balances", []):
        if bal.get("asset_type") == "native":
            native_balance = str(bal.get("balance", "0.0000000"))
        else:
            trustlines.append(
                TrustlineSummary(
                    asset_code=str(bal.get("asset_code", "")),
                    asset_issuer=str(bal.get("asset_issuer", "")),
                    balance=str(bal.get("balance", "0.0000000")),
                    limit=str(bal.get("limit", "0")),
                    is_authorized=bool(bal.get("is_authorized", False)),
                )
            )

    thresh_raw = raw.get("thresholds", {})
    thresholds = {
        "low_threshold": int(thresh_raw.get("low_threshold", 0)),
        "med_threshold": int(thresh_raw.get("med_threshold", 0)),
        "high_threshold": int(thresh_raw.get("high_threshold", 0)),
    }

    return HorizonAccountSummaryResult(
        status="ok",
        network=net,
        account_id=cleaned_acct,
        native_balance=native_balance,
        sequence=str(raw.get("sequence", "0")),
        thresholds=thresholds,
        trustlines=tuple(trustlines),
        last_modified_ledger=int(raw.get("last_modified_ledger", 0)),
        last_modified_time=str(raw.get("last_modified_time", "")),
    )
