"""Network-aware 7-decimal Stellar amount formatter with non-custodial secret guard (#629)."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from decimal import ROUND_HALF_UP, Decimal
from typing import Any

PUBLIC_PASSPHRASE = "Public Global Stellar Network ; September 2015"
TESTNET_PASSPHRASE = "Test SDF Network ; September 2015"
STROOPS_PER_UNIT = Decimal("10000000")
_SECRET_KEY_RE = re.compile(r"\bS[A-Z2-7]{55}\b")
_PUBLIC_KEY_RE = re.compile(r"^G[A-Z2-7]{55}$")
_TX_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_SEED_PHRASE_RE = re.compile(r"\b(mnemonic|seed\s+phrase|12\s+words|24\s+words|private\s+key)\b", re.IGNORECASE)
_INJECTION_RE = re.compile(
    r"(ignore\s+(all\s+)?(previous|prior)\s+instructions|system\s*prompt|send\s+funds\s+to)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class StellarWaveCheckResult:
    """Deterministic, network-aware, non-custodial result envelope (#629)."""

    issue_number: int
    operation: str
    network: str
    network_passphrase: str
    valid: bool
    non_custodial: bool
    summary: str
    details: dict[str, Any]
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def contains_secret_material(text: str) -> bool:
    """Return True if text contains a Stellar S-seed or recovery phrase."""
    return bool(_SECRET_KEY_RE.search(text) or _SEED_PHRASE_RE.search(text))


def resolve_network(network: str) -> tuple[str, str]:
    """Normalize network identifier and return (network_name, canonical_passphrase)."""
    cleaned = network.strip().lower()
    if cleaned in ("public", "mainnet", "pubnet"):
        return ("public", PUBLIC_PASSPHRASE)
    if cleaned == "testnet":
        return ("testnet", TESTNET_PASSPHRASE)
    raise ValueError(f"Unsupported Stellar network: {network!r}")


def format_stroops_to_decimal(stroops: int) -> str:
    """Convert integer stroops to a 7-decimal Stellar amount string."""
    if isinstance(stroops, bool) or not isinstance(stroops, int) or stroops < 0:
        raise ValueError("stroops must be a non-negative integer")
    val = (Decimal(stroops) / STROOPS_PER_UNIT).quantize(Decimal("0.0000001"), rounding=ROUND_HALF_UP)
    return f"{val:.7f}"


def format_stellar_amount(
    *,
    network: str = "testnet",
    payload: str = "",
    amount_stroops: int = 10_000_000,
    asset_code: str = "USDC",
    issuer: str = "GBBD47IF6LWK7P7MDEVSCWR7DPUWV3NY3DTQEVFL4NAT4AQH3ZLLFLA5",
    ledger_sequence: int = 1_024,
    tx_hash: str = "a" * 64,
) -> StellarWaveCheckResult:
    """Execute network-aware 7-decimal stellar amount formatter with non-custodial secret guard (#629)."""
    combined = f"{payload} {asset_code} {issuer} {tx_hash}"
    if contains_secret_material(combined):
        raise ValueError("Refusing secret keys or seed phrases in non-custodial Stellar tool")

    net_name, passphrase = resolve_network(network)
    warnings: list[str] = []

    if _INJECTION_RE.search(payload):
        warnings.append("untrusted_metadata_injection_blocked")
        clean_payload = "[REDACTED_UNTRUSTED_STELLAR_CONTENT]"
    else:
        clean_payload = "".join(ch for ch in payload if ch.isprintable()).strip()[:240]

    norm_code = asset_code.strip().upper()
    is_native = norm_code == "XLM"
    issuer_ok = True if is_native else bool(_PUBLIC_KEY_RE.match(issuer.strip()))
    if not issuer_ok:
        warnings.append("invalid_issuer_public_key")

    norm_hash = tx_hash.strip().lower()
    hash_ok = bool(_TX_HASH_RE.match(norm_hash))
    if not hash_ok:
        warnings.append("invalid_transaction_hash")

    ledger_ok = isinstance(ledger_sequence, int) and not isinstance(ledger_sequence, bool) and ledger_sequence > 0
    if not ledger_ok:
        warnings.append("invalid_ledger_sequence")

    try:
        formatted_amount = format_stroops_to_decimal(amount_stroops)
        amount_ok = True
    except ValueError:
        formatted_amount = "0.0000000"
        amount_ok = False
        warnings.append("invalid_amount_stroops")

    if net_name == "testnet":
        warnings.append("testnet_environment_not_real_funds")

    valid = (
        issuer_ok and hash_ok and ledger_ok and amount_ok and ("untrusted_metadata_injection_blocked" not in warnings)
    )

    details: dict[str, Any] = {
        "asset_code": norm_code,
        "issuer": None if is_native else issuer.strip(),
        "formatted_amount": f"{formatted_amount} {norm_code} ({net_name})",
        "amount_stroops": amount_stroops,
        "ledger_sequence": ledger_sequence,
        "tx_hash": norm_hash if hash_ok else "",
        "sanitized_payload": clean_payload,
        "explorer_url": (
            f"https://stellar.expert/explorer/{'public' if net_name == 'public' else 'testnet'}/tx/{norm_hash}"
            if hash_ok
            else ""
        ),
        "arabic_term": "شبكة ستيلر غير الوصائية",
        "swahili_term": "Mtandao wa Stellar usioshikilia funguo za siri",
    }

    return StellarWaveCheckResult(
        issue_number=629,
        operation="format_stellar_amount",
        network=net_name,
        network_passphrase=passphrase,
        valid=valid,
        non_custodial=True,
        summary=f"[{net_name.upper()}] Network-aware 7-decimal Stellar amount formatter with non-custodial secret guard: {formatted_amount} {norm_code}",
        details=details,
        warnings=tuple(warnings),
    )
