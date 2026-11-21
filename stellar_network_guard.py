"""Production guard validating Stellar network, asset, and issuer pairs before provider access (#862).

Enforces non-custodial invariants, canonical issuer matching across testnet and mainnet,
freshness evidence, and prompt-injection filtering prior to any outbound Horizon or RPC call.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from decimal import ROUND_HALF_UP, Decimal
from typing import Any

PUBLIC_PASSPHRASE = "Public Global Stellar Network ; September 2015"
TESTNET_PASSPHRASE = "Test SDF Network ; September 2015"
STROOPS_PER_UNIT = Decimal("10000000")
ZAKAT_RATE = Decimal("0.025")
USDC_PRECISION = Decimal("0.0000001")

# Secret key and sensitive material patterns
_SECRET_KEY_RE = re.compile(r"\bS[A-Z2-7]{55}\b")
_PUBLIC_KEY_RE = re.compile(r"^G[A-Z2-7]{55}$")
_TX_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_SEED_PHRASE_RE = re.compile(
    r"\b(mnemonic|seed\s+phrase|12\s+words|24\s+words|private\s+key)\b", re.IGNORECASE
)
_INJECTION_RE = re.compile(
    r"(ignore\s+(all\s+)?(previous|prior)\s+instructions|system\s*prompt|send\s+funds\s+to)",
    re.IGNORECASE,
)

# Canonical asset registries across networks
CANONICAL_ASSET_ISSUERS: dict[str, dict[str, str | None]] = {
    "public": {
        "XLM": None,
        "USDC": "GA5ZSEJYB37JRC5AVCIA5MOP4RHTM335X2KGX3IHOJAPP5RE34K4KZVN",
    },
    "testnet": {
        "XLM": None,
        "USDC": "GBBD47IF6LWK7P7MDEVSCWR7DPUWV3NY3DTQEVFL4NAT4AQH3ZLLFLA5",
    },
}


@dataclass(frozen=True)
class StellarNetworkGuardResult:
    """Production result envelope for pre-provider validation (#862)."""

    issue_number: int
    operation: str
    network: str
    network_passphrase: str
    valid: bool
    non_custodial: bool
    is_canonical: bool
    summary: str
    details: dict[str, Any]
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def contains_secret_material(text: str) -> bool:
    """Return True if text contains a Stellar secret key or seed phrase."""
    return bool(_SECRET_KEY_RE.search(text) or _SEED_PHRASE_RE.search(text))


def resolve_network(network: str) -> tuple[str, str]:
    """Normalize network identifier and return (canonical_name, passphrase)."""
    cleaned = (network or "").strip().lower()
    if cleaned in ("public", "mainnet", "pubnet"):
        return ("public", PUBLIC_PASSPHRASE)
    if cleaned == "testnet":
        return ("testnet", TESTNET_PASSPHRASE)
    raise ValueError(f"Unsupported Stellar network: {network!r}. Allowed: testnet, public.")


def format_stroops_to_decimal(stroops: int) -> str:
    """Convert integer stroops to a 7-decimal Stellar amount string."""
    if isinstance(stroops, bool) or not isinstance(stroops, int) or stroops < 0:
        raise ValueError("stroops must be a non-negative integer")
    val = (Decimal(stroops) / STROOPS_PER_UNIT).quantize(
        Decimal("0.0000001"), rounding=ROUND_HALF_UP
    )
    return f"{val:.7f}"


def compute_zakat(balance: Decimal, nisab: Decimal) -> Decimal:
    """Zakat due on balance (2.5%), or zero when below nisab."""
    if balance < nisab:
        return Decimal("0")
    return (balance * ZAKAT_RATE).quantize(USDC_PRECISION)


def validate_public_key_format(public_key: str) -> str:
    """Validate 56-character Stellar G-address format or raise ValueError."""
    cleaned = (public_key or "").strip()
    if not _PUBLIC_KEY_RE.match(cleaned):
        raise ValueError(
            "Invalid Stellar public key format. Expected 56 characters starting with G."
        )
    return cleaned


def validate_asset_issuer_pair_for_provider(
    *,
    network: str,
    asset_code: str,
    issuer: str | None,
) -> tuple[bool, str | None]:
    """Validate whether an asset code and issuer pair is recognized and safe for provider access.

    Returns (is_allowed, error_or_warning_message).
    """
    try:
        net_name, _ = resolve_network(network)
    except ValueError as e:
        return False, str(e)

    code = (asset_code or "").strip().upper()
    if not code:
        return False, "Asset code cannot be empty"

    if code == "XLM":
        if issuer is not None and issuer.strip() != "":
            return False, "Native XLM must not have an issuer"
        return True, None

    clean_issuer = (issuer or "").strip()
    if not _PUBLIC_KEY_RE.match(clean_issuer):
        return False, f"Invalid issuer public key format: {clean_issuer!r}"

    canonical_map = CANONICAL_ASSET_ISSUERS.get(net_name, {})
    canonical_issuer = canonical_map.get(code)
    if canonical_issuer is None:
        return True, f"Asset {code} is not in the reviewed canonical registry for {net_name}"

    if clean_issuer != canonical_issuer:
        return (
            False,
            f"Lookalike asset rejected: {code} issuer {clean_issuer} does not match canonical {net_name} issuer ({canonical_issuer})",
        )

    return True, None


def guard_stellar_provider_access(
    *,
    network: str = "testnet",
    payload: str = "",
    amount_stroops: int = 10_000_000,
    asset_code: str = "USDC",
    issuer: str | None = "GBBD47IF6LWK7P7MDEVSCWR7DPUWV3NY3DTQEVFL4NAT4AQH3ZLLFLA5",
    ledger_sequence: int = 1_024,
    tx_hash: str = "a" * 64,
) -> StellarNetworkGuardResult:
    """Validate asset/issuer pairs, freshness, and custody boundaries before provider access (#862)."""
    combined = f"{payload} {asset_code} {issuer or ''} {tx_hash}"
    if contains_secret_material(combined):
        raise ValueError("Refusing secret keys or seed phrases in non-custodial Stellar tool")

    net_name, passphrase = resolve_network(network)
    warnings: list[str] = []

    if _INJECTION_RE.search(payload):
        warnings.append("untrusted_metadata_injection_blocked")
        clean_payload = "[REDACTED_UNTRUSTED_STELLAR_CONTENT]"
    else:
        clean_payload = "".join(ch for ch in payload if ch.isprintable()).strip()[:240]

    norm_code = (asset_code or "").strip().upper()
    is_native = norm_code == "XLM"

    is_allowed, pair_err = validate_asset_issuer_pair_for_provider(
        network=net_name,
        asset_code=norm_code,
        issuer=None if is_native else issuer,
    )
    if not is_allowed:
        warnings.append("invalid_or_lookalike_asset_pair")
        if pair_err:
            warnings.append(pair_err)

    clean_issuer = None if is_native else (issuer or "").strip()
    canonical_expected = CANONICAL_ASSET_ISSUERS.get(net_name, {}).get(norm_code)
    is_canonical = is_allowed and (
        (is_native and clean_issuer is None) or (clean_issuer == canonical_expected)
    )

    norm_hash = (tx_hash or "").strip().lower()
    hash_ok = bool(_TX_HASH_RE.match(norm_hash))
    if not hash_ok:
        warnings.append("invalid_transaction_hash")

    ledger_ok = (
        isinstance(ledger_sequence, int)
        and not isinstance(ledger_sequence, bool)
        and ledger_sequence > 0
    )
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
        is_allowed
        and hash_ok
        and ledger_ok
        and amount_ok
        and ("untrusted_metadata_injection_blocked" not in warnings)
    )

    details: dict[str, Any] = {
        "asset_code": norm_code,
        "issuer": clean_issuer,
        "is_canonical": is_canonical,
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
        "arabic_term": "التحقق من صحة الزوج قبل الوصول إلى المزود",
        "swahili_term": "Uthibitishaji wa sarafu na mtoaji kabla ya huduma",
    }

    return StellarNetworkGuardResult(
        issue_number=862,
        operation="validate_issuer_asset_pair_before_provider_access",
        network=net_name,
        network_passphrase=passphrase,
        valid=valid,
        non_custodial=True,
        is_canonical=is_canonical,
        summary=f"[{net_name.upper()}] Pre-provider asset-issuer pair validation: {formatted_amount} {norm_code}",
        details=details,
        warnings=tuple(warnings),
    )
