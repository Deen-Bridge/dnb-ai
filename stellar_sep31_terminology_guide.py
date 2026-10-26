"""SEP-31 sending anchor, receiving anchor, and quotes/KYC field terminology guide (#640)."""

from __future__ import annotations

import base64
import binascii
import re
from dataclasses import asdict, dataclass
from typing import Any

PUBLIC_PASSPHRASE = "Public Global Stellar Network ; September 2015"
TESTNET_PASSPHRASE = "Test SDF Network ; September 2015"
MAX_INT64 = 2**63 - 1
STROOPS_PER_UNIT_INT = 10_000_000
_SECRET_KEY_RE = re.compile(r"(?<![A-Z2-7])S[A-Z2-7]{55}(?![A-Z2-7])")
_PUBLIC_KEY_RE = re.compile(r"^G[A-Z2-7]{55}$")
_ASSET_CODE_RE = re.compile(r"^[A-Za-z0-9]{1,12}$")
_TX_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_MNEMONIC_WORD_RE = re.compile(r"^[a-z]{3,8}$")
_SEED_PHRASE_RE = re.compile(
    r"\b(mnemonic|seed\s+phrase|recovery\s+phrase|recovery\s+words|12\s+words|24\s+words|private\s+key|secret\s+seed)\b",
    re.IGNORECASE,
)
_INJECTION_RE = re.compile(
    r"(ignore\s+(all\s+)?(previous|prior)\s+instructions|system\s*prompt|send\s+funds\s+to)",
    re.IGNORECASE,
)
_ED25519_PUBLIC_KEY_VERSION_BYTE = 6 << 3  # 48 ('G')


@dataclass(frozen=True)
class StellarWaveCheckResult:
    """Deterministic, network-aware, non-custodial result envelope (#640)."""

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


def _crc16_xmodem(data: bytes) -> int:
    crc = 0x0000
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ 0x1021) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc


def is_valid_stellar_account_id(value: str) -> bool:
    """Validate a Stellar SEP-0023 ``G...`` Ed25519 public key including CRC16-XMODEM checksum."""
    candidate = value.strip()
    if not _PUBLIC_KEY_RE.match(candidate):
        return False
    try:
        raw = base64.b32decode(candidate, casefold=False)
    except (binascii.Error, ValueError):
        return False
    if len(raw) != 35 or raw[0] != _ED25519_PUBLIC_KEY_VERSION_BYTE:
        return False
    canonical = base64.b32encode(raw).decode("ascii").rstrip("=")
    if canonical != candidate:
        return False
    expected_crc = _crc16_xmodem(raw[:33]).to_bytes(2, "little")
    return raw[33:] == expected_crc


def _has_unlabeled_recovery_phrase(text: str) -> bool:
    for segment in re.split(r"[\r\n:;,\-]+", text):
        tokens = segment.strip().split()
        if len(tokens) in (12, 15, 18, 21, 24) and all(_MNEMONIC_WORD_RE.fullmatch(tok) for tok in tokens):
            return True
    return False


def contains_secret_material(text: str) -> bool:
    """Return True if text contains a Stellar S-seed or recovery phrase."""
    return bool(_SECRET_KEY_RE.search(text) or _SEED_PHRASE_RE.search(text) or _has_unlabeled_recovery_phrase(text))


def resolve_network(network: str) -> tuple[str, str]:
    """Normalize network identifier and return (network_name, canonical_passphrase)."""
    if contains_secret_material(network):
        raise ValueError("Refusing secret keys or seed phrases in non-custodial Stellar tool")
    cleaned = network.strip().lower()
    if cleaned in ("public", "mainnet", "pubnet"):
        return ("public", PUBLIC_PASSPHRASE)
    if cleaned == "testnet":
        return ("testnet", TESTNET_PASSPHRASE)
    raise ValueError("Unsupported Stellar network; expected 'testnet' or 'public'")


def format_stroops_to_decimal(stroops: int) -> str:
    """Convert integer stroops (0..2**63-1) to a 7-decimal Stellar amount string."""
    if isinstance(stroops, bool) or not isinstance(stroops, int) or not 0 <= stroops <= MAX_INT64:
        raise ValueError("stroops must be an integer between 0 and 2**63 - 1")
    whole, frac = divmod(stroops, STROOPS_PER_UNIT_INT)
    return f"{whole}.{frac:07d}"


def explain_sep31_payment_role(
    *,
    network: str = "testnet",
    payload: str = "",
    amount_stroops: int = 10_000_000,
    asset_code: str = "USDC",
    issuer: str = "GBBD47IF6LWK7P7MDEVSCWR7DPUWV3NY3DTQEVFL4NAT4AQH3ZLLFLA5",
    ledger_sequence: int = 1_024,
    tx_hash: str = "a" * 64,
) -> StellarWaveCheckResult:
    """Execute SEP-31 sending anchor, receiving anchor, and quotes/KYC field terminology guide (#640)."""
    for field_val in (network, payload, asset_code, issuer, tx_hash):
        if contains_secret_material(field_val):
            raise ValueError("Refusing secret keys or seed phrases in non-custodial Stellar tool")

    net_name, passphrase = resolve_network(network)
    warnings: list[str] = []

    stripped_payload = "".join(ch for ch in payload if ch.isprintable()).strip()[:240]
    stripped_code = "".join(ch for ch in asset_code if ch.isprintable()).strip()
    if (
        _INJECTION_RE.search(payload)
        or _INJECTION_RE.search(stripped_payload)
        or _INJECTION_RE.search(asset_code)
        or _INJECTION_RE.search(stripped_code)
    ):
        warnings.append("untrusted_metadata_injection_blocked")
        clean_payload = "[REDACTED_UNTRUSTED_STELLAR_CONTENT]"
    else:
        clean_payload = stripped_payload

    raw_code = asset_code.strip()
    code_ok = bool(_ASSET_CODE_RE.fullmatch(raw_code))
    norm_code = raw_code.upper() if code_ok else "INVALID"
    if not code_ok:
        warnings.append("invalid_asset_code")

    is_native = code_ok and norm_code == "XLM"
    issuer_ok = True if is_native else is_valid_stellar_account_id(issuer)
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
        code_ok
        and issuer_ok
        and hash_ok
        and ledger_ok
        and amount_ok
        and ("untrusted_metadata_injection_blocked" not in warnings)
    )

    details: dict[str, Any] = {
        "asset_code": norm_code if code_ok else "",
        "issuer": None if is_native else (issuer.strip() if issuer_ok else ""),
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
        issue_number=640,
        operation="explain_sep31_payment_role",
        network=net_name,
        network_passphrase=passphrase,
        valid=valid,
        non_custodial=True,
        summary=f"[{net_name.upper()}] SEP-31 sending anchor, receiving anchor, and quotes/KYC field terminology guide: {formatted_amount} {norm_code}",
        details=details,
        warnings=tuple(warnings),
    )
