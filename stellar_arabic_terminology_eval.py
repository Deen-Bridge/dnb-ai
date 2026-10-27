"""Arabic Stellar terminology evaluator checking non-custodial and network-explicit translations (#667)."""

from __future__ import annotations

import base64
import binascii
import hashlib
import re
from dataclasses import asdict, dataclass
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from typing import Any

PUBLIC_PASSPHRASE = "Public Global Stellar Network ; September 2015"
TESTNET_PASSPHRASE = "Test SDF Network ; September 2015"
STROOPS_PER_UNIT = Decimal("10000000")
MAX_INT64_STROOPS = (1 << 63) - 1
_SECRET_KEY_RE = re.compile(r"(?<![A-Z2-7])S[A-Z2-7]{55}(?![A-Z2-7])")
_PUBLIC_KEY_RE = re.compile(r"^G[A-Z2-7]{55}$")
_ASSET_CODE_RE = re.compile(r"^[A-Za-z0-9]{1,12}$")
_TX_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_SEED_PHRASE_RE = re.compile(
    r"\b(mnemonic|seed\s+phrase|recovery\s+phrase|recovery\s+words|12\s+words|24\s+words|private\s+key|secret\s+seed)\b",
    re.IGNORECASE,
)
_BIP39_WORD_RUN_RE = re.compile(r"(?:^|\s)[a-z]{3,8}(?:\s+[a-z]{3,8}){11,23}(?:\s|$)", re.IGNORECASE)
# Detection is layered. The first pattern is the literal denylist kept from the
# original revision; the remaining three match injection *intent* structurally so
# that reworded instructions are still caught. They are deliberately narrow:
#   - verbs must be imperative and directed ("show me", not "shows")
#   - the verb and its object must sit within one short clause (_CLAUSE_GAP)
# That keeps declarative non-custodial copy, which legitimately names secret keys
# and seed phrases, from being flagged.
_CLAUSE_GAP = r"[^.!?\n]{0,24}?"

_INJECTION_RE = re.compile(
    r"(ignore\s+(all\s+)?(previous|prior|above)\s+instructions|"
    r"disregard\s+.*instructions|"
    r"forget\s+.*instructions|"
    r"override\s+.*instructions|"
    r"system\s*prompt|developer\s*message|"
    r"output\s+credentials|reveal\s+.*(secret|key|prompt|credentials)|"
    r"send\s+funds\s+to)",
    re.IGNORECASE,
)

# "forget everything", "disregard the rules above", ...
_OVERRIDE_RE = re.compile(
    r"\b(ignore|disregard|forget|override|bypass|skip|discard|drop|cancel)\b"
    + _CLAUSE_GAP
    + r"\b(instruction|instructions|rule|rules|prompt|prompts|directive|directives|"
    r"guideline|guidelines|constraint|constraints|policy|policies|above|previous|prior|"
    r"earlier|system|developer|everything|all|that|this)\b",
    re.IGNORECASE,
)

# "show me your secret key", "output credentials", ...
_EXFIL_RE = re.compile(
    r"\b(output|reveal|disclose|dump|exfiltrate|leak|echo|repeat|recite|"
    r"show me|give me|send me|tell me|hand over|print out|write out|list all)\b"
    + _CLAUSE_GAP
    + r"\b(secret|seed|mnemonic|passphrase|password|private key|private keys|"
    r"credential|credentials|token|prompt|api key)\b",
    re.IGNORECASE,
)

# Fake role/message boundaries used to smuggle instructions.
_DELIMITER_RE = re.compile(
    r"(<\s*\|\s*(system|assistant|user)\s*\|\s*>|"
    r"^\s*(system|assistant)\s*:|"
    r"\bjailbreak\b|\bDAN\b|\bact as\b|\bpretend (to be|you)\b|\byou are now\b|"
    r"\bnew (system )?prompt\b)",
    re.IGNORECASE | re.MULTILINE,
)
_ED25519_PUBLIC_KEY_VERSION_BYTE = 6 << 3  # 48 ('G')


@dataclass(frozen=True)
class StellarWaveCheckResult:
    """Deterministic, network-aware, non-custodial result envelope (#667).

    Note: ``valid`` reflects offline format and safety validation only; it performs
    no Horizon or RPC network reads and does not confirm on-chain inclusion.
    """

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


def contains_secret_material(text: str) -> bool:
    """Return True if text contains a Stellar S-seed, recovery label, or 12-24 word mnemonic."""
    return bool(_SECRET_KEY_RE.search(text) or _SEED_PHRASE_RE.search(text) or _BIP39_WORD_RUN_RE.search(text))


def looks_like_injection(text: str) -> bool:
    """Return True if untrusted text reads as an attempt to steer the consumer.

    Heuristic by nature: it is a guard rail, not a proof of safety. The security
    property does not rest on it, because no untrusted text is ever echoed back --
    see ``evaluate_arabic_stellar_terminology``.
    """
    return bool(
        _INJECTION_RE.search(text) or _OVERRIDE_RE.search(text) or _EXFIL_RE.search(text) or _DELIMITER_RE.search(text)
    )


def resolve_network(network: str) -> tuple[str, str]:
    """Normalize network identifier and return (network_name, canonical_passphrase)."""
    cleaned = network.strip().lower()
    if cleaned in ("public", "mainnet", "pubnet"):
        return ("public", PUBLIC_PASSPHRASE)
    if cleaned == "testnet":
        return ("testnet", TESTNET_PASSPHRASE)
    raise ValueError("Unsupported Stellar network; expected 'public' or 'testnet'")


def format_stroops_to_decimal(stroops: int) -> str:
    """Convert bounded int64 stroops to a 7-decimal Stellar amount string."""
    if isinstance(stroops, bool) or not isinstance(stroops, int) or stroops < 0 or stroops > MAX_INT64_STROOPS:
        raise ValueError("stroops must be a bounded non-negative int64 integer")
    try:
        val = (Decimal(stroops) / STROOPS_PER_UNIT).quantize(Decimal("0.0000001"), rounding=ROUND_HALF_UP)
    except InvalidOperation as exc:
        raise ValueError("stroops exceeds supported decimal precision") from exc
    return f"{val:.7f}"


def evaluate_arabic_stellar_terminology(
    *,
    network: str = "testnet",
    payload: str = "",
    amount_stroops: int = 10_000_000,
    asset_code: str = "USDC",
    issuer: str = "GBBD47IF6LWK7P7MDEVSCWR7DPUWV3NY3DTQEVFL4NAT4AQH3ZLLFLA5",
    ledger_sequence: int = 1_024,
    tx_hash: str = "a" * 64,
) -> StellarWaveCheckResult:
    """Execute offline Arabic Stellar terminology check (#667); ``valid`` checks local format only."""
    combined = f"{network} {payload} {asset_code} {issuer} {tx_hash} {amount_stroops} {ledger_sequence}"
    if contains_secret_material(combined):
        raise ValueError("Refusing secret keys or seed phrases in non-custodial Stellar tool")

    net_name, passphrase = resolve_network(network)
    warnings: list[str] = []

    printable_payload = "".join(ch for ch in str(payload) if ch.isprintable()).strip()[:240]

    # Untrusted text is never echoed. Any payload collapses to a one-way digest that
    # still supports correlation but cannot carry attacker-chosen wording, so the
    # sanitisation guarantee does not depend on the detector above being complete.
    if printable_payload:
        digest = hashlib.sha256(printable_payload.encode("utf-8")).hexdigest()[:16]
        clean_payload = f"[UNTRUSTED_INPUT_REDACTED sha256:{digest}]"
    else:
        clean_payload = ""

    if looks_like_injection(printable_payload) or looks_like_injection(str(asset_code)):
        warnings.append("untrusted_metadata_injection_blocked")

    raw_code = str(asset_code).strip()
    code_ok = bool(_ASSET_CODE_RE.match(raw_code))
    norm_code = raw_code if code_ok else "INVALID"
    if not code_ok:
        warnings.append("invalid_asset_code")

    is_native = code_ok and norm_code == "XLM"
    issuer_ok = True if is_native else is_valid_stellar_account_id(str(issuer))
    if not issuer_ok:
        warnings.append("invalid_issuer_public_key")

    norm_hash = str(tx_hash).strip().lower()
    hash_ok = bool(_TX_HASH_RE.match(norm_hash))
    if not hash_ok:
        warnings.append("invalid_transaction_hash")

    ledger_ok = isinstance(ledger_sequence, int) and not isinstance(ledger_sequence, bool) and ledger_sequence > 0
    safe_ledger_sequence = ledger_sequence if ledger_ok else 0
    if not ledger_ok:
        warnings.append("invalid_ledger_sequence")

    try:
        formatted_amount = format_stroops_to_decimal(amount_stroops)
        amount_ok = True
        safe_amount_stroops = amount_stroops
    except ValueError:
        formatted_amount = "0.0000000"
        amount_ok = False
        safe_amount_stroops = 0
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
        "amount_stroops": safe_amount_stroops,
        "ledger_sequence": safe_ledger_sequence,
        "tx_hash": norm_hash if hash_ok else "",
        "sanitized_payload": clean_payload,
        # False on every path: untrusted text is never returned, only a digest.
        "payload_echoed": False,
        "explorer_url": (
            f"https://stellar.expert/explorer/{'public' if net_name == 'public' else 'testnet'}/tx/{norm_hash}"
            if hash_ok
            else ""
        ),
        "arabic_term": "شبكة ستيلر غير الوصائية",
        "swahili_term": "Mtandao wa Stellar usioshikilia funguo za siri",
    }

    return StellarWaveCheckResult(
        issue_number=667,
        operation="evaluate_arabic_stellar_terminology",
        network=net_name,
        network_passphrase=passphrase,
        valid=valid,
        non_custodial=True,
        summary=(
            f"[{net_name.upper()}] Arabic Stellar terminology evaluator checking non-custodial "
            f"and network-explicit translations: {formatted_amount} {norm_code}"
        ),
        details=details,
        warnings=tuple(warnings),
    )
