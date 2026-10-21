"""Explain Stellar memo types and payment memo privacy limitations (#562)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

PUBLIC_MEMO_PRIVACY_WARNING = (
    "Stellar transaction memos are recorded permanently on the public ledger and "
    "must NEVER contain secret keys, passwords, or personal identifiable information (PII)."
)

MEMO_TYPE_DESCRIPTIONS: dict[str, tuple[str, str]] = {
    "none": ("MEMO_NONE", "No memo attached to the transaction."),
    "text": ("MEMO_TEXT", "UTF-8 text string up to 28 bytes."),
    "id": ("MEMO_ID", "Unsigned 64-bit integer identifier, often required by exchanges/anchors."),
    "hash": ("MEMO_HASH", "32-byte cryptographic hash referencing external metadata."),
    "return": ("MEMO_RETURN", "32-byte hash of the original transaction being refunded."),
}


@dataclass(frozen=True)
class StellarMemoExplanation:
    """Source-grounded explanation of a Stellar memo type and its constraints."""

    memo_type: str
    protocol_label: str
    constraint_summary: str
    is_public_on_chain: bool
    privacy_warning: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def explain_stellar_memo(memo_type: str) -> StellarMemoExplanation:
    """Return a grounded explanation and privacy warning for a Stellar memo type."""
    key = memo_type.strip().lower()
    if key not in MEMO_TYPE_DESCRIPTIONS:
        raise ValueError(f"Unsupported Stellar memo_type: {memo_type}")
    label, summary = MEMO_TYPE_DESCRIPTIONS[key]
    return StellarMemoExplanation(
        memo_type=key,
        protocol_label=label,
        constraint_summary=summary,
        is_public_on_chain=True,
        privacy_warning=PUBLIC_MEMO_PRIVACY_WARNING,
    )
