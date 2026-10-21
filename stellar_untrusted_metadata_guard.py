"""Treat Stellar RPC metadata and on-chain strings as untrusted retrieved content (#585)."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

_INJECTION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"ignore\s+(all\s+)?(previous|prior)\s+instructions", re.IGNORECASE),
    re.compile(r"system\s*prompt", re.IGNORECASE),
    re.compile(r"send\s+(all\s+)?funds\s+to", re.IGNORECASE),
    re.compile(r"reveal\s+(your\s+)?secret", re.IGNORECASE),
)


@dataclass(frozen=True)
class UntrustedStellarMetadataEnvelope:
    """Isolated untrusted data wrapper for on-chain memos and contract strings."""

    source_provider: str
    field_name: str
    sanitized_text: str
    injection_detected: bool
    channel_role: str  # Always "untrusted_retrieved_data", never "system" or "developer"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def wrap_untrusted_stellar_metadata(
    *,
    source_provider: str,
    field_name: str,
    raw_value: str,
    max_length: int = 280,
) -> UntrustedStellarMetadataEnvelope:
    """Sanitize and wrap external Stellar strings so they cannot enter instruction channels."""
    cleaned = "".join(ch for ch in raw_value if ch.isprintable()).strip()[:max_length]
    detected = any(pat.search(cleaned) for pat in _INJECTION_PATTERNS)
    if detected:
        sanitized = "[REDACTED_UNTRUSTED_ONCHAIN_METADATA]"
    else:
        sanitized = cleaned

    return UntrustedStellarMetadataEnvelope(
        source_provider=source_provider.strip().lower(),
        field_name=field_name.strip().lower(),
        sanitized_text=sanitized,
        injection_detected=detected,
        channel_role="untrusted_retrieved_data",
    )
