"""Prevent stale cached Stellar ledger data from being presented as current (#577)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any

DEFAULT_FRESHNESS_WINDOWS_SECONDS: dict[str, int] = {
    "account_state": 15,
    "transaction_status": 30,
    "static_protocol_fact": 86400,
}


@dataclass(frozen=True)
class CachedStellarResultEnvelope:
    """Cached Stellar response carrying explicit freshness and ledger provenance."""

    category: str
    network: str
    ledger_sequence: int | None
    observed_at: str
    age_seconds: int
    is_current: bool
    freshness_label: str  # "current" or "stale_cached_snapshot"
    data: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_cached_stellar_result(
    *,
    category: str,
    network: str,
    observed_at: datetime,
    now: datetime,
    ledger_sequence: int | None,
    data: dict[str, Any],
    windows_seconds: dict[str, int] | None = None,
) -> CachedStellarResultEnvelope:
    """Attach freshness status and downgrade stale cached ledger responses."""
    cat = category.strip().lower()
    win_map = windows_seconds or DEFAULT_FRESHNESS_WINDOWS_SECONDS
    if cat not in win_map:
        raise ValueError(f"Unsupported freshness category: {category}")

    if observed_at.tzinfo is None:
        observed_at = observed_at.replace(tzinfo=UTC)
    if now.tzinfo is None:
        now = now.replace(tzinfo=UTC)

    age = max(0, int((now - observed_at).total_seconds()))
    max_age = win_map[cat]
    is_current = age <= max_age
    return CachedStellarResultEnvelope(
        category=cat,
        network=network.strip().lower(),
        ledger_sequence=ledger_sequence,
        observed_at=observed_at.isoformat(),
        age_seconds=age,
        is_current=is_current,
        freshness_label="current" if is_current else "stale_cached_snapshot",
        data=dict(data),
    )
