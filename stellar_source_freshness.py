"""Track protocol version and freshness for indexed Stellar sources (#552)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from typing import Any


@dataclass(frozen=True)
class IndexedSourceFreshnessRecord:
    """Freshness metadata stored alongside an indexed Stellar source chunk."""

    chunk_id: str
    source_url: str
    last_checked_at: datetime
    content_hash: str
    protocol_version: str


@dataclass(frozen=True)
class RetrievalFreshnessSummary:
    """Small internal freshness summary without leaking operational internals."""

    chunk_id: str
    protocol_version: str
    age_days: int
    is_stale: bool
    freshness_state: str  # "unchanged", "updated", "stale", "missing"

    def to_metadata(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "protocol_version": self.protocol_version,
            "age_days": self.age_days,
            "is_stale": self.is_stale,
            "freshness_state": self.freshness_state,
        }


class StellarSourceFreshnessTracker:
    """Manage freshness and protocol version tracking for indexed documentation chunks."""

    def __init__(self, stale_threshold_days: int = 30) -> None:
        if stale_threshold_days <= 0:
            raise ValueError("stale_threshold_days must be positive")
        self.stale_threshold_days = stale_threshold_days
        self._records: dict[str, IndexedSourceFreshnessRecord] = {}
        self._previous_hashes: dict[str, str] = {}

    def record_chunk(
        self,
        *,
        chunk_id: str,
        source_url: str,
        content: str,
        protocol_version: str,
        checked_at: datetime,
    ) -> IndexedSourceFreshnessRecord:
        if checked_at.tzinfo is None:
            checked_at = checked_at.replace(tzinfo=timezone.utc)
        digest = hashlib.sha256(content.strip().encode("utf-8")).hexdigest()
        existing = self._records.get(chunk_id)
        if existing is not None:
            self._previous_hashes[chunk_id] = existing.content_hash
        rec = IndexedSourceFreshnessRecord(
            chunk_id=chunk_id.strip(),
            source_url=source_url.strip(),
            last_checked_at=checked_at,
            content_hash=digest,
            protocol_version=protocol_version.strip(),
        )
        self._records[rec.chunk_id] = rec
        return rec

    def evaluate_freshness(
        self, chunk_id: str, *, now: datetime
    ) -> RetrievalFreshnessSummary:
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        rec = self._records.get(chunk_id)
        if rec is None:
            return RetrievalFreshnessSummary(
                chunk_id=chunk_id,
                protocol_version="unknown",
                age_days=-1,
                is_stale=True,
                freshness_state="missing",
            )
        delta_seconds = max(0.0, (now - rec.last_checked_at).total_seconds())
        age_days = int(delta_seconds // 86400)
        is_stale = age_days > self.stale_threshold_days
        prev_hash = self._previous_hashes.get(chunk_id)
        if is_stale:
            state = "stale"
        elif prev_hash is not None and prev_hash != rec.content_hash:
            state = "updated"
        else:
            state = "unchanged"
        return RetrievalFreshnessSummary(
            chunk_id=rec.chunk_id,
            protocol_version=rec.protocol_version,
            age_days=age_days,
            is_stale=is_stale,
            freshness_state=state,
        )
