"""Explain Stellar claimable balance lifecycle and claimant conditions (#563)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any


@dataclass(frozen=True)
class ClaimantConditionSummary:
    """Evaluated claimant condition in UTC."""

    destination: str
    predicate_type: str  # "unconditional", "before_utc", "not_before_utc", "complex"
    deadline_utc: str | None
    is_currently_claimable: bool


@dataclass(frozen=True)
class ClaimableBalanceExplanation:
    """Grounded explanation of a public Stellar claimable balance record."""

    balance_id: str
    sponsor: str | None
    asset: str
    amount: str
    claimants: tuple[ClaimantConditionSummary, ...]
    lifecycle_note: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def explain_claimable_balance(
    record: dict[str, Any], *, now_utc: datetime | None = None
) -> ClaimableBalanceExplanation:
    """Explain a verified claimable balance record and its UTC time predicates."""
    if now_utc is None:
        now_utc = datetime.now(UTC)
    elif now_utc.tzinfo is None:
        now_utc = now_utc.replace(tzinfo=UTC)

    balance_id = str(record.get("id", "")).strip()
    if not balance_id:
        raise ValueError("Claimable balance record requires a non-empty 'id'")
    asset = str(record.get("asset", "native")).strip()
    amount = str(record.get("amount", "0.0000000")).strip()
    sponsor = str(record["sponsor"]).strip() if record.get("sponsor") else None

    claimant_summaries: list[ClaimantConditionSummary] = []
    for c in record.get("claimants", []):
        dest = str(c.get("destination", "")).strip()
        pred = c.get("predicate", {})
        if "unconditional" in pred and pred["unconditional"] is True:
            claimant_summaries.append(
                ClaimantConditionSummary(
                    destination=dest,
                    predicate_type="unconditional",
                    deadline_utc=None,
                    is_currently_claimable=True,
                )
            )
        elif "abs_before" in pred:
            ts_str = str(pred["abs_before"]).strip()
            dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            claimant_summaries.append(
                ClaimantConditionSummary(
                    destination=dest,
                    predicate_type="before_utc",
                    deadline_utc=dt.astimezone(UTC).isoformat(),
                    is_currently_claimable=now_utc < dt,
                )
            )
        else:
            claimant_summaries.append(
                ClaimantConditionSummary(
                    destination=dest,
                    predicate_type="complex",
                    deadline_utc=None,
                    is_currently_claimable=False,
                )
            )

    return ClaimableBalanceExplanation(
        balance_id=balance_id,
        sponsor=sponsor,
        asset=asset,
        amount=amount,
        claimants=tuple(claimant_summaries),
        lifecycle_note=(
            "A claimable balance is held in a ledger entry until a listed claimant "
            "satisfies its predicate and submits a claim_claimable_balance operation."
        ),
    )
