from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from fractions import Fraction
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Constants and shared data
# ---------------------------------------------------------------------------

DISCLAIMER = (
    "This calculation follows the classical Sunni (majority) position on faraid. "
    "Islamic inheritance is a serious matter with school-specific differences. "
    "Please consult a qualified scholar for a final ruling."
)

# Qur'anic references for the fixed shares (furud).
# Surah an-Nisa 4:11 covers children and parents; 4:12 covers spouses;
# 4:176 covers kalala (siblings). These are the only verses that specify
# fractional shares in inheritance.
QURAN_4_11 = "Surah an-Nisa 4:11"
QURAN_4_12 = "Surah an-Nisa 4:12"
QURAN_4_176 = "Surah an-Nisa 4:176"

# Juristic principles for awl, radd, and hajb.
AWL_BASIS = "Juristic principle of awl (proportional reduction) — recognized by all four Sunni schools."
RADD_BASIS = "Juristic principle of radd (return of surplus) — majority Sunni position, excluding the spouse."
HAJB_BASIS = "Juristic principle of hajb (blocking) — a nearer relative blocks a more distant one."

# ---------------------------------------------------------------------------
# Heir definitions
# ---------------------------------------------------------------------------

# Each heir is defined by:
# - key: unique identifier used in requests
# - name: display name
# - category: 'fard' (fixed share) or 'asaba' (residuary)
# - base_share: Fraction or None for asaba
# - blocked_by: list of heir keys that block this heir (hajb)
# - radd_eligible: whether this heir can receive from radd (spouses are excluded)

HEIRS: Dict[str, Dict[str, Any]] = {
    "son": {
        "name": "Son",
        "category": "asaba",
        "base_share": None,
        "blocked_by": [],
        "radd_eligible": False,
    },
    "daughter": {
        "name": "Daughter",
        "category": "fard",
        "base_share": Fraction(1, 2),
        "blocked_by": [],
        "radd_eligible": True,
    },
    "wife": {
        "name": "Wife",
        "category": "fard",
        "base_share": Fraction(1, 4),
        "blocked_by": [],
        "radd_eligible": False,
    },
    "husband": {
        "name": "Husband",
        "category": "fard",
        "base_share": Fraction(1, 2),
        "blocked_by": [],
        "radd_eligible": False,
    },
    "father": {
        "name": "Father",
        "category": "fard",
        "base_share": Fraction(1, 6),
        "blocked_by": [],
        "radd_eligible": True,
    },
    "mother": {
        "name": "Mother",
        "category": "fard",
        "base_share": Fraction(1, 6),
        "blocked_by": [],
        "radd_eligible": True,
    },
    "full_sister": {
        "name": "Full Sister",
        "category": "fard",
        "base_share": Fraction(1, 2),
        "blocked_by": ["son", "daughter", "father", "grandfather"],
        "radd_eligible": True,
    },
    "full_brother": {
        "name": "Full Brother",
        "category": "asaba",
        "base_share": None,
        "blocked_by": ["son", "grandson", "father", "grandfather"],
        "radd_eligible": False,
    },
    "paternal_grandfather": {
        "name": "Paternal Grandfather",
        "category": "fard",
        "base_share": Fraction(1, 6),
        "blocked_by": ["father"],
        "radd_eligible": True,
    },
    "paternal_grandmother": {
        "name": "Paternal Grandmother",
        "category": "fard",
        "base_share": Fraction(1, 6),
        "blocked_by": ["mother", "father"],
        "radd_eligible": True,
    },
    "maternal_grandmother": {
        "name": "Maternal Grandmother",
        "category": "fard",
        "base_share": Fraction(1, 6),
        "blocked_by": ["mother", "father"],
        "radd_eligible": True,
    },
    "grandson": {
        "name": "Grandson (son's son)",
        "category": "asaba",
        "base_share": None,
        "blocked_by": ["son"],
        "radd_eligible": False,
    },
    "granddaughter": {
        "name": "Granddaughter (son's daughter)",
        "category": "fard",
        "base_share": Fraction(1, 2),
        "blocked_by": ["son", "daughter"],
        "radd_eligible": True,
    },
}

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class FaraidRequest(BaseModel):
    estate: Decimal = Field(..., gt=0, description="Total estate value")
    heirs: List[str] = Field(..., min_length=1, description="List of heir keys")


class HeirShare(BaseModel):
    heir: str
    name: str
    category: str
    fraction: str
    amount: str
    basis: str
    blocked_by: Optional[str] = None


class FaraidResponse(BaseModel):
    estate: str
    shares: List[HeirShare]
    disclaimer: str
    adjustments: List[str]


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


@dataclass
class FaraidResult:
    shares: Dict[str, Fraction]
    steps: List[str]
    adjustments: List[str]


def _get_heir(key: str) -> Dict[str, Any]:
    if key not in HEIRS:
        raise ValueError(f"Unknown heir: {key}")
    return HEIRS[key]


def _apply_hajb(heirs: List[str]) -> Tuple[List[str], Dict[str, str]]:
    """Return (active_heirs, blocked_by_map)."""
    active = list(heirs)
    blocked_by: Dict[str, str] = {}
    for heir in heirs:
        for blocker in _get_heir(heir)["blocked_by"]:
            if blocker in heirs:
                active.remove(heir)
                blocked_by[heir] = blocker
                break
    return active, blocked_by


def _furud_shares(active: List[str]) -> Dict[str, Fraction]:
    """Assign fixed shares to fard heirs. Asaba heirs get None."""
    shares: Dict[str, Optional[Fraction]] = {}
    for heir in active:
        info = _get_heir(heir)
        if info["category"] == "fard":
            shares[heir] = info["base_share"]
        else:
            shares[heir] = None
    return shares


def _asaba_share(active: List[str], shares: Dict[str, Optional[Fraction]]) -> Optional[Fraction]:
    """Compute the total asaba share (residue) and per-asaba fractions."""
    asaba_heirs = [h for h in active if shares[h] is None]
    if not asaba_heirs:
        return None
    fixed_sum = sum(shares[h] for h in active if shares[h] is not None)
    residue = Fraction(1) - fixed_sum
    if residue <= 0:
        return None
    return residue


def _distribute_asaba(
    active: List[str], shares: Dict[str, Optional[Fraction]], residue: Fraction
) -> Dict[str, Fraction]:
    """Distribute residue among asaba heirs, male gets twice female."""
    asaba_heirs = [h for h in active if shares[h] is None]
    if not asaba_heirs:
        return {}
    # Count units: male = 2, female = 1
    units = 0
    for h in asaba_heirs:
        if "son" in h or "brother" in h or "father" in h or "grandfather" in h:
            units += 2
        else:
            units += 1
    result = {}
    for h in asaba_heirs:
        if "son" in h or "brother" in h or "father" in h or "grandfather" in h:
            result[h] = residue * Fraction(2, units)
        else:
            result[h] = residue * Fraction(1, units)
    return result


def _apply_awl(shares: Dict[str, Fraction]) -> Tuple[Dict[str, Fraction], bool]:
    """Scale down shares if they sum to more than 1."""
    total = sum(shares.values())
    if total <= 1:
        return shares, False
    factor = Fraction(1) / total
    return {k: v * factor for k, v in shares.items()}, True


def _apply_radd(
    shares: Dict[str, Fraction], active: List[str]
) -> Tuple[Dict[str, Fraction], bool]:
    """Return surplus to eligible sharers, excluding spouses."""
    total = sum(shares.values())
    if total >= 1:
        return shares, False
    surplus = Fraction(1) - total
    eligible = [h for h in active if _get_heir(h)["radd_eligible"]]
    if not eligible:
        return shares, False
    eligible_total = sum(shares[h] for h in eligible)
    if eligible_total == 0:
        return shares, False
    for h in eligible:
        shares[h] += surplus * (shares[h] / eligible_total)
    return shares, True


def distribute(estate: Decimal, heirs: List[str]) -> FaraidResult:
    """Compute faraid shares for the given heirs and estate."""
    if estate <= 0:
        raise ValueError("Estate must be positive")
    if not heirs:
        raise ValueError("At least one heir required")

    steps: List[str] = []
    adjustments: List[str] = []

    # 1. Hajb
    active, blocked_by = _apply_hajb(heirs)
    for heir, blocker in blocked_by.items():
        steps.append(f"{HEIRS[heir]['name']} blocked by {HEIRS[blocker]['name']} ({HAJB_BASIS})")
        adjustments.append(f"hajb: {HEIRS[heir]['name']} blocked by {HEIRS[blocker]['name']}")

    # 2. Furud
    shares = _furud_shares(active)
    for heir in active:
        if shares[heir] is not None:
            basis = _basis_for(heir)
            steps.append(f"{HEIRS[heir]['name']} gets {shares[heir]} ({basis})")

    # 3. Asaba
    residue = _asaba_share(active, shares)
    asaba_shares: Dict[str, Fraction] = {}
    if residue is not None:
        asaba_shares = _distribute_asaba(active, shares, residue)
        for heir, frac in asaba_shares.items():
            steps.append(f"{HEIRS[heir]['name']} gets residue {frac} as asaba")

    # Combine
    final_shares: Dict[str, Fraction] = {}
    for heir in active:
        if shares[heir] is not None:
            final_shares[heir] = shares[heir]
        elif heir in asaba_shares:
            final_shares[heir] = asaba_shares[heir]
        else:
            final_shares[heir] = Fraction(0)

    # 4. Awl
    final_shares, awl_applied = _apply_awl(final_shares)
    if awl_applied:
        steps.append(f"Awl applied: shares scaled down proportionally ({AWL_BASIS})")
        adjustments.append("awl: over-subscription, shares reduced proportionally")

    # 5. Radd
    final_shares, radd_applied = _apply_radd(final_shares, active)
    if radd_applied:
        steps.append(f"Radd applied: surplus returned to eligible sharers ({RADD_BASIS})")
        adjustments.append("radd: surplus returned to sharers, spouse excluded")

    # Add blocked heirs with zero
    for heir in blocked_by:
        final_shares[heir] = Fraction(0)

    return FaraidResult(shares=final_shares, steps=steps, adjustments=adjustments)


def _basis_for(heir: str) -> str:
    """Return the fiqh basis for a fard heir."""
    if heir in ["wife", "husband"]:
        return QURAN_4_12
    if heir in ["full_sister", "full_brother"]:
        return QURAN_4_176
    return QURAN_4_11


# ---------------------------------------------------------------------------
# Router and endpoint
# ---------------------------------------------------------------------------

from fastapi import APIRouter, HTTPException

router = APIRouter(tags=["faraid"])


@router.post("/faraid", response_model=FaraidResponse)
def faraid_endpoint(request: FaraidRequest) -> FaraidResponse:
    try:
        result = distribute(request.estate, request.heirs)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    estate_frac = Fraction(request.estate)
    shares = []
    for heir, frac in result.shares.items():
        info = _get_heir(heir)
        amount = estate_frac * frac
        shares.append(
            HeirShare(
                heir=heir,
                name=info["name"],
                category=info["category"],
                fraction=str(frac),
                amount=str(amount),
                basis=_basis_for(heir) if frac > 0 else HAJB_BASIS,
                blocked_by=result.steps[0] if heir in result.shares and frac == 0 else None,
            )
        )

    return FaraidResponse(
        estate=str(request.estate),
        shares=shares,
        disclaimer=DISCLAIMER,
        adjustments=result.adjustments,
    )
