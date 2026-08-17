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
    "uterine_brother": {
        "name": "Uterine Brother",
        "category": "fard",
        "base_share": Fraction(1, 6),
        "blocked_by": ["son", "daughter", "father", "grandfather"],
        "radd_eligible": True,
    },
    "uterine_sister": {
        "name": "Uterine Sister",
        "category": "fard",
        "base_share": Fraction(1, 6),
        "blocked_by": ["son", "daughter", "father", "grandfather"],
        "radd_eligible": True,
    },
}

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

class FaraidRequest(BaseModel):
    estate: Decimal = Field(..., gt=0, description="Total estate value")
    heirs: List[str] = Field(..., min_length=1, description="List of heir keys")


class FaraidStep(BaseModel):
    heir: str
    category: str
    fraction: str
    amount: str
    basis: str
    blocked_by: Optional[str] = None
    awl_applied: bool = False
    radd_applied: bool = False


class FaraidResponse(BaseModel):
    estate: str
    steps: List[FaraidStep]
    disclaimer: str


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

@dataclass
class FaraidResult:
    shares: Dict[str, Fraction]
    steps: List[Dict[str, Any]]
    awl_applied: bool
    radd_applied: bool


def _get_heir(key: str) -> Dict[str, Any]:
    if key not in HEIRS:
        raise ValueError(f"Unknown heir: {key}")
    return HEIRS[key]


def _apply_hajb(heirs: List[str]) -> Tuple[List[str], Dict[str, str]]:
    """Return (active_heirs, blocked_by_map)."""
    active = set(heirs)
    blocked_by: Dict[str, str] = {}
    for heir in heirs:
        for blocker in _get_heir(heir)["blocked_by"]:
            if blocker in active:
                blocked_by[heir] = blocker
                active.discard(heir)
                break
    return list(active), blocked_by


def _furud_shares(heirs: List[str]) -> Dict[str, Fraction]:
    """Assign fixed shares, adjusting for multiple heirs of the same category."""
    shares: Dict[str, Fraction] = {}
    # Count categories that change the share
    daughters = [h for h in heirs if h == "daughter"]
    full_sisters = [h for h in heirs if h == "full_sister"]
    uterine_siblings = [h for h in heirs if h in ("uterine_brother", "uterine_sister")]

    for heir in heirs:
        info = _get_heir(heir)
        if info["category"] != "fard":
            continue
        share = info["base_share"]
        # Adjust for multiple daughters: 2+ daughters get 2/3
        if heir == "daughter" and len(daughters) >= 2:
            share = Fraction(2, 3)
        # Adjust for multiple full sisters: 2+ get 2/3
        if heir == "full_sister" and len(full_sisters) >= 2:
            share = Fraction(2, 3)
        # Uterine siblings: one gets 1/6, two+ get 1/3
        if heir in ("uterine_brother", "uterine_sister"):
            if len(uterine_siblings) == 1:
                share = Fraction(1, 6)
            else:
                share = Fraction(1, 3)
        shares[heir] = share
    return shares


def _asaba_heirs(heirs: List[str]) -> List[str]:
    return [h for h in heirs if _get_heir(h)["category"] == "asaba"]


def _distribute_asaba(residue: Fraction, asaba_heirs: List[str]) -> Dict[str, Fraction]:
    """Distribute residue among asaba heirs, male gets twice female."""
    if not asaba_heirs:
        return {}
    # Count male and female asaba
    male_count = sum(1 for h in asaba_heirs if "son" in h or "brother" in h or "father" in h or "grandfather" in h)
    female_count = len(asaba_heirs) - male_count
    # Each male counts as 2 units, each female as 1
    total_units = male_count * 2 + female_count
    unit = residue / total_units
    shares = {}
    for h in asaba_heirs:
        if "son" in h or "brother" in h or "father" in h or "grandfather" in h:
            shares[h] = unit * 2
        else:
            shares[h] = unit
    return shares


def _apply_awl(shares: Dict[str, Fraction]) -> Tuple[Dict[str, Fraction], bool]:
    """Scale down shares if they sum to more than 1."""
    total = sum(shares.values(), Fraction(0))
    if total > 1:
        factor = Fraction(1, total)
        return {k: v * factor for k, v in shares.items()}, True
    return shares, False


def _apply_radd(shares: Dict[str, Fraction], heirs: List[str]) -> Tuple[Dict[str, Fraction], bool]:
    """Return surplus to eligible sharers, excluding spouses."""
    total = sum(shares.values(), Fraction(0))
    if total >= 1:
        return shares, False
    surplus = Fraction(1) - total
    eligible = [h for h in heirs if _get_heir(h)["radd_eligible"]]
    if not eligible:
        return shares, False
    # Distribute surplus proportionally among eligible
    eligible_shares = {h: shares[h] for h in eligible}
    eligible_total = sum(eligible_shares.values(), Fraction(0))
    if eligible_total == 0:
        return shares, False
    for h in eligible:
        shares[h] += surplus * (eligible_shares[h] / eligible_total)
    return shares, True


def distribute(estate: Decimal, heirs: List[str]) -> FaraidResult:
    """Main faraid engine entry point."""
    # Validate
    if estate <= 0:
        raise ValueError("Estate must be positive")
    if not heirs:
        raise ValueError("At least one heir required")
    for h in heirs:
        _get_heir(h)

    # Apply hajb
    active_heirs, blocked_by = _apply_hajb(heirs)

    # Assign furud
    shares = _furud_shares(active_heirs)

    # Apply awl
    shares, awl_applied = _apply_awl(shares)

    # Distribute residue to asaba
    asaba = _asaba_heirs(active_heirs)
    if asaba:
        total_furud = sum(shares.values(), Fraction(0))
        if total_furud < 1:
            residue = Fraction(1) - total_furud
            asaba_shares = _distribute_asaba(residue, asaba)
            shares.update(asaba_shares)
    else:
        # Apply radd if no asaba
        shares, radd_applied = _apply_radd(shares, active_heirs)
    else:
        radd_applied = False

    # Build steps
    estate_frac = Fraction(estate)
    steps = []
    for heir in heirs:
        info = _get_heir(heir)
        if heir in blocked_by:
            steps.append({
                "heir": info["name"],
                "category": "blocked",
                "fraction": "0",
                "amount": "0",
                "basis": HAJB_BASIS,
                "blocked_by": blocked_by[heir],
                "awl_applied": awl_applied,
                "radd_applied": False,
            })
        elif heir in shares:
            frac = shares[heir]
            amount = estate_frac * frac
            basis = _get_basis(heir, active_heirs)
            steps.append({
                "heir": info["name"],
                "category": info["category"],
                "fraction": f"{frac.numerator}/{frac.denominator}",
                "amount": str(amount),
                "basis": basis,
                "blocked_by": None,
                "awl_applied": awl_applied,
                "radd_applied": radd_applied if heir in _get_radd_eligible(active_heirs) else False,
            })
        else:
            # Should not happen, but safety
            steps.append({
                "heir": info["name"],
                "category": "unknown",
                "fraction": "0",
                "amount": "0",
                "basis": "No basis",
                "blocked_by": None,
                "awl_applied": awl_applied,
                "radd_applied": False,
            })

    return FaraidResult(
        shares=shares,
        steps=steps,
        awl_applied=awl_applied,
        radd_applied=radd_applied,
    )


def _get_basis(heir: str, active_heirs: List[str]) -> str:
    """Return the fiqh basis for a given heir's share."""
    info = _get_heir(heir)
    if heir in ("son", "daughter"):
        return QURAN_4_11
    if heir in ("wife", "husband"):
        return QURAN_4_12
    if heir in ("father", "mother"):
        return QURAN_4_11
    if heir in ("full_sister", "full_brother", "uterine_brother", "uterine_sister"):
        return QURAN_4_176
    if "grand" in heir:
        return QURAN_4_11
    return QURAN_4_11


def _get_radd_eligible(heirs: List[str]) -> List[str]:
    return [h for h in heirs if _get_heir(h)["radd_eligible"]]


# ---------------------------------------------------------------------------
# Router and endpoint
# ---------------------------------------------------------------------------

from fastapi import APIRouter, HTTPException

router = APIRouter(tags=["faraid"])


@router.post("/faraid", response_model=FaraidResponse)
async def faraid_endpoint(request: FaraidRequest) -> FaraidResponse:
    try:
        result = distribute(request.estate, request.heirs)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    steps = [
        FaraidStep(
            heir=step["heir"],
            category=step["category"],
            fraction=step["fraction"],
            amount=step["amount"],
            basis=step["basis"],
            blocked_by=step.get("blocked_by"),
            awl_applied=step["awl_applied"],
            radd_applied=step["radd_applied"],
        )
        for step in result.steps
    ]

    return FaraidResponse(
        estate=str(request.estate),
        steps=steps,
        disclaimer=DISCLAIMER,
    )
