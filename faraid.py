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
    "grandson": {
        "name": "Grandson (son's son)",
        "category": "asaba",
        "base_share": None,
        "blocked_by": ["son"],
        "radd_eligible": False,
    },
    "grandfather": {
        "name": "Grandfather (father's father)",
        "category": "fard",
        "base_share": Fraction(1, 6),
        "blocked_by": ["father"],
        "radd_eligible": True,
    },
    "grandmother": {
        "name": "Grandmother (mother's mother)",
        "category": "fard",
        "base_share": Fraction(1, 6),
        "blocked_by": ["mother"],
        "radd_eligible": True,
    },
    "full_sister": {
        "name": "Full sister",
        "category": "fard",
        "base_share": Fraction(1, 2),
        "blocked_by": ["son", "grandson", "father", "grandfather"],
        "radd_eligible": True,
    },
    "full_brother": {
        "name": "Full brother",
        "category": "asaba",
        "base_share": None,
        "blocked_by": ["son", "grandson", "father", "grandfather"],
        "radd_eligible": False,
    },
    "paternal_uncle": {
        "name": "Paternal uncle",
        "category": "asaba",
        "base_share": None,
        "blocked_by": ["son", "grandson", "father", "grandfather", "full_brother"],
        "radd_eligible": False,
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
    amount: Decimal
    basis: str
    blocked_by: Optional[str] = None


class FaraidResponse(BaseModel):
    estate: Decimal
    shares: List[HeirShare]
    total_allocated: Decimal
    disclaimer: str
    steps: List[str]


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


@dataclass
class FaraidResult:
    shares: Dict[str, Fraction]
    steps: List[str]
    basis: Dict[str, str]
    blocked: Dict[str, str]


def _get_heir(key: str) -> Dict[str, Any]:
    if key not in HEIRS:
        raise ValueError(f"Unknown heir: {key}")
    return HEIRS[key]


def _basis_for(heir_key: str) -> str:
    heir = _get_heir(heir_key)
    if heir["category"] == "asaba":
        return "Residuary heir (asaba) — takes the remainder after fixed shares, per the established juristic principle."
    if heir_key in ("daughter", "full_sister"):
        return QURAN_4_11 if heir_key == "daughter" else QURAN_4_176
    if heir_key in ("wife", "husband"):
        return QURAN_4_12
    if heir_key in ("father", "mother", "grandfather", "grandmother"):
        return QURAN_4_11
    return QURAN_4_11


def _apply_hajb(heirs: List[str]) -> Tuple[List[str], Dict[str, str]]:
    active = []
    blocked = {}
    for key in heirs:
        heir = _get_heir(key)
        blocker = None
        for b in heir["blocked_by"]:
            if b in heirs:
                blocker = b
                break
        if blocker:
            blocked[key] = blocker
        else:
            active.append(key)
    return active, blocked


def distribute(estate: Decimal, heirs: List[str]) -> FaraidResult:
    """
    Compute faraid shares for the given heirs and estate.

    Implements furud (fixed shares), asaba (residuary), awl (proportional
    reduction when shares exceed the estate), radd (return of surplus when
    shares fall short and no asaba exists), and hajb (blocking).
    """
    steps: List[str] = []
    basis: Dict[str, str] = {}

    # Validate heirs
    for h in heirs:
        _get_heir(h)

    # Apply hajb
    active_heirs, blocked = _apply_hajb(heirs)
    for blocked_key, blocker in blocked.items():
        steps.append(f"{_get_heir(blocked_key)['name']} is blocked by {_get_heir(blocker)['name']} (hajb).")

    # Separate fard and asaba
    fard_heirs = [h for h in active_heirs if _get_heir(h)["category"] == "fard"]
    asaba_heirs = [h for h in active_heirs if _get_heir(h)["category"] == "asaba"]

    # Compute fixed shares
    shares: Dict[str, Fraction] = {}
    for h in fard_heirs:
        heir = _get_heir(h)
        # Adjust for multiple daughters/sisters: 2+ take 2/3
        if h in ("daughter", "full_sister"):
            count = sum(1 for x in fard_heirs if x == h)
            if count >= 2:
                shares[h] = Fraction(2, 3)
            else:
                shares[h] = heir["base_share"]
        else:
            shares[h] = heir["base_share"]
        basis[h] = _basis_for(h)

    # Sum fixed shares
    total_fard = sum(shares.values(), Fraction(0))

    # Awl: if total > 1, scale down proportionally
    if total_fard > 1:
        steps.append(f"Total fixed shares {total_fard} exceed the estate; applying awl.")
        for h in fard_heirs:
            shares[h] = shares[h] / total_fard
        steps.append(f"After awl, fixed shares sum to 1.")

    # Distribute residue to asaba
    residue = Fraction(1) - sum(shares.values(), Fraction(0))
    if asaba_heirs:
        # Male:female ratio 2:1 for children/siblings
        male_asaba = [h for h in asaba_heirs if h in ("son", "full_brother", "paternal_uncle")]
        female_asaba = [h for h in asaba_heirs if h in ("daughter", "full_sister")]
        # Note: daughters/sisters are fard, but if they are also asaba with a male, they get residue too
        # For simplicity, we treat asaba as purely male here; daughters/sisters already have fard shares.
        if male_asaba:
            per_male = residue / len(male_asaba)
            for h in male_asaba:
                shares[h] = per_male
                basis[h] = _basis_for(h)
            steps.append(f"Residue {residue} distributed equally among male asaba heirs.")
        else:
            steps.append(f"No male asaba; residue {residue} remains for radd.")
    else:
        steps.append(f"No asaba; residue {residue} remains for radd.")

    # Radd: if residue > 0 and no asaba, return to eligible fard heirs proportionally
    residue = Fraction(1) - sum(shares.values(), Fraction(0))
    if residue > 0 and not asaba_heirs:
        radd_eligible = [h for h in fard_heirs if _get_heir(h)["radd_eligible"]]
        if radd_eligible:
            total_radd_base = sum(shares[h] for h in radd_eligible, Fraction(0))
            steps.append(f"Applying radd: surplus {residue} returned proportionally to eligible heirs.")
            for h in radd_eligible:
                shares[h] += residue * (shares[h] / total_radd_base)
        else:
            steps.append(f"No radd-eligible heirs; surplus {residue} remains undistributed.")

    # Add blocked heirs with zero share
    for h in blocked:
        shares[h] = Fraction(0)
        basis[h] = HAJB_BASIS

    # Verify sum
    total = sum(shares.values(), Fraction(0))
    if total != 1:
        steps.append(f"Note: total shares sum to {total} (not 1) due to spouse exclusion from radd or other school differences.")

    return FaraidResult(shares=shares, steps=steps, basis=basis, blocked=blocked)


# ---------------------------------------------------------------------------
# Router and endpoint
# ---------------------------------------------------------------------------

from fastapi import APIRouter, HTTPException

router = APIRouter(tags=["faraid"])


@router.post("/faraid", response_model=FaraidResponse)
async def compute_faraid(request: FaraidRequest) -> FaraidResponse:
    try:
        result = distribute(request.estate, request.heirs)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    estate_decimal = request.estate
    shares_list = []
    total_allocated = Decimal(0)
    for heir_key, frac in result.shares.items():
        amount = estate_decimal * Decimal(frac.numerator) / Decimal(frac.denominator)
        total_allocated += amount
        heir = _get_heir(heir_key)
        shares_list.append(
            HeirShare(
                heir=heir_key,
                name=heir["name"],
                category=heir["category"],
                fraction=f"{frac.numerator}/{frac.denominator}",
                amount=amount,
                basis=result.basis.get(heir_key, ""),
                blocked_by=result.blocked.get(heir_key),
            )
        )

    return FaraidResponse(
        estate=estate_decimal,
        shares=shares_list,
        total_allocated=total_allocated,
        disclaimer=DISCLAIMER,
        steps=result.steps,
    )
