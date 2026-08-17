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
        "blocked_by": ["son"],
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
        "blocked_by": ["son", "daughter", "father", "grandfather"],
        "radd_eligible": True,
    },
    "full_brother": {
        "name": "Full brother",
        "category": "asaba",
        "base_share": None,
        "blocked_by": ["son", "grandson", "father", "grandfather"],
        "radd_eligible": False,
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


class FaraidResponse(BaseModel):
    estate: str
    steps: List[FaraidStep]
    disclaimer: str
    total_allocated: str


# ---------------------------------------------------------------------------
# Core engine
# ---------------------------------------------------------------------------

@dataclass
class FaraidResult:
    allocations: Dict[str, Fraction]
    steps: List[Dict[str, Any]]
    total: Fraction


def _apply_hajb(heirs: List[str]) -> Tuple[List[str], Dict[str, str]]:
    """Remove blocked heirs and record the blocker."""
    active = list(heirs)
    blocked_by: Dict[str, str] = {}
    changed = True
    while changed:
        changed = False
        for heir in list(active):
            for blocker in HEIRS[heir]["blocked_by"]:
                if blocker in active:
                    active.remove(heir)
                    blocked_by[heir] = blocker
                    changed = True
                    break
    return active, blocked_by


def distribute(estate: Decimal, heirs: List[str]) -> FaraidResult:
    """
    Distribute an estate among heirs using the classical faraid algorithm.

    Steps:
    1. Apply hajb (blocking) to remove heirs who are blocked by a nearer relative.
    2. Assign fixed shares (furud) to eligible heirs.
    3. If the fixed shares sum to more than the estate, apply awl (proportional reduction).
    4. If the fixed shares sum to less than the estate and there is no asaba, apply radd (return surplus to sharers, excluding spouse).
    5. If there is an asaba, give the residue to the asaba (male takes twice female).
    """
    active, blocked_by = _apply_hajb(heirs)

    # Count heirs by type
    fard_heirs = [h for h in active if HEIRS[h]["category"] == "fard"]
    asaba_heirs = [h for h in active if HEIRS[h]["category"] == "asaba"]

    # Determine if any asaba exists (excluding those who are blocked)
    has_asaba = len(asaba_heirs) > 0

    # Assign fixed shares
    shares: Dict[str, Fraction] = {}
    for h in fard_heirs:
        # If there is a son, daughters become asaba and lose their fixed share
        if h == "daughter" and "son" in active:
            shares[h] = Fraction(0)  # will be handled as asaba
        else:
            shares[h] = HEIRS[h]["base_share"]

    # Sum of fixed shares
    fixed_sum = sum(shares.values(), Fraction(0))

    # Apply awl if fixed shares exceed 1
    awl_factor = Fraction(1)
    if fixed_sum > 1:
        awl_factor = Fraction(1, fixed_sum)

    # Apply radd if fixed shares are less than 1 and no asaba
    radd_factor = Fraction(1)
    if fixed_sum < 1 and not has_asaba:
        # Exclude spouses from radd
        radd_eligible = [h for h in fard_heirs if HEIRS[h]["radd_eligible"]]
        if radd_eligible:
            eligible_sum = sum(HEIRS[h]["base_share"] for h in radd_eligible)
            if eligible_sum > 0:
                radd_factor = Fraction(1 - fixed_sum, eligible_sum) + 1

    # Apply factors to fard shares
    for h in fard_heirs:
        if h == "daughter" and "son" in active:
            continue  # handled as asaba
        shares[h] = shares[h] * awl_factor
        if radd_factor != 1 and HEIRS[h]["radd_eligible"]:
            shares[h] = shares[h] * radd_factor

    # Distribute residue to asaba
    if has_asaba:
        # Asaba take the residue, male twice female
        residue = Fraction(1) - sum(shares.values(), Fraction(0))
        # Count asaba units: each male counts as 2, each female as 1
        male_asaba = [h for h in asaba_heirs if "son" in h or "brother" in h]
        female_asaba = [h for h in asaba_heirs if h not in male_asaba]
        # For simplicity, treat all asaba as male unless explicitly female
        # In this implementation, only 'son' and 'full_brother' are asaba and they are male
        total_units = len(asaba_heirs) * 2  # each male asaba gets 2 units
        # Actually, if there are daughters with sons, they become asaba too
        daughters_with_sons = [h for h in fard_heirs if h == "daughter" and "son" in active]
        total_units = len(asaba_heirs) * 2 + len(daughters_with_sons) * 1
        if total_units > 0:
            per_unit = residue / total_units
            for h in asaba_heirs:
                shares[h] = per_unit * 2
            for h in daughters_with_sons:
                shares[h] = per_unit * 1

    # Build steps
    steps: List[Dict[str, Any]] = []
    for h in active:
        category = HEIRS[h]["category"]
        if h == "daughter" and "son" in active:
            category = "asaba"
        basis = ""
        if category == "fard":
            if h in ["daughter", "father", "mother", "grandfather", "grandmother"]:
                basis = QURAN_4_11
            elif h in ["wife", "husband"]:
                basis = QURAN_4_12
            elif h in ["full_sister"]:
                basis = QURAN_4_176
        elif category == "asaba":
            basis = "Residuary heir (asaba) — takes the residue after fixed shares, per the established juristic principle."
        steps.append({
            "heir": h,
            "category": category,
            "fraction": str(shares.get(h, Fraction(0))),
            "amount": str(estate * shares.get(h, Fraction(0))),
            "basis": basis,
            "blocked_by": blocked_by.get(h),
        })

    # Add blocked heirs with zero share
    for h in blocked_by:
        steps.append({
            "heir": h,
            "category": "blocked",
            "fraction": "0",
            "amount": "0",
            "basis": HAJB_BASIS,
            "blocked_by": blocked_by[h],
        })

    total = sum(shares.values(), Fraction(0))
    return FaraidResult(allocations=shares, steps=steps, total=total)


# ---------------------------------------------------------------------------
# Router and endpoint
# ---------------------------------------------------------------------------

from fastapi import APIRouter

router = APIRouter(tags=["faraid"])


@router.post("/faraid", response_model=FaraidResponse)
def faraid_endpoint(request: FaraidRequest) -> FaraidResponse:
    result = distribute(request.estate, request.heirs)
    steps = [
        FaraidStep(
            heir=s["heir"],
            category=s["category"],
            fraction=s["fraction"],
            amount=s["amount"],
            basis=s["basis"],
            blocked_by=s.get("blocked_by"),
        )
        for s in result.steps
    ]
    return FaraidResponse(
        estate=str(request.estate),
        steps=steps,
        disclaimer=DISCLAIMER,
        total_allocated=str(result.total),
    )
