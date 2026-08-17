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
# Pydantic models
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
    """Return (active_heirs, blocked_map) where blocked_map maps blocked heir to blocker."""
    active = set(heirs)
    blocked_map: Dict[str, str] = {}
    for heir in heirs:
        for blocker in _get_heir(heir)["blocked_by"]:
            if blocker in active:
                blocked_map[heir] = blocker
                active.discard(heir)
                break
    return list(active), blocked_map


def distribute(estate: Decimal, heirs: List[str]) -> FaraidResult:
    """
    Compute faraid shares for the given heirs.

    Implements furud (fixed shares), asaba (residuary), awl (proportional
    reduction when shares exceed estate), radd (return of surplus when no
    asaba exists), and hajb (blocking). Uses exact rational arithmetic.
    """
    # Validate heirs
    for h in heirs:
        _get_heir(h)

    # Apply hajb
    active_heirs, blocked_map = _apply_hajb(heirs)

    # Separate fard and asaba
    fard_heirs = [h for h in active_heirs if _get_heir(h)["category"] == "fard"]
    asaba_heirs = [h for h in active_heirs if _get_heir(h)["category"] == "asaba"]

    # Compute fixed shares
    shares: Dict[str, Fraction] = {}
    for h in fard_heirs:
        shares[h] = _get_heir(h)["base_share"]

    # Sum fixed shares
    fixed_sum = sum(shares.values(), Fraction(0))

    awl_applied = False
    radd_applied = False

    if fixed_sum > 1:
        # Awl: scale all shares down proportionally
        awl_applied = True
        scale = Fraction(1, 1) / fixed_sum
        for h in shares:
            shares[h] = shares[h] * scale
    elif fixed_sum < 1 and not asaba_heirs:
        # Radd: return surplus to fard heirs (excluding spouse)
        radd_applied = True
        surplus = Fraction(1, 1) - fixed_sum
        radd_eligible = [h for h in fard_heirs if _get_heir(h)["radd_eligible"]]
        if radd_eligible:
            radd_total = sum(shares[h] for h in radd_eligible)
            for h in radd_eligible:
                shares[h] += surplus * (shares[h] / radd_total)
        else:
            # No eligible heirs, surplus goes to Bait-ul-Mal (not distributed)
            pass

    # Distribute residue to asaba
    if asaba_heirs:
        residue = Fraction(1, 1) - sum(shares.values(), Fraction(0))
        if residue > 0:
            # Male gets twice female
            male_count = sum(1 for h in asaba_heirs if "son" in h or "brother" in h or "uncle" in h)
            female_count = len(asaba_heirs) - male_count
            total_units = male_count * 2 + female_count
            for h in asaba_heirs:
                if "son" in h or "brother" in h or "uncle" in h:
                    shares[h] = residue * Fraction(2, total_units)
                else:
                    shares[h] = residue * Fraction(1, total_units)

    # Build steps
    steps: List[Dict[str, Any]] = []
    for h in heirs:
        if h in blocked_map:
            steps.append({
                "heir": h,
                "category": "blocked",
                "fraction": "0",
                "amount": "0",
                "basis": HAJB_BASIS,
                "blocked_by": blocked_map[h],
                "awl_applied": awl_applied,
                "radd_applied": radd_applied,
            })
        else:
            frac = shares.get(h, Fraction(0))
            amount = estate * Decimal(frac.numerator) / Decimal(frac.denominator)
            basis = _get_heir(h)["basis"] if "basis" in _get_heir(h) else _get_basis(h)
            steps.append({
                "heir": h,
                "category": _get_heir(h)["category"],
                "fraction": f"{frac.numerator}/{frac.denominator}",
                "amount": str(amount),
                "basis": basis,
                "blocked_by": None,
                "awl_applied": awl_applied,
                "radd_applied": radd_applied,
            })

    return FaraidResult(shares=shares, steps=steps, awl_applied=awl_applied, radd_applied=radd_applied)


def _get_basis(heir: str) -> str:
    """Return the fiqh basis for a given heir."""
    if heir in ["wife", "husband"]:
        return QURAN_4_12
    if heir in ["daughter", "full_sister"]:
        return QURAN_4_11 if heir == "daughter" else QURAN_4_176
    if heir in ["father", "mother", "grandfather", "grandmother"]:
        return QURAN_4_11
    if heir in ["son", "grandson", "full_brother", "paternal_uncle"]:
        return "Residuary (asaba) — takes the remainder after fixed shares, per the established juristic principle."
    return "Juristic principle"


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

from fastapi import APIRouter, HTTPException

router = APIRouter(tags=["faraid"])


@router.post("/faraid", response_model=FaraidResponse)
def faraid_endpoint(request: FaraidRequest) -> FaraidResponse:
    try:
        result = distribute(request.estate, request.heirs)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    steps = [
        FaraidStep(
            heir=s["heir"],
            category=s["category"],
            fraction=s["fraction"],
            amount=s["amount"],
            basis=s["basis"],
            blocked_by=s.get("blocked_by"),
            awl_applied=s["awl_applied"],
            radd_applied=s["radd_applied"],
        )
        for s in result.steps
    ]

    return FaraidResponse(
        estate=str(request.estate),
        steps=steps,
        disclaimer=DISCLAIMER,
    )
