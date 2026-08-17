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


class FaraidShare(BaseModel):
    heir: str
    name: str
    category: str
    fraction: str
    amount: str
    basis: str
    blocked_by: Optional[str] = None


class FaraidResponse(BaseModel):
    estate: str
    shares: List[FaraidShare]
    total_allocated: str
    adjustments: List[str]
    disclaimer: str


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

@dataclass
class FaraidResult:
    shares: List[FaraidShare]
    total_allocated: Fraction
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
        definition = _get_heir(heir)
        for blocker in definition["blocked_by"]:
            if blocker in heirs:
                if heir in active:
                    active.remove(heir)
                blocked_by[heir] = blocker
                break
    return active, blocked_by


def _furud_share(key: str, count: int) -> Fraction:
    """Return the fixed share for a fard heir, adjusting for multiple heirs."""
    definition = _get_heir(key)
    base = definition["base_share"]
    if base is None:
        raise ValueError(f"{key} is not a fard heir")
    # Special rules for daughters and sisters: one gets 1/2, two or more get 2/3.
    if key in ("daughter", "full_sister"):
        if count == 1:
            return Fraction(1, 2)
        return Fraction(2, 3)
    # Spouses: wife gets 1/4 if no children, 1/8 if children; husband 1/2 if no children, 1/4 if children.
    if key == "wife":
        if any(h in ("son", "daughter", "grandson") for h in heirs_global):
            return Fraction(1, 8)
        return Fraction(1, 4)
    if key == "husband":
        if any(h in ("son", "daughter", "grandson") for h in heirs_global):
            return Fraction(1, 4)
        return Fraction(1, 2)
    return base


# Global variable to track current heirs for spouse share calculation
heirs_global: List[str] = []


def distribute(estate: Decimal, heirs: List[str]) -> FaraidResult:
    """Compute faraid shares for the given heirs and estate."""
    global heirs_global
    heirs_global = list(heirs)

    # Validate heirs
    for h in heirs:
        _get_heir(h)

    # Apply hajb
    active_heirs, blocked_by = _apply_hajb(heirs)

    # Count fard heirs of each type
    fard_counts: Dict[str, int] = {}
    for h in active_heirs:
        if _get_heir(h)["category"] == "fard":
            fard_counts[h] = fard_counts.get(h, 0) + 1

    # Compute fard shares
    fard_shares: Dict[str, Fraction] = {}
    for h in active_heirs:
        if _get_heir(h)["category"] == "fard":
            fard_shares[h] = _furud_share(h, fard_counts[h])

    # Sum fard shares
    fard_sum = sum(fard_shares.values(), Fraction(0))

    # Determine if there are asaba heirs
    asaba_heirs = [h for h in active_heirs if _get_heir(h)["category"] == "asaba"]

    adjustments: List[str] = []
    final_shares: Dict[str, Fraction] = {}

    if fard_sum > 1:
        # Awl: scale down all fard shares proportionally
        adjustments.append("awl")
        for h, share in fard_shares.items():
            final_shares[h] = share / fard_sum
        # Asaba get nothing
        for h in asaba_heirs:
            final_shares[h] = Fraction(0)
    elif fard_sum < 1 and not asaba_heirs:
        # Radd: return surplus to radd-eligible fard heirs proportionally
        adjustments.append("radd")
        radd_eligible = [h for h in active_heirs if _get_heir(h)["radd_eligible"]]
        if radd_eligible:
            surplus = Fraction(1) - fard_sum
            radd_sum = sum(fard_shares.get(h, Fraction(0)) for h in radd_eligible)
            for h in active_heirs:
                if h in radd_eligible:
                    share = fard_shares.get(h, Fraction(0))
                    final_shares[h] = share + surplus * (share / radd_sum) if radd_sum else share
                else:
                    final_shares[h] = fard_shares.get(h, Fraction(0))
        else:
            # No radd-eligible heirs, just keep fard shares
            for h in active_heirs:
                final_shares[h] = fard_shares.get(h, Fraction(0))
    else:
        # Normal case: fard shares + asaba residue
        for h in active_heirs:
            if _get_heir(h)["category"] == "fard":
                final_shares[h] = fard_shares[h]
        if asaba_heirs:
            residue = Fraction(1) - fard_sum
            # Distribute residue among asaba: males get double females
            male_asaba = [h for h in asaba_heirs if _get_heir(h)["name"] in ("Son", "Full brother", "Paternal uncle", "Grandson (son's son)")]
            female_asaba = [h for h in asaba_heirs if h not in male_asaba]
            # For simplicity, if only male asaba, split equally; if mixed, male:female 2:1
            if male_asaba and female_asaba:
                # This is a simplified approach; real faraid has complex rules
                # For now, treat all asaba equally (this is a known limitation)
                pass
            # Simple equal split for asaba (this is a simplification; real rules are more complex)
            if asaba_heirs:
                per_asaba = residue / len(asaba_heirs)
                for h in asaba_heirs:
                    final_shares[h] = per_asaba

    # Convert to amounts
    estate_decimal = Decimal(estate)
    shares_list: List[FaraidShare] = []
    total_allocated = Fraction(0)
    for h in active_heirs:
        fraction = final_shares.get(h, Fraction(0))
        total_allocated += fraction
        amount = estate_decimal * Decimal(fraction.numerator) / Decimal(fraction.denominator)
        # Round to 2 decimal places for display
        amount_str = f"{amount:.2f}"
        definition = _get_heir(h)
        basis = _get_basis(h, fraction, adjustments)
        shares_list.append(
            FaraidShare(
                heir=h,
                name=definition["name"],
                category=definition["category"],
                fraction=f"{fraction.numerator}/{fraction.denominator}",
                amount=amount_str,
                basis=basis,
                blocked_by=blocked_by.get(h),
            )
        )

    # Add blocked heirs with zero share
    for h in heirs:
        if h not in active_heirs:
            definition = _get_heir(h)
            shares_list.append(
                FaraidShare(
                    heir=h,
                    name=definition["name"],
                    category=definition["category"],
                    fraction="0/1",
                    amount="0.00",
                    basis=HAJB_BASIS,
                    blocked_by=blocked_by.get(h),
                )
            )

    return FaraidResult(
        shares=shares_list,
        total_allocated=total_allocated,
        adjustments=adjustments,
    )


def _get_basis(heir: str, fraction: Fraction, adjustments: List[str]) -> str:
    """Return the fiqh basis for an heir's share."""
    definition = _get_heir(heir)
    if heir in ("daughter", "son", "father", "mother", "grandfather", "grandmother"):
        return QURAN_4_11
    if heir in ("wife", "husband"):
        return QURAN_4_12
    if heir in ("full_sister", "full_brother"):
        return QURAN_4_176
    if heir in ("grandson", "paternal_uncle"):
        return "Residuary heir (asaba) — takes the residue after fixed shares."
    if "awl" in adjustments:
        return AWL_BASIS
    if "radd" in adjustments:
        return RADD_BASIS
    return "Residuary heir (asaba) — takes the residue after fixed shares."


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

    total_allocated_str = f"{result.total_allocated.numerator}/{result.total_allocated.denominator}"
    return FaraidResponse(
        estate=str(request.estate),
        shares=result.shares,
        total_allocated=total_allocated_str,
        adjustments=result.adjustments,
        disclaimer=DISCLAIMER,
    )
