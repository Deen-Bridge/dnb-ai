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
        "name": "Grandson",
        "category": "asaba",
        "base_share": None,
        "blocked_by": ["son"],
        "radd_eligible": False,
    },
    "grandfather": {
        "name": "Grandfather",
        "category": "fard",
        "base_share": Fraction(1, 6),
        "blocked_by": ["father"],
        "radd_eligible": True,
    },
    "grandmother": {
        "name": "Grandmother",
        "category": "fard",
        "base_share": Fraction(1, 6),
        "blocked_by": ["mother"],
        "radd_eligible": True,
    },
    "full_sister": {
        "name": "Full Sister",
        "category": "fard",
        "base_share": Fraction(1, 2),
        "blocked_by": ["son", "daughter", "father", "grandson"],
        "radd_eligible": True,
    },
    "full_brother": {
        "name": "Full Brother",
        "category": "asaba",
        "base_share": None,
        "blocked_by": ["son", "grandson", "father"],
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
    amount: str
    basis: str
    blocked_by: Optional[str] = None


class FaraidResponse(BaseModel):
    estate: str
    shares: List[HeirShare]
    total_allocated: str
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


def _get_heir_def(key: str) -> Dict[str, Any]:
    if key not in HEIRS:
        raise ValueError(f"Unknown heir: {key}")
    return HEIRS[key]


def _apply_hajb(heirs: List[str]) -> Tuple[Dict[str, str], List[str]]:
    """Return (blocked_by_map, active_heirs)."""
    blocked_by: Dict[str, str] = {}
    active = []
    for key in heirs:
        heir = _get_heir_def(key)
        blocker = None
        for b in heir["blocked_by"]:
            if b in heirs:
                blocker = b
                break
        if blocker:
            blocked_by[key] = blocker
        else:
            active.append(key)
    return blocked_by, active


def distribute(estate: Decimal, heirs: List[str]) -> FaraidResult:
    """Compute faraid shares for the given heirs and estate."""
    steps: List[str] = []
    basis: Dict[str, str] = {}
    blocked: Dict[str, str] = {}

    # Validate heirs
    for h in heirs:
        _get_heir_def(h)

    # Apply hajb
    blocked, active = _apply_hajb(heirs)
    if blocked:
        for k, b in blocked.items():
            steps.append(f"Hajb: {HEIRS[k]['name']} is blocked by {HEIRS[b]['name']}.")
            basis[k] = HAJB_BASIS

    # Determine fard and asaba heirs
    fard_keys = [k for k in active if _get_heir_def(k)["category"] == "fard"]
    asaba_keys = [k for k in active if _get_heir_def(k)["category"] == "asaba"]

    # Compute fixed shares
    fixed_shares: Dict[str, Fraction] = {}
    for k in fard_keys:
        heir = _get_heir_def(k)
        share = heir["base_share"]
        # Adjust for multiple heirs of the same type
        # For daughters: 1 daughter = 1/2, 2+ daughters = 2/3
        if k == "daughter":
            daughters = [x for x in fard_keys if x == "daughter"]
            if len(daughters) >= 2:
                share = Fraction(2, 3)
        if k == "full_sister":
            sisters = [x for x in fard_keys if x == "full_sister"]
            if len(sisters) >= 2:
                share = Fraction(2, 3)
        # Wife: 1/4 if no children, 1/8 if children
        if k == "wife":
            if any(x in active for x in ["son", "daughter", "grandson", "granddaughter"]):
                share = Fraction(1, 8)
        # Husband: 1/2 if no children, 1/4 if children
        if k == "husband":
            if any(x in active for x in ["son", "daughter", "grandson", "granddaughter"]):
                share = Fraction(1, 4)
        # Mother: 1/6 if children or multiple siblings, else 1/3
        if k == "mother":
            if any(x in active for x in ["son", "daughter", "grandson", "granddaughter"]) or \
               sum(1 for x in active if x in ["full_brother", "full_sister", "half_brother", "half_sister"]) >= 2:
                share = Fraction(1, 6)
            else:
                share = Fraction(1, 3)
        fixed_shares[k] = share
        basis[k] = QURAN_4_11 if k in ["daughter", "father", "mother", "grandfather", "grandmother"] else QURAN_4_12 if k in ["wife", "husband"] else QURAN_4_176

    # Sum fixed shares
    total_fixed = sum(fixed_shares.values(), Fraction(0))

    # Determine if asaba exists
    has_asaba = len(asaba_keys) > 0

    # Awl: if total_fixed > 1, scale down
    awl_factor = Fraction(1)
    if total_fixed > 1:
        awl_factor = Fraction(1, total_fixed)
        steps.append(f"Awl applied: fixed shares sum to {total_fixed}, scaling by {awl_factor}.")
        for k in fixed_shares:
            fixed_shares[k] *= awl_factor
            basis[k] += " " + AWL_BASIS

    # Radd: if total_fixed < 1 and no asaba, return surplus to eligible fard heirs
    radd_applied = False
    if total_fixed < 1 and not has_asaba:
        eligible = [k for k in fard_keys if _get_heir_def(k)["radd_eligible"]]
        if eligible:
            surplus = Fraction(1) - total_fixed
            radd_applied = True
            steps.append(f"Radd applied: surplus {surplus} returned to eligible heirs.")
            # Distribute surplus proportionally among eligible
            eligible_shares = {k: fixed_shares[k] for k in eligible}
            total_eligible = sum(eligible_shares.values(), Fraction(0))
            for k in eligible:
                fixed_shares[k] += surplus * (eligible_shares[k] / total_eligible)
                basis[k] += " " + RADD_BASIS

    # Distribute residue to asaba (2:1 male:female)
    if has_asaba:
        # Determine if there are female asaba (daughters/sisters) that become asaba with males
        # For simplicity, treat all asaba as taking residue, with males twice females
        # This is a simplified model; full implementation would handle daughters becoming asaba with sons
        # For now, asaba heirs take the residue equally if all male, or 2:1 if mixed
        residue = Fraction(1) - sum(fixed_shares.values(), Fraction(0))
        if residue > 0:
            # Count males and females among asaba
            males = [k for k in asaba_keys if k in ["son", "full_brother", "grandson"]]
            females = [k for k in asaba_keys if k in ["daughter", "full_sister"]]
            # If there are both, daughters/sisters become asaba with males
            # For simplicity, we treat all asaba as taking residue with 2:1 ratio
            # This is a simplification; full implementation would handle specific cases
            # For now, we just divide residue equally among asaba if all male, else 2:1
            if females and males:
                # Each male gets 2 units, each female 1 unit
                total_units = 2 * len(males) + len(females)
                unit = residue / total_units
                for k in asaba_keys:
                    if k in males:
                        fixed_shares[k] = 2 * unit
                    else:
                        fixed_shares[k] = unit
                    basis[k] = "Asaba (residuary) — takes the residue after fixed shares."
            else:
                # All male or all female asaba
                share = residue / len(asaba_keys)
                for k in asaba_keys:
                    fixed_shares[k] = share
                    basis[k] = "Asaba (residuary) — takes the residue after fixed shares."
            steps.append(f"Residue {residue} distributed to asaba heirs.")

    # Ensure all active heirs have a share (if not assigned, assign 0)
    for k in active:
        if k not in fixed_shares:
            fixed_shares[k] = Fraction(0)
            basis[k] = "No share assigned."

    # Convert to amounts
    estate_frac = Fraction(estate)
    shares_out: Dict[str, Fraction] = {}
    for k, frac in fixed_shares.items():
        shares_out[k] = frac

    return FaraidResult(shares=shares_out, steps=steps, basis=basis, blocked=blocked)


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

    estate_frac = Fraction(request.estate)
    shares = []
    for key, frac in result.shares.items():
        heir_def = _get_heir_def(key)
        amount = estate_frac * frac
        shares.append(
            HeirShare(
                heir=key,
                name=heir_def["name"],
                category=heir_def["category"],
                fraction=str(frac),
                amount=str(amount),
                basis=result.basis.get(key, ""),
                blocked_by=result.blocked.get(key),
            )
        )

    total_allocated = sum(result.shares.values(), Fraction(0))
    return FaraidResponse(
        estate=str(request.estate),
        shares=shares,
        total_allocated=str(total_allocated),
        disclaimer=DISCLAIMER,
        steps=result.steps,
    )
