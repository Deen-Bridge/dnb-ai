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
    "granddaughter": {
        "name": "Granddaughter (son's daughter)",
        "category": "fard",
        "base_share": Fraction(1, 2),
        "blocked_by": ["son"],
        "radd_eligible": True,
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
    "consanguine_sister": {
        "name": "Consanguine sister",
        "category": "fard",
        "base_share": Fraction(1, 2),
        "blocked_by": ["son", "grandson", "father", "grandfather", "full_brother", "full_sister"],
        "radd_eligible": True,
    },
    "consanguine_brother": {
        "name": "Consanguine brother",
        "category": "asaba",
        "base_share": None,
        "blocked_by": ["son", "grandson", "father", "grandfather", "full_brother"],
        "radd_eligible": False,
    },
    "uterine_sister": {
        "name": "Uterine sister",
        "category": "fard",
        "base_share": Fraction(1, 6),
        "blocked_by": ["son", "grandson", "father", "grandfather", "full_brother", "full_sister", "consanguine_brother", "consanguine_sister"],
        "radd_eligible": True,
    },
    "uterine_brother": {
        "name": "Uterine brother",
        "category": "fard",
        "base_share": Fraction(1, 6),
        "blocked_by": ["son", "grandson", "father", "grandfather", "full_brother", "full_sister", "consanguine_brother", "consanguine_sister"],
        "radd_eligible": True,
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
    awl_applied: bool = False
    radd_applied: bool = False


def _get_heir(key: str) -> Dict[str, Any]:
    if key not in HEIRS:
        raise ValueError(f"Unknown heir: {key}")
    return HEIRS[key]


def _is_blocked(heir_key: str, present_heirs: List[str]) -> Optional[str]:
    """Return the key of the blocker if this heir is blocked, else None."""
    heir = _get_heir(heir_key)
    for blocker in heir["blocked_by"]:
        if blocker in present_heirs:
            return blocker
    return None


def distribute(estate: Decimal, heirs: List[str]) -> FaraidResult:
    """
    Compute faraid shares for the given heirs and estate.

    Implements furud (fixed shares), asaba (residuary), awl (proportional
    reduction when shares over-subscribe), radd (return of surplus when
    under-subscribed and no asaba), and hajb (blocking).

    Uses exact rational arithmetic (Fraction) throughout.
    """
    # Validate heirs
    for h in heirs:
        _get_heir(h)

    # Apply hajb: remove blocked heirs
    active_heirs = []
    blocked_info = {}
    for h in heirs:
        blocker = _is_blocked(h, heirs)
        if blocker:
            blocked_info[h] = blocker
        else:
            active_heirs.append(h)

    # Determine if there is any asaba heir (excluding those blocked)
    has_asaba = any(
        _get_heir(h)["category"] == "asaba" for h in active_heirs
    )

    # Compute fixed shares (furud)
    # For multiple daughters/sisters, adjust the share: 2+ daughters get 2/3,
    # 2+ sisters get 2/3, etc.
    fixed_shares: Dict[str, Fraction] = {}
    for h in active_heirs:
        heir = _get_heir(h)
        if heir["category"] == "fard":
            base = heir["base_share"]
            # Adjust for multiple heirs of the same type
            count = sum(1 for x in active_heirs if x == h)
            if count >= 2:
                if h in ("daughter", "granddaughter", "full_sister", "consanguine_sister"):
                    base = Fraction(2, 3)
                elif h in ("uterine_sister", "uterine_brother"):
                    base = Fraction(1, 3)
            fixed_shares[h] = base
        else:
            fixed_shares[h] = Fraction(0)

    # Sum of fixed shares
    total_fixed = sum(fixed_shares.values(), Fraction(0))

    # Determine the common denominator and apply awl if needed
    # We'll work with a common denominator that is the LCM of all denominators
    # For simplicity, we use the product of denominators (or a smarter LCM).
    denominators = [f.denominator for f in fixed_shares.values() if f != 0]
    if not denominators:
        common_den = 1
    else:
        # LCM of denominators
        from math import gcd
        common_den = 1
        for d in denominators:
            common_den = common_den * d // gcd(common_den, d)

    # Convert fixed shares to numerator over common_den
    fixed_numerators = {h: f.numerator * (common_den // f.denominator) for h, f in fixed_shares.items() if f != 0}
    total_fixed_num = sum(fixed_numerators.values(), 0)

    awl_applied = False
    radd_applied = False

    if total_fixed_num > common_den:
        # Awl: increase the denominator to total_fixed_num
        awl_applied = True
        new_den = total_fixed_num
        # Each share becomes numerator / new_den
        shares = {h: Fraction(num, new_den) for h, num in fixed_numerators.items()}
        # Asaba get nothing
        for h in active_heirs:
            if _get_heir(h)["category"] == "asaba":
                shares[h] = Fraction(0)
    else:
        # No awl; fixed shares as is
        shares = {h: Fraction(num, common_den) for h, num in fixed_numerators.items()}
        # Asaba get the residue
        residue = Fraction(1) - sum(shares.values(), Fraction(0))
        if has_asaba:
            # Distribute residue among asaba, male gets twice female
            asaba_heirs = [h for h in active_heirs if _get_heir(h)["category"] == "asaba"]
            # Compute weights: male=2, female=1
            weights = {}
            for h in asaba_heirs:
                # Determine gender from heir key (simplistic: 'son', 'brother', 'grandson' are male)
                male = h in ("son", "full_brother", "consanguine_brother", "grandson")
                weights[h] = 2 if male else 1
            total_weight = sum(weights.values())
            for h in asaba_heirs:
                shares[h] = residue * Fraction(weights[h], total_weight)
        else:
            # Radd: return surplus to eligible fard heirs (excluding spouse)
            # Only if there is surplus and no asaba
            surplus = Fraction(1) - sum(shares.values(), Fraction(0))
            if surplus > 0:
                radd_eligible = [h for h in active_heirs if _get_heir(h)["radd_eligible"]]
                if radd_eligible:
                    radd_applied = True
                    # Distribute surplus proportionally to their fixed shares
                    eligible_fixed = {h: shares[h] for h in radd_eligible}
                    total_eligible = sum(eligible_fixed.values(), Fraction(0))
                    if total_eligible > 0:
                        for h in radd_eligible:
                            shares[h] += surplus * Fraction(eligible_fixed[h], total_eligible)
                    else:
                        # If no eligible shares (shouldn't happen), give to all equally
                        for h in radd_eligible:
                            shares[h] += surplus / len(radd_eligible)

    # Build steps
    steps = []
    for h in heirs:
        if h in blocked_info:
            steps.append({
                "heir": h,
                "category": "blocked",
                "fraction": "0",
                "amount": "0",
                "basis": HAJB_BASIS,
                "blocked_by": blocked_info[h],
                "awl_applied": awl_applied,
                "radd_applied": radd_applied,
            })
        else:
            heir = _get_heir(h)
            frac = shares.get(h, Fraction(0))
            amount = estate * Decimal(frac.numerator) / Decimal(frac.denominator)
            # Determine basis
            if heir["category"] == "fard":
                if h in ("wife", "husband"):
                    basis = QURAN_4_12
                elif h in ("daughter", "granddaughter", "full_sister", "consanguine_sister", "uterine_sister", "uterine_brother"):
                    basis = QURAN_4_176 if "sister" in h or "brother" in h else QURAN_4_11
                else:
                    basis = QURAN_4_11
            else:
                basis = "Residuary (asaba) — the residue after fixed shares, per the Sunnah and consensus."
            steps.append({
                "heir": h,
                "category": heir["category"],
                "fraction": f"{frac.numerator}/{frac.denominator}",
                "amount": str(amount),
                "basis": basis,
                "blocked_by": None,
                "awl_applied": awl_applied,
                "radd_applied": radd_applied,
            })

    return FaraidResult(
        shares=shares,
        steps=steps,
        awl_applied=awl_applied,
        radd_applied=radd_applied,
    )


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
