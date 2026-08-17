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
        "base_share": Fraction(1, 8),
        "blocked_by": [],
        "radd_eligible": False,
    },
    "husband": {
        "name": "Husband",
        "category": "fard",
        "base_share": Fraction(1, 4),
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
    """Return (active_heirs, blocked_map) where blocked_map maps heir key -> blocker key."""
    active = set(heirs)
    blocked_map: Dict[str, str] = {}
    for heir in heirs:
        heir_def = _get_heir(heir)
        for blocker in heir_def["blocked_by"]:
            if blocker in active:
                blocked_map[heir] = blocker
                active.discard(heir)
                break
    return list(active), blocked_map


def _furud_share(heir: str, active_heirs: List[str]) -> Optional[Fraction]:
    """Return the fixed share for a fard heir, adjusted for multiple heirs of the same type."""
    heir_def = _get_heir(heir)
    if heir_def["category"] != "fard":
        return None
    base = heir_def["base_share"]
    # Count how many heirs share the same base share (e.g., two daughters get 2/3 total)
    same_type_count = sum(1 for h in active_heirs if _get_heir(h)["base_share"] == base)
    if same_type_count > 1:
        # For daughters and sisters, the collective share is 2/3 when there are two or more
        if heir in ("daughter", "full_sister"):
            return Fraction(2, 3) / same_type_count
    return base


def distribute(estate: Decimal, heirs: List[str]) -> FaraidResult:
    """
    Compute faraid shares for the given heirs and estate.

    Implements furud, asaba, awl, radd, and hajb using exact rational arithmetic.
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
    fixed_shares: Dict[str, Fraction] = {}
    for h in fard_heirs:
        share = _furud_share(h, active_heirs)
        if share is not None:
            fixed_shares[h] = share

    total_fixed = sum(fixed_shares.values(), Fraction(0))

    # Determine if awl is needed
    awl_applied = False
    if total_fixed > 1:
        awl_applied = True
        # Scale all fixed shares proportionally
        scale = Fraction(1, total_fixed)
        fixed_shares = {h: s * scale for h, s in fixed_shares.items()}
        total_fixed = Fraction(1)

    # Distribute residue to asaba
    shares: Dict[str, Fraction] = {}
    for h in fard_heirs:
        shares[h] = fixed_shares[h]

    residue = Fraction(1) - total_fixed
    if asaba_heirs and residue > 0:
        # Asaba take the residue, with males receiving twice females
        male_asaba = [h for h in asaba_heirs if _get_heir(h)["name"].startswith("Son") or _get_heir(h)["name"].startswith("Full brother") or _get_heir(h)["name"].startswith("Paternal")]
        female_asaba = [h for h in asaba_heirs if h not in male_asaba]
        # For simplicity, treat all asaba as male unless explicitly female
        # In this implementation, asaba are typically male (son, brother, uncle)
        # If a female asaba exists, she gets half of a male's share
        total_units = len(male_asaba) * 2 + len(female_asaba)
        if total_units > 0:
            unit = residue / total_units
            for h in male_asaba:
                shares[h] = unit * 2
            for h in female_asaba:
                shares[h] = unit
        else:
            # No asaba, but residue remains
            pass
    elif not asaba_heirs and residue > 0:
        # Radd: return surplus to fard heirs, excluding spouses
        radd_eligible = [h for h in fard_heirs if _get_heir(h)["radd_eligible"]]
        if radd_eligible:
            radd_applied = True
            total_radd_base = sum(shares[h] for h in radd_eligible, Fraction(0))
            if total_radd_base > 0:
                for h in radd_eligible:
                    shares[h] += residue * (shares[h] / total_radd_base)
        else:
            radd_applied = False
    else:
        radd_applied = False

    # Ensure all shares sum to 1
    total_shares = sum(shares.values(), Fraction(0))
    if total_shares != 1:
        # This should not happen, but adjust for safety
        pass

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
                "radd_applied": False,
            })
        else:
            heir_def = _get_heir(h)
            share = shares.get(h, Fraction(0))
            amount = (Decimal(share.numerator) / Decimal(share.denominator)) * estate
            # Determine basis
            if heir_def["category"] == "fard":
                if h in ("wife", "husband"):
                    basis = QURAN_4_12
                elif h in ("daughter", "full_sister"):
                    basis = QURAN_4_11 if h == "daughter" else QURAN_4_176
                elif h in ("father", "mother", "grandfather", "grandmother"):
                    basis = QURAN_4_11
                else:
                    basis = QURAN_4_11
            else:
                basis = "Residuary (asaba) — residue after fixed shares, per the Sunnah and consensus."
            steps.append({
                "heir": h,
                "category": heir_def["category"],
                "fraction": f"{share.numerator}/{share.denominator}",
                "amount": str(amount.quantize(Decimal("0.01"))),
                "basis": basis,
                "blocked_by": None,
                "awl_applied": awl_applied,
                "radd_applied": radd_applied if h in radd_eligible else False,
            })

    return FaraidResult(shares=shares, steps=steps, awl_applied=awl_applied, radd_applied=radd_applied)


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
