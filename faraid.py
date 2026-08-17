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


def _apply_hajb(heirs: List[str]) -> Dict[str, Optional[str]]:
    """Return a mapping of heir key -> blocker key (or None if not blocked)."""
    blocked_by: Dict[str, Optional[str]] = {h: None for h in heirs}
    for heir in heirs:
        heir_def = _get_heir(heir)
        for blocker in heir_def["blocked_by"]:
            if blocker in heirs:
                blocked_by[heir] = blocker
                break
    return blocked_by


def _furud_shares(heirs: List[str]) -> Dict[str, Fraction]:
    """Assign fixed shares to fard heirs, respecting hajb."""
    blocked_by = _apply_hajb(heirs)
    shares: Dict[str, Fraction] = {}
    for heir in heirs:
        if blocked_by[heir] is not None:
            shares[heir] = Fraction(0)
            continue
        heir_def = _get_heir(heir)
        if heir_def["category"] == "fard":
            shares[heir] = heir_def["base_share"]
        else:
            shares[heir] = Fraction(0)  # asaba gets residue later
    return shares


def _has_asaba(heirs: List[str]) -> bool:
    blocked_by = _apply_hajb(heirs)
    return any(
        _get_heir(h)["category"] == "asaba" and blocked_by[h] is None
        for h in heirs
    )


def _asaba_share(heirs: List[str], residue: Fraction) -> Dict[str, Fraction]:
    """Distribute residue among asaba heirs, male:female 2:1."""
    blocked_by = _apply_hajb(heirs)
    asaba = [h for h in heirs if _get_heir(h)["category"] == "asaba" and blocked_by[h] is None]
    if not asaba:
        return {}
    # Simple 2:1 for male:female asaba. For this engine, we treat all asaba
    # as male (2 units) unless the key contains 'sister' or 'daughter'.
    # In a full implementation, gender would be a field; here we approximate
    # by key name for the common cases.
    units = 0
    for h in asaba:
        if "sister" in h or "daughter" in h:
            units += 1
        else:
            units += 2
    per_unit = residue / units
    result = {}
    for h in asaba:
        if "sister" in h or "daughter" in h:
            result[h] = per_unit
        else:
            result[h] = per_unit * 2
    return result


def _apply_awl(shares: Dict[str, Fraction]) -> Tuple[Dict[str, Fraction], bool]:
    total = sum(shares.values(), Fraction(0))
    if total > 1:
        factor = Fraction(1) / total
        return {k: v * factor for k, v in shares.items()}, True
    return shares, False


def _apply_radd(shares: Dict[str, Fraction], heirs: List[str]) -> Tuple[Dict[str, Fraction], bool]:
    total = sum(shares.values(), Fraction(0))
    if total >= 1:
        return shares, False
    # Only radd to eligible heirs (exclude spouses and asaba)
    eligible = [h for h in heirs if _get_heir(h)["radd_eligible"] and shares.get(h, Fraction(0)) > 0]
    if not eligible:
        return shares, False
    surplus = Fraction(1) - total
    # Distribute surplus proportionally to eligible shares
    eligible_total = sum(shares[h] for h in eligible)
    if eligible_total == 0:
        return shares, False
    for h in eligible:
        shares[h] += surplus * (shares[h] / eligible_total)
    return shares, True


def distribute(estate: Decimal, heirs: List[str]) -> FaraidResult:
    """Compute faraid shares for the given heirs and estate."""
    # Validate heirs
    for h in heirs:
        _get_heir(h)

    blocked_by = _apply_hajb(heirs)
    shares = _furud_shares(heirs)

    # Apply awl if needed
    shares, awl_applied = _apply_awl(shares)

    # Distribute residue to asaba
    residue = Fraction(1) - sum(shares.values(), Fraction(0))
    if residue > 0 and _has_asaba(heirs):
        asaba_shares = _asaba_share(heirs, residue)
        for h, v in asaba_shares.items():
            shares[h] = v

    # Apply radd if no asaba and surplus remains
    radd_applied = False
    if not _has_asaba(heirs):
        shares, radd_applied = _apply_radd(shares, heirs)

    # Build steps
    steps = []
    for h in heirs:
        heir_def = _get_heir(h)
        fraction = shares.get(h, Fraction(0))
        amount = (estate * Decimal(fraction.numerator) / Decimal(fraction.denominator)).quantize(Decimal("0.01"))
        basis = ""
        if blocked_by[h] is not None:
            basis = HAJB_BASIS
        elif heir_def["category"] == "fard":
            if h in ["wife", "husband"]:
                basis = QURAN_4_12
            elif h in ["full_sister", "full_brother"]:
                basis = QURAN_4_176
            else:
                basis = QURAN_4_11
        else:
            basis = "Residuary (asaba) — residue after fixed shares, per the Sunnah and consensus."
        steps.append({
            "heir": h,
            "category": "blocked" if blocked_by[h] is not None else heir_def["category"],
            "fraction": f"{fraction.numerator}/{fraction.denominator}",
            "amount": str(amount),
            "basis": basis,
            "blocked_by": blocked_by[h],
            "awl_applied": awl_applied,
            "radd_applied": radd_applied,
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
    return FaraidResponse(
        estate=str(request.estate),
        steps=[FaraidStep(**step) for step in result.steps],
        disclaimer=DISCLAIMER,
    )
