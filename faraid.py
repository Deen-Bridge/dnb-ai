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
        "blocked_by": ["son", "daughter"],
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
    "full_brother": {
        "name": "Full brother",
        "category": "asaba",
        "base_share": None,
        "blocked_by": ["son", "grandson", "father", "grandfather"],
        "radd_eligible": False,
    },
    "full_sister": {
        "name": "Full sister",
        "category": "fard",
        "base_share": Fraction(1, 2),
        "blocked_by": ["son", "grandson", "daughter", "granddaughter", "father", "grandfather"],
        "radd_eligible": True,
    },
    "consanguine_brother": {
        "name": "Consanguine brother (father's side)",
        "category": "asaba",
        "base_share": None,
        "blocked_by": ["son", "grandson", "father", "grandfather", "full_brother"],
        "radd_eligible": False,
    },
    "consanguine_sister": {
        "name": "Consanguine sister (father's side)",
        "category": "fard",
        "base_share": Fraction(1, 2),
        "blocked_by": ["son", "grandson", "daughter", "granddaughter", "father", "grandfather", "full_brother", "full_sister"],
        "radd_eligible": True,
    },
    "uterine_brother": {
        "name": "Uterine brother (mother's side)",
        "category": "fard",
        "base_share": Fraction(1, 6),
        "blocked_by": ["son", "grandson", "daughter", "granddaughter", "father", "grandfather"],
        "radd_eligible": True,
    },
    "uterine_sister": {
        "name": "Uterine sister (mother's side)",
        "category": "fard",
        "base_share": Fraction(1, 6),
        "blocked_by": ["son", "grandson", "daughter", "granddaughter", "father", "grandfather"],
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
    awl_applied: bool = False
    radd_applied: bool = False


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
    awl_applied: bool
    radd_applied: bool


def _get_heir(key: str) -> Dict[str, Any]:
    if key not in HEIRS:
        raise ValueError(f"Unknown heir: {key}")
    return HEIRS[key]


def _is_blocked(key: str, present: List[str]) -> Optional[str]:
    """Return the key of the blocker if this heir is blocked, else None."""
    heir = _get_heir(key)
    for blocker in heir["blocked_by"]:
        if blocker in present:
            return blocker
    return None


def _furud_share(key: str, present: List[str]) -> Optional[Fraction]:
    """Return the fixed share for a fard heir, adjusted for multiple heirs."""
    heir = _get_heir(key)
    if heir["category"] != "fard":
        return None
    base = heir["base_share"]
    # Special rules for daughters and sisters when multiple
    if key in ("daughter", "granddaughter", "full_sister", "consanguine_sister"):
        # Count same-category heirs
        same = [k for k in present if k == key]
        if len(same) >= 2:
            return Fraction(2, 3)
        return base
    if key in ("uterine_brother", "uterine_sister"):
        # Uterine siblings share 1/3 when multiple
        same = [k for k in present if k in ("uterine_brother", "uterine_sister")]
        if len(same) >= 2:
            return Fraction(1, 3)
        return base
    return base


def distribute(estate: Decimal, heirs: List[str]) -> FaraidResult:
    """Compute faraid shares for the given heirs and estate."""
    steps: List[str] = []
    present = list(dict.fromkeys(heirs))  # deduplicate, preserve order

    # Validate all heirs
    for key in present:
        _get_heir(key)

    # Apply hajb: remove blocked heirs
    active = []
    for key in present:
        blocker = _is_blocked(key, present)
        if blocker:
            steps.append(f"{HEIRS[key]['name']} is blocked by {HEIRS[blocker]['name']} (hajb).")
        else:
            active.append(key)

    # Compute furud shares
    furud: Dict[str, Fraction] = {}
    asaba: List[str] = []
    for key in active:
        heir = _get_heir(key)
        if heir["category"] == "fard":
            share = _furud_share(key, active)
            if share is not None:
                furud[key] = share
                basis = _basis_for(key)
                steps.append(f"{HEIRS[key]['name']} gets {share} ({basis}).")
        else:
            asaba.append(key)

    # Sum furud
    total_furud = sum(furud.values(), Fraction(0))

    # Check for awl
    awl_applied = False
    if total_furud > 1:
        awl_applied = True
        steps.append(f"Total fixed shares {total_furud} exceed the estate; applying awl.")
        # Scale all furud shares proportionally
        furud = {k: v / total_furud for k, v in furud.items()}
        steps.append(f"All shares scaled by 1/{total_furud}.")

    # Compute residue
    residue = Fraction(1) - sum(furud.values(), Fraction(0))

    # Distribute residue to asaba
    if asaba:
        # Male asaba take double the female asaba (2:1)
        male_asaba = [k for k in asaba if _get_heir(k)["name"].startswith("Son") or "brother" in k]
        female_asaba = [k for k in asaba if k not in male_asaba]
        # For simplicity, treat all asaba as male unless explicitly female
        # In this implementation, only sons and brothers are asaba
        total_asaba_units = len(male_asaba) * 2 + len(female_asaba)
        if total_asaba_units > 0:
            unit = residue / total_asaba_units
            for k in male_asaba:
                furud[k] = unit * 2
                steps.append(f"{HEIRS[k]['name']} takes residue as asaba: {furud[k]}.")
            for k in female_asaba:
                furud[k] = unit
                steps.append(f"{HEIRS[k]['name']} takes residue as asaba: {furud[k]}.")
        else:
            steps.append("No asaba heirs; residue remains.")
    else:
        steps.append("No asaba heirs; residue remains for radd.")

    # Apply radd if residue > 0 and no asaba
    radd_applied = False
    if residue > 0 and not asaba:
        # Exclude spouses from radd
        radd_eligible = [k for k in furud if _get_heir(k)["radd_eligible"]]
        if radd_eligible:
            radd_applied = True
            total_radd_units = sum(furud[k] for k in radd_eligible)
            if total_radd_units > 0:
                steps.append(f"Applying radd: surplus {residue} returned proportionally to eligible heirs.")
                for k in radd_eligible:
                    furud[k] += residue * (furud[k] / total_radd_units)
                    steps.append(f"{HEIRS[k]['name']} receives radd share.")
            else:
                steps.append("No radd-eligible heirs; surplus remains undistributed.")
        else:
            steps.append("No radd-eligible heirs; surplus remains undistributed.")

    # Ensure blocked heirs get zero
    for key in present:
        if key not in furud:
            furud[key] = Fraction(0)

    # Verify sum
    total = sum(furud.values(), Fraction(0))
    if total != 1:
        steps.append(f"Note: total allocated {total} (may be less than 1 if no radd).")

    return FaraidResult(shares=furud, steps=steps, awl_applied=awl_applied, radd_applied=radd_applied)


def _basis_for(key: str) -> str:
    """Return the fiqh basis for a fard heir."""
    if key in ("daughter", "granddaughter", "full_sister", "consanguine_sister"):
        return QURAN_4_11
    if key in ("wife", "husband"):
        return QURAN_4_12
    if key in ("full_brother", "full_sister", "consanguine_brother", "consanguine_sister", "uterine_brother", "uterine_sister"):
        return QURAN_4_176
    if key in ("father", "mother", "grandfather", "grandmother"):
        return QURAN_4_11
    return "Juristic principle"


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

    estate_frac = Fraction(request.estate)
    shares = []
    for key, frac in result.shares.items():
        heir = _get_heir(key)
        amount = estate_frac * frac
        basis = _basis_for(key)
        if frac == 0:
            basis = HAJB_BASIS
        shares.append(
            HeirShare(
                heir=key,
                name=heir["name"],
                category=heir["category"],
                fraction=str(frac),
                amount=str(amount),
                basis=basis,
                blocked_by=_is_blocked(key, request.heirs),
                awl_applied=result.awl_applied,
                radd_applied=result.radd_applied,
            )
        )

    total_allocated = sum((Fraction(s.fraction) for s in shares), Fraction(0))
    return FaraidResponse(
        estate=str(request.estate),
        shares=shares,
        total_allocated=str(total_allocated),
        disclaimer=DISCLAIMER,
        steps=result.steps,
    )
