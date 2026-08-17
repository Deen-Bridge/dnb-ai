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
    estate: Fraction
    steps: List[FaraidStep]
    disclaimer: str = DISCLAIMER


def _get_heir(key: str) -> Dict[str, Any]:
    if key not in HEIRS:
        raise ValueError(f"Unknown heir: {key}")
    return HEIRS[key]


def _apply_hajb(heirs: List[str]) -> Dict[str, Optional[str]]:
    """Return a mapping of heir key -> blocker key (or None if not blocked)."""
    blocked_by: Dict[str, Optional[str]] = {h: None for h in heirs}
    for heir in heirs:
        for blocker in HEIRS[heir]["blocked_by"]:
            if blocker in heirs:
                blocked_by[heir] = blocker
                break
    return blocked_by


def distribute(estate: Decimal, heirs: List[str]) -> FaraidResult:
    """
    Compute faraid shares for the given heirs.

    Implements furud (fixed shares), asaba (residuary), awl (proportional
    reduction when shares exceed the estate), radd (return of surplus when
    shares fall short and no asaba exists), and hajb (blocking).
    """
    # Validate and normalize
    estate_frac = Fraction(estate)
    if len(heirs) == 0:
        raise ValueError("At least one heir is required")
    for h in heirs:
        _get_heir(h)

    # Apply hajb
    blocked_by = _apply_hajb(heirs)
    active_heirs = [h for h in heirs if blocked_by[h] is None]

    # Assign fixed shares (furud)
    shares: Dict[str, Fraction] = {}
    for h in active_heirs:
        info = HEIRS[h]
        if info["category"] == "fard":
            shares[h] = info["base_share"]
        else:
            shares[h] = Fraction(0)  # asaba gets residue later

    # Sum fixed shares
    fixed_sum = sum(shares.values(), Fraction(0))

    # Determine asaba heirs (those with category asaba and not blocked)
    asaba_heirs = [h for h in active_heirs if HEIRS[h]["category"] == "asaba"]

    # Awl: if fixed shares exceed 1, scale down proportionally
    awl_applied = False
    if fixed_sum > 1:
        awl_applied = True
        scale = Fraction(1) / fixed_sum
        for h in shares:
            shares[h] *= scale
        fixed_sum = Fraction(1)

    # Residue for asaba
    residue = Fraction(1) - fixed_sum
    if asaba_heirs:
        # Distribute residue among asaba: males take twice females
        # For simplicity, we treat all asaba as male unless key contains 'sister' or 'daughter'
        # In a full implementation, we would need gender info per heir.
        # Here we assume asaba heirs are male unless they are sisters/daughters.
        # This is a simplification; the engine should be extended for mixed asaba.
        male_asaba = [h for h in asaba_heirs if "sister" not in h and "daughter" not in h]
        female_asaba = [h for h in asaba_heirs if "sister" in h or "daughter" in h]
        # For this issue, we handle the common case: sons and daughters as asaba together.
        # If both sons and daughters are present, sons take twice daughters.
        # We'll implement a general rule: each male unit = 2, each female unit = 1.
        # But we need to know gender. We'll infer from key.
        # For simplicity, we'll treat all asaba as male unless key contains 'sister' or 'daughter'.
        # This is a known limitation; the engine can be extended.
        # For the acceptance criteria, the key cases are:
        # - wife + sons + daughter: sons and daughter as asaba 2:1
        # - full brother as asaba alone
        # - paternal uncle as asaba alone
        # We'll implement a generic 2:1 for mixed gender asaba.
        # Determine gender from key.
        def is_female(key: str) -> bool:
            return "sister" in key or "daughter" in key

        female_count = sum(1 for h in asaba_heirs if is_female(h))
        male_count = len(asaba_heirs) - female_count
        total_units = male_count * 2 + female_count
        if total_units > 0:
            unit_share = residue / total_units
            for h in asaba_heirs:
                if is_female(h):
                    shares[h] = unit_share
                else:
                    shares[h] = unit_share * 2
        else:
            # No asaba units (shouldn't happen)
            pass
    else:
        # Radd: if no asaba and fixed_sum < 1, return surplus to radd-eligible sharers
        radd_applied = False
        if fixed_sum < 1:
            radd_eligible = [h for h in active_heirs if HEIRS[h]["radd_eligible"]]
            if radd_eligible:
                radd_applied = True
                surplus = Fraction(1) - fixed_sum
                # Distribute surplus proportionally among radd-eligible
                # But note: if there is only one radd-eligible, they get all surplus
                # For multiple, distribute proportionally to their fixed shares
                eligible_shares = {h: shares[h] for h in radd_eligible}
                eligible_sum = sum(eligible_shares.values(), Fraction(0))
                if eligible_sum > 0:
                    for h in radd_eligible:
                        shares[h] += surplus * (eligible_shares[h] / eligible_sum)
                else:
                    # If no fixed shares (shouldn't happen), split equally
                    for h in radd_eligible:
                        shares[h] += surplus / len(radd_eligible)
            # If no radd-eligible, surplus goes to the state (not distributed)

    # Build steps
    steps: List[FaraidStep] = []
    for h in heirs:
        info = HEIRS[h]
        if blocked_by[h] is not None:
            steps.append(
                FaraidStep(
                    heir=info["name"],
                    category="blocked",
                    fraction="0",
                    amount="0",
                    basis=HAJB_BASIS,
                    blocked_by=HEIRS[blocked_by[h]]["name"],
                )
            )
        else:
            frac = shares.get(h, Fraction(0))
            amount = estate_frac * frac
            # Determine basis
            if info["category"] == "fard":
                if h in ["wife", "husband"]:
                    basis = QURAN_4_12
                elif h in ["full_sister", "full_brother"]:
                    basis = QURAN_4_176
                else:
                    basis = QURAN_4_11
            else:
                basis = "Residuary (asaba) — the residue after fixed shares, per the established juristic principle."
            steps.append(
                FaraidStep(
                    heir=info["name"],
                    category=info["category"],
                    fraction=str(frac),
                    amount=str(amount),
                    basis=basis,
                    awl_applied=awl_applied,
                    radd_applied=radd_applied and h in [x for x in active_heirs if HEIRS[x]["radd_eligible"]],
                )
            )

    return FaraidResult(estate=estate_frac, steps=steps)


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
        estate=str(result.estate),
        steps=result.steps,
        disclaimer=result.disclaimer,
    )
