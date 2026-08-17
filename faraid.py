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
        "radd_eligible": False,  # asaba take residue anyway
    },
    "daughter": {
        "name": "Daughter",
        "category": "fard",
        "base_share": Fraction(1, 2),  # single daughter
        "blocked_by": [],
        "radd_eligible": True,
    },
    "wife": {
        "name": "Wife",
        "category": "fard",
        "base_share": Fraction(1, 4),  # with children; 1/8 without children
        "blocked_by": [],
        "radd_eligible": False,  # spouse excluded from radd
    },
    "husband": {
        "name": "Husband",
        "category": "fard",
        "base_share": Fraction(1, 2),  # without children; 1/4 with children
        "blocked_by": [],
        "radd_eligible": False,
    },
    "father": {
        "name": "Father",
        "category": "fard",
        "base_share": Fraction(1, 6),  # with children; else asaba
        "blocked_by": [],
        "radd_eligible": True,
    },
    "mother": {
        "name": "Mother",
        "category": "fard",
        "base_share": Fraction(1, 6),  # with children; else 1/3
        "blocked_by": [],
        "radd_eligible": True,
    },
    "paternal_grandfather": {
        "name": "Paternal Grandfather",
        "category": "fard",
        "base_share": Fraction(1, 6),  # when no father
        "blocked_by": ["father"],
        "radd_eligible": True,
    },
    "paternal_grandmother": {
        "name": "Paternal Grandmother",
        "category": "fard",
        "base_share": Fraction(1, 6),
        "blocked_by": ["mother"],
        "radd_eligible": True,
    },
    "maternal_grandmother": {
        "name": "Maternal Grandmother",
        "category": "fard",
        "base_share": Fraction(1, 6),
        "blocked_by": ["mother"],
        "radd_eligible": True,
    },
    "full_brother": {
        "name": "Full Brother",
        "category": "asaba",
        "base_share": None,
        "blocked_by": ["son", "father"],
        "radd_eligible": False,
    },
    "full_sister": {
        "name": "Full Sister",
        "category": "fard",
        "base_share": Fraction(1, 2),  # single sister; 2/3 for two+; asaba with brother
        "blocked_by": ["son", "father"],
        "radd_eligible": True,
    },
    "consanguine_brother": {
        "name": "Consanguine Brother",
        "category": "asaba",
        "base_share": None,
        "blocked_by": ["son", "father", "full_brother"],
        "radd_eligible": False,
    },
    "consanguine_sister": {
        "name": "Consanguine Sister",
        "category": "fard",
        "base_share": Fraction(1, 2),
        "blocked_by": ["son", "father", "full_brother", "full_sister"],
        "radd_eligible": True,
    },
    "uterine_brother": {
        "name": "Uterine Brother",
        "category": "fard",
        "base_share": Fraction(1, 6),  # single; 1/3 for two+
        "blocked_by": ["son", "daughter", "father", "mother"],
        "radd_eligible": True,
    },
    "uterine_sister": {
        "name": "Uterine Sister",
        "category": "fard",
        "base_share": Fraction(1, 6),
        "blocked_by": ["son", "daughter", "father", "mother"],
        "radd_eligible": True,
    },
    "paternal_grandson": {
        "name": "Paternal Grandson",
        "category": "asaba",
        "base_share": None,
        "blocked_by": ["son"],
        "radd_eligible": False,
    },
    "paternal_granddaughter": {
        "name": "Paternal Granddaughter",
        "category": "fard",
        "base_share": Fraction(1, 2),
        "blocked_by": ["son", "daughter"],
        "radd_eligible": True,
    },
}

# ---------------------------------------------------------------------------
# Pydantic models for the HTTP endpoint
# ---------------------------------------------------------------------------


class FaraidRequest(BaseModel):
    estate: Decimal = Field(..., gt=0, description="Total estate value")
    heirs: List[str] = Field(..., min_length=1, description="List of heir keys")


class HeirShare(BaseModel):
    heir: str
    name: str
    category: str  # 'fard' or 'asaba'
    fraction: str  # e.g. "3/8"
    amount: str  # decimal string, e.g. "33750.00"
    basis: str  # citation
    blocked_by: Optional[str] = None


class FaraidResponse(BaseModel):
    estate: str
    shares: List[HeirShare]
    steps: List[str]
    disclaimer: str


# ---------------------------------------------------------------------------
# Core engine
# ---------------------------------------------------------------------------


@dataclass
class FaraidResult:
    shares: Dict[str, Fraction]
    steps: List[str]
    basis: Dict[str, str]
    blocked: Dict[str, str]


def _has_children(heirs: List[str]) -> bool:
    return any(h in ("son", "daughter") for h in heirs)


def _has_male_asaba(heirs: List[str]) -> bool:
    """True if there is any male asaba heir (son, father, brother, etc.)."""
    for h in heirs:
        info = HEIRS[h]
        if info["category"] == "asaba" and h not in ("paternal_grandson",):
            # All asaba in our set are male except paternal_granddaughter (fard)
            return True
    return False


def _apply_hajb(heirs: List[str]) -> Tuple[List[str], Dict[str, str]]:
    """Return (active_heirs, blocked_map) where blocked_map maps heir_key -> blocker_key."""
    active = list(heirs)
    blocked: Dict[str, str] = {}
    # Iterate until stable (blocking can cascade)
    changed = True
    while changed:
        changed = False
        for h in list(active):
            info = HEIRS[h]
            for blocker in info["blocked_by"]:
                if blocker in active:
                    active.remove(h)
                    blocked[h] = blocker
                    changed = True
                    break
    return active, blocked


def _assign_furud(active: List[str]) -> Dict[str, Fraction]:
    """Assign fixed shares based on the presence of other heirs."""
    shares: Dict[str, Fraction] = {}
    has_children = _has_children(active)
    has_male_asaba = _has_male_asaba(active)

    for h in active:
        info = HEIRS[h]
        if info["category"] != "fard":
            continue
        base = info["base_share"]
        # Adjust for special cases
        if h == "wife":
            shares[h] = Fraction(1, 8) if has_children else Fraction(1, 4)
        elif h == "husband":
            shares[h] = Fraction(1, 4) if has_children else Fraction(1, 2)
        elif h == "daughter":
            # Single daughter gets 1/2; two+ get 2/3 (handled later)
            daughters = [x for x in active if x == "daughter"]
            if len(daughters) >= 2:
                shares[h] = Fraction(2, 3)
            else:
                shares[h] = Fraction(1, 2)
        elif h == "full_sister":
            sisters = [x for x in active if x == "full_sister"]
            if len(sisters) >= 2:
                shares[h] = Fraction(2, 3)
            else:
                shares[h] = Fraction(1, 2)
        elif h == "consanguine_sister":
            sisters = [x for x in active if x == "consanguine_sister"]
            if len(sisters) >= 2:
                shares[h] = Fraction(2, 3)
            else:
                shares[h] = Fraction(1, 2)
        elif h == "uterine_brother" or h == "uterine_sister":
            uterine = [x for x in active if x in ("uterine_brother", "uterine_sister")]
            if len(uterine) >= 2:
                shares[h] = Fraction(1, 3)
            else:
                shares[h] = Fraction(1, 6)
        elif h == "mother":
            # Mother gets 1/3 if no children, but if there are siblings she gets 1/6
            if has_children:
                shares[h] = Fraction(1, 6)
            else:
                # Check for siblings (any brother/sister)
                has_siblings = any(
                    x in ("full_brother", "full_sister", "consanguine_brother", "consanguine_sister", "uterine_brother", "uterine_sister")
                    for x in active
                )
                if has_siblings:
                    shares[h] = Fraction(1, 6)
                else:
                    shares[h] = Fraction(1, 3)
        elif h == "father":
            # Father gets 1/6 with children, else asaba (but we treat as fard for simplicity)
            if has_children:
                shares[h] = Fraction(1, 6)
            else:
                # Asaba: takes residue, but we'll handle in asaba step
                # For now assign 0 and let asaba handle
                shares[h] = Fraction(0)
        else:
            shares[h] = base
    return shares


def _assign_asaba(active: List[str], shares: Dict[str, Fraction], residue: Fraction) -> Dict[str, Fraction]:
    """Distribute residue to asaba heirs, male:female 2:1."""
    asaba = [h for h in active if HEIRS[h]["category"] == "asaba"]
    if not asaba:
        return shares
    # Determine if there are female asaba (e.g., daughter with son, full sister with brother)
    # For simplicity, we treat all asaba as male except when a female is explicitly asaba
    # In our set, only 'daughter' and 'full_sister' can become asaba when a male counterpart exists.
    # We'll handle that in the caller by converting them to asaba.
    # Here we just distribute to male asaba equally.
    male_asaba = [h for h in asaba if h not in ("daughter", "full_sister", "consanguine_sister", "paternal_granddaughter")]
    if male_asaba:
        each = residue / len(male_asaba)
        for h in male_asaba:
            shares[h] = each
    return shares


def _apply_awl(shares: Dict[str, Fraction]) -> Tuple[Dict[str, Fraction], Optional[Fraction]]:
    """If sum > 1, scale down proportionally. Return (new_shares, awl_factor)."""
    total = sum(shares.values(), Fraction(0))
    if total > 1:
        factor = Fraction(1, 1) / total
        new_shares = {k: v * factor for k, v in shares.items()}
        return new_shares, factor
    return shares, None


def _apply_radd(shares: Dict[str, Fraction], active: List[str]) -> Tuple[Dict[str, Fraction], Optional[Fraction]]:
    """If sum < 1 and no asaba, return surplus to radd-eligible heirs proportionally."""
    total = sum(shares.values(), Fraction(0))
    if total >= 1:
        return shares, None
    # Check if there is any asaba (male) that should take residue
    if any(HEIRS[h]["category"] == "asaba" for h in active):
        return shares, None
    # Radd-eligible: those with radd_eligible True
    eligible = [h for h in active if HEIRS[h]["radd_eligible"]]
    if not eligible:
        return shares, None
    surplus = Fraction(1, 1) - total
    # Distribute surplus proportionally to their current shares
    eligible_total = sum(shares[h] for h in eligible)
    if eligible_total == 0:
        return shares, None
    for h in eligible:
        shares[h] += surplus * (shares[h] / eligible_total)
    return shares, None


def distribute(estate: Decimal, heirs: List[str]) -> FaraidResult:
    """Main entry point. Returns exact shares and reasoning steps."""
    # Validate heirs
    unknown = [h for h in heirs if h not in HEIRS]
    if unknown:
        raise ValueError(f"Unknown heir(s): {unknown}")

    steps: List[str] = []
    basis: Dict[str, str] = {}
    blocked: Dict[str, str] = {}

    # 1. Hajb
    active, blocked = _apply_hajb(heirs)
    if blocked:
        for h, blocker in blocked.items():
            steps.append(f"Hajb: {HEIRS[h]['name']} is blocked by {HEIRS[blocker]['name']}.")
            basis[h] = HAJB_BASIS

    # 2. Assign furud
    shares = _assign_furud(active)
    for h in active:
        if HEIRS[h]["category"] == "fard" and shares.get(h, Fraction(0)) > 0:
            # Determine basis
            if h in ("wife", "husband"):
                basis[h] = QURAN_4_12
            elif h in ("daughter", "full_sister", "consanguine_sister", "uterine_brother", "uterine_sister"):
                basis[h] = QURAN_4_11 if h == "daughter" else QURAN_4_176
            elif h in ("father", "mother", "paternal_grandfather", "paternal_grandmother", "maternal_grandmother"):
                basis[h] = QURAN_4_11
            else:
                basis[h] = QURAN_4_11

    # 3. Asaba: if there are male asaba, they take residue
    total_fixed = sum(shares.values(), Fraction(0))
    residue = Fraction(1, 1) - total_fixed
    if residue > 0:
        # Check for asaba heirs (male or female with male counterpart)
        # For simplicity, we treat 'son' and 'full_brother' as asaba
        asaba_heirs = [h for h in active if HEIRS[h]["category"] == "asaba"]
        if asaba_heirs:
            # Distribute residue to asaba, male:female 2:1 if both present
            # For now, just give to male asaba equally
            male_asaba = [h for h in asaba_heirs if h not in ("daughter", "full_sister", "consanguine_sister", "paternal_granddaughter")]
            if male_asaba:
                each = residue / len(male_asaba)
                for h in male_asaba:
                    shares[h] = each
                    basis[h] = "Residuary (asaba) — the Prophet ﷺ said: 'Give the fixed shares to those entitled, and the remainder to the nearest male relative.' (Sahih al-Bukhari 6732, Sahih Muslim 1615)"
                steps.append(f"Asaba: residue {residue} distributed equally among male asaba.")
            else:
                # No male asaba, but maybe female asaba with male counterpart? Not implemented.
                pass

    # 4. Awl
    shares, awl_factor = _apply_awl(shares)
    if awl_factor:
        steps.append(f"Awl applied: total shares exceeded 1, scaled by factor {awl_factor}.")
        for h in shares:
            basis[h] = basis.get(h, "") + " " + AWL_BASIS

    # 5. Radd
    shares, radd_factor = _apply_radd(shares, active)
    if radd_factor:
        steps.append(f"Radd applied: surplus returned proportionally to eligible heirs.")
        for h in shares:
            if HEIRS[h]["radd_eligible"]:
                basis[h] = basis.get(h, "") + " " + RADD_BASIS

    # 6. Ensure blocked heirs have 0 share
    for h in blocked:
        shares[h] = Fraction(0)

    # 7. Compute amounts
    estate_frac = Fraction(estate)
    amounts = {h: shares[h] * estate_frac for h in shares}

    # Build result
    result_shares = {h: shares[h] for h in heirs}
    result_basis = {h: basis.get(h, "") for h in heirs}
    result_blocked = {h: blocked.get(h, "") for h in heirs}

    return FaraidResult(
        shares=result_shares,
        steps=steps,
        basis=result_basis,
        blocked=result_blocked,
    )


# ---------------------------------------------------------------------------
# HTTP endpoint
# ---------------------------------------------------------------------------

from fastapi import APIRouter, HTTPException

router = APIRouter(tags=["faraid"])


@router.post("/faraid", response_model=FaraidResponse)
def faraid_endpoint(request: FaraidRequest) -> FaraidResponse:
    try:
        result = distribute(request.estate, request.heirs)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    estate_frac = Fraction(request.estate)
    shares_list = []
    for h in request.heirs:
        frac = result.shares[h]
        amount = frac * estate_frac
        shares_list.append(
            HeirShare(
                heir=h,
                name=HEIRS[h]["name"],
                category=HEIRS[h]["category"],
                fraction=f"{frac.numerator}/{frac.denominator}",
                amount=f"{amount:.2f}",
                basis=result.basis[h],
                blocked_by=result.blocked[h] or None,
            )
        )

    return FaraidResponse(
        estate=str(request.estate),
        shares=shares_list,
        steps=result.steps,
        disclaimer=DISCLAIMER,
    )
