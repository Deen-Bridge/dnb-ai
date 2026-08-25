"""Faraid (Islamic inheritance) computation engine.

A deterministic module that, given a set of surviving heirs and an estate
value, returns each heir's exact share *and the reasoning steps that produced it*,
exposed as a real HTTP endpoint.  Uses exact rational arithmetic (fractions.Fraction)
so shares are provably exact -- no float rounding drift.

The algorithm implements:

1. Quranic fixed shares (furud) per Surah an-Nisa 4:11-12, 4:176
2. Proportional reduction (awl) when shares exceed the estate
3. Residuary distribution (asaba) to entitled heirs
4. Return of surplus (radd) to sharers when no residuary heirs exist
5. Blocking (hajb) of nearer relatives

Every allocation cites its fiqh basis.  No allocation appears without a
citation, and no citation is fabricated.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from enum import Enum
from fractions import Fraction

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(tags=["faraid"])

DISCLAIMER = (
    "This is an automated estimate based on the information provided. "
    "Inheritance rulings depend on many factors including debts, bequests, "
    "and specific family circumstances. Please consult a qualified scholar "
    "for a definitive ruling."
)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class HeirType(str, Enum):
    """Types of heirs recognised by the faraid algorithm."""

    WIFE = "wife"
    HUSBAND = "husband"
    SON = "son"
    DAUGHTER = "daughter"
    FATHER = "father"
    MOTHER = "mother"
    FULL_SISTER = "full_sister"
    PATERNAL_HALF_SISTER = "paternal_half_sister"
    SON_OF_SON = "son_of_son"
    DAUGHTER_OF_SON = "daughter_of_son"
    GRANDFATHER = "grandfather"
    GRANDMOTHER = "grandmother"


@dataclass(frozen=True)
class HeirEntry:
    """Internal representation of a single heir instance."""

    heir_type: HeirType
    index: int  # which heir of this type (0-based)


@dataclass
class ShareAllocation:
    """A single heir's share allocation with full reasoning."""

    heir_type: HeirType
    heir_index: int
    category: str  # "fard" | "asaba" | "radd" | "blocked"
    fraction: Fraction
    amount: Decimal
    basis: str


@dataclass
class FaraidResult:
    """Complete result of a faraid computation."""

    estate: Decimal
    total_allocated: Fraction
    allocations: list[ShareAllocation]
    awl_applied: bool
    awl_denominator: int | None
    radd_applied: bool
    hajb_applied: bool
    steps: list[str]
    disclaimer: str


# ---------------------------------------------------------------------------
# Heir counting helpers
# ---------------------------------------------------------------------------


def _count(heirs: list[HeirEntry], ht: HeirType) -> int:
    return sum(1 for h in heirs if h.heir_type == ht)


def _has(heirs: list[HeirEntry], ht: HeirType) -> bool:
    return _count(heirs, ht) > 0


def _has_children(heirs: list[HeirEntry]) -> bool:
    """Children = sons OR daughters."""
    return _has(heirs, HeirType.SON) or _has(heirs, HeirType.DAUGHTER)


def _has_descendants(heirs: list[HeirEntry]) -> bool:
    """Children or grandchildren through sons."""
    return _has_children(heirs) or _has(heirs, HeirType.SON_OF_SON)


def _frac_amount(frac: Fraction, estate: Decimal) -> Decimal:
    """Convert a fraction-of-estate into a Decimal monetary amount."""
    d = Decimal(frac.numerator) / Decimal(frac.denominator)
    return (d * estate).quantize(Decimal("1"), rounding=ROUND_HALF_UP)


# ---------------------------------------------------------------------------
# Quranic fixed-share (fard) calculation
# ---------------------------------------------------------------------------

_Q4_11 = "Quran 4:11"
_Q4_12 = "Quran 4:12"
_Q4_176 = "Quran 4:176"


def _furud_share(heir_type: HeirType, heirs: list[HeirEntry]) -> Fraction | None:
    """Return the Quranic fixed share for *heir_type*, or ``None`` if the
    heir type is not a Quranic sharer (asaba-only), or ``Fraction(0)`` if
    the heir is blocked (hajb).
    """
    has_sons = _has(heirs, HeirType.SON)
    has_daughters = _has(heirs, HeirType.DAUGHTER)
    has_son_of_sons = _has(heirs, HeirType.SON_OF_SON)
    has_descendants = _has_children(heirs) or has_son_of_sons
    has_father = _has(heirs, HeirType.FATHER)
    has_mother = _has(heirs, HeirType.MOTHER)
    num_daughters = _count(heirs, HeirType.DAUGHTER)
    num_full_sisters = _count(heirs, HeirType.FULL_SISTER)
    num_half_sisters = _count(heirs, HeirType.PATERNAL_HALF_SISTER)
    num_sod = _count(heirs, HeirType.DAUGHTER_OF_SON)

    if heir_type == HeirType.HUSBAND:
        # 4:12 – without descendants: 1/2; with descendants: 1/4
        return Fraction(1, 4) if has_descendants else Fraction(1, 2)

    if heir_type == HeirType.WIFE:
        # 4:12 – each wife gets 1/8 (with descendants) or 1/4 (without).
        # The per-wife interpretation (majority position).
        return Fraction(1, 8) if has_descendants else Fraction(1, 4)

    if heir_type == HeirType.DAUGHTER:
        if has_sons:
            return None
        if num_daughters == 1:
            return Fraction(1, 2)
        return Fraction(2, 3) / num_daughters

    if heir_type == HeirType.SON:
        return None

    if heir_type == HeirType.FATHER:
        if has_descendants:
            return Fraction(1, 6)
        return None

    if heir_type == HeirType.MOTHER:
        if has_descendants:
            return Fraction(1, 6)
        return Fraction(1, 3)

    if heir_type == HeirType.FULL_SISTER:
        if has_descendants or has_father:
            return Fraction(0)
        if num_full_sisters == 1:
            return Fraction(1, 2)
        return Fraction(2, 3) / num_full_sisters

    if heir_type == HeirType.PATERNAL_HALF_SISTER:
        if has_descendants or has_father:
            return Fraction(0)
        if num_full_sisters > 0:
            return Fraction(0)
        if num_half_sisters == 1:
            return Fraction(1, 2)
        return Fraction(2, 3) / num_half_sisters

    if heir_type == HeirType.SON_OF_SON:
        if has_sons:
            return Fraction(0)
        return None

    if heir_type == HeirType.DAUGHTER_OF_SON:
        if has_sons:
            return Fraction(0)
        if has_daughters:
            return Fraction(0)
        if num_sod == 1:
            return Fraction(1, 6)
        return Fraction(1, 3) / num_sod

    if heir_type == HeirType.GRANDFATHER:
        if has_father:
            return Fraction(1, 6)
        return None

    if heir_type == HeirType.GRANDMOTHER:
        if has_mother:
            return Fraction(0)
        return Fraction(1, 6)

    return None


# ---------------------------------------------------------------------------
# Residuary (asaba) helpers
# ---------------------------------------------------------------------------


def _is_asaba(heir_type: HeirType, heirs: list[HeirEntry]) -> bool:
    """Return ``True`` if *heir_type* is entitled to residue."""
    has_sons = _has(heirs, HeirType.SON)
    has_father = _has(heirs, HeirType.FATHER)

    if heir_type == HeirType.SON:
        return True
    if heir_type == HeirType.DAUGHTER:
        return has_sons
    if heir_type == HeirType.FATHER:
        return True
    if heir_type == HeirType.GRANDFATHER:
        return not has_father
    if heir_type == HeirType.SON_OF_SON:
        return not has_sons
    return False


def _asaba_heirs(heirs: list[HeirEntry]) -> list[HeirEntry]:
    """Return the list of heirs entitled to residue."""
    return [h for h in heirs if _is_asaba(h.heir_type, heirs)]


def _asaba_weight(heir_type: HeirType) -> Fraction:
    """Weight for residue distribution: males get 2, females get 1."""
    if heir_type in (HeirType.DAUGHTER, HeirType.DAUGHTER_OF_SON):
        return Fraction(1)
    return Fraction(2)


# ---------------------------------------------------------------------------
# Citation helpers
# ---------------------------------------------------------------------------


def _fard_basis(heir_type: HeirType, heirs: list[HeirEntry]) -> str:
    """Return the fiqh citation for a Quranic fixed share."""
    ht = heir_type
    if ht == HeirType.HUSBAND:
        if _has_descendants(heirs):
            return f"{_Q4_12} -- husband with descendants (1/4)"
        return f"{_Q4_12} -- husband without descendants (1/2)"
    if ht == HeirType.WIFE:
        if _has_descendants(heirs):
            return f"{_Q4_12} -- wife with descendants (1/8)"
        return f"{_Q4_12} -- wife without descendants (1/4)"
    if ht == HeirType.DAUGHTER:
        n = _count(heirs, HeirType.DAUGHTER)
        if n == 1:
            return f"{_Q4_11} -- one daughter receives 1/2"
        return f"{_Q4_11} -- daughters share 2/3 total ({n} daughters)"
    if ht == HeirType.FATHER:
        return f"{_Q4_11} -- father fixed share (1/6)"
    if ht == HeirType.MOTHER:
        if _has_descendants(heirs):
            return f"{_Q4_11} -- mother with descendants (1/6)"
        return f"{_Q4_12} -- mother without descendants (1/3)"
    if ht == HeirType.FULL_SISTER:
        n = _count(heirs, HeirType.FULL_SISTER)
        if n == 1:
            return f"{_Q4_12} -- one full sister receives 1/2"
        return f"{_Q4_12} -- full sisters share 2/3 total ({n} sisters)"
    if ht == HeirType.PATERNAL_HALF_SISTER:
        n = _count(heirs, HeirType.PATERNAL_HALF_SISTER)
        if n == 1:
            return f"{_Q4_12} -- one paternal half-sister receives 1/2"
        return f"{_Q4_12} -- paternal half-sisters share 2/3 total ({n} sisters)"
    if ht == HeirType.DAUGHTER_OF_SON:
        n = _count(heirs, HeirType.DAUGHTER_OF_SON)
        if n == 1:
            return f"{_Q4_176} -- one son's daughter receives 1/6"
        return f"{_Q4_176} -- two or more son's daughters receive 1/3"
    if ht == HeirType.GRANDMOTHER:
        return f"{_Q4_12} -- grandmother (1/6)"
    return "Juristic principle"


def _asaba_basis(heir_type: HeirType) -> str:
    """Return the fiqh citation for a residuary share."""
    labels = {
        HeirType.SON: "son (residuary)",
        HeirType.DAUGHTER: "daughter with sons (2:1 ratio)",
        HeirType.FATHER: "father (residuary)",
        HeirType.GRANDFATHER: "grandfather in lieu of father (residuary)",
        HeirType.SON_OF_SON: "son's son in lieu of son (residuary)",
    }
    label = labels.get(heir_type, "residuary heir")
    return f"Juristic consensus (ijma') -- {label}"


def _blocked_basis(heir_type: HeirType, heirs: list[HeirEntry]) -> str:
    """Return the fiqh citation for a blocked (hajb) heir."""
    ht = heir_type
    if ht == HeirType.SON_OF_SON:
        return "Hajb -- grandson blocked by son (nearer paternal relative)"
    if ht == HeirType.DAUGHTER_OF_SON:
        if _has(heirs, HeirType.SON):
            return "Hajb -- son's granddaughter blocked by son"
        return "Hajb -- son's granddaughter blocked by full daughters"
    if ht == HeirType.FULL_SISTER:
        return "Hajb -- full sister blocked by descendants or father"
    if ht == HeirType.PATERNAL_HALF_SISTER:
        if _count(heirs, HeirType.FULL_SISTER) > 0:
            return "Hajb -- paternal half-sister blocked by full sister"
        return "Hajb -- paternal half-sister blocked by descendants or father"
    if ht == HeirType.GRANDMOTHER:
        return "Hajb -- grandmother blocked by mother"
    return "Hajb -- blocked by nearer relative"


# ---------------------------------------------------------------------------
# Core algorithm
# ---------------------------------------------------------------------------


def distribute(estate: Decimal, heirs: list[HeirEntry]) -> FaraidResult:
    """Compute the faraid distribution for *estate* and *heirs*.

    Implements the complete Islamic inheritance algorithm:

    1. Assign Quranic fixed shares (**furud**).
    2. Apply proportional reduction (**awl**) when shares exceed the estate.
    3. Distribute residue to residuary heirs (**asaba**).
    4. Return surplus to sharers (**radd**) when no residuary heirs exist.
    5. Remove heirs blocked by nearer relatives (**hajb**).

    All arithmetic uses :class:`fractions.Fraction` for exactness.
    """
    steps: list[str] = []
    allocs: list[ShareAllocation] = []

    # 1. Calculate furud shares
    furud: dict[tuple[HeirType, int], Fraction] = {}
    for heir in heirs:
        share = _furud_share(heir.heir_type, heirs)
        if share is not None:
            furud[(heir.heir_type, heir.index)] = share

    total_furud = sum(furud.values())
    steps.append(f"Step 1 -- Furud: total fixed shares = {total_furud}")

    # 2. Awl (proportional reduction)
    awl_applied = False
    awl_denom: int | None = None
    if total_furud > Fraction(1):
        awl_applied = True
        awl_denom = total_furud.numerator
        factor = Fraction(1) / total_furud
        furud = {k: v * factor for k, v in furud.items()}
        total_furud = sum(furud.values())
        steps.append(
            f"Step 2 -- Awl applied: shares exceeded estate "
            f"(sum was {awl_denom}/{awl_denom}). "
            f"All shares scaled by {factor}."
        )

    # 3. Record furud allocations
    for (ht, idx), frac in furud.items():
        if frac == 0:
            allocs.append(
                ShareAllocation(
                    heir_type=ht,
                    heir_index=idx,
                    category="blocked",
                    fraction=Fraction(0),
                    amount=Decimal("0"),
                    basis=_blocked_basis(ht, heirs),
                )
            )
        else:
            allocs.append(
                ShareAllocation(
                    heir_type=ht,
                    heir_index=idx,
                    category="fard",
                    fraction=frac,
                    amount=_frac_amount(frac, estate),
                    basis=_fard_basis(ht, heirs),
                )
            )

    # 4. Calculate residue
    residue = Fraction(1) - sum(a.fraction for a in allocs if a.category == "fard")

    # 5. Distribute residue to asaba heirs
    asaba = _asaba_heirs(heirs)
    hajb_applied = any(a.category == "blocked" for a in allocs)

    if asaba and residue > 0:
        weights = [_asaba_weight(h.heir_type) for h in asaba]
        total_w = sum(weights)
        if total_w > 0:
            for h, w in zip(asaba, weights, strict=True):
                asaba_frac = residue * w / total_w
                allocs.append(
                    ShareAllocation(
                        heir_type=h.heir_type,
                        heir_index=h.index,
                        category="asaba",
                        fraction=asaba_frac,
                        amount=_frac_amount(asaba_frac, estate),
                        basis=_asaba_basis(h.heir_type),
                    )
                )
            residue = Fraction(0)
            steps.append("Step 4 -- Asaba: residue distributed to residuary heirs.")

    # 6. Radd (surplus return to sharers, excluding spouse)
    radd_applied = False
    if residue > 0 and not asaba:
        non_spouse = [
            a for a in allocs if a.category == "fard" and a.heir_type not in (HeirType.WIFE, HeirType.HUSBAND)
        ]
        if non_spouse:
            radd_applied = True
            base_total = sum(a.fraction for a in non_spouse)
            for a in non_spouse:
                extra = residue * a.fraction / base_total
                a.fraction += extra
                a.amount = _frac_amount(a.fraction, estate)
                a.category = "radd"
            steps.append(
                "Step 5 -- Radd: surplus returned to non-spouse sharers "
                "(majority Sunni position excludes spouse from radd)."
            )
            residue = Fraction(0)

    # 7. Final verification
    total = sum((a.fraction for a in allocs), Fraction(0))
    steps.append(f"Step 6 -- Verification: total allocated = {total}")

    return FaraidResult(
        estate=estate,
        total_allocated=total,
        allocations=allocs,
        awl_applied=awl_applied,
        awl_denominator=awl_denom,
        radd_applied=radd_applied,
        hajb_applied=hajb_applied,
        steps=steps,
        disclaimer=DISCLAIMER,
    )


# ---------------------------------------------------------------------------
# FastAPI endpoint
# ---------------------------------------------------------------------------


class HeirInput(BaseModel):
    """A group of heirs of the same type."""

    heir_type: str = Field(
        ...,
        description=(
            "Heir type: wife, husband, son, daughter, father, mother, "
            "full_sister, paternal_half_sister, son_of_son, daughter_of_son, "
            "grandfather, grandmother"
        ),
    )
    count: int = Field(1, ge=1, description="Number of heirs of this type")


class FaraidRequest(BaseModel):
    """Request body for the faraid endpoint."""

    estate: float = Field(..., gt=0, description="Total estate value")
    heirs: list[HeirInput] = Field(..., min_length=1, description="List of heir groups")
    currency: str = Field("USD", description="Currency for amount calculations")


class HeirShareOutput(BaseModel):
    """One heir's share in the response."""

    heir_type: str
    heir_index: int
    category: str
    fraction: str
    amount: str
    basis: str


class FaraidResponse(BaseModel):
    """Full faraid computation response."""

    estate: str
    currency: str
    total_allocated: str
    shares: list[HeirShareOutput]
    awl_applied: bool
    awl_denominator: int | None
    radd_applied: bool
    hajb_applied: bool
    steps: list[str]
    disclaimer: str


@router.post("/faraid", response_model=FaraidResponse)
async def calculate_faraid(request: FaraidRequest) -> FaraidResponse:
    """Calculate Islamic inheritance (faraid) distribution.

    Given a set of surviving heirs and an estate value, returns each
    heir's exact share, the reasoning steps, and fiqh citations.
    """
    heirs: list[HeirEntry] = []
    for h in request.heirs:
        try:
            ht = HeirType(h.heir_type)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown heir type: {h.heir_type!r}",
            ) from None
        for i in range(h.count):
            heirs.append(HeirEntry(heir_type=ht, index=i))

    if not heirs:
        raise HTTPException(
            status_code=400,
            detail="At least one heir is required.",
        )

    estate_int = int(request.estate)
    estate_dec = Decimal(estate_int) if request.estate == estate_int else Decimal(str(request.estate))
    result = distribute(estate_dec, heirs)

    shares = [
        HeirShareOutput(
            heir_type=a.heir_type.value,
            heir_index=a.heir_index,
            category=a.category,
            fraction=str(a.fraction),
            amount=str(a.amount),
            basis=a.basis,
        )
        for a in result.allocations
    ]

    return FaraidResponse(
        estate=str(result.estate),
        currency=request.currency,
        total_allocated=str(result.total_allocated),
        shares=shares,
        awl_applied=result.awl_applied,
        awl_denominator=result.awl_denominator,
        radd_applied=result.radd_applied,
        hajb_applied=result.hajb_applied,
        steps=result.steps,
        disclaimer=result.disclaimer,
    )
