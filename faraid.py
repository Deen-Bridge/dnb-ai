"""Faraid (Islamic inheritance) computation engine.

A deterministic module that, given a set of surviving heirs and an estate
value, returns each heir's exact share *and the reasoning steps that produced
it*, exposed as a real HTTP endpoint. Uses exact rational arithmetic
(:class:`fractions.Fraction`) so shares are provably exact -- no float rounding
drift.

The algorithm implements the majority Sunni position:

1. Quranic fixed shares (furud) per Surah an-Nisa 4:11-12, 4:176
2. Proportional reduction (awl) when shares exceed the estate
3. Residuary distribution (asaba) to entitled heirs
4. Return of surplus (radd) to sharers when no residuary heirs exist
5. Blocking (hajb) of nearer relatives

Every allocation cites its fiqh basis, every response reports the versioned
rule set it was computed under, and every response carries an ``uncertain`` flag
plus warnings for cases this engine handles only approximately (for example, a
grandfather inheriting alongside siblings). The calculation is advisory: it does
not account for debts, bequests (wasiyyah), or heirs outside the supported set.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from enum import Enum
from fractions import Fraction

from fastapi import APIRouter
from pydantic import BaseModel, Field

from errors import APIException

logger = logging.getLogger(__name__)

router = APIRouter(tags=["faraid"])

# ---------------------------------------------------------------------------
# Versioned rule set and provenance
# ---------------------------------------------------------------------------

FARAID_RULESET_VERSION = "1.0.0"
FARAID_MADHAB = "sunni_majority"
FARAID_BASIS = (
    "Quranic fixed shares (furud) of Surah an-Nisa 4:11-12 and 4:176 together "
    "with the juristic principles of awl, radd, asaba, and hajb, following the "
    "majority Sunni position."
)
FARAID_SOURCES = [
    "Quran 4:11-12, 4:176 (fixed shares)",
    "Juristic consensus (ijma') on asaba (residuary heirs)",
    "Juristic principle of awl (proportional reduction)",
    "Juristic principle of radd (return of surplus; majority Sunni position excludes the spouse)",
    "Juristic principle of hajb (blocking by a nearer relative)",
]
STANDARD_QURANIC_FURUD = (
    Fraction(1, 2),
    Fraction(1, 4),
    Fraction(1, 8),
    Fraction(2, 3),
    Fraction(1, 3),
    Fraction(1, 6),
)
MAX_HEIRS = 100

DISCLAIMER = (
    "This is an automated estimate based on the information provided. "
    "Inheritance rulings depend on many factors including debts, bequests, "
    "and specific family circumstances. Please consult a qualified scholar "
    "for a definitive ruling."
)

UNCERTAINTY_RULES = [
    "Grandfather with siblings (al-jadd wa al-ikhwa): simplified residuary treatment; schools differ.",
    "No eligible residuary or radd recipient: the remaining residue is left unallocated.",
    "Distant heirs (dhawu al-arham), maternal half-siblings, and wasiyyah are not modelled.",
]


class FaraidProvenance(BaseModel):
    """Versioned provenance for a single faraid computation."""

    ruleset_version: str
    madhab: str
    basis: str
    sources: list[str]


class FaraidAssumptions(BaseModel):
    """Versioned record of the assumptions behind the faraid rules."""

    version: str
    madhab: str
    basis: str
    sources: list[str]
    supported_heirs: list[str]
    fixed_furud: list[str]
    awl_rule: str
    radd_rule: str
    hajb_rule: str
    uncertainty_rules: list[str]
    valid: bool
    errors: list[str] = Field(default_factory=list)


def faraid_provenance() -> FaraidProvenance:
    """Return the versioned provenance attached to every computation."""
    return FaraidProvenance(
        ruleset_version=FARAID_RULESET_VERSION,
        madhab=FARAID_MADHAB,
        basis=FARAID_BASIS,
        sources=list(FARAID_SOURCES),
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
    FULL_BROTHER = "full_brother"
    PATERNAL_HALF_SISTER = "paternal_half_sister"
    SON_OF_SON = "son_of_son"
    DAUGHTER_OF_SON = "daughter_of_son"
    GRANDFATHER = "grandfather"
    GRANDMOTHER = "grandmother"


def _normalize_heir_key(raw: str | None) -> str:
    """Fold an English or Arabic heir label to a stable lookup key."""
    text = unicodedata.normalize("NFKD", raw or "")
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = text.lower().replace("’", "'").strip()
    for variant, canon in (("أ", "ا"), ("إ", "ا"), ("آ", "ا"), ("ٱ", "ا"), ("ى", "ي"), ("ة", "ه")):
        text = text.replace(variant, canon)
    text = re.sub(r"[\s\-]+", "_", text)
    return re.sub(r"[^0-9a-z_\u0600-\u06ff]", "", text)


_HEIR_ALIASES_RAW: dict[str, str] = {
    "wife": "wife",
    "wives": "wife",
    "زوجة": "wife",
    "زوجه": "wife",
    "husband": "husband",
    "زوج": "husband",
    "son": "son",
    "sons": "son",
    "ابن": "son",
    "ابناء": "son",
    "daughter": "daughter",
    "daughters": "daughter",
    "بنت": "daughter",
    "بنات": "daughter",
    "father": "father",
    "اب": "father",
    "mother": "mother",
    "ام": "mother",
    "والدة": "mother",
    "full_sister": "full_sister",
    "full sister": "full_sister",
    "sister": "full_sister",
    "اخت_شقيقه": "full_sister",
    "شقيقة": "full_sister",
    "full_brother": "full_brother",
    "full brother": "full_brother",
    "brother": "full_brother",
    "اخ_شقيق": "full_brother",
    "اخ": "full_brother",
    "paternal_half_sister": "paternal_half_sister",
    "paternal half sister": "paternal_half_sister",
    "half sister": "paternal_half_sister",
    "اخت_لاب": "paternal_half_sister",
    "son_of_son": "son_of_son",
    "grandson": "son_of_son",
    "son's son": "son_of_son",
    "ابن_الابن": "son_of_son",
    "حفيد": "son_of_son",
    "daughter_of_son": "daughter_of_son",
    "granddaughter": "daughter_of_son",
    "son's daughter": "daughter_of_son",
    "بنت_الابن": "daughter_of_son",
    "حفيدة": "daughter_of_son",
    "grandfather": "grandfather",
    "paternal_grandfather": "grandfather",
    "جد": "grandfather",
    "grandmother": "grandmother",
    "paternal_grandmother": "grandmother",
    "maternal_grandmother": "grandmother",
    "جدة": "grandmother",
    "جده": "grandmother",
}

_HEIR_ALIASES: dict[str, HeirType] = {
    _normalize_heir_key(alias): HeirType(target) for alias, target in _HEIR_ALIASES_RAW.items()
}
for _heir in HeirType:
    _HEIR_ALIASES[_normalize_heir_key(_heir.value)] = _heir


def resolve_heir_type(raw: str | None) -> HeirType | None:
    """Resolve an English or Arabic heir label to a :class:`HeirType`."""
    return _HEIR_ALIASES.get(_normalize_heir_key(raw))


@dataclass(frozen=True)
class HeirEntry:
    """Internal representation of a single heir instance."""

    heir_type: HeirType
    index: int = 0


@dataclass
class ShareAllocation:
    """A single heir's share allocation with full reasoning."""

    heir_type: HeirType
    heir_index: int
    category: str
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
    uncertain: bool
    warnings: list[str]
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
    return _has(heirs, HeirType.SON) or _has(heirs, HeirType.DAUGHTER)


def _has_descendants(heirs: list[HeirEntry]) -> bool:
    return _has_children(heirs) or _has(heirs, HeirType.SON_OF_SON)


def _has_female_descendants(heirs: list[HeirEntry]) -> bool:
    return _has(heirs, HeirType.DAUGHTER) or _has(heirs, HeirType.DAUGHTER_OF_SON)


def _siblings_blocked(heirs: list[HeirEntry]) -> bool:
    return (
        _has(heirs, HeirType.SON)
        or _has(heirs, HeirType.SON_OF_SON)
        or _has(heirs, HeirType.FATHER)
        or _has(heirs, HeirType.GRANDFATHER)
    )


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
    """Return the fixed share for *heir_type*, ``None`` if asaba-only, or ``0`` if blocked."""
    has_sons = _has(heirs, HeirType.SON)
    has_son_of_sons = _has(heirs, HeirType.SON_OF_SON)
    has_descendants = _has_children(heirs) or has_son_of_sons
    has_father = _has(heirs, HeirType.FATHER)
    has_mother = _has(heirs, HeirType.MOTHER)
    num_daughters = _count(heirs, HeirType.DAUGHTER)
    num_full_sisters = _count(heirs, HeirType.FULL_SISTER)
    num_half_sisters = _count(heirs, HeirType.PATERNAL_HALF_SISTER)
    num_sod = _count(heirs, HeirType.DAUGHTER_OF_SON)
    num_wives = _count(heirs, HeirType.WIFE)
    num_grandmothers = _count(heirs, HeirType.GRANDMOTHER)
    num_siblings = num_full_sisters + num_half_sisters + _count(heirs, HeirType.FULL_BROTHER)

    if heir_type == HeirType.HUSBAND:
        return Fraction(1, 4) if has_descendants else Fraction(1, 2)

    if heir_type == HeirType.WIFE:
        share = Fraction(1, 8) if has_descendants else Fraction(1, 4)
        return share / num_wives if num_wives else None

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
        if has_descendants or num_siblings >= 2:
            return Fraction(1, 6)
        return Fraction(1, 3)

    if heir_type == HeirType.FULL_SISTER:
        if _siblings_blocked(heirs):
            return Fraction(0)
        if _has(heirs, HeirType.FULL_BROTHER) or _has_female_descendants(heirs):
            return None
        if num_full_sisters == 1:
            return Fraction(1, 2)
        return Fraction(2, 3) / num_full_sisters

    if heir_type == HeirType.FULL_BROTHER:
        if _siblings_blocked(heirs):
            return Fraction(0)
        return None

    if heir_type == HeirType.PATERNAL_HALF_SISTER:
        if _siblings_blocked(heirs) or _has(heirs, HeirType.FULL_BROTHER):
            return Fraction(0)
        if num_full_sisters == 0:
            if num_half_sisters == 1:
                return Fraction(1, 2)
            return Fraction(2, 3) / num_half_sisters
        if num_full_sisters == 1:
            return Fraction(1, 6) / num_half_sisters
        return Fraction(0)

    if heir_type == HeirType.SON_OF_SON:
        if has_sons:
            return Fraction(0)
        return None

    if heir_type == HeirType.DAUGHTER_OF_SON:
        if has_sons:
            return Fraction(0)
        if has_son_of_sons:
            return None
        if num_daughters >= 2:
            return Fraction(0)
        if num_daughters == 1:
            return Fraction(1, 6) / num_sod
        if num_sod == 1:
            return Fraction(1, 2)
        return Fraction(2, 3) / num_sod

    if heir_type == HeirType.GRANDFATHER:
        if has_father:
            return Fraction(0)
        if has_descendants:
            return Fraction(1, 6)
        return None

    if heir_type == HeirType.GRANDMOTHER:
        if has_mother:
            return Fraction(0)
        return Fraction(1, 6) / num_grandmothers if num_grandmothers else None

    return None


# ---------------------------------------------------------------------------
# Residuary (asaba) helpers
# ---------------------------------------------------------------------------


def _is_asaba(heir_type: HeirType, heirs: list[HeirEntry]) -> bool:
    """Return ``True`` if *heir_type* is entitled to residue."""
    has_sons = _has(heirs, HeirType.SON)
    has_father = _has(heirs, HeirType.FATHER)
    siblings_blocked = _siblings_blocked(heirs)

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
    if heir_type == HeirType.DAUGHTER_OF_SON:
        return _has(heirs, HeirType.SON_OF_SON) and not has_sons
    if heir_type == HeirType.FULL_BROTHER:
        return not siblings_blocked
    if heir_type == HeirType.FULL_SISTER:
        if siblings_blocked:
            return False
        return _has(heirs, HeirType.FULL_BROTHER) or _has_female_descendants(heirs)
    return False


def _asaba_heirs(heirs: list[HeirEntry]) -> list[HeirEntry]:
    """Return the list of heirs entitled to residue."""
    return [h for h in heirs if _is_asaba(h.heir_type, heirs)]


def _asaba_weight(heir_type: HeirType) -> Fraction:
    """Weight for residue distribution: males get 2, females get 1."""
    if heir_type in (HeirType.DAUGHTER, HeirType.DAUGHTER_OF_SON, HeirType.FULL_SISTER):
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
            return f"{_Q4_12} -- wife with descendants (1/8, shared)"
        return f"{_Q4_12} -- wife without descendants (1/4, shared)"
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
        if (
            _count(heirs, HeirType.FULL_SISTER)
            + _count(heirs, HeirType.PATERNAL_HALF_SISTER)
            + _count(heirs, HeirType.FULL_BROTHER)
            >= 2
        ):
            return f"{_Q4_11} -- mother with multiple siblings (1/6)"
        return f"{_Q4_12} -- mother without descendants or multiple siblings (1/3)"
    if ht == HeirType.FULL_SISTER:
        n = _count(heirs, HeirType.FULL_SISTER)
        if n == 1:
            return f"{_Q4_12} -- one full sister receives 1/2"
        return f"{_Q4_12} -- full sisters share 2/3 total ({n} sisters)"
    if ht == HeirType.PATERNAL_HALF_SISTER:
        n = _count(heirs, HeirType.PATERNAL_HALF_SISTER)
        if _count(heirs, HeirType.FULL_SISTER) == 1:
            return f"{_Q4_12} -- paternal half-sisters receive the 1/6 complement"
        if n == 1:
            return f"{_Q4_12} -- one paternal half-sister receives 1/2"
        return f"{_Q4_12} -- paternal half-sisters share 2/3 total ({n} sisters)"
    if ht == HeirType.DAUGHTER_OF_SON:
        if _count(heirs, HeirType.DAUGHTER) == 1:
            return f"{_Q4_176} -- son's daughter receives the 1/6 complement"
        if _count(heirs, HeirType.DAUGHTER_OF_SON) == 1:
            return f"{_Q4_176} -- one son's daughter receives 1/2"
        return f"{_Q4_176} -- son's daughters share 2/3 total"
    if ht == HeirType.GRANDFATHER:
        return f"{_Q4_11} -- grandfather fixed share in lieu of father (1/6)"
    if ht == HeirType.GRANDMOTHER:
        return f"{_Q4_12} -- grandmother (1/6, shared)"
    return "Juristic principle"


def _asaba_basis(heir_type: HeirType) -> str:
    """Return the fiqh citation for a residuary share."""
    labels = {
        HeirType.SON: "son (residuary)",
        HeirType.DAUGHTER: "daughter with sons (2:1 ratio)",
        HeirType.FATHER: "father (residuary)",
        HeirType.GRANDFATHER: "grandfather in lieu of father (residuary)",
        HeirType.SON_OF_SON: "son's son in lieu of son (residuary)",
        HeirType.DAUGHTER_OF_SON: "son's daughter with son's son (2:1 ratio)",
        HeirType.FULL_BROTHER: "full brother (residuary)",
        HeirType.FULL_SISTER: "full sister as residuary (with brother or with daughters)",
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
        return "Hajb -- son's granddaughter blocked by two or more daughters"
    if ht == HeirType.FULL_SISTER:
        return "Hajb -- full sister blocked by descendants, father, or grandfather"
    if ht == HeirType.FULL_BROTHER:
        return "Hajb -- full brother blocked by descendants, father, or grandfather"
    if ht == HeirType.PATERNAL_HALF_SISTER:
        return "Hajb -- paternal half-sister blocked by a nearer heir"
    if ht == HeirType.GRANDFATHER:
        return "Hajb -- grandfather blocked by father (nearer paternal relative)"
    if ht == HeirType.GRANDMOTHER:
        return "Hajb -- grandmother blocked by mother"
    return "Hajb -- blocked by nearer relative"


# ---------------------------------------------------------------------------
# Core algorithm
# ---------------------------------------------------------------------------


def distribute(estate: Decimal, heirs: list[HeirEntry]) -> FaraidResult:
    """Compute the faraid distribution for *estate* and *heirs*.

    All arithmetic uses :class:`fractions.Fraction` for exactness. Cases this
    engine only approximates are surfaced through ``uncertain`` and ``warnings``
    rather than silently returned as definitive.
    """
    if estate <= 0:
        raise ValueError("Estate must be positive")
    if not heirs:
        raise ValueError("At least one heir is required")

    warnings: list[str] = []
    uncertain = False
    steps: list[str] = []
    allocs: list[ShareAllocation] = []

    if _has(heirs, HeirType.GRANDFATHER) and (
        _has(heirs, HeirType.FULL_SISTER)
        or _has(heirs, HeirType.FULL_BROTHER)
        or _has(heirs, HeirType.PATERNAL_HALF_SISTER)
    ):
        warnings.append(
            "Grandfather inheriting alongside siblings (al-jadd wa al-ikhwa) is a case of "
            "scholarly disagreement; this engine uses a simplified residuary treatment. "
            "Confirm with a qualified scholar."
        )
        uncertain = True

    furud: dict[tuple[HeirType, int], Fraction] = {}
    for heir in heirs:
        share = _furud_share(heir.heir_type, heirs)
        if share is not None:
            furud[(heir.heir_type, heir.index)] = share

    total_furud = sum(furud.values(), Fraction(0))
    steps.append(f"Step 1 -- Furud: total fixed shares = {total_furud}")

    awl_applied = False
    awl_denom: int | None = None
    if total_furud > Fraction(1):
        awl_applied = True
        awl_denom = total_furud.numerator
        factor = Fraction(1) / total_furud
        furud = {key: value * factor for key, value in furud.items()}
        total_furud = sum(furud.values(), Fraction(0))
        steps.append(
            f"Step 2 -- Awl applied: shares exceeded the estate. "
            f"All fixed shares scaled by {factor} (denominator {awl_denom})."
        )

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

    residue = Fraction(1) - sum((a.fraction for a in allocs if a.category == "fard"), Fraction(0))

    asaba = _asaba_heirs(heirs)
    hajb_applied = any(a.category == "blocked" for a in allocs)

    if asaba and residue > 0:
        weights = [_asaba_weight(h.heir_type) for h in asaba]
        total_w = sum(weights, Fraction(0))
        if total_w > 0:
            for h, weight in zip(asaba, weights, strict=True):
                asaba_frac = residue * weight / total_w
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

    radd_applied = False
    if residue > 0 and not asaba:
        non_spouse = [
            a for a in allocs if a.category == "fard" and a.heir_type not in (HeirType.WIFE, HeirType.HUSBAND)
        ]
        if non_spouse:
            radd_applied = True
            base_total = sum((a.fraction for a in non_spouse), Fraction(0))
            for a in non_spouse:
                extra = residue * a.fraction / base_total
                a.fraction += extra
                a.amount = _frac_amount(a.fraction, estate)
                a.category = "radd"
            steps.append(
                "Step 5 -- Radd: surplus returned to non-spouse sharers "
                "(majority Sunni position excludes the spouse from radd)."
            )
            residue = Fraction(0)

    if residue > 0:
        warnings.append(
            "Part of the estate remains unallocated under this rule set (no eligible residuary "
            "or radd recipient). The remainder would pass to the bayt al-mal or dhawu al-arham "
            "depending on jurisdiction. Confirm with a qualified scholar."
        )
        uncertain = True

    total = sum((a.fraction for a in allocs), Fraction(0))
    if total > Fraction(1):
        warnings.append("Allocated shares exceed the estate; the result is inconsistent.")
        uncertain = True
    steps.append(f"Step 6 -- Verification: total allocated = {total}")

    return FaraidResult(
        estate=estate,
        total_allocated=total,
        allocations=allocs,
        awl_applied=awl_applied,
        awl_denominator=awl_denom,
        radd_applied=radd_applied,
        hajb_applied=hajb_applied,
        uncertain=uncertain,
        warnings=warnings,
        steps=steps,
        disclaimer=DISCLAIMER,
    )


# ---------------------------------------------------------------------------
# Assumption validation - explicit, versioned, and testable
# ---------------------------------------------------------------------------


def _entries(pairs: list[tuple[HeirType, int]]) -> list[HeirEntry]:
    heirs: list[HeirEntry] = []
    for heir_type, count in pairs:
        for index in range(count):
            heirs.append(HeirEntry(heir_type=heir_type, index=index))
    return heirs


def _shares_by_type(result: FaraidResult) -> dict[str, Fraction]:
    out: dict[str, Fraction] = {}
    for allocation in result.allocations:
        key = allocation.heir_type.value
        out[key] = out.get(key, Fraction(0)) + allocation.fraction
    return out


def validate_faraid_assumptions() -> FaraidAssumptions:
    """Check the faraid rule set's invariants and return a versioned record.

    Designed to be called at startup, in tests, and during audits. It never
    raises: callers inspect ``valid`` and ``errors`` so a drift is surfaced
    rather than silenced. Pure and offline.
    """
    errors: list[str] = []

    try:
        if len(FARAID_RULESET_VERSION.split(".")) != 3:
            errors.append(f"FARAID_RULESET_VERSION {FARAID_RULESET_VERSION!r} is not MAJOR.MINOR.PATCH.")
        if set(STANDARD_QURANIC_FURUD) != {
            Fraction(1, 2),
            Fraction(1, 4),
            Fraction(1, 8),
            Fraction(2, 3),
            Fraction(1, 3),
            Fraction(1, 6),
        }:
            errors.append("STANDARD_QURANIC_FURUD no longer matches the six Quranic fractions.")
        if not FARAID_SOURCES or not FARAID_BASIS:
            errors.append("Provenance basis and sources must be non-empty.")
        if {h.value for h in HeirType} < {"wife", "husband", "son", "daughter", "father", "mother"}:
            errors.append("Core heir types are missing from HeirType.")

        wife_case = distribute(
            Decimal("80000"), _entries([(HeirType.WIFE, 1), (HeirType.SON, 2), (HeirType.DAUGHTER, 1)])
        )
        if _shares_by_type(wife_case).get("wife") != Fraction(1, 8):
            errors.append("Wife-with-descendants share drifted from 1/8.")
        if wife_case.total_allocated != Fraction(1):
            errors.append("Wife + sons + daughter case no longer allocates the whole estate.")

        awl_case = distribute(Decimal("70000"), _entries([(HeirType.HUSBAND, 1), (HeirType.FULL_SISTER, 2)]))
        if not awl_case.awl_applied or awl_case.awl_denominator != 7 or awl_case.total_allocated != Fraction(1):
            errors.append("Awl case (husband + two full sisters) no longer matches 7/6 -> 1.")

        radd_case = distribute(Decimal("100000"), _entries([(HeirType.DAUGHTER, 1)]))
        if not radd_case.radd_applied or _shares_by_type(radd_case).get("daughter") != Fraction(1):
            errors.append("Radd case (single daughter) no longer returns the whole estate.")

        hajb_case = distribute(Decimal("120000"), _entries([(HeirType.SON, 1), (HeirType.SON_OF_SON, 1)]))
        if not hajb_case.hajb_applied or _shares_by_type(hajb_case).get("son_of_son") != Fraction(0):
            errors.append("Hajb case (son blocks son's son) no longer blocks the grandson.")

        for case in (wife_case, awl_case, radd_case, hajb_case):
            for allocation in case.allocations:
                if not allocation.basis:
                    errors.append(f"Allocation {allocation.heir_type.value}#{allocation.heir_index} has no basis.")
                if not isinstance(allocation.fraction, Fraction):
                    errors.append(f"Allocation {allocation.heir_type.value}#{allocation.heir_index} is not exact.")
    except Exception as exc:  # noqa: BLE001 - validation must never raise
        errors.append(f"validate_faraid_assumptions() raised unexpectedly: {exc}")

    return FaraidAssumptions(
        version=FARAID_RULESET_VERSION,
        madhab=FARAID_MADHAB,
        basis=FARAID_BASIS,
        sources=list(FARAID_SOURCES),
        supported_heirs=[h.value for h in HeirType],
        fixed_furud=[str(f) for f in STANDARD_QURANIC_FURUD],
        awl_rule="If fixed shares exceed the estate, every fixed share is scaled by 1/sum(furud).",
        radd_rule="If residue remains and no residuary heir exists, it is returned to non-spouse sharers.",
        hajb_rule="A nearer relative blocks a more distant one (e.g. son blocks son's son).",
        uncertainty_rules=list(UNCERTAINTY_RULES),
        valid=not errors,
        errors=errors,
    )


# ---------------------------------------------------------------------------
# FastAPI endpoint
# ---------------------------------------------------------------------------


class HeirInput(BaseModel):
    """A group of heirs of the same type."""

    heir_type: str = Field(
        ...,
        description=(
            "Heir type in English or Arabic, e.g. wife, husband, son, daughter, father, "
            "mother, full_sister, full_brother, paternal_half_sister, son_of_son, "
            "daughter_of_son, grandfather, grandmother."
        ),
    )
    count: int = Field(1, ge=1, le=MAX_HEIRS, description="Number of heirs of this type")


class FaraidRequest(BaseModel):
    """Request body for the faraid endpoint."""

    estate: float = Field(..., gt=0, description="Total estate value")
    heirs: list[HeirInput] = Field(..., min_length=1, description="List of heir groups")
    currency: str = Field("USD", min_length=1, max_length=8, description="Currency label for the amounts")


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
    uncertain: bool
    warnings: list[str]
    steps: list[str]
    provenance: FaraidProvenance
    disclaimer: str


def _build_heirs(groups: list[HeirInput]) -> list[HeirEntry]:
    counts: dict[HeirType, int] = {}
    for group in groups:
        heir_type = resolve_heir_type(group.heir_type)
        if heir_type is None:
            supported = ", ".join(sorted(h.value for h in HeirType))
            raise APIException(
                status_code=400,
                detail=f"Unknown heir type: {group.heir_type!r}",
                hint=f"Use one of: {supported} (Arabic labels such as زوجة/ابن are also accepted).",
            )
        counts[heir_type] = counts.get(heir_type, 0) + group.count

    total = sum(counts.values())
    if total > MAX_HEIRS:
        raise APIException(
            status_code=400,
            detail=f"Too many heirs: {total} (maximum {MAX_HEIRS}).",
            hint="Split the estate across at most 100 recorded heirs.",
        )
    if counts.get(HeirType.HUSBAND, 0) > 1:
        raise APIException(
            status_code=400,
            detail="At most one husband can be recorded.",
            hint="A decedent has at most one husband.",
        )
    if counts.get(HeirType.WIFE) and counts.get(HeirType.HUSBAND):
        raise APIException(
            status_code=400,
            detail="A decedent cannot leave both a husband and a wife.",
            hint="Record the spouse of the decedent only.",
        )

    heirs: list[HeirEntry] = []
    for heir_type, count in counts.items():
        for index in range(count):
            heirs.append(HeirEntry(heir_type=heir_type, index=index))
    if not heirs:
        raise APIException(
            status_code=400,
            detail="At least one heir is required.",
            hint="Provide at least one heir group.",
        )
    return heirs


def _estate_to_decimal(estate: float) -> Decimal:
    estate_int = int(estate)
    return Decimal(estate_int) if estate == estate_int else Decimal(str(estate))


@router.post("/faraid", response_model=FaraidResponse)
async def calculate_faraid(request: FaraidRequest) -> FaraidResponse:
    """Calculate Islamic inheritance (faraid) distribution.

    Given a set of surviving heirs and an estate value, returns each heir's
    exact share, the reasoning steps, fiqh citations, versioned rule-set
    provenance, and an uncertainty flag for cases the engine only approximates.
    """
    heirs = _build_heirs(request.heirs)
    result = distribute(_estate_to_decimal(request.estate), heirs)

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
        uncertain=result.uncertain,
        warnings=result.warnings,
        steps=result.steps,
        provenance=faraid_provenance(),
        disclaimer=result.disclaimer,
    )


@router.get("/faraid/assumptions", response_model=FaraidAssumptions)
async def faraid_assumptions() -> FaraidAssumptions:
    """Expose the versioned assumptions and invariant checks behind the engine."""
    return validate_faraid_assumptions()
