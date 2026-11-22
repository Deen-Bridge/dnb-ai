"""Failure states and known limitations of the faraid calculator (#848).

The calculator applies fixed base shares without conditioning them on the other
heirs present, so several rulings stated explicitly in Qur'an 4:11-12 are not
yet produced. Those cases are recorded below as strict expected failures: they
document the gap in CI and will start failing (XPASS) once the rules are fixed,
which is the signal to turn them into ordinary tests. Correcting the rules
needs content review and is tracked separately; see
docs/PUBLIC_KNOWLEDGE_GOVERNANCE.md.
"""

from __future__ import annotations

from decimal import Decimal
from fractions import Fraction

import pytest

from faraid import DISCLAIMER, distribute


def _shares(*heirs: str) -> dict[str, Fraction]:
    return distribute(Decimal(1200), list(heirs)).shares


# ---------------------------------------------------------------------------
# Failure states: fail closed instead of answering
# ---------------------------------------------------------------------------


def test_unknown_heir_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown heir"):
        distribute(Decimal(100), ["cousin_of_neighbour"])


def test_empty_heir_list_is_rejected() -> None:
    with pytest.raises(ValueError, match="At least one heir"):
        distribute(Decimal(100), [])


@pytest.mark.parametrize("estate", [Decimal(0), Decimal(-5)])
def test_non_positive_estate_is_rejected(estate: Decimal) -> None:
    with pytest.raises(ValueError, match="positive"):
        distribute(estate, ["son"])


def test_repeated_heir_type_is_refused_rather_than_merged() -> None:
    # Previously returned a single 9/8 share: more than the whole estate.
    with pytest.raises(ValueError, match="not supported"):
        distribute(Decimal(1200), ["daughter", "daughter"])


@pytest.mark.parametrize(
    "heirs",
    [
        ["son"],
        ["wife", "son"],
        ["husband", "mother", "full_sister"],
        ["mother", "father"],
        ["father", "son", "full_brother"],
        ["wife", "mother", "daughter", "granddaughter"],
    ],
)
def test_shares_never_exceed_the_estate(heirs: list[str]) -> None:
    shares = _shares(*heirs)
    assert sum(shares.values()) <= 1
    assert all(share >= 0 for share in shares.values())


def test_blocked_heir_receives_nothing() -> None:
    shares = _shares("son", "full_brother")
    assert shares["full_brother"] == 0
    assert shares["son"] == 1


def test_disclaimer_names_the_school_and_advises_a_scholar() -> None:
    assert "Sunni" in DISCLAIMER
    assert "scholar" in DISCLAIMER


# ---------------------------------------------------------------------------
# Known gaps against explicit Qur'anic shares (strict expected failures)
# ---------------------------------------------------------------------------

_GAP = "faraid base shares are not yet conditioned on the presence of children or other heirs"


@pytest.mark.xfail(strict=True, reason=_GAP)
def test_wife_receives_one_eighth_when_there_are_children() -> None:
    # Qur'an 4:12
    assert _shares("wife", "son")["wife"] == Fraction(1, 8)


@pytest.mark.xfail(strict=True, reason=_GAP)
def test_husband_receives_one_quarter_when_there_are_children() -> None:
    # Qur'an 4:12
    assert _shares("husband", "daughter")["husband"] == Fraction(1, 4)


@pytest.mark.xfail(strict=True, reason=_GAP)
def test_son_receives_twice_the_daughter() -> None:
    # Qur'an 4:11
    shares = _shares("son", "daughter")
    assert shares["son"] == 2 * shares["daughter"]


@pytest.mark.xfail(strict=True, reason=_GAP)
def test_mother_receives_one_third_without_children_or_siblings() -> None:
    # Qur'an 4:11: with only the parents inheriting, the mother has a third.
    assert _shares("mother", "father")["mother"] == Fraction(1, 3)
