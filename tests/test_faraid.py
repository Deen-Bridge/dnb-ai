"""Tests for the faraid (Islamic inheritance) computation engine.

Everything runs offline: no network calls, no API keys needed.
"""

from decimal import Decimal
from fractions import Fraction

import pytest
from fastapi.testclient import TestClient

from faraid import (
    FaraidResult,
    HeirEntry,
    HeirType,
    distribute,
)
from main import app

client = TestClient(app)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _heirs(*pairs: tuple[str, int]) -> list[HeirEntry]:
    """Build a flat list of HeirEntry from (type_name, count) pairs."""
    heirs: list[HeirEntry] = []
    for type_name, count in pairs:
        ht = HeirType(type_name)
        for i in range(count):
            heirs.append(HeirEntry(heir_type=ht, index=i))
    return heirs


def _shares_by_type(result: FaraidResult) -> dict[str, Fraction]:
    """Aggregate shares by heir type (summing across instances)."""
    out: dict[str, Fraction] = {}
    for a in result.allocations:
        key = a.heir_type.value
        out[key] = out.get(key, Fraction(0)) + a.fraction
    return out


# ---------------------------------------------------------------------------
# Furud + asaba: wife (1/8) + sons + daughter
# ---------------------------------------------------------------------------


class TestFurudPlusAsaba:
    def test_wife_plus_sons_plus_daughter(self):
        """Wife gets 1/8 (fard); sons and daughter split residue 2:1 (asaba)."""
        heirs = _heirs(("wife", 1), ("son", 2), ("daughter", 1))
        result = distribute(Decimal("80000"), heirs)
        shares = _shares_by_type(result)

        assert shares["wife"] == Fraction(1, 8)
        # Residue = 7/8, split 2:1 among 2 sons + 1 daughter
        # Total male parts = 4, female parts = 1, total = 5
        # Each son: 2/5 * 7/8 = 7/20, total sons = 14/20 = 7/10
        assert shares["son"] == Fraction(2, 5) * Fraction(7, 8) * 2  # two sons
        assert shares["daughter"] == Fraction(1, 5) * Fraction(7, 8)  # 7/40

    def test_amounts_sum_to_estate(self):
        heirs = _heirs(("wife", 1), ("son", 2), ("daughter", 1))
        result = distribute(Decimal("80000"), heirs)
        total = sum(a.amount for a in result.allocations)
        assert total == Decimal("80000")

    def test_no_awl_no_radd(self):
        heirs = _heirs(("wife", 1), ("son", 2), ("daughter", 1))
        result = distribute(Decimal("80000"), heirs)
        assert result.awl_applied is False
        assert result.radd_applied is False


# ---------------------------------------------------------------------------
# Awl: husband (1/2) + two full sisters (2/3)
# ---------------------------------------------------------------------------


class TestAwl:
    def test_awl_applied(self):
        """Husband 1/2 + 2 sisters 2/3 = 7/6 > 1, so awl is applied."""
        heirs = _heirs(("husband", 1), ("full_sister", 2))
        result = distribute(Decimal("70000"), heirs)
        assert result.awl_applied is True
        assert result.awl_denominator == 7

    def test_correct_fractions_after_awl(self):
        heirs = _heirs(("husband", 1), ("full_sister", 2))
        result = distribute(Decimal("70000"), heirs)
        shares = _shares_by_type(result)
        # After awl: husband = 3/7, each sister = 2/7, total sisters = 4/7
        assert shares["husband"] == Fraction(3, 7)
        assert shares["full_sister"] == Fraction(4, 7)

    def test_amounts_sum_to_estate(self):
        heirs = _heirs(("husband", 1), ("full_sister", 2))
        result = distribute(Decimal("70000"), heirs)
        total = sum(a.amount for a in result.allocations)
        assert total == Decimal("70000")

    def test_awl_denominator_is_exact(self):
        """Awl must use exact Fraction arithmetic, never float."""
        heirs = _heirs(("husband", 1), ("full_sister", 2))
        result = distribute(Decimal("70000"), heirs)
        for a in result.allocations:
            assert isinstance(a.fraction, Fraction)


# ---------------------------------------------------------------------------
# Radd: single daughter (1/2) with no asaba
# ---------------------------------------------------------------------------


class TestRadd:
    def test_single_daughter_gets_whole_estate(self):
        """One daughter gets 1/2 (fard), remaining 1/2 returns via radd."""
        heirs = _heirs(("daughter", 1))
        result = distribute(Decimal("100000"), heirs)
        shares = _shares_by_type(result)
        assert shares["daughter"] == Fraction(1)
        assert result.radd_applied is True

    def test_daughter_category_is_radd(self):
        heirs = _heirs(("daughter", 1))
        result = distribute(Decimal("100000"), heirs)
        daught = [a for a in result.allocations if a.heir_type == HeirType.DAUGHTER][0]
        assert daught.category == "radd"

    def test_amounts_sum_to_estate(self):
        heirs = _heirs(("daughter", 1))
        result = distribute(Decimal("100000"), heirs)
        total = sum(a.amount for a in result.allocations)
        assert total == Decimal("100000")


# ---------------------------------------------------------------------------
# Radd with spouse: spouse excluded from radd
# ---------------------------------------------------------------------------


class TestRaddWithSpouse:
    def test_wife_excluded_from_radd(self):
        """Wife gets 1/8 (fard); daughter gets 1/2 + all residue (radd)."""
        heirs = _heirs(("wife", 1), ("daughter", 1))
        result = distribute(Decimal("240000"), heirs)
        shares = _shares_by_type(result)

        # Wife: 1/8 (fard only, no radd)
        assert shares["wife"] == Fraction(1, 8)
        # Daughter: 1/2 (fard) + 3/8 (radd) = 7/8
        assert shares["daughter"] == Fraction(7, 8)

    def test_wife_category_stays_fard(self):
        heirs = _heirs(("wife", 1), ("daughter", 1))
        result = distribute(Decimal("240000"), heirs)
        wife = [a for a in result.allocations if a.heir_type == HeirType.WIFE][0]
        assert wife.category == "fard"

    def test_daughter_category_becomes_radd(self):
        heirs = _heirs(("wife", 1), ("daughter", 1))
        result = distribute(Decimal("240000"), heirs)
        daught = [a for a in result.allocations if a.heir_type == HeirType.DAUGHTER][0]
        assert daught.category == "radd"

    def test_amounts_sum_to_estate(self):
        heirs = _heirs(("wife", 1), ("daughter", 1))
        result = distribute(Decimal("240000"), heirs)
        total = sum(a.amount for a in result.allocations)
        assert total == Decimal("240000")


# ---------------------------------------------------------------------------
# Hajb: grandson blocked by son
# ---------------------------------------------------------------------------


class TestHajb:
    def test_grandson_blocked_by_son(self):
        """Son's son gets 0 when a son is present."""
        heirs = _heirs(("son", 1), ("son_of_son", 1))
        result = distribute(Decimal("120000"), heirs)
        shares = _shares_by_type(result)

        assert shares["son"] == Fraction(1)
        assert shares["son_of_son"] == Fraction(0)
        assert result.hajb_applied is True

    def test_blocked_heir_has_zero_amount(self):
        heirs = _heirs(("son", 1), ("son_of_son", 1))
        result = distribute(Decimal("120000"), heirs)
        blocked = [a for a in result.allocations if a.category == "blocked"]
        assert len(blocked) == 1
        assert blocked[0].amount == Decimal("0")

    def test_blocked_heir_has_basis(self):
        heirs = _heirs(("son", 1), ("son_of_son", 1))
        result = distribute(Decimal("120000"), heirs)
        blocked = [a for a in result.allocations if a.category == "blocked"]
        assert blocked[0].basis  # non-empty


# ---------------------------------------------------------------------------
# Mother's share
# ---------------------------------------------------------------------------


class TestMother:
    def test_mother_with_children_gets_one_sixth(self):
        heirs = _heirs(("son", 1), ("mother", 1))
        result = distribute(Decimal("60000"), heirs)
        shares = _shares_by_type(result)
        assert shares["mother"] == Fraction(1, 6)

    def test_mother_without_children_gets_one_third(self):
        heirs = _heirs(("mother", 1), ("father", 1))
        result = distribute(Decimal("60000"), heirs)
        shares = _shares_by_type(result)
        assert shares["mother"] == Fraction(1, 3)


# ---------------------------------------------------------------------------
# Father
# ---------------------------------------------------------------------------


class TestFather:
    def test_father_with_children_is_sixth_plus_residue(self):
        """Father gets 1/6 (fard) + residue (asaba) when children exist."""
        heirs = _heirs(("father", 1), ("daughter", 1))
        result = distribute(Decimal("120000"), heirs)
        shares = _shares_by_type(result)
        # Father: 1/6 (fard) + 1/3 (asaba) = 1/2
        # Daughter: 1/2 (fard)
        assert shares["father"] == Fraction(1, 2)
        assert shares["daughter"] == Fraction(1, 2)


# ---------------------------------------------------------------------------
# Multiple daughters (2/3)
# ---------------------------------------------------------------------------


class TestMultipleDaughters:
    def test_two_daughters_share_two_thirds(self):
        heirs = _heirs(("daughter", 2))
        result = distribute(Decimal("90000"), heirs)
        shares = _shares_by_type(result)
        # Each daughter: 1/3 (fard), total 2/3; radd returns 1/3
        # Final: each gets 1/2, total 1
        assert shares["daughter"] == Fraction(1)

    def test_radd_returns_surplus_to_daughters(self):
        """Two daughters: 2/3 furud, 1/3 residue returns via radd."""
        heirs = _heirs(("daughter", 2))
        result = distribute(Decimal("90000"), heirs)
        assert result.radd_applied is True


# ---------------------------------------------------------------------------
# Grandmother blocked by mother
# ---------------------------------------------------------------------------


class TestGrandmother:
    def test_grandmother_blocked_by_mother(self):
        heirs = _heirs(("mother", 1), ("grandmother", 1))
        result = distribute(Decimal("60000"), heirs)
        shares = _shares_by_type(result)
        assert shares["grandmother"] == Fraction(0)
        assert result.hajb_applied is True

    def test_grandmother_gets_one_sixth_when_no_mother(self):
        heirs = _heirs(("grandmother", 1), ("son", 1))
        result = distribute(Decimal("60000"), heirs)
        shares = _shares_by_type(result)
        assert shares["grandmother"] == Fraction(1, 6)


# ---------------------------------------------------------------------------
# Simple son-only case
# ---------------------------------------------------------------------------


class TestSimpleSon:
    def test_son_gets_all(self):
        heirs = _heirs(("son", 1))
        result = distribute(Decimal("50000"), heirs)
        shares = _shares_by_type(result)
        assert shares["son"] == Fraction(1)

    def test_two_sons_split_equally(self):
        heirs = _heirs(("son", 2))
        result = distribute(Decimal("50000"), heirs)
        shares = _shares_by_type(result)
        assert shares["son"] == Fraction(1)


# ---------------------------------------------------------------------------
# Every allocation has a cited basis
# ---------------------------------------------------------------------------


class TestBasisPresent:
    @pytest.mark.parametrize(
        "heirs_input,estate",
        [
            ([("wife", 1), ("son", 2), ("daughter", 1)], "80000"),
            ([("husband", 1), ("full_sister", 2)], "70000"),
            ([("daughter", 1)], "100000"),
            ([("son", 1), ("son_of_son", 1)], "120000"),
        ],
    )
    def test_every_allocation_has_basis(self, heirs_input, estate):
        heirs = _heirs(*heirs_input)
        result = distribute(Decimal(estate), heirs)
        for a in result.allocations:
            assert a.basis, f"{a.heir_type.value}#{a.heir_index} missing basis"


# ---------------------------------------------------------------------------
# Exact arithmetic
# ---------------------------------------------------------------------------


class TestExactArithmetic:
    def test_all_fractions_are_exact(self):
        """No float rounding drift -- every share is a Fraction."""
        heirs = _heirs(("husband", 1), ("full_sister", 2))
        result = distribute(Decimal("70000"), heirs)
        for a in result.allocations:
            assert isinstance(a.fraction, Fraction)

    def test_shares_sum_to_one(self):
        heirs = _heirs(("wife", 1), ("son", 2), ("daughter", 1))
        result = distribute(Decimal("80000"), heirs)
        assert result.total_allocated == Fraction(1)


# ---------------------------------------------------------------------------
# Running-app endpoint test
# ---------------------------------------------------------------------------


class TestFaraidEndpoint:
    def test_post_faraid_returns_200(self):
        payload = {
            "estate": 90000,
            "heirs": [
                {"heir_type": "wife", "count": 1},
                {"heir_type": "son", "count": 2},
                {"heir_type": "daughter", "count": 1},
            ],
        }
        response = client.post("/faraid", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["estate"] == "90000"
        assert len(data["shares"]) > 0
        assert data["disclaimer"]

    def test_post_faraid_awl_case(self):
        payload = {
            "estate": 70000,
            "heirs": [
                {"heir_type": "husband", "count": 1},
                {"heir_type": "full_sister", "count": 2},
            ],
        }
        response = client.post("/faraid", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["awl_applied"] is True
        assert data["awl_denominator"] == 7

    def test_post_faraid_hajb_case(self):
        payload = {
            "estate": 120000,
            "heirs": [
                {"heir_type": "son", "count": 1},
                {"heir_type": "son_of_son", "count": 1},
            ],
        }
        response = client.post("/faraid", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["hajb_applied"] is True

    def test_post_faraid_invalid_heir_type(self):
        payload = {
            "estate": 10000,
            "heirs": [{"heir_type": "alien", "count": 1}],
        }
        response = client.post("/faraid", json=payload)
        assert response.status_code == 400

    def test_post_faraid_zero_heirs(self):
        payload = {"estate": 10000, "heirs": []}
        response = client.post("/faraid", json=payload)
        assert response.status_code == 422  # Pydantic validation
