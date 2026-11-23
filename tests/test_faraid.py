"""Tests for the faraid (Islamic inheritance) computation engine.

Everything runs offline: no network calls, no API keys needed.
"""

from decimal import Decimal
from fractions import Fraction

import pytest
from fastapi.testclient import TestClient

from faraid import (
    FARAID_RULESET_VERSION,
    FaraidResult,
    HeirEntry,
    HeirType,
    distribute,
    faraid_provenance,
    resolve_heir_type,
    validate_faraid_assumptions,
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


# ---------------------------------------------------------------------------
# Edge cases: aggregate shares, grandfather, grandmothers, siblings, grandchildren
# ---------------------------------------------------------------------------


class TestMultipleWives:
    def test_wives_share_the_collective_eighth(self):
        heirs = _heirs(("wife", 2), ("son", 1))
        result = distribute(Decimal("160000"), heirs)
        shares = _shares_by_type(result)
        assert shares["wife"] == Fraction(1, 8)
        wife_parts = [a.fraction for a in result.allocations if a.heir_type == HeirType.WIFE]
        assert wife_parts == [Fraction(1, 16), Fraction(1, 16)]
        assert shares["son"] == Fraction(7, 8)


class TestMotherSiblingsRule:
    def test_two_siblings_reduce_mother_to_one_sixth(self):
        heirs = _heirs(("mother", 1), ("full_brother", 1), ("full_sister", 1))
        result = distribute(Decimal("90000"), heirs)
        shares = _shares_by_type(result)
        assert shares["mother"] == Fraction(1, 6)
        assert shares["full_brother"] == Fraction(5, 9)
        assert shares["full_sister"] == Fraction(5, 18)
        assert result.total_allocated == Fraction(1)


class TestGrandfather:
    def test_grandfather_blocked_by_father(self):
        heirs = _heirs(("father", 1), ("grandfather", 1))
        result = distribute(Decimal("60000"), heirs)
        shares = _shares_by_type(result)
        assert shares["grandfather"] == Fraction(0)
        assert shares["father"] == Fraction(1)
        assert result.hajb_applied is True

    def test_grandfather_gets_sixth_plus_residue(self):
        heirs = _heirs(("grandfather", 1), ("daughter", 1))
        result = distribute(Decimal("120000"), heirs)
        shares = _shares_by_type(result)
        assert shares["grandfather"] == Fraction(1, 2)
        assert shares["daughter"] == Fraction(1, 2)


class TestGrandmothers:
    def test_grandmothers_share_a_single_sixth(self):
        heirs = _heirs(("grandmother", 2), ("son", 1))
        result = distribute(Decimal("120000"), heirs)
        shares = _shares_by_type(result)
        assert shares["grandmother"] == Fraction(1, 6)
        assert shares["son"] == Fraction(5, 6)


class TestFullSiblings:
    def test_brother_and_sister_share_residue_two_to_one(self):
        heirs = _heirs(("full_brother", 1), ("full_sister", 1))
        result = distribute(Decimal("90000"), heirs)
        shares = _shares_by_type(result)
        assert shares["full_brother"] == Fraction(2, 3)
        assert shares["full_sister"] == Fraction(1, 3)

    def test_full_sister_becomes_residuary_with_daughters(self):
        heirs = _heirs(("daughter", 1), ("full_sister", 1))
        result = distribute(Decimal("90000"), heirs)
        shares = _shares_by_type(result)
        assert shares["daughter"] == Fraction(1, 2)
        assert shares["full_sister"] == Fraction(1, 2)


class TestGrandchildren:
    def test_son_and_daughter_of_son_share_two_to_one(self):
        heirs = _heirs(("son_of_son", 1), ("daughter_of_son", 1))
        result = distribute(Decimal("90000"), heirs)
        shares = _shares_by_type(result)
        assert shares["son_of_son"] == Fraction(2, 3)
        assert shares["daughter_of_son"] == Fraction(1, 3)

    def test_son_daughter_alone_receives_half_then_radd(self):
        result = distribute(Decimal("90000"), _heirs(("daughter_of_son", 1)))
        assert _shares_by_type(result)["daughter_of_son"] == Fraction(1)
        assert result.radd_applied is True

    def test_son_daughter_receives_sixth_complement(self):
        heirs = _heirs(("daughter", 1), ("daughter_of_son", 1), ("full_brother", 1))
        result = distribute(Decimal("60000"), heirs)
        shares = _shares_by_type(result)
        assert shares["daughter"] == Fraction(1, 2)
        assert shares["daughter_of_son"] == Fraction(1, 6)
        assert shares["full_brother"] == Fraction(1, 3)


class TestUncertainty:
    def test_only_spouse_leaves_residue_unallocated(self):
        result = distribute(Decimal("100000"), _heirs(("wife", 1)))
        assert result.total_allocated == Fraction(1, 4)
        assert result.uncertain is True
        assert any("unallocated" in warning for warning in result.warnings)

    def test_grandfather_with_siblings_is_flagged_uncertain(self):
        result = distribute(Decimal("100000"), _heirs(("grandfather", 1), ("full_sister", 1)))
        assert result.uncertain is True
        assert any("Grandfather" in warning for warning in result.warnings)


class TestProvenanceAndAliases:
    def test_provenance_is_versioned(self):
        provenance = faraid_provenance()
        assert provenance.ruleset_version == FARAID_RULESET_VERSION
        assert provenance.madhab == "sunni_majority"
        assert provenance.sources

    def test_resolves_english_arabic_and_legacy_names(self):
        assert resolve_heir_type("Wife") == HeirType.WIFE
        assert resolve_heir_type("زوجة") == HeirType.WIFE
        assert resolve_heir_type("ابن") == HeirType.SON
        assert resolve_heir_type("grandson") == HeirType.SON_OF_SON
        assert resolve_heir_type("paternal_grandfather") == HeirType.GRANDFATHER
        assert resolve_heir_type("not-an-heir") is None

    def test_assumptions_validate_on_unmodified_code(self):
        record = validate_faraid_assumptions()
        assert record.valid is True
        assert record.errors == []
        assert record.version == FARAID_RULESET_VERSION


class TestInvalidDistributionInputs:
    def test_zero_estate_rejected(self):
        with pytest.raises(ValueError):
            distribute(Decimal("0"), _heirs(("son", 1)))

    def test_empty_heirs_rejected(self):
        with pytest.raises(ValueError):
            distribute(Decimal("1000"), [])


# ---------------------------------------------------------------------------
# Endpoint hardening: aliases, aggregation, conflicting spouses, provenance
# ---------------------------------------------------------------------------


class TestHardenedEndpoint:
    def test_duplicate_groups_are_aggregated(self):
        payload = {
            "estate": 100000,
            "heirs": [{"heir_type": "son", "count": 1}, {"heir_type": "son", "count": 1}],
        }
        response = client.post("/faraid", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert len(data["shares"]) == 2
        assert sum(Fraction(s["fraction"]) for s in data["shares"]) == Fraction(1)

    def test_arabic_heir_aliases_accepted(self):
        payload = {
            "estate": 80000,
            "heirs": [{"heir_type": "زوجة", "count": 1}, {"heir_type": "ابن", "count": 1}],
        }
        response = client.post("/faraid", json=payload)
        assert response.status_code == 200
        fractions = {s["heir_type"]: Fraction(s["fraction"]) for s in response.json()["shares"]}
        assert fractions["wife"] == Fraction(1, 8)
        assert fractions["son"] == Fraction(7, 8)

    def test_unknown_heir_type_includes_hint(self):
        response = client.post("/faraid", json={"estate": 1000, "heirs": [{"heir_type": "alien", "count": 1}]})
        assert response.status_code == 400
        body = response.json()
        assert "Unknown heir type" in body["detail"]
        assert "hint" in body

    def test_husband_and_wife_conflict_rejected(self):
        payload = {
            "estate": 1000,
            "heirs": [{"heir_type": "husband", "count": 1}, {"heir_type": "wife", "count": 1}],
        }
        response = client.post("/faraid", json=payload)
        assert response.status_code == 400

    def test_multiple_husbands_rejected(self):
        payload = {"estate": 1000, "heirs": [{"heir_type": "husband", "count": 2}]}
        response = client.post("/faraid", json=payload)
        assert response.status_code == 400

    def test_zero_estate_rejected_by_schema(self):
        response = client.post("/faraid", json={"estate": 0, "heirs": [{"heir_type": "son", "count": 1}]})
        assert response.status_code == 422

    def test_zero_count_rejected_by_schema(self):
        response = client.post("/faraid", json={"estate": 1000, "heirs": [{"heir_type": "son", "count": 0}]})
        assert response.status_code == 422

    def test_response_exposes_provenance_and_uncertainty(self):
        payload = {"estate": 100000, "heirs": [{"heir_type": "wife", "count": 1}]}
        data = client.post("/faraid", json=payload).json()
        assert data["provenance"]["ruleset_version"] == FARAID_RULESET_VERSION
        assert data["uncertain"] is True
        assert data["warnings"]

    def test_assumptions_endpoint(self):
        response = client.get("/faraid/assumptions")
        assert response.status_code == 200
        body = response.json()
        assert body["valid"] is True
        assert body["version"] == FARAID_RULESET_VERSION
