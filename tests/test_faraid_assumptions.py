"""Validate faraid assumptions and provenance - PROD-AUDIT AI-059.

Acceptance criteria
-------------------
- Outputs are grounded in versioned data with provenance and uncertainty.
- Arabic/English edge cases and empty/failure states are tested.
- The engine is stateless (no user-scoped data to isolate or reset).
- A reviewer can reproduce every test without network access or secrets.

Design
------
Everything here is offline and deterministic: the faraid engine is pure
:class:`fractions.Fraction` arithmetic over the committed rule set. The
assumption validator is also pure, so it is safe to call at startup or in CI.
"""

from __future__ import annotations

from decimal import Decimal
from fractions import Fraction

import pytest

import faraid
from faraid import (
    FARAID_RULESET_VERSION,
    STANDARD_QURANIC_FURUD,
    FaraidAssumptions,
    HeirType,
    distribute,
    resolve_heir_type,
    validate_faraid_assumptions,
)

# ---------------------------------------------------------------------------
# 1. Versioned assumption record
# ---------------------------------------------------------------------------


class TestFaraidAssumptionsRecord:
    def test_returns_faraid_assumptions_instance(self):
        assert isinstance(validate_faraid_assumptions(), FaraidAssumptions)

    def test_valid_on_unmodified_codebase(self):
        record = validate_faraid_assumptions()
        assert record.valid is True, record.errors

    def test_no_errors_on_unmodified_codebase(self):
        assert validate_faraid_assumptions().errors == []

    def test_version_matches_constant(self):
        assert validate_faraid_assumptions().version == FARAID_RULESET_VERSION

    def test_version_is_semver_shaped(self):
        parts = validate_faraid_assumptions().version.split(".")
        assert len(parts) == 3
        assert all(part.isdigit() for part in parts)

    def test_madhab_and_basis_are_explicit(self):
        record = validate_faraid_assumptions()
        assert record.madhab == "sunni_majority"
        assert "sunni" in record.basis.lower()
        assert "quran" in record.basis.lower()

    def test_sources_are_cited(self):
        record = validate_faraid_assumptions()
        assert record.sources
        assert any("4:11" in source for source in record.sources)
        assert any("hajb" in source.lower() for source in record.sources)


# ---------------------------------------------------------------------------
# 2. Rule-set invariants
# ---------------------------------------------------------------------------


class TestRuleSetInvariants:
    def test_six_quranic_furud(self):
        record = validate_faraid_assumptions()
        assert set(record.fixed_furud) == {"1/2", "1/4", "1/8", "2/3", "1/3", "1/6"}
        assert set(STANDARD_QURANIC_FURUD) == {
            Fraction(1, 2),
            Fraction(1, 4),
            Fraction(1, 8),
            Fraction(2, 3),
            Fraction(1, 3),
            Fraction(1, 6),
        }

    def test_supported_heirs_include_core_relatives(self):
        supported = set(validate_faraid_assumptions().supported_heirs)
        assert {"wife", "husband", "son", "daughter", "father", "mother"} <= supported
        assert {"full_sister", "full_brother", "grandfather", "grandmother"} <= supported

    def test_awl_radd_hajb_rules_documented(self):
        record = validate_faraid_assumptions()
        assert record.awl_rule
        assert record.radd_rule
        assert record.hajb_rule

    def test_uncertainty_rules_cover_known_gaps(self):
        rules = " ".join(validate_faraid_assumptions().uncertainty_rules).lower()
        assert "grandfather" in rules
        assert "unallocated" in rules
        assert "dhawu al-arham" in rules

    def test_record_round_trips_through_dict(self):
        original = validate_faraid_assumptions()
        restored = FaraidAssumptions(**original.model_dump())
        assert restored.version == original.version
        assert restored.valid == original.valid
        assert restored.errors == original.errors


# ---------------------------------------------------------------------------
# 3. Drift detection
# ---------------------------------------------------------------------------


class TestDriftDetection:
    def test_detects_non_semver_version(self, monkeypatch):
        monkeypatch.setattr(faraid, "FARAID_RULESET_VERSION", "not-a-version")
        record = validate_faraid_assumptions()
        assert record.valid is False
        assert any("FARAID_RULESET_VERSION" in error for error in record.errors)

    def test_detects_tampered_furud_table(self, monkeypatch):
        monkeypatch.setattr(faraid, "STANDARD_QURANIC_FURUD", (Fraction(1, 2),))
        record = validate_faraid_assumptions()
        assert record.valid is False
        assert any("STANDARD_QURANIC_FURUD" in error for error in record.errors)

    def test_detects_engine_exception_without_raising(self, monkeypatch):
        def broken(*args, **kwargs):
            raise RuntimeError("engine drifted")

        monkeypatch.setattr(faraid, "distribute", broken)
        record = validate_faraid_assumptions()
        assert record.valid is False
        assert any("raised unexpectedly" in error for error in record.errors)

    def test_validator_never_raises(self):
        try:
            validate_faraid_assumptions()
        except Exception as exc:  # noqa: BLE001
            pytest.fail(f"validate_faraid_assumptions() raised unexpectedly: {exc}")


# ---------------------------------------------------------------------------
# 4. Arabic/English aliases and empty/failure states
# ---------------------------------------------------------------------------


class TestEdgeCases:
    @pytest.mark.parametrize(
        "label,expected",
        [
            ("wife", HeirType.WIFE),
            ("WIVES", HeirType.WIFE),
            ("زوجة", HeirType.WIFE),
            ("husband", HeirType.HUSBAND),
            ("زوج", HeirType.HUSBAND),
            ("son", HeirType.SON),
            ("ابن", HeirType.SON),
            ("daughter", HeirType.DAUGHTER),
            ("بنت", HeirType.DAUGHTER),
            ("grandson", HeirType.SON_OF_SON),
            ("granddaughter", HeirType.DAUGHTER_OF_SON),
            ("paternal_grandfather", HeirType.GRANDFATHER),
            ("maternal_grandmother", HeirType.GRANDMOTHER),
            ("full brother", HeirType.FULL_BROTHER),
            ("أخت شقيقة", HeirType.FULL_SISTER),
        ],
    )
    def test_alias_resolution(self, label, expected):
        assert resolve_heir_type(label) == expected

    def test_unknown_and_empty_labels(self):
        assert resolve_heir_type("") is None
        assert resolve_heir_type("   ") is None
        assert resolve_heir_type(None) is None
        assert resolve_heir_type("dragon") is None

    def test_zero_or_negative_estate_is_rejected(self):
        heirs = faraid._entries([(HeirType.SON, 1)])
        with pytest.raises(ValueError):
            distribute(Decimal("0"), heirs)
        with pytest.raises(ValueError):
            distribute(Decimal("-1"), heirs)

    def test_empty_heir_list_is_rejected(self):
        with pytest.raises(ValueError):
            distribute(Decimal("1000"), [])

    def test_uncertainty_flags_unallocated_residue(self):
        result = distribute(Decimal("100000"), faraid._entries([(HeirType.WIFE, 1)]))
        assert result.uncertain is True
        assert result.warnings
        assert result.total_allocated == Fraction(1, 4)
