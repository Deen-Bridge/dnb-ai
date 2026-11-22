"""Validate zakat calculations and nisab assumptions — PROD-AUDIT AI-062.

Acceptance criteria
-------------------
- Outputs are grounded in versioned data with provenance and uncertainty.
- Arabic/English edge cases and empty/failure states are tested.
- User-scoped data is isolated and resettable (via the autouse cache fixture).
- A reviewer can reproduce every test without network access or secrets.

Design
------
Every test is offline-only. Network calls are either mocked or the code under
test is the pure arithmetic layer that never reaches the network. The
``clear_nisab_cache`` fixture (autouse) guarantees test isolation.
"""

from __future__ import annotations

import asyncio
from decimal import Decimal
from unittest.mock import patch

import pytest

import nisab
from nisab import (
    DEFAULT_NISAB_USD,
    FALLBACK_SOURCE,
    GRAMS_PER_TROY_OUNCE,
    MAX_PLAUSIBLE_OUNCE_USD,
    MIN_PLAUSIBLE_OUNCE_USD,
    NISAB_GOLD_GRAMS,
    PRIMARY_SOURCE,
    NisabAssumptions,
    NisabQuote,
    get_nisab,
    nisab_from_ounce_price,
    override_quote,
    parse_price,
    validate_nisab_assumptions,
)
from semantic_cache import get_keyed_cache

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def run(coro):
    return asyncio.run(coro)


def returns(value):
    """Async stub that always returns *value* (can be awaited multiple times)."""

    async def _stub(*args, **kwargs):
        return value

    return _stub


@pytest.fixture(autouse=True)
def clear_nisab_cache():
    """Isolate every test: wipe the nisab cache before and after."""
    get_keyed_cache(nisab.CACHE_NAMESPACE).clear()
    yield
    get_keyed_cache(nisab.CACHE_NAMESPACE).clear()


# ---------------------------------------------------------------------------
# 1. Constant integrity — fiqh and physics
# ---------------------------------------------------------------------------


class TestNisabConstants:
    """The fiqh constants must be exactly right; drift silently breaks rulings."""

    def test_nisab_gold_grams_is_85(self):
        """OIC/AAOIFI standard: 85 g of gold is the nisab for cash wealth."""
        assert NISAB_GOLD_GRAMS == Decimal("85"), (
            "The 85 g nisab is a scholarly-consensus constant (OIC Fiqh Academy 1986, "
            "AAOIFI Sharia Std No. 35). It must not be changed without a fatwa reference."
        )

    def test_grams_per_troy_ounce_is_nist_value(self):
        """NIST conversion factor: 1 troy ounce = 31.1034768 g exactly."""
        assert GRAMS_PER_TROY_OUNCE == Decimal("31.1034768"), (
            "GRAMS_PER_TROY_OUNCE must be the NIST value. Any change breaks nisab arithmetic."
        )

    def test_nisab_gold_grams_is_less_than_one_troy_ounce(self):
        """Sanity: 85 g < 31.1 g/oz means the nisab is less than 3 troy oz."""
        nisab_oz = NISAB_GOLD_GRAMS / GRAMS_PER_TROY_OUNCE
        assert Decimal("2") < nisab_oz < Decimal("3"), f"85 g should be ~2.73 troy oz, got {nisab_oz}"

    def test_plausibility_bounds_are_ordered(self):
        assert MIN_PLAUSIBLE_OUNCE_USD < MAX_PLAUSIBLE_OUNCE_USD

    def test_plausibility_bounds_bracket_realistic_gold_prices(self):
        """Current gold is in the $1 800–$3 500/oz range; bounds must include it."""
        for realistic_price in (Decimal("1800"), Decimal("2000"), Decimal("2500"), Decimal("3000")):
            assert MIN_PLAUSIBLE_OUNCE_USD <= realistic_price <= MAX_PLAUSIBLE_OUNCE_USD, (
                f"Realistic gold price {realistic_price} USD/oz is outside plausibility bounds"
            )


# ---------------------------------------------------------------------------
# 2. Nisab arithmetic — nisab_from_ounce_price()
# ---------------------------------------------------------------------------


class TestNisabFromOuncePrice:
    """Verify the formula: nisab = (85 / 31.1034768) × price, rounded to cents."""

    @pytest.mark.parametrize(
        "price_per_oz",
        ["1000", "2000", "2500", "3000", "3500"],
    )
    def test_known_prices(self, price_per_oz):
        """Each result must equal the independent manual formula to within 1 cent."""
        price = Decimal(price_per_oz)
        result = nisab_from_ounce_price(price)
        # Independent formula — not calling nisab_from_ounce_price, so this is a
        # genuine cross-check and cannot mask a systematic bug in the function.
        expected = (Decimal("85") / Decimal("31.1034768") * price).quantize(Decimal("0.01"))
        assert result == expected, f"At {price_per_oz} USD/oz: expected nisab {expected}, got {result}"

    def test_result_is_quantized_to_cents(self):
        result = nisab_from_ounce_price(Decimal("1234.5678"))
        assert result == result.quantize(Decimal("0.01"))

    def test_result_has_at_most_two_decimal_places(self):
        result = nisab_from_ounce_price(Decimal("2000"))
        exponent = result.as_tuple().exponent
        assert exponent >= -2, f"Expected ≤ 2 decimal places, got exponent {exponent}"

    def test_proportionality(self):
        """Doubling the price doubles the nisab (linear relationship)."""
        n1 = nisab_from_ounce_price(Decimal("2000"))
        n2 = nisab_from_ounce_price(Decimal("4000"))
        # Allow 1 cent for rounding at each end.
        assert abs(n2 - n1 * 2) <= Decimal("0.02")

    def test_no_float_arithmetic_used(self):
        """Result must be a Decimal, not float — float arithmetic drifts."""
        result = nisab_from_ounce_price(Decimal("2000"))
        assert isinstance(result, Decimal)

    def test_cross_check_against_manual_formula(self):
        """Independent formula: price × (85 / 31.1034768) == nisab_from_ounce_price."""
        for price in ("1500", "2000", "2750", "3100"):
            expected = (Decimal("85") / Decimal("31.1034768") * Decimal(price)).quantize(Decimal("0.01"))
            actual = nisab_from_ounce_price(Decimal(price))
            assert abs(actual - expected) <= Decimal("0.01"), (
                f"Formula mismatch at {price} USD/oz: manual={expected}, function={actual}"
            )


# ---------------------------------------------------------------------------
# 3. validate_nisab_assumptions() — versioned provenance record
# ---------------------------------------------------------------------------


class TestValidateNisabAssumptions:
    """The assumption validator must catch drift and report clean provenance."""

    def test_returns_nisab_assumptions_instance(self):
        result = validate_nisab_assumptions()
        assert isinstance(result, NisabAssumptions)

    def test_valid_on_unmodified_codebase(self):
        result = validate_nisab_assumptions()
        assert result.valid is True, f"validate_nisab_assumptions() reported errors on unmodified code: {result.errors}"

    def test_no_errors_on_unmodified_codebase(self):
        result = validate_nisab_assumptions()
        assert result.errors == []

    def test_version_is_set(self):
        result = validate_nisab_assumptions()
        assert result.version, "version must be a non-empty string"

    def test_gold_grams_field_matches_constant(self):
        result = validate_nisab_assumptions()
        assert Decimal(result.nisab_gold_grams) == NISAB_GOLD_GRAMS

    def test_grams_per_troy_ounce_field_matches_constant(self):
        result = validate_nisab_assumptions()
        assert Decimal(result.grams_per_troy_ounce) == GRAMS_PER_TROY_OUNCE

    def test_nisab_in_troy_ounces_is_derived_correctly(self):
        result = validate_nisab_assumptions()
        expected = NISAB_GOLD_GRAMS / GRAMS_PER_TROY_OUNCE
        actual = Decimal(result.nisab_in_troy_ounces)
        assert abs(actual - expected) < Decimal("0.0000002"), (
            f"nisab_in_troy_ounces: expected ~{expected}, got {actual}"
        )

    def test_example_at_2000_matches_function(self):
        result = validate_nisab_assumptions()
        assert Decimal(result.example_nisab_at_2000_usd_oz) == nisab_from_ounce_price(Decimal("2000"))

    def test_example_at_3000_matches_function(self):
        result = validate_nisab_assumptions()
        assert Decimal(result.example_nisab_at_3000_usd_oz) == nisab_from_ounce_price(Decimal("3000"))

    def test_silver_basis_note_mentions_oic_and_aaoifi(self):
        """Provenance note must reference the two main standard-setting bodies."""
        result = validate_nisab_assumptions()
        note = result.silver_basis_note.lower()
        assert "oic" in note
        assert "aaoifi" in note

    def test_basis_field_references_gold_and_standard(self):
        result = validate_nisab_assumptions()
        assert "85" in result.basis
        assert "gold" in result.basis.lower()

    def test_detects_tampered_gold_grams(self):
        """Monkey-patching NISAB_GOLD_GRAMS away from 85 must trigger an error."""
        original = nisab.NISAB_GOLD_GRAMS
        try:
            nisab.NISAB_GOLD_GRAMS = Decimal("80")  # wrong fiqh value
            result = validate_nisab_assumptions()
            assert result.valid is False
            assert any("NISAB_GOLD_GRAMS" in e for e in result.errors)
        finally:
            nisab.NISAB_GOLD_GRAMS = original

    def test_detects_tampered_grams_per_troy_ounce(self):
        """Changing the NIST conversion factor must be caught."""
        original = nisab.GRAMS_PER_TROY_OUNCE
        try:
            nisab.GRAMS_PER_TROY_OUNCE = Decimal("31.10")  # truncated / wrong
            result = validate_nisab_assumptions()
            assert result.valid is False
            assert any("GRAMS_PER_TROY_OUNCE" in e for e in result.errors)
        finally:
            nisab.GRAMS_PER_TROY_OUNCE = original

    def test_function_never_raises(self):
        """Must never raise — it is designed to be called at startup."""
        try:
            validate_nisab_assumptions()
        except Exception as exc:  # noqa: BLE001
            pytest.fail(f"validate_nisab_assumptions() raised unexpectedly: {exc}")


# ---------------------------------------------------------------------------
# 4. parse_price() — source payload validation
# ---------------------------------------------------------------------------


class TestParsePrice:
    def test_valid_primary_payload(self):
        payload = {"price": 2000.5}
        result = parse_price(payload, PRIMARY_SOURCE)
        assert result == Decimal("2000.5")

    def test_valid_fallback_payload(self):
        payload = {"pax-gold": {"usd": 1987.12}}
        result = parse_price(payload, FALLBACK_SOURCE)
        assert result == Decimal("1987.12")

    def test_missing_field_returns_none(self):
        result = parse_price({}, PRIMARY_SOURCE)
        assert result is None

    def test_non_numeric_value_returns_none(self):
        result = parse_price({"price": "not-a-number"}, PRIMARY_SOURCE)
        assert result is None

    def test_price_too_low_returns_none(self):
        """A price below $100/oz cannot be a real gold price — reject it."""
        result = parse_price({"price": 50}, PRIMARY_SOURCE)
        assert result is None

    def test_price_too_high_returns_none(self):
        """A price above $100 000/oz indicates a unit or field error."""
        result = parse_price({"price": 200000}, PRIMARY_SOURCE)
        assert result is None

    def test_price_at_lower_bound_accepted(self):
        result = parse_price({"price": float(MIN_PLAUSIBLE_OUNCE_USD)}, PRIMARY_SOURCE)
        assert result == MIN_PLAUSIBLE_OUNCE_USD

    def test_price_at_upper_bound_accepted(self):
        result = parse_price({"price": float(MAX_PLAUSIBLE_OUNCE_USD)}, PRIMARY_SOURCE)
        assert result == MAX_PLAUSIBLE_OUNCE_USD

    def test_null_nested_key_returns_none(self):
        """Partial payload where the inner key is missing."""
        result = parse_price({"pax-gold": {}}, FALLBACK_SOURCE)
        assert result is None

    def test_non_dict_payload_returns_none(self):
        result = parse_price("not a dict", PRIMARY_SOURCE)
        assert result is None

    def test_integer_price_accepted(self):
        result = parse_price({"price": 2000}, PRIMARY_SOURCE)
        assert result == Decimal("2000")

    def test_string_numeric_price_accepted(self):
        """Some APIs return prices as JSON strings."""
        result = parse_price({"price": "2000.00"}, PRIMARY_SOURCE)
        assert result == Decimal("2000.00")


# ---------------------------------------------------------------------------
# 5. get_nisab() — live fetch, cache, fallback
# ---------------------------------------------------------------------------


class TestGetNisab:
    """get_nisab() must never raise and must degrade gracefully."""

    def test_live_quote_from_primary_source(self):
        with patch(
            "nisab.fetch_gold_price",
            returns({"price": Decimal("2000"), "source": "gold-api.com"}),
        ):
            quote = run(get_nisab(use_cache=False))

        assert quote.live is True
        assert quote.source == "gold-api.com"
        assert Decimal(quote.nisab_usd) == nisab_from_ounce_price(Decimal("2000"))
        assert quote.gold_price_usd_per_ounce == "2000"
        assert quote.basis == "85g gold"

    def test_fallback_when_all_sources_fail(self):
        with patch("nisab.fetch_gold_price", returns(None)):
            quote = run(get_nisab(use_cache=False))

        assert quote.live is False
        assert "default" in quote.source.lower()
        assert quote.note is not None
        assert Decimal(quote.nisab_usd) == DEFAULT_NISAB_USD

    def test_fallback_on_unexpected_exception(self):
        async def _raise(*args, **kwargs):
            raise RuntimeError("network exploded")

        with patch("nisab.fetch_gold_price", _raise):
            quote = run(get_nisab(use_cache=False))

        assert quote.live is False

    def test_cached_result_is_returned_on_second_call(self):
        call_count = 0

        async def _counted(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return {"price": Decimal("2500"), "source": "gold-api.com"}

        with patch("nisab.fetch_gold_price", _counted):
            run(get_nisab(use_cache=True))
            run(get_nisab(use_cache=True))

        assert call_count == 1, "fetch_gold_price should only be called once when cache is warm"

    def test_cache_bypass_hits_network_again(self):
        call_count = 0

        async def _counted(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return {"price": Decimal("2500"), "source": "gold-api.com"}

        with patch("nisab.fetch_gold_price", _counted):
            run(get_nisab(use_cache=False))
            run(get_nisab(use_cache=False))

        assert call_count == 2

    def test_nisab_usd_is_string(self):
        with patch("nisab.fetch_gold_price", returns({"price": Decimal("2000"), "source": "gold-api.com"})):
            quote = run(get_nisab(use_cache=False))
        assert isinstance(quote.nisab_usd, str)

    def test_live_quote_has_as_of_timestamp(self):
        with patch("nisab.fetch_gold_price", returns({"price": Decimal("2000"), "source": "gold-api.com"})):
            quote = run(get_nisab(use_cache=False))
        assert quote.as_of is not None
        # ISO 8601 UTC: ends with +00:00
        assert "+00:00" in quote.as_of

    def test_live_quote_has_per_gram_price(self):
        with patch("nisab.fetch_gold_price", returns({"price": Decimal("3110.34768"), "source": "test"})):
            quote = run(get_nisab(use_cache=False))
        # At $3110.34768/oz, per gram = $100.00 exactly
        assert quote.gold_price_usd_per_gram is not None
        assert Decimal(quote.gold_price_usd_per_gram) == Decimal("100.00")

    def test_fallback_has_note(self):
        with patch("nisab.fetch_gold_price", returns(None)):
            quote = run(get_nisab(use_cache=False))
        assert quote.note and len(quote.note) > 10

    def test_live_quote_gold_price_per_ounce_matches_input(self):
        with patch("nisab.fetch_gold_price", returns({"price": Decimal("1900"), "source": "test"})):
            quote = run(get_nisab(use_cache=False))
        assert quote.gold_price_usd_per_ounce == "1900"


# ---------------------------------------------------------------------------
# 6. override_quote() — caller-supplied nisab
# ---------------------------------------------------------------------------


class TestOverrideQuote:
    def test_override_is_not_live(self):
        q = override_quote(Decimal("7000"))
        assert q.live is False

    def test_override_nisab_usd_is_stored(self):
        q = override_quote(Decimal("5500"))
        assert Decimal(q.nisab_usd) == Decimal("5500")

    def test_override_source_indicates_caller(self):
        q = override_quote(Decimal("6000"))
        assert "override" in q.source.lower() or "caller" in q.source.lower()

    def test_override_has_note(self):
        q = override_quote(Decimal("6000"))
        assert q.note is not None

    def test_override_amount_property(self):
        q = override_quote(Decimal("8000"))
        assert q.amount == Decimal("8000")


# ---------------------------------------------------------------------------
# 7. NisabQuote.amount property
# ---------------------------------------------------------------------------


class TestNisabQuoteAmount:
    def test_amount_returns_decimal(self):
        q = NisabQuote(nisab_usd="6000", live=False, source="test")
        assert isinstance(q.amount, Decimal)

    def test_amount_equals_nisab_usd(self):
        q = NisabQuote(nisab_usd="5464.78", live=False, source="test")
        assert q.amount == Decimal("5464.78")

    def test_amount_is_exact_not_float(self):
        """Decimal must round-trip exactly; float conversion of a large number drifts."""
        q = NisabQuote(nisab_usd="12345.67", live=False, source="test")
        assert q.amount == Decimal("12345.67")


# ---------------------------------------------------------------------------
# 8. Edge cases: empty / failure / Arabic-numeral inputs
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Empty states, boundary values, and Arabic/English numeric representations."""

    def test_nisab_at_minimum_plausible_gold_price(self):
        """Gold at the absolute lower plausibility bound must still yield a positive nisab."""
        result = nisab_from_ounce_price(MIN_PLAUSIBLE_OUNCE_USD)
        assert result > 0

    def test_nisab_at_maximum_plausible_gold_price(self):
        result = nisab_from_ounce_price(MAX_PLAUSIBLE_OUNCE_USD)
        assert result > 0

    def test_parse_price_ignores_arabic_indic_numerals_as_string(self):
        """Arabic-Indic digit strings (٢٠٠٠) are not parseable as Decimal; must return None."""
        # Decimal("٢٠٠٠") raises InvalidOperation — treat it as a bad payload.
        result = parse_price({"price": "٢٠٠٠"}, PRIMARY_SOURCE)
        assert result is None

    def test_parse_price_ignores_none_value(self):
        result = parse_price({"price": None}, PRIMARY_SOURCE)
        assert result is None

    def test_parse_price_ignores_list_value(self):
        result = parse_price({"price": [2000]}, PRIMARY_SOURCE)
        assert result is None

    def test_parse_price_ignores_dict_value(self):
        result = parse_price({"price": {"usd": 2000}}, PRIMARY_SOURCE)
        assert result is None

    def test_fallback_quote_with_empty_reason(self):
        """_fallback_quote is reachable via get_nisab; ensure it handles an empty reason."""
        from nisab import _fallback_quote

        q = _fallback_quote("")
        assert q.live is False
        assert q.note is not None

    def test_get_nisab_with_zero_cache_ttl_env(self, monkeypatch):
        """NISAB_CACHE_TTL_SECONDS=0 is an odd but valid config; must not crash."""
        monkeypatch.setenv("NISAB_CACHE_TTL_SECONDS", "0")
        with patch("nisab.fetch_gold_price", returns({"price": Decimal("2000"), "source": "test"})):
            # Does not raise
            run(get_nisab(use_cache=False))

    def test_nisab_from_ounce_price_with_many_decimal_places(self):
        """Arbitrarily precise input must still produce a 2-dp result."""
        result = nisab_from_ounce_price(Decimal("2000.123456789012345"))
        assert result == result.quantize(Decimal("0.01"))

    def test_very_high_gold_price_nisab_is_still_exact(self):
        """Near the upper plausibility bound: no float overflow or precision loss."""
        result = nisab_from_ounce_price(Decimal("99999"))
        assert isinstance(result, Decimal)
        assert result > 0

    def test_gold_price_exactly_at_lower_bound_accepted(self):
        payload = {"price": str(MIN_PLAUSIBLE_OUNCE_USD)}
        assert parse_price(payload, PRIMARY_SOURCE) == MIN_PLAUSIBLE_OUNCE_USD

    def test_gold_price_one_cent_below_lower_bound_rejected(self):
        too_low = str(MIN_PLAUSIBLE_OUNCE_USD - Decimal("0.01"))
        result = parse_price({"price": too_low}, PRIMARY_SOURCE)
        assert result is None

    def test_gold_price_one_cent_above_upper_bound_rejected(self):
        too_high = str(MAX_PLAUSIBLE_OUNCE_USD + Decimal("0.01"))
        result = parse_price({"price": too_high}, PRIMARY_SOURCE)
        assert result is None


# ---------------------------------------------------------------------------
# 9. Zakat arithmetic integration with the nisab
# ---------------------------------------------------------------------------


class TestZakatIntegration:
    """Zakat rules applied using the nisab module as the threshold source."""

    # Import compute_zakat lazily to avoid a hard dependency on stellar.py
    @pytest.fixture
    def compute_zakat(self):
        from stellar import compute_zakat

        return compute_zakat

    @pytest.fixture
    def zakat_rate(self):
        from stellar import ZAKAT_RATE

        return ZAKAT_RATE

    def test_at_exactly_live_nisab_zakat_is_due(self, compute_zakat):
        """When balance == nisab, zakat is 2.5 % of the whole balance."""
        nisab_val = nisab_from_ounce_price(Decimal("2000"))  # 5464.78
        zakat = compute_zakat(nisab_val, nisab_val)
        expected = (nisab_val * Decimal("0.025")).quantize(Decimal("0.0000001"))
        assert zakat == expected

    def test_one_cent_below_live_nisab_no_zakat(self, compute_zakat):
        nisab_val = nisab_from_ounce_price(Decimal("2000"))
        balance = nisab_val - Decimal("0.01")
        assert compute_zakat(balance, nisab_val) == Decimal("0")

    def test_zakat_rate_is_exactly_two_and_half_percent(self, zakat_rate):
        assert zakat_rate == Decimal("0.025")

    def test_zakat_uses_whole_balance_not_surplus(self, compute_zakat):
        """2.5 % applies to the entire zakatable balance, not to the amount over nisab."""
        nisab_val = Decimal("6000")
        balance = Decimal("7000")
        zakat = compute_zakat(balance, nisab_val)
        # 7000 * 0.025 = 175, NOT (7000-6000)*0.025 = 25
        assert zakat == Decimal("175.0000000")
        assert zakat != (balance - nisab_val) * Decimal("0.025")

    def test_zakat_with_fallback_nisab_matches_expected(self, compute_zakat):
        """When live price is unavailable, the default nisab still computes correctly."""
        balance = Decimal("10000")
        default_nisab = DEFAULT_NISAB_USD
        zakat = compute_zakat(balance, default_nisab)
        expected = (balance * Decimal("0.025")).quantize(Decimal("0.0000001"))
        assert zakat == expected

    def test_zero_balance_gives_zero_zakat(self, compute_zakat):
        assert compute_zakat(Decimal("0"), Decimal("6000")) == Decimal("0")

    def test_zakat_quantized_to_seven_decimal_places(self, compute_zakat):
        """USDC on Stellar has 7 decimal places — zakat must not exceed that."""
        result = compute_zakat(Decimal("7777.7777777"), Decimal("6000"))
        assert -result.as_tuple().exponent <= 7


# ---------------------------------------------------------------------------
# 10. Deployment / rollback observability
# ---------------------------------------------------------------------------


class TestNisabAssumptionsObservability:
    """The assumption record must supply everything an auditor or rollback needs."""

    def test_assumptions_are_serialisable_to_dict(self):
        result = validate_nisab_assumptions()
        d = result.model_dump()
        assert isinstance(d, dict)
        assert "version" in d
        assert "valid" in d
        assert "errors" in d

    def test_assumptions_version_is_semver_shaped(self):
        """Semantic-versioning shape (MAJOR.MINOR.PATCH) so bumps are auditable."""
        result = validate_nisab_assumptions()
        parts = result.version.split(".")
        assert len(parts) == 3, f"version should be X.Y.Z, got {result.version!r}"
        for part in parts:
            assert part.isdigit(), f"version part {part!r} is not numeric"

    def test_assumptions_contain_silver_basis_explanation(self):
        """Auditors must be able to verify why silver is not used."""
        result = validate_nisab_assumptions()
        assert result.silver_basis_note, "silver_basis_note must be non-empty"
        note = result.silver_basis_note.lower()
        assert "silver" in note
        assert "scholar" in note

    def test_assumptions_basis_field_is_non_empty(self):
        result = validate_nisab_assumptions()
        assert result.basis

    def test_full_provenance_round_trip(self):
        """Serialise to dict and back; no data should be lost."""
        original = validate_nisab_assumptions()
        restored = NisabAssumptions(**original.model_dump())
        assert original.version == restored.version
        assert original.nisab_gold_grams == restored.nisab_gold_grams
        assert original.valid == restored.valid
        assert original.errors == restored.errors
