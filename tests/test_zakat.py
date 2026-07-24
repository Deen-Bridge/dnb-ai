"""Tests for zakat calculation, the live nisab, and the chat integration.

Everything runs offline: Horizon is mocked, the gold-price API is stubbed, and
no Stellar network call is made. No API keys needed.
"""

import asyncio
from decimal import Decimal
from unittest.mock import MagicMock, patch

import httpx
import pytest
from fastapi import HTTPException

import nisab
import stellar
from nisab import (
    FALLBACK_SOURCE,
    GRAMS_PER_TROY_OUNCE,
    NISAB_GOLD_GRAMS,
    PRIMARY_SOURCE,
    NisabQuote,
    get_nisab,
    nisab_from_ounce_price,
    override_quote,
    parse_price,
)
from semantic_cache import get_keyed_cache
from stellar import (
    ZAKAT_RATE,
    build_chat_zakat_context,
    build_zakat_response,
    compute_zakat,
    contains_secret_key,
    extract_public_key,
    fetch_usdc_balance,
    is_zakat_question,
    redact_secret_keys,
    validate_public_key,
)

# A structurally valid testnet public key (checksum included) and its
# secret-key counterpart's shape. Neither controls anything.
VALID_KEY = "GBBD47IF6LWK7P7MDEVSCWR7DPUWV3NY3DTQEVFL4NAT4AQH3ZLLFLA5"
OTHER_KEY = "GA5ZSEJYB37JRC5AVCIA5MOP4RHTM335X2KGX3IHOJAPP5RE34K4KZVN"

FIXED_NISAB = NisabQuote(
    nisab_usd="6000",
    live=False,
    source="test fixture",
)


def run(coro):
    return asyncio.run(coro)


def returns(value):
    """An async stub that yields *value* every time it is awaited.

    A patch target that is awaited more than once needs a fresh coroutine per
    call, so this patches in a function rather than a single coroutine object.
    """

    async def _stub(*args, **kwargs):
        return value

    return _stub


@pytest.fixture(autouse=True)
def clear_nisab_cache():
    get_keyed_cache(nisab.CACHE_NAMESPACE).clear()
    yield
    get_keyed_cache(nisab.CACHE_NAMESPACE).clear()


def horizon_account(balances):
    """A Horizon account payload with the given balance entries."""
    return {"balances": balances}


def usdc_balance_entry(amount, issuer=None):
    return {
        "asset_code": "USDC",
        "asset_issuer": issuer or stellar.usdc_issuer(),
        "balance": amount,
    }


# ---------------------------------------------------------------------------
# Zakat arithmetic
# ---------------------------------------------------------------------------


class TestComputeZakat:
    def test_above_nisab_is_two_and_a_half_percent(self):
        assert compute_zakat(Decimal("10000"), Decimal("6000")) == Decimal("250.0000000")

    def test_below_nisab_is_zero(self):
        assert compute_zakat(Decimal("5999.99"), Decimal("6000")) == Decimal("0")

    def test_exactly_at_nisab_is_due(self):
        """The threshold is inclusive — wealth *reaching* the nisab is liable."""
        assert compute_zakat(Decimal("6000"), Decimal("6000")) == Decimal("150.0000000")

    def test_zakat_is_on_the_whole_amount_not_the_excess(self):
        """A common misreading: 2.5% applies to everything, not to the surplus."""
        due = compute_zakat(Decimal("8000"), Decimal("6000"))
        assert due == Decimal("200.0000000")
        assert due != (Decimal("8000") - Decimal("6000")) * ZAKAT_RATE

    def test_zero_balance(self):
        assert compute_zakat(Decimal("0"), Decimal("6000")) == Decimal("0")

    @pytest.mark.parametrize("balance,expected", [
        # Quantized to USDC's 7 decimal places, never more.
        ("6000.1234567", "150.0030864"),
        ("9999.9999999", "250.0000000"),
        ("12345.6789012", "308.6419725"),
    ])
    def test_quantized_to_usdc_precision(self, balance, expected):
        due = compute_zakat(Decimal(balance), Decimal("6000"))
        assert due == Decimal(expected)
        assert -due.as_tuple().exponent <= 7

    def test_very_large_balance_stays_exact(self):
        """Decimal, not float — no binary rounding drift on big balances."""
        assert compute_zakat(Decimal("1000000.10"), Decimal("6000")) == Decimal("25000.0025000")


# ---------------------------------------------------------------------------
# Key validation
# ---------------------------------------------------------------------------


class TestKeyValidation:
    def test_valid_key_passes(self):
        assert validate_public_key(VALID_KEY) == VALID_KEY

    def test_whitespace_is_trimmed(self):
        assert validate_public_key(f"  {VALID_KEY}\n") == VALID_KEY

    @pytest.mark.parametrize("key", [
        "",
        "   ",
        "not-a-key",
        "GINVALID",
        VALID_KEY[:-1],           # too short
        VALID_KEY + "A",          # too long
        VALID_KEY[:-1] + "X",     # bad checksum
        "MBBD47IF6LWK7P7MDEVSCWR7DPUWV3NY3DTQEVFL4NAT4AQH3ZLLFLA5",  # wrong prefix
    ])
    def test_invalid_key_is_400(self, key):
        with pytest.raises(HTTPException) as exc:
            validate_public_key(key)
        assert exc.value.status_code == 400

    def test_secret_key_is_rejected_like_any_other_bad_input(self):
        """Read-only by construction: an S... key can never reach Horizon."""
        secret = "SBBD47IF6LWK7P7MDEVSCWR7DPUWV3NY3DTQEVFL4NAT4AQH3ZLLFLA5"
        with pytest.raises(HTTPException) as exc:
            validate_public_key(secret)
        assert exc.value.status_code == 400


# ---------------------------------------------------------------------------
# Horizon lookup (mocked)
# ---------------------------------------------------------------------------


class TestFetchBalance:
    def _server_returning(self, payload):
        server = MagicMock()
        server.accounts.return_value.account_id.return_value.call.return_value = payload
        return server

    def test_trustline_returns_balance(self):
        server = self._server_returning(
            horizon_account([usdc_balance_entry("1234.5670000")])
        )
        with patch.object(stellar, "Server", return_value=server):
            assert fetch_usdc_balance(VALID_KEY) == Decimal("1234.5670000")

    def test_no_trustline_returns_none(self):
        server = self._server_returning(
            horizon_account([{"asset_type": "native", "balance": "100.0"}])
        )
        with patch.object(stellar, "Server", return_value=server):
            assert fetch_usdc_balance(VALID_KEY) is None

    def test_usdc_from_another_issuer_is_ignored(self):
        """Only the platform's USDC counts — a lookalike asset is not USDC."""
        server = self._server_returning(
            horizon_account([usdc_balance_entry("500.0", issuer=OTHER_KEY)])
        )
        with patch.object(stellar, "Server", return_value=server):
            assert fetch_usdc_balance(VALID_KEY) is None

    def test_unknown_account_is_404(self):
        from stellar_sdk.exceptions import NotFoundError

        server = MagicMock()
        server.accounts.return_value.account_id.return_value.call.side_effect = (
            NotFoundError(MagicMock())
        )
        with patch.object(stellar, "Server", return_value=server):
            with pytest.raises(HTTPException) as exc:
                fetch_usdc_balance(VALID_KEY)
        assert exc.value.status_code == 404

    def test_lookup_never_sees_a_secret_key(self):
        """The endpoint validates before calling Horizon."""
        server = self._server_returning(horizon_account([]))
        with patch.object(stellar, "Server", return_value=server):
            with pytest.raises(HTTPException):
                validate_public_key("SBBD47IF6LWK7P7MDEVSCWR7DPUWV3NY3DTQEVFL4NAT4AQH3ZLLFLA5")
        server.accounts.assert_not_called()


# ---------------------------------------------------------------------------
# Response assembly (recorded shape)
# ---------------------------------------------------------------------------


class TestBuildResponse:
    def test_above_nisab_response_shape(self):
        response = build_zakat_response(VALID_KEY, Decimal("10000"), FIXED_NISAB)
        assert response.model_dump().keys() >= {
            "network", "public_key", "has_usdc_trustline", "usdc_balance",
            "nisab_usd", "zakat_rate", "zakat_due", "message", "disclaimer",
            "nisab",
        }
        assert response.has_usdc_trustline is True
        assert response.usdc_balance == "10000"
        assert response.zakat_due == "250.0000000"
        assert response.zakat_rate == str(ZAKAT_RATE)
        assert "meets" in response.message
        assert response.disclaimer == stellar.DISCLAIMER

    def test_below_nisab_response(self):
        response = build_zakat_response(VALID_KEY, Decimal("100"), FIXED_NISAB)
        assert response.zakat_due == "0"
        assert "below" in response.message
        assert response.has_usdc_trustline is True

    def test_no_trustline_response(self):
        response = build_zakat_response(VALID_KEY, None, FIXED_NISAB)
        assert response.has_usdc_trustline is False
        assert response.usdc_balance == "0"
        assert response.zakat_due == "0"
        assert "trustline" in response.message

    def test_every_response_carries_the_scholar_disclaimer(self):
        for balance in (None, Decimal("0"), Decimal("100"), Decimal("10000")):
            response = build_zakat_response(VALID_KEY, balance, FIXED_NISAB)
            assert "qualified scholar" in response.disclaimer

    def test_response_reports_where_the_nisab_came_from(self):
        live = NisabQuote(
            nisab_usd="11111.11",
            live=True,
            source="gold-api.com",
            gold_price_usd_per_ounce="4066.20",
        )
        response = build_zakat_response(VALID_KEY, Decimal("20000"), live)
        assert response.nisab.live is True
        assert "gold-api.com" in response.message
        assert response.nisab_usd == "11111.11"

    def test_never_echoes_anything_but_the_public_key(self):
        response = build_zakat_response(VALID_KEY, Decimal("10000"), FIXED_NISAB)
        assert response.public_key == VALID_KEY
        assert not response.public_key.startswith("S")


# ---------------------------------------------------------------------------
# Live nisab
# ---------------------------------------------------------------------------


class TestNisabMath:
    def test_85g_gold_conversion(self):
        # 85g / 31.1034768 g-per-oz × price
        expected = (Decimal("4000") / GRAMS_PER_TROY_OUNCE * NISAB_GOLD_GRAMS).quantize(
            Decimal("0.01")
        )
        assert nisab_from_ounce_price(Decimal("4000")) == expected

    def test_conversion_is_monotonic(self):
        assert nisab_from_ounce_price(Decimal("5000")) > nisab_from_ounce_price(Decimal("4000"))

    def test_known_value(self):
        """A $2,000/oz gold price puts the 85g nisab at $5,465.63."""
        assert nisab_from_ounce_price(Decimal("2000")) == Decimal("5465.63")


class TestPriceParsing:
    def test_primary_source_shape(self):
        payload = {"price": 4066.199951, "currency": "USD", "symbol": "XAU"}
        assert parse_price(payload, PRIMARY_SOURCE) == Decimal("4066.199951")

    def test_fallback_source_shape(self):
        payload = {"pax-gold": {"usd": 4059.87}}
        assert parse_price(payload, FALLBACK_SOURCE) == Decimal("4059.87")

    @pytest.mark.parametrize("payload", [
        {},
        {"error": "rate limited"},
        {"price": None},
        {"price": "not-a-number"},
    ])
    def test_unusable_payload_returns_none(self, payload):
        assert parse_price(payload, PRIMARY_SOURCE) is None

    @pytest.mark.parametrize("price", [0, 1, 99, 100001, 1e9])
    def test_implausible_price_is_refused(self, price):
        """A unit change upstream must degrade, not produce a wrong nisab."""
        assert parse_price({"price": price}, PRIMARY_SOURCE) is None


class TestGetNisab:
    def _client(self, handler):
        return httpx.AsyncClient(transport=httpx.MockTransport(handler))

    def test_live_price_produces_a_live_quote(self):
        def handler(request):
            return httpx.Response(200, json={"price": 4000.0})

        quote = run(get_nisab(client=self._client(handler), use_cache=False))
        assert quote.live is True
        assert quote.source == PRIMARY_SOURCE.name
        assert quote.gold_price_usd_per_ounce == "4000.0"
        assert quote.amount == nisab_from_ounce_price(Decimal("4000.0"))
        assert quote.as_of

    def test_falls_back_to_second_source(self):
        def handler(request):
            if "gold-api" in str(request.url):
                return httpx.Response(503)
            return httpx.Response(200, json={"pax-gold": {"usd": 4100.0}})

        quote = run(get_nisab(client=self._client(handler), use_cache=False))
        assert quote.live is True
        assert quote.source == FALLBACK_SOURCE.name

    def test_all_sources_down_falls_back_to_configured_default(self):
        def handler(request):
            return httpx.Response(500)

        quote = run(get_nisab(client=self._client(handler), use_cache=False))
        assert quote.live is False
        assert quote.amount == nisab.DEFAULT_NISAB_USD
        assert "default" in quote.source
        assert quote.note

    def test_network_error_falls_back(self):
        def handler(request):
            raise httpx.ConnectError("no route to host")

        quote = run(get_nisab(client=self._client(handler), use_cache=False))
        assert quote.live is False

    def test_implausible_price_falls_back_rather_than_using_it(self):
        def handler(request):
            return httpx.Response(200, json={"price": 4.06, "pax-gold": {"usd": 4.06}})

        quote = run(get_nisab(client=self._client(handler), use_cache=False))
        assert quote.live is False

    def test_result_is_cached(self):
        calls = {"n": 0}

        def handler(request):
            calls["n"] += 1
            return httpx.Response(200, json={"price": 4000.0})

        first = run(get_nisab(client=self._client(handler)))
        second = run(get_nisab(client=self._client(handler)))
        assert calls["n"] == 1
        assert first.nisab_usd == second.nisab_usd

    def test_fallback_is_not_cached(self):
        """A price outage must not pin the default in place for hours."""
        def down(request):
            return httpx.Response(500)

        def up(request):
            return httpx.Response(200, json={"price": 4000.0})

        assert run(get_nisab(client=self._client(down))).live is False
        assert run(get_nisab(client=self._client(up))).live is True

    def test_override_is_reported_as_an_override(self):
        quote = override_quote(Decimal("7500"))
        assert quote.live is False
        assert quote.amount == Decimal("7500")
        assert "override" in quote.source


# ---------------------------------------------------------------------------
# Chat integration
# ---------------------------------------------------------------------------


class TestNoLiveNetworkInCI:
    """Detection must be free — an ordinary prompt reaches no network at all."""

    def test_non_zakat_prompt_touches_neither_horizon_nor_the_price_api(self):
        with patch.object(stellar, "Server") as server, \
             patch.object(stellar, "get_nisab") as get_nisab_mock:
            assert run(build_chat_zakat_context("How do I make wudu?")) is None
        server.assert_not_called()
        get_nisab_mock.assert_not_called()

    def test_zakat_prompt_without_a_key_touches_neither(self):
        with patch.object(stellar, "Server") as server, \
             patch.object(stellar, "get_nisab") as get_nisab_mock:
            context = run(build_chat_zakat_context("How much zakat do I owe?"))
        assert context.info.calculated is False
        server.assert_not_called()
        get_nisab_mock.assert_not_called()


class TestChatDetection:
    @pytest.mark.parametrize("prompt,expected", [
        ("How much zakat do I owe?", True),
        ("What is the nisab this year?", True),
        ("Explain zakah on savings", True),
        ("What time is Maghrib?", False),
        ("How do I pray?", False),
        ("", False),
    ])
    def test_zakat_intent(self, prompt, expected):
        assert is_zakat_question(prompt) is expected

    def test_extracts_a_public_key_from_a_sentence(self):
        prompt = f"How much zakat do I owe on my wallet {VALID_KEY}?"
        assert extract_public_key(prompt) == VALID_KEY

    def test_extracts_from_the_context_field(self):
        assert extract_public_key("How much zakat do I owe?", VALID_KEY) == VALID_KEY

    @pytest.mark.parametrize("text", [
        "How much zakat do I owe?",
        "My wallet is GINVALID",
        f"My wallet is {VALID_KEY[:-1]}X",  # right shape, bad checksum
    ])
    def test_no_valid_key_found(self, text):
        assert extract_public_key(text) is None

    def test_detects_a_secret_key_shape(self):
        secret = "SBBD47IF6LWK7P7MDEVSCWR7DPUWV3NY3DTQEVFL4NAT4AQH3ZLLFLA5"
        assert contains_secret_key(f"my key is {secret}") is True
        assert contains_secret_key(f"my key is {VALID_KEY}") is False


class TestSecretKeyRedaction:
    """A volunteered secret key must not reach the model provider or storage."""

    SECRET = "SBBD47IF6LWK7P7MDEVSCWR7DPUWV3NY3DTQEVFL4NAT4AQH3ZLLFLA5"

    def test_secret_key_is_replaced(self):
        redacted = redact_secret_keys(f"my key is {self.SECRET} ok")
        assert self.SECRET not in redacted
        assert "[REDACTED STELLAR SECRET KEY]" in redacted

    def test_multiple_keys_are_all_replaced(self):
        other = "SA5ZSEJYB37JRC5AVCIA5MOP4RHTM335X2KGX3IHOJAPP5RE34K4KZVN"
        redacted = redact_secret_keys(f"{self.SECRET} and {other}")
        assert self.SECRET not in redacted and other not in redacted

    def test_public_keys_are_left_alone(self):
        """Redaction must not break the feature it protects."""
        text = f"zakat for {VALID_KEY}"
        assert redact_secret_keys(text) == text

    @pytest.mark.parametrize("text", [None, "", "no keys here"])
    def test_harmless_text_is_unchanged(self, text):
        assert redact_secret_keys(text) == text


class TestChatZakatContext:
    def test_non_zakat_prompt_returns_none(self):
        assert run(build_chat_zakat_context("How do I make wudu?")) is None

    def test_zakat_question_without_a_key_explains_instead(self):
        context = run(build_chat_zakat_context("How much zakat do I owe?"))
        assert context is not None
        assert context.info.calculated is False
        assert "public key" in context.prompt_block
        assert "secret" in context.prompt_block.lower()

    def test_secret_key_is_refused_and_warned_about(self):
        secret = "SBBD47IF6LWK7P7MDEVSCWR7DPUWV3NY3DTQEVFL4NAT4AQH3ZLLFLA5"
        context = run(build_chat_zakat_context(f"zakat for {secret}"))
        assert context.info.secret_key_detected is True
        assert context.info.calculated is False
        # The key itself is never echoed into the prompt block.
        assert secret not in context.prompt_block
        assert "never be shared" in context.prompt_block

    def test_computes_from_the_real_balance(self):
        quote = NisabQuote(nisab_usd="6000", live=True, source="gold-api.com")
        with patch.object(stellar, "fetch_usdc_balance", return_value=Decimal("10000")), \
             patch.object(stellar, "get_nisab", returns(quote)):
            context = run(build_chat_zakat_context(
                f"How much zakat do I owe on my wallet {VALID_KEY}?"
            ))
        assert context.info.calculated is True
        assert context.info.public_key == VALID_KEY
        assert context.info.usdc_balance == "10000"
        assert context.info.zakat_due == "250.0000000"
        assert context.info.nisab_source == "gold-api.com"
        # The model is given the real figures and told to keep the disclaimer.
        assert "10000" in context.prompt_block
        assert "250.0000000" in context.prompt_block
        assert stellar.DISCLAIMER in context.prompt_block

    def test_unknown_account_degrades_to_an_explanation(self):
        def missing(_key):
            raise HTTPException(status_code=404, detail="Account not found.")

        with patch.object(stellar, "fetch_usdc_balance", side_effect=missing), \
             patch.object(stellar, "get_nisab", returns(FIXED_NISAB)):
            context = run(build_chat_zakat_context(f"zakat for {VALID_KEY}"))
        assert context.info.calculated is False
        assert context.info.public_key == VALID_KEY
        assert "could not be read" in context.prompt_block

    def test_prompt_block_names_the_nisab_basis(self):
        live = NisabQuote(
            nisab_usd="11111.11",
            live=True,
            source="gold-api.com",
            gold_price_usd_per_ounce="4066.20",
        )
        with patch.object(stellar, "fetch_usdc_balance", return_value=Decimal("20000")), \
             patch.object(stellar, "get_nisab", returns(live)):
            context = run(build_chat_zakat_context(f"zakat for {VALID_KEY}"))
        assert "85g gold" in context.prompt_block
        assert "live gold price" in context.prompt_block
        assert "4066.20" in context.prompt_block
