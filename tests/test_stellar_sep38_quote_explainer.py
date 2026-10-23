"""Tests for SEP-38 anchor RFQ quote response explainer (#643)."""

from __future__ import annotations

from datetime import UTC, datetime

from stellar_sep38_quote_explainer import (
    OFFICIAL_SEP38_DOC_URL,
    explain_sep38_protocol,
    explain_sep38_quote,
    parse_sep38_asset,
)


def test_parse_sep38_asset_variants() -> None:
    fiat = parse_sep38_asset("iso4217:USD")
    assert fiat.asset_type == "iso4217"
    assert fiat.code == "USD"
    assert fiat.issuer is None
    assert fiat.is_onchain_stellar is False
    assert fiat.explorer_url is None

    native = parse_sep38_asset("stellar:native", network="public")
    assert native.asset_type == "stellar_native"
    assert native.code == "XLM"
    assert native.is_onchain_stellar is True
    assert native.explorer_url == "https://stellar.expert/explorer/public/asset/XLM"

    native_xlm = parse_sep38_asset("stellar:XLM", network="testnet")
    assert native_xlm.asset_type == "stellar_native"
    assert native_xlm.explorer_url == "https://stellar.expert/explorer/testnet/asset/XLM"

    issued = parse_sep38_asset(
        "stellar:USDC:GA5ZSEJYB37JRC5AVCIA5MOP4RHTM335X2KGX3IHOJAPP5RE34K4KZVN", network="public"
    )
    assert issued.asset_type == "stellar_issued"
    assert issued.code == "USDC"
    assert issued.issuer == "GA5ZSEJYB37JRC5AVCIA5MOP4RHTM335X2KGX3IHOJAPP5RE34K4KZVN"
    assert issued.is_onchain_stellar is True
    assert "USDC-GA5ZSEJYB37JRC5AVCIA5MOP4RHTM335X2KGX3IHOJAPP5RE34K4KZVN" in (issued.explorer_url or "")

    contract_id = "C" + "A" * 55
    contract_asset = parse_sep38_asset(f"stellar:{contract_id}", network="testnet")
    assert contract_asset.asset_type == "stellar_contract"
    assert contract_asset.code == contract_id
    assert contract_asset.is_onchain_stellar is True
    assert f"/testnet/contract/{contract_id}" in (contract_asset.explorer_url or "")

    unknown = parse_sep38_asset("custom:asset", network="public")
    assert unknown.asset_type == "unknown"
    assert unknown.is_onchain_stellar is False


def test_rejects_secret_key_in_quote_dict() -> None:
    secret_seed = "S" + "B" * 55
    quote_payload = {
        "id": "quote-12345",
        "sell_asset": "iso4217:USD",
        "buy_asset": "stellar:native",
        "secret": secret_seed,
    }
    res = explain_sep38_quote(quote_payload)
    assert res.status == "rejected_secret_input"
    assert res.flow_type == "unknown"
    assert res.quote_id is None
    assert secret_seed not in (res.rejection_reason or "")
    assert secret_seed not in res.safety_policy


def test_rejects_secret_key_in_json_string() -> None:
    secret_seed = "S" + "C" * 55
    raw_json = f'{{"id": "test", "anchor_auth": "{secret_seed}", "sell_asset": "iso4217:USD"}}'
    res = explain_sep38_quote(raw_json)
    assert res.status == "rejected_secret_input"
    assert secret_seed not in (res.rejection_reason or "")


def test_rejects_seed_phrase() -> None:
    seed_phrase = "abandon ability able about above absent absorb abstract absurd abuse access accident"
    res = explain_sep38_protocol(f"Can I use this seed phrase {seed_phrase}?")
    assert res.status == "rejected_secret_input"
    assert seed_phrase not in res.summary


def test_on_ramp_firm_quote_explanation() -> None:
    now = datetime(2026, 10, 22, 10, 0, 0, tzinfo=UTC)
    quote_data = {
        "id": "123e4567-e89b-12d3-a456-426614174000",
        "expires_at": "2026-10-22T10:15:00Z",
        "total_price": "1.03",
        "price": "1.00",
        "sell_asset": "iso4217:USD",
        "sell_amount": "100.00",
        "buy_asset": "stellar:USDC:GA5ZSEJYB37JRC5AVCIA5MOP4RHTM335X2KGX3IHOJAPP5RE34K4KZVN",
        "buy_amount": "97.087",
        "fee": {
            "total": "3.00",
            "asset": "iso4217:USD",
            "details": [
                {"name": "Service fee", "amount": "2.00", "description": "Anchor processing fee"},
                {"name": "Payment rail fee", "amount": "1.00", "description": "ACH transfer fee"},
            ],
        },
    }

    res = explain_sep38_quote(
        quote_data,
        network="public",
        anchor_domain="testanchor.stellar.org",
        now_utc=now,
    )

    assert res.status == "ok"
    assert res.flow_type == "on_ramp"
    assert res.is_firm_quote is True
    assert res.is_expired is False
    assert res.seconds_until_expiration == 900
    assert res.sell_asset is not None and res.sell_asset.code == "USD"
    assert res.buy_asset is not None and res.buy_asset.code == "USDC"
    assert res.fee is not None
    assert res.fee.total == "3.00"
    assert res.fee.percentage_of_sell_amount == "3.00%"
    assert len(res.fee.details) == 2
    assert res.fee.details[0].name == "Service fee"
    assert res.rate_summary is not None
    assert "Effective rate: 1 USD ≈ 0.97087 USDC" in res.rate_summary
    assert "Total price (including fees): 1.03" in res.rate_summary
    assert len(res.execution_steps) == 4
    assert "trustline" in res.execution_steps[0]
    assert "testanchor.stellar.org" in res.execution_steps[1]
    assert res.official_sep_url == OFFICIAL_SEP38_DOC_URL


def test_off_ramp_firm_quote_explanation() -> None:
    now = datetime(2026, 10, 22, 12, 0, 0, tzinfo=UTC)
    quote_data = {
        "id": "quote-offramp-999",
        "expires_at": "2026-10-22T12:05:00Z",
        "total_price": "0.95",
        "price": "0.98",
        "sell_asset": "stellar:USDC:GA5ZSEJYB37JRC5AVCIA5MOP4RHTM335X2KGX3IHOJAPP5RE34K4KZVN",
        "sell_amount": "500.00",
        "buy_asset": "iso4217:EUR",
        "buy_amount": "475.00",
        "fee": {
            "total": "10.00",
            "asset": "stellar:USDC:GA5ZSEJYB37JRC5AVCIA5MOP4RHTM335X2KGX3IHOJAPP5RE34K4KZVN",
            "details": [{"name": "SEPA payout fee", "amount": "10.00"}],
        },
    }

    res = explain_sep38_quote(
        quote_data,
        network="public",
        anchor_domain="anchoreur.com",
        now_utc=now,
    )

    assert res.status == "ok"
    assert res.flow_type == "off_ramp"
    assert res.is_firm_quote is True
    assert res.is_expired is False
    assert res.seconds_until_expiration == 300
    assert res.fee is not None
    assert res.fee.percentage_of_sell_amount == "2.00%"
    assert any("Sign On-Chain Payment" in step for step in res.execution_steps)


def test_stellar_cross_asset_quote() -> None:
    quote_data = {
        "id": "swap-123",
        "expires_at": "2026-10-22T14:00:00Z",
        "sell_asset": "stellar:native",
        "sell_amount": "1000.00",
        "buy_asset": "stellar:USDC:GA5ZSEJYB37JRC5AVCIA5MOP4RHTM335X2KGX3IHOJAPP5RE34K4KZVN",
        "buy_amount": "115.00",
    }
    res = explain_sep38_quote(quote_data, network="public")
    assert res.status == "ok"
    assert res.flow_type == "stellar_cross_asset"
    assert res.is_firm_quote is True


def test_off_chain_exchange_quote() -> None:
    quote_data = {
        "id": "fx-456",
        "expires_at": "2026-10-22T15:00:00Z",
        "sell_asset": "iso4217:EUR",
        "sell_amount": "100.00",
        "buy_asset": "iso4217:USD",
        "buy_amount": "108.00",
    }
    res = explain_sep38_quote(quote_data)
    assert res.status == "ok"
    assert res.flow_type == "off_chain_exchange"


def test_expired_quote_guidance() -> None:
    now = datetime(2026, 10, 22, 16, 0, 0, tzinfo=UTC)
    quote_data = {
        "id": "expired-quote",
        "expires_at": "2026-10-22T15:50:00Z",
        "sell_asset": "iso4217:USD",
        "sell_amount": "100.00",
        "buy_asset": "stellar:native",
        "buy_amount": "800.00",
    }
    res = explain_sep38_quote(quote_data, now_utc=now)
    assert res.is_expired is True
    assert res.seconds_until_expiration == 0
    assert any("Quote Expired" in step for step in res.execution_steps)
    assert any("Do Not Send Funds" in step for step in res.execution_steps)


def test_indicative_quote_without_firm_lock() -> None:
    # Indicative price endpoint response (/price) lacks quote ID
    price_data = {
        "price": "0.10",
        "total_price": "0.103",
        "sell_asset": "iso4217:USD",
        "buy_asset": "stellar:native",
    }
    res = explain_sep38_quote(price_data)
    assert res.status == "ok"
    assert res.is_firm_quote is False
    assert res.quote_id is None
    assert res.expires_at is None
    assert res.is_expired is None


def test_explain_sep38_protocol_overview() -> None:
    guide = explain_sep38_protocol("How do I request a SEP-38 quote?")
    assert guide.status == "ok"
    assert len(guide.supported_endpoints) == 5
    assert any("/quote" in ep for ep in guide.supported_endpoints)
    assert len(guide.quote_lifecycle_steps) == 5
    assert "non-custodial" in guide.safety_policy.lower()


def test_to_dict_serialization() -> None:
    quote_data = {
        "id": "q-1",
        "sell_asset": "iso4217:USD",
        "sell_amount": "10",
        "buy_asset": "stellar:native",
        "buy_amount": "80",
    }
    res = explain_sep38_quote(quote_data)
    d = res.to_dict()
    assert isinstance(d, dict)
    assert d["status"] == "ok"
    assert d["flow_type"] == "on_ramp"
    assert d["sell_asset"]["code"] == "USD"
    assert d["buy_asset"]["code"] == "XLM"


def test_quote_without_fee() -> None:
    quote_data = {
        "id": "nofee-1",
        "expires_at": "2026-10-22T17:00:00Z",
        "sell_asset": "iso4217:USD",
        "sell_amount": "50.00",
        "buy_asset": "stellar:native",
        "buy_amount": "400.00",
    }
    res = explain_sep38_quote(quote_data)
    assert res.status == "ok"
    assert res.fee is None


def test_quote_invalid_json_string() -> None:
    res = explain_sep38_quote("not a valid json string {")
    assert res.status == "ok"
    assert res.flow_type == "unknown"


def test_quote_invalid_expiration_timestamp() -> None:
    quote_data = {
        "id": "bad-time",
        "expires_at": "invalid-timestamp",
        "sell_asset": "iso4217:USD",
        "buy_asset": "stellar:native",
    }
    res = explain_sep38_quote(quote_data)
    assert res.status == "ok"
    assert res.is_expired is None
    assert res.seconds_until_expiration is None


def test_secret_in_nested_fee_details() -> None:
    secret = "S" + "D" * 55
    quote_data = {
        "id": "q-fee-secret",
        "sell_asset": "iso4217:USD",
        "buy_asset": "stellar:native",
        "fee": {
            "total": "1.00",
            "asset": "iso4217:USD",
            "details": [{"name": "memo", "description": f"transfer with {secret}"}],
        },
    }
    res = explain_sep38_quote(quote_data)
    assert res.status == "rejected_secret_input"
    assert secret not in (res.rejection_reason or "")
