"""Offline tests for the Soroban contract event topic decoder.

Most cases are driven by tests/fixtures/stellar_contract_events.json: base64
topics and values shaped like Soroban RPC getEvents entries, paired with
hand-written expected decodings. Fixture keys come from fixed, publicly known
test seeds, so nothing here touches a network or a real account.
"""

import asyncio
import base64
import json
import random
import struct
from pathlib import Path

import pytest
from fastapi import HTTPException
from pydantic import ValidationError
from stellar_sdk import Address, Asset, Keypair, Network, scval, xdr

import stellar_fee_bump
from stellar import stellar_info
from stellar_contract_events import (
    CONTRACT_EVENT_DISCLAIMER,
    MAX_EVENT_TOPICS,
    MAX_SCVAL_DEPTH,
    ContractEventDecodeError,
    ContractEventDecodeRequest,
    decode_contract_event,
    decode_contract_event_endpoint,
    decode_scval,
    interpret_token_event,
)

FIXTURES = json.loads((Path(__file__).parent / "fixtures" / "stellar_contract_events.json").read_text("utf-8"))
SCVAL_FIXTURES = FIXTURES["scvals"]
EVENT_FIXTURES = FIXTURES["events"]
EVENTS_BY_NAME = {event["name"]: event for event in EVENT_FIXTURES}

ALICE_KP = Keypair.from_raw_ed25519_seed(bytes(32))
ALICE = ALICE_KP.public_key
BOB = Keypair.from_raw_ed25519_seed(bytes([1]) * 32).public_key

USDC_PUBLIC_SAC = "CCW67TSZV3SSS2HXMBQ5JFGCKJNXKZM7UQUWUZPUTHXSTZLEO7SJMI75"
USDC_TESTNET_SAC = "CBIELTK6YBZJU5UP2WWQEUCYKLPU6AUNZ2BQ4WWFEIE3USCIHMXQDAMA"


def run(coro):
    return asyncio.run(coro)


def decode_fixture(event: dict):
    """Decode a fixture the way the endpoint does, through its getEvents field names."""
    body = ContractEventDecodeRequest.model_validate(event)
    return decode_contract_event(body.topics, body.value, body.contract_id, body.network)


def explanation_of(name: str) -> str:
    return " ".join(decode_fixture(EVENTS_BY_NAME[name]).explanation)


# ---------------------------------------------------------------------------
# Fixture integrity
# ---------------------------------------------------------------------------


def test_fixture_names_are_unique():
    for group in (SCVAL_FIXTURES, EVENT_FIXTURES):
        names = [case["name"] for case in group]
        assert len(names) == len(set(names))


def test_fixture_asset_contract_ids_are_the_real_ones():
    """Pins the well-known USDC contracts so a fixture regeneration cannot drift."""
    public_usdc = Asset("USDC", "GA5ZSEJYB37JRC5AVCIA5MOP4RHTM335X2KGX3IHOJAPP5RE34K4KZVN")
    testnet_usdc = Asset("USDC", "GBBD47IF6LWK7P7MDEVSCWR7DPUWV3NY3DTQEVFL4NAT4AQH3ZLLFLA5")
    assert public_usdc.contract_id(Network.PUBLIC_NETWORK_PASSPHRASE) == USDC_PUBLIC_SAC
    assert testnet_usdc.contract_id(Network.TESTNET_NETWORK_PASSPHRASE) == USDC_TESTNET_SAC
    assert EVENTS_BY_NAME["sac_usdc_transfer_public"]["contractId"] == USDC_PUBLIC_SAC
    assert EVENTS_BY_NAME["sac_usdc_transfer_testnet"]["contractId"] == USDC_TESTNET_SAC


@pytest.mark.parametrize("event", EVENT_FIXTURES, ids=lambda event: event["name"])
def test_fixture_xdr_is_canonical(event):
    """Every fixture topic re-encodes byte for byte, so the fixtures are well-formed XDR."""
    for topic in [*event["topic"], event["value"]]:
        if topic is not None:
            assert xdr.SCVal.from_xdr(topic).to_xdr() == topic


def test_fixtures_respect_the_topic_limit():
    assert all(1 <= len(event["topic"]) <= MAX_EVENT_TOPICS for event in EVENT_FIXTURES)


# ---------------------------------------------------------------------------
# SCVal decoding
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", SCVAL_FIXTURES, ids=lambda case: case["name"])
def test_scval_fixtures_decode(case):
    assert decode_scval(case["xdr"]).model_dump(exclude_none=True) == case["expected"]


def test_large_integers_survive_json():
    decoded = decode_scval(next(case["xdr"] for case in SCVAL_FIXTURES if case["name"] == "u256_max"))
    assert json.loads(decoded.model_dump_json())["value"] == str(2**256 - 1)


def test_non_utf8_symbol_is_replaced_not_raised():
    raw = xdr.SCVal(xdr.SCValType.SCV_SYMBOL, sym=xdr.SCSymbol(b"tr\xffnsfer")).to_xdr()
    assert decode_scval(raw).value == "tr�nsfer"


def test_whitespace_wrapped_scval_is_accepted():
    xdr_b64 = scval.to_string("a longer string that wraps across several lines").to_xdr()
    wrapped = "\n".join(xdr_b64[i : i + 16] for i in range(0, len(xdr_b64), 16))
    assert decode_scval(f"  {wrapped}\n").value == "a longer string that wraps across several lines"


def _nested_vec(depth: int) -> str:
    # Built from raw XDR (SCV_VEC = 16, present, one item) because the SDK's
    # own encoder recurses too deeply to build the hostile cases.
    return base64.b64encode(struct.pack(">III", 16, 1, 1) * depth + struct.pack(">I", 1)).decode()


def test_nesting_up_to_the_limit_decodes():
    decoded = decode_scval(_nested_vec(MAX_SCVAL_DEPTH))
    for _ in range(MAX_SCVAL_DEPTH):
        assert decoded.type == "vec" and decoded.items is not None
        decoded = decoded.items[0]
    assert decoded.type == "void"


@pytest.mark.parametrize("depth", [MAX_SCVAL_DEPTH + 1, 5000])
def test_excessive_nesting_is_rejected(depth):
    with pytest.raises(ContractEventDecodeError):
        decode_scval(_nested_vec(depth))


# ---------------------------------------------------------------------------
# Whole events, from the fixtures
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("event", EVENT_FIXTURES, ids=lambda event: event["name"])
def test_event_fixtures_decode(event):
    expected = event["expected"]
    result = decode_fixture(event)

    assert result.network == event["network"]
    assert result.contract_id == event["contractId"]
    assert result.event_name == expected["event_name"]
    assert [topic.type for topic in result.topics] == expected["topic_types"]
    token_event = result.token_event.model_dump(exclude_none=True) if result.token_event else None
    assert token_event == expected["token_event"]
    assert result.asset_contract_id == expected["asset_contract_id"]
    assert result.asset_contract_status == expected["asset_contract_status"]
    assert result.is_platform_usdc is expected["is_platform_usdc"]
    assert result.disclaimer == CONTRACT_EVENT_DISCLAIMER
    assert (result.value is None) is (event["value"] is None)


def test_genuine_usdc_transfer_is_explained():
    text = explanation_of("sac_usdc_transfer_public")
    assert f"A SEP-41 transfer of 1.2500000 USDC (12500000 base units) from {ALICE} to {BOB}." in text
    assert f"{USDC_PUBLIC_SAC} is the genuine Stellar Asset Contract" in text
    assert "Deen Bridge uses for course and book purchases" in text


def test_spoofed_usdc_transfer_is_flagged():
    text = explanation_of("spoofed_usdc_transfer")
    assert "Warning" in text
    assert "is NOT the Stellar Asset Contract" in text
    assert USDC_PUBLIC_SAC in text
    assert "did not move the real asset" in text


def test_contract_from_the_other_network_is_flagged():
    text = explanation_of("public_usdc_event_decoded_for_testnet")
    assert "Stellar Asset Contract for" in text
    assert "on public, not testnet" in text


def test_missing_contract_id_names_the_contract_to_check():
    text = explanation_of("transfer_topics_only")
    assert "No contract ID was supplied" in text
    assert USDC_TESTNET_SAC in text
    assert "amount not supplied" in text


def test_muxed_destination_is_described():
    assert "muxed with ID 42" in explanation_of("sac_transfer_to_muxed_cap67")


def test_native_asset_amount_is_in_xlm():
    assert "1.0000000 XLM" in explanation_of("sac_native_xlm_transfer")


def test_custom_token_amount_stays_in_base_units():
    text = explanation_of("custom_token_burn")
    assert "1000 base units" in text
    assert "decimals are set by its contract" in text


def test_approval_expiry_is_described():
    assert "valid until ledger 600000" in explanation_of("sac_approve")


def test_admin_and_authorization_events_are_described():
    assert f"handed admin rights to {BOB}" in explanation_of("sac_set_admin")
    assert f"{ALICE} was deauthorized" in explanation_of("sac_set_authorized_cap67")


def test_non_symbol_first_topic_is_described():
    assert "is a bytes, not a symbol" in explanation_of("hash_first_topic")


def test_token_name_with_wrong_shape_is_not_guessed():
    text = explanation_of("transfer_with_unexpected_value")
    assert "not read as a token event" in text


def test_invalid_asset_topic_gets_no_contract_or_display_amount():
    result = decode_fixture(EVENTS_BY_NAME["invalid_asset_topic"])
    assert result.token_event is not None
    assert result.token_event.amount_display is None
    assert "not a valid Stellar asset" in " ".join(result.explanation)


def test_custom_event_keeps_its_structured_value():
    result = decode_fixture(EVENTS_BY_NAME["course_enrollment_custom_event"])
    assert result.value is not None and result.value.entries is not None
    fields = {entry.key.value: entry.value.model_dump(exclude_none=True) for entry in result.value.entries}
    assert fields == {"paid": {"type": "i128", "value": "50000000"}, "at": {"type": "timepoint", "value": "1761091200"}}


def test_default_network_follows_service_config(monkeypatch):
    event = EVENTS_BY_NAME["sac_usdc_transfer_testnet"]
    monkeypatch.setattr(stellar_fee_bump, "STELLAR_NETWORK", "testnet")
    assert decode_contract_event(event["topic"], event["value"], event["contractId"]).network == "testnet"
    monkeypatch.setattr(stellar_fee_bump, "STELLAR_NETWORK", "mainnet")
    result = decode_contract_event(event["topic"], event["value"], event["contractId"])
    assert result.network == "public"
    assert result.asset_contract_status == "other_network"


def test_too_many_topics_in_a_token_event_is_not_a_token_event():
    sym, addr = scval.to_symbol, scval.to_address
    topics = [decode_scval(t.to_xdr()) for t in (sym("burn"), addr(ALICE), addr(BOB), addr(ALICE))]
    assert interpret_token_event(topics, None) is None


# ---------------------------------------------------------------------------
# Secret keys
# ---------------------------------------------------------------------------


def test_secret_key_inside_an_event_string_is_redacted():
    secret = ALICE_KP.secret
    topics = [scval.to_symbol("memo").to_xdr(), scval.to_string(f"backup: {secret}").to_xdr()]
    result = decode_contract_event(topics, scval.to_vec([scval.to_string(secret)]).to_xdr(), network="testnet")
    assert secret not in result.model_dump_json()
    assert "REDACTED" in str(result.topics[1].value)
    assert any("treat it as compromised" in line for line in result.explanation)


def test_endpoint_refuses_secret_keys_without_echoing_them():
    secret = ALICE_KP.secret
    with pytest.raises(HTTPException) as excinfo:
        run(decode_contract_event_endpoint(ContractEventDecodeRequest(topics=[secret])))
    assert excinfo.value.status_code == 400
    assert "secret key" in excinfo.value.detail
    assert secret not in excinfo.value.detail
    assert secret not in excinfo.value.hint


def test_endpoint_refuses_a_secret_key_in_the_contract_id():
    body = ContractEventDecodeRequest(topics=[scval.to_symbol("transfer").to_xdr()], contract_id=ALICE_KP.secret)
    with pytest.raises(HTTPException) as excinfo:
        run(decode_contract_event_endpoint(body))
    assert "secret key" in excinfo.value.detail


# ---------------------------------------------------------------------------
# Malformed input
# ---------------------------------------------------------------------------

TRANSFER = scval.to_symbol("transfer").to_xdr()


@pytest.mark.parametrize(
    "topics,message",
    [
        ([], "No topics"),
        ([TRANSFER] * (MAX_EVENT_TOPICS + 1), "at most 4 topics"),
        ([""], "Topic 1 is empty"),
        ([TRANSFER, "  \n"], "Topic 2 is empty"),
        ([TRANSFER, "not base64!"], "Topic 2 is not base64"),
        (["<script>alert(1)</script>"], "not base64"),
        (["A" * 64_001], "too large"),
        (["AAAADwAAAAh0cmFu"], "could not be decoded"),
        (["AAAADwAAAAh0cmFuc2ZlcgAAAAA="], "could not be decoded"),
    ],
)
def test_malformed_topics_are_rejected(topics, message):
    with pytest.raises(ContractEventDecodeError, match=message) as excinfo:
        decode_contract_event(topics, network="testnet")
    assert excinfo.value.hint


def test_malformed_value_is_named_in_the_error():
    with pytest.raises(ContractEventDecodeError, match="The value could not be decoded"):
        decode_contract_event([TRANSFER], "AAAACg==", network="testnet")


@pytest.mark.parametrize("contract_id", [ALICE, "C123", "c" + USDC_PUBLIC_SAC[1:]])
def test_invalid_contract_id_is_rejected(contract_id):
    with pytest.raises(ContractEventDecodeError, match="not a valid Soroban contract address"):
        decode_contract_event([TRANSFER], contract_id=contract_id, network="testnet")


def test_blank_contract_id_counts_as_absent():
    assert decode_contract_event([TRANSFER], contract_id="  ", network="testnet").contract_id is None


def test_corrupted_events_only_ever_raise_decode_errors():
    """Regression: hostile bytes must surface as a 400, never a 500."""
    rng = random.Random(647)
    for event in EVENT_FIXTURES:
        for _ in range(40):
            mutated = []
            for topic in event["topic"]:
                raw = bytearray(base64.b64decode(topic))
                for _ in range(rng.randint(0, 4)):
                    raw[rng.randrange(len(raw))] = rng.randrange(256)
                mutated.append(base64.b64encode(bytes(raw)).decode())
            try:
                decode_contract_event(mutated, event["value"], event["contractId"], event["network"])
            except ContractEventDecodeError:
                pass


def test_unknown_address_bytes_are_a_decode_error():
    # SCV_ADDRESS with an address type tag the XDR schema does not define.
    raw = struct.pack(">II", 18, 99) + bytes(32)
    with pytest.raises(ContractEventDecodeError):
        decode_scval(base64.b64encode(raw).decode())


# ---------------------------------------------------------------------------
# HTTP endpoint and request model
# ---------------------------------------------------------------------------


def test_endpoint_accepts_a_get_events_entry_as_is():
    event = {**EVENTS_BY_NAME["sac_usdc_transfer_public"], "ledger": 123, "txHash": "ab" * 32, "type": "contract"}
    result = run(decode_contract_event_endpoint(ContractEventDecodeRequest.model_validate(event)))
    assert result.asset_contract_status == "verified"
    assert result.token_event is not None and result.token_event.amount == "12500000"
    assert "nothing was signed or submitted" in result.disclaimer


def test_endpoint_maps_decode_errors_to_400():
    body = ContractEventDecodeRequest(topics=["not base64!"], network="testnet")
    with pytest.raises(HTTPException) as excinfo:
        run(decode_contract_event_endpoint(body))
    assert excinfo.value.status_code == 400
    assert "not base64" in excinfo.value.detail
    assert excinfo.value.hint


@pytest.mark.parametrize(
    "overrides",
    [
        {"topics": []},
        {"topics": [TRANSFER] * (MAX_EVENT_TOPICS + 1)},
        {"topics": ["A" * 64_001]},
        {"value": "A" * 64_001},
        {"network": "futurenet"},
        {"contract_id": "C" * 65},
    ],
)
def test_request_validation(overrides):
    with pytest.raises(ValidationError):
        ContractEventDecodeRequest(**{"topics": [TRANSFER], **overrides})


def test_request_accepts_both_field_spellings():
    contract = Address.from_raw_contract(bytes([7]) * 32).address
    snake = ContractEventDecodeRequest.model_validate({"topics": [TRANSFER], "contract_id": contract})
    camel = ContractEventDecodeRequest.model_validate({"topic": [TRANSFER], "contractId": contract})
    assert snake == camel


def test_stellar_info_advertises_the_decoder():
    assert "contract-event-decoder" in run(stellar_info())["features"]
