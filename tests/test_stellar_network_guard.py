"""Offline tests for the Stellar network passphrase mismatch guard.

Envelopes are built locally with stellar_sdk from fixed, publicly known test
seeds and signed for each network in turn, so every hash and signature is
reproducible. No Horizon or Soroban RPC call is made and nothing is submitted.
"""

import asyncio
import base64
import random

import pytest
from fastapi import HTTPException
from pydantic import ValidationError
from stellar_sdk import Account, Asset, Keypair, Network, TransactionBuilder, TransactionEnvelope

import stellar
import stellar_fee_bump
from stellar import explorer_url, network_passphrase, stellar_info
from stellar_network_guard import (
    KNOWN_PASSPHRASES,
    MAX_PASSPHRASE_CHARS,
    NETWORK_GUARD_DISCLAIMER,
    NetworkGuardError,
    NetworkGuardRequest,
    check_network_endpoint,
    check_passphrase,
    describe_reported_network,
    guard_network,
)

TESTNET = Network.TESTNET_NETWORK_PASSPHRASE
PUBLIC = Network.PUBLIC_NETWORK_PASSPHRASE
FUTURENET = Network.FUTURENET_NETWORK_PASSPHRASE

# Deterministic throwaway keys: these seeds are fixed bytes and control
# nothing on any network.
SOURCE_KP = Keypair.from_raw_ed25519_seed(bytes(32))
FEE_KP = Keypair.from_raw_ed25519_seed(bytes([1]) * 32)
DESTINATION = Keypair.from_raw_ed25519_seed(bytes([2]) * 32).public_key
OPERATION_KP = Keypair.from_raw_ed25519_seed(bytes([3]) * 32)
STRANGER_KP = Keypair.from_raw_ed25519_seed(bytes([4]) * 32)

TX_HASH = "a" * 64


def run(coro):
    return asyncio.run(coro)


def build_tx(passphrase: str = TESTNET, *, operation_source: str | None = None, v1: bool = True) -> TransactionEnvelope:
    builder = TransactionBuilder(Account(SOURCE_KP.public_key, 1234), passphrase, base_fee=100, v1=v1)
    builder.append_payment_op(DESTINATION, Asset.native(), "1", source=operation_source)
    return builder.set_timeout(0).build()


def signed_tx(passphrase: str = TESTNET) -> str:
    envelope = build_tx(passphrase)
    envelope.sign(SOURCE_KP)
    return envelope.to_xdr()


def fee_bump(outer_passphrase: str = TESTNET, inner_passphrase: str = TESTNET) -> str:
    inner = build_tx(inner_passphrase)
    inner.sign(SOURCE_KP)
    envelope = TransactionBuilder.build_fee_bump_transaction(FEE_KP.public_key, 200, inner, outer_passphrase)
    envelope.sign(FEE_KP)
    return envelope.to_xdr()


def codes(report) -> list[str]:
    return [finding.code for finding in report.findings]


def finding(report, code: str):
    matches = [f for f in report.findings if f.code == code]
    assert matches, f"expected a {code} finding, got {codes(report)}"
    return matches[0]


# ---------------------------------------------------------------------------
# Passphrase classification
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("reported", [None, "", "   \n"])
def test_nothing_reported_is_not_a_finding(reported):
    assert check_passphrase(reported, "testnet") is None


def test_exact_passphrase_matches():
    check = check_passphrase(TESTNET, "testnet")
    assert (check.status, check.reported_network) == ("match", "testnet")


def test_surrounding_whitespace_is_a_paste_artifact_not_a_mismatch():
    assert check_passphrase(f"  {PUBLIC}\n", "public").status == "match"


def test_other_supported_network_is_named():
    check = check_passphrase(PUBLIC, "testnet")
    assert (check.status, check.reported_network) == ("other_network", "public")


def test_well_known_unsupported_network_is_named():
    check = check_passphrase(FUTURENET, "testnet")
    assert (check.status, check.reported_network) == ("other_network", "futurenet")
    assert describe_reported_network(check) == "the futurenet network, which Deen Bridge does not use"


@pytest.mark.parametrize(
    "reported, resembles",
    [
        (TESTNET.lower(), "testnet"),
        (TESTNET.replace(" ", "  "), "testnet"),
        (TESTNET.replace(" ", "\t"), "testnet"),
        (PUBLIC.upper(), "public"),
    ],
)
def test_case_and_spacing_near_misses_are_caught(reported, resembles):
    check = check_passphrase(reported, "testnet")
    assert (check.status, check.reported_network) == ("near_miss", resembles)


def test_dropping_characters_is_not_a_near_miss():
    # Removing the spaces around ";" changes the words, not just the spacing.
    assert check_passphrase(TESTNET.replace(" ; ", ";"), "testnet").status == "unknown"


def test_every_known_passphrase_is_distinct():
    assert len(set(KNOWN_PASSPHRASES.values())) == len(KNOWN_PASSPHRASES)
    assert KNOWN_PASSPHRASES["testnet"] == TESTNET
    assert KNOWN_PASSPHRASES["public"] == PUBLIC


# ---------------------------------------------------------------------------
# Passphrase reports
# ---------------------------------------------------------------------------


def test_matching_passphrase_confirms_the_network():
    report = guard_network("testnet", TESTNET)
    assert report.findings == []
    assert report.network_confirmed is True
    assert report.mismatch_detected is False
    assert report.expected_passphrase == TESTNET
    assert "matches testnet exactly" in " ".join(report.explanation)


def test_mainnet_is_an_alias_for_public():
    report = guard_network("mainnet", PUBLIC)
    assert report.network == "public"
    assert report.network_confirmed is True


def test_default_network_follows_the_service(monkeypatch):
    monkeypatch.setattr(stellar_fee_bump, "STELLAR_NETWORK", "public")
    assert guard_network(network_passphrase=PUBLIC).network_confirmed is True
    assert guard_network(network_passphrase=TESTNET).mismatch_detected is True


def test_passphrase_for_the_other_network_is_an_error():
    report = guard_network("testnet", PUBLIC)
    mismatch = finding(report, "passphrase_mismatch")
    assert mismatch.severity == "error"
    assert "the public network" in mismatch.message
    assert report.mismatch_detected is True
    assert report.network_confirmed is False
    assert "Do not sign or submit" in " ".join(report.explanation)


def test_near_miss_of_the_expected_network_is_an_error():
    report = guard_network("testnet", TESTNET.lower())
    near_miss = finding(report, "passphrase_near_miss")
    assert near_miss.severity == "error"
    assert "byte for byte" in near_miss.message
    assert "expects" not in near_miss.message
    assert report.network_confirmed is False


def test_near_miss_of_the_other_network_names_the_expected_one():
    report = guard_network("testnet", PUBLIC.lower())
    assert "Deen Bridge expects testnet" in finding(report, "passphrase_near_miss").message


def test_unknown_passphrase_is_never_echoed():
    passphrase = "Standalone Network ; February 2017 <script>alert(1)</script>"
    report = guard_network("testnet", passphrase)
    assert "neither testnet nor public" in finding(report, "unknown_passphrase").message
    assert passphrase not in report.model_dump_json()
    assert "<script>" not in report.model_dump_json()


def test_nothing_to_check_is_an_input_error():
    with pytest.raises(NetworkGuardError, match="Nothing to check"):
        guard_network("testnet", "  ", "  ")


# ---------------------------------------------------------------------------
# Envelope signatures
# ---------------------------------------------------------------------------


def test_envelope_signed_for_the_expected_network_is_confirmed():
    report = guard_network("testnet", envelope_xdr=signed_tx(TESTNET))
    envelope = report.envelope
    assert envelope.kind == "transaction"
    assert envelope.signed_for == ["testnet"]
    assert envelope.transaction_hash == build_tx(TESTNET).hash_hex()
    assert [(s.layer, s.signer, s.verdict) for s in envelope.signatures] == [
        ("transaction", SOURCE_KP.public_key, "verified")
    ]
    assert report.findings == []
    assert report.network_confirmed is True


def test_envelope_signed_for_the_other_network_is_an_error():
    report = guard_network("testnet", envelope_xdr=signed_tx(PUBLIC))
    signature = report.envelope.signatures[0]
    assert (signature.verdict, signature.valid_on) == ("other_network", ["public"])
    error = finding(report, "signed_for_other_network")
    assert SOURCE_KP.public_key in error.message
    assert "testnet will reject it" in error.message
    assert report.mismatch_detected is True
    assert report.network_confirmed is False


def test_hash_is_reported_for_the_expected_network_even_when_signed_for_another():
    report = guard_network("public", envelope_xdr=signed_tx(TESTNET))
    assert report.envelope.transaction_hash == build_tx(PUBLIC).hash_hex()


def test_envelope_signed_for_futurenet_is_identified():
    report = guard_network("testnet", envelope_xdr=signed_tx(FUTURENET))
    assert report.envelope.signed_for == ["futurenet"]
    assert "futurenet" in finding(report, "signed_for_other_network").message


def test_mixed_network_signatures_are_caught():
    envelope = build_tx(TESTNET, operation_source=OPERATION_KP.public_key)
    envelope.sign(SOURCE_KP)
    # The operation source signed the same transaction, but on the public network.
    on_public = build_tx(PUBLIC, operation_source=OPERATION_KP.public_key)
    on_public.sign(OPERATION_KP)
    envelope.signatures.append(on_public.signatures[0])

    report = guard_network("testnet", envelope_xdr=envelope.to_xdr())
    verdicts = {s.signer: s.verdict for s in report.envelope.signatures}
    assert verdicts == {SOURCE_KP.public_key: "verified", OPERATION_KP.public_key: "other_network"}
    assert report.envelope.signed_for == ["testnet", "public"]
    # One good signature does not excuse the bad one.
    assert report.mismatch_detected is True
    assert report.network_confirmed is True


def test_operation_source_signatures_are_checked():
    envelope = build_tx(TESTNET, operation_source=OPERATION_KP.public_key)
    envelope.sign(OPERATION_KP)
    report = guard_network("testnet", envelope_xdr=envelope.to_xdr())
    assert [(s.signer, s.verdict) for s in report.envelope.signatures] == [(OPERATION_KP.public_key, "verified")]


def test_unsigned_envelope_cannot_confirm_the_network():
    report = guard_network("testnet", envelope_xdr=build_tx(PUBLIC).to_xdr())
    assert codes(report) == ["unsigned_envelope"]
    assert finding(report, "unsigned_envelope").severity == "info"
    assert report.mismatch_detected is False
    assert report.network_confirmed is False
    assert "could confirm the network" in report.explanation[-1]


def test_signature_from_an_unrelated_key_is_unverifiable_not_wrong():
    envelope = build_tx(TESTNET)
    envelope.sign(STRANGER_KP)
    report = guard_network("testnet", envelope_xdr=envelope.to_xdr())
    assert [(s.signer, s.verdict) for s in report.envelope.signatures] == [(None, "unknown_signer")]
    assert codes(report) == ["unverifiable_signature"]
    assert report.mismatch_detected is False


def test_forged_signature_is_invalid():
    envelope = build_tx(TESTNET)
    envelope.sign(SOURCE_KP)
    envelope.signatures[0].signature = bytes(64)
    report = guard_network("testnet", envelope_xdr=envelope.to_xdr())
    assert report.envelope.signatures[0].verdict == "invalid"
    assert report.envelope.signed_for == []
    assert finding(report, "invalid_signature").severity == "error"


def test_v0_envelopes_are_checked():
    envelope = build_tx(PUBLIC, v1=False)
    envelope.sign(SOURCE_KP)
    report = guard_network("testnet", envelope_xdr=envelope.to_xdr())
    assert report.envelope.signed_for == ["public"]
    assert "signed_for_other_network" in codes(report)


def test_fee_bump_layers_are_checked_separately():
    report = guard_network("testnet", envelope_xdr=fee_bump(outer_passphrase=PUBLIC, inner_passphrase=TESTNET))
    assert report.envelope.kind == "fee_bump"
    layers = {s.layer: (s.signer, s.verdict) for s in report.envelope.signatures}
    assert layers == {
        "fee_bump": (FEE_KP.public_key, "other_network"),
        "inner_transaction": (SOURCE_KP.public_key, "verified"),
    }
    assert "fee bump signature" in finding(report, "signed_for_other_network").message


def test_clean_fee_bump_is_confirmed():
    xdr = fee_bump()
    report = guard_network("testnet", envelope_xdr=xdr)
    assert report.findings == []
    assert report.network_confirmed is True
    # The reported hash is the outer fee bump's, as the fee-bump explainer computes it.
    assert report.envelope.transaction_hash == stellar_fee_bump.explain_fee_bump(xdr, "testnet").fee_bump_hash
    assert "fee-bump transaction" in " ".join(report.explanation)


def test_passphrase_and_envelope_are_both_checked():
    report = guard_network("testnet", network_passphrase=PUBLIC, envelope_xdr=signed_tx(TESTNET))
    assert codes(report) == ["passphrase_mismatch"]
    assert report.mismatch_detected is True
    # The envelope itself is fine; the wallet's network setting is not.
    assert report.network_confirmed is True


def test_wrapped_envelope_is_accepted():
    xdr = signed_tx(TESTNET)
    wrapped = "\n".join(xdr[i : i + 40] for i in range(0, len(xdr), 40))
    assert guard_network("testnet", envelope_xdr=wrapped).network_confirmed is True


# ---------------------------------------------------------------------------
# Malformed envelopes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "envelope_xdr, message",
    [
        ("not base64!", "not base64"),
        ("A" * 200_001, "too large"),
        (base64.b64encode(b"\x00\x00\x00\x02junk").decode(), "could not be decoded"),
    ],
)
def test_bad_envelopes_raise_guard_errors(envelope_xdr, message):
    with pytest.raises(NetworkGuardError, match=message):
        guard_network("testnet", envelope_xdr=envelope_xdr)


def test_corrupted_envelopes_only_ever_raise_guard_errors():
    """Regression: hostile bytes must surface as a 400, never a 500."""
    raw = bytearray(base64.b64decode(fee_bump()))
    rng = random.Random(649)
    for _ in range(300):
        mutated = bytearray(raw)
        for _ in range(rng.randint(1, 8)):
            mutated[rng.randrange(len(mutated))] = rng.randrange(256)
        try:
            guard_network("testnet", envelope_xdr=base64.b64encode(bytes(mutated)).decode())
        except NetworkGuardError:
            pass


# ---------------------------------------------------------------------------
# HTTP endpoint
# ---------------------------------------------------------------------------


def test_endpoint_reports_a_mismatch():
    body = NetworkGuardRequest(network="testnet", network_passphrase=PUBLIC, envelope_xdr=signed_tx(PUBLIC))
    report = run(check_network_endpoint(body))
    assert report.mismatch_detected is True
    assert set(codes(report)) == {"passphrase_mismatch", "signed_for_other_network"}
    assert report.disclaimer == NETWORK_GUARD_DISCLAIMER
    assert "nothing was signed or submitted" in report.disclaimer


@pytest.mark.parametrize("field", ["network_passphrase", "envelope_xdr"])
def test_endpoint_refuses_secret_keys_without_echoing_them(field):
    secret = SOURCE_KP.secret
    body = NetworkGuardRequest(**{field: f"here it is {secret}"})
    with pytest.raises(HTTPException) as excinfo:
        run(check_network_endpoint(body))
    assert excinfo.value.status_code == 400
    assert "secret key" in excinfo.value.detail
    assert secret not in str(excinfo.value.detail)


def test_endpoint_turns_input_errors_into_400s():
    with pytest.raises(HTTPException) as excinfo:
        run(check_network_endpoint(NetworkGuardRequest(envelope_xdr="not base64!")))
    assert excinfo.value.status_code == 400
    assert "base64" in excinfo.value.detail


@pytest.mark.parametrize(
    "overrides",
    [
        {"network": "futurenet"},
        {"network_passphrase": "x" * (MAX_PASSPHRASE_CHARS + 1)},
        {"envelope_xdr": "A" * 200_001},
    ],
)
def test_request_validation(overrides):
    with pytest.raises(ValidationError):
        NetworkGuardRequest(**overrides)


@pytest.mark.parametrize(
    "body",
    [
        {"networkPassphrase": TESTNET, "envelopeXdr": "AAAA"},  # Freighter / JS SDK
        {"passphrase": TESTNET, "envelope_xdr": "AAAA"},  # Soroban RPC getNetwork
        {"network_passphrase": TESTNET, "envelope_xdr": "AAAA"},
    ],
)
def test_request_accepts_wallet_and_rpc_field_names(body):
    request = NetworkGuardRequest.model_validate(body)
    assert (request.network_passphrase, request.envelope_xdr) == (TESTNET, "AAAA")


# ---------------------------------------------------------------------------
# Service configuration
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "configured, expected",
    [("testnet", TESTNET), ("public", PUBLIC), ("mainnet", PUBLIC), ("unknown", TESTNET)],
)
def test_stellar_info_publishes_the_expected_passphrase(monkeypatch, configured, expected):
    monkeypatch.setattr(stellar, "STELLAR_NETWORK", configured)
    assert network_passphrase() == expected
    info = run(stellar_info())
    assert info["network_passphrase"] == expected
    assert "network-passphrase-guard" in info["features"]


def test_mainnet_explorer_links_point_at_the_public_network(monkeypatch):
    """Regression: STELLAR_NETWORK=mainnet used to produce testnet explorer links."""
    monkeypatch.setattr(stellar, "STELLAR_NETWORK", "mainnet")
    assert explorer_url(TX_HASH) == f"https://stellar.expert/explorer/public/tx/{TX_HASH}"
