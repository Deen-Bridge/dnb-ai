"""Offline tests for the Stellar fee-bump explainer.

Envelopes are built locally with stellar_sdk from fixed, publicly known test
seeds, so every hash and signature is reproducible. No Horizon call is made
and nothing is submitted anywhere.
"""

import asyncio
import base64
import random

import pytest
from fastapi import HTTPException
from pydantic import ValidationError
from stellar_sdk import (
    Account,
    Asset,
    FeeBumpTransaction,
    FeeBumpTransactionEnvelope,
    Keypair,
    MuxedAccount,
    Network,
    SorobanDataBuilder,
    TransactionBuilder,
    TransactionEnvelope,
)

import stellar_fee_bump
from stellar import SECRET_KEY_WARNING, stellar_info
from stellar_fee_bump import (
    FEE_BUMP_CONCEPTS,
    FeeBumpDecodeError,
    FeeBumpExplainRequest,
    build_chat_fee_bump_context,
    canonical_network,
    explain_fee_bump,
    explain_fee_bump_endpoint,
    format_xlm,
    is_fee_bump_question,
)

TESTNET = Network.TESTNET_NETWORK_PASSPHRASE
PUBLIC = Network.PUBLIC_NETWORK_PASSPHRASE

# Deterministic throwaway keys: these seeds are all-zero / all-one bytes and
# control nothing on any network.
INNER_KP = Keypair.from_raw_ed25519_seed(bytes(32))
FEE_KP = Keypair.from_raw_ed25519_seed(bytes([1]) * 32)
DESTINATION = Keypair.from_raw_ed25519_seed(bytes([2]) * 32).public_key


def run(coro):
    return asyncio.run(coro)


def build_inner(
    passphrase: str = TESTNET,
    *,
    operations: int = 2,
    base_fee: int = 100,
    resource_fee: int | None = None,
    sign: bool = True,
) -> TransactionEnvelope:
    builder = TransactionBuilder(Account(INNER_KP.public_key, 1234), passphrase, base_fee=base_fee)
    for amount in range(1, operations + 1):
        builder.append_payment_op(DESTINATION, Asset.native(), str(amount))
    if resource_fee is not None:
        builder.set_soroban_data(SorobanDataBuilder().set_resource_fee(resource_fee).build())
    envelope = builder.set_timeout(0).build()
    if sign:
        envelope.sign(INNER_KP)
    return envelope


def build_fee_bump(
    passphrase: str = TESTNET,
    *,
    base_fee: int = 500,
    fee_source: str | MuxedAccount | None = None,
    sign: bool = True,
    sign_inner: bool = True,
    **inner_kwargs,
) -> FeeBumpTransactionEnvelope:
    inner = build_inner(passphrase, sign=sign_inner, **inner_kwargs)
    envelope = TransactionBuilder.build_fee_bump_transaction(
        fee_source or FEE_KP.public_key, base_fee, inner, passphrase
    )
    if sign:
        envelope.sign(FEE_KP)
    return envelope


def raw_fee_bump(fee: int, inner: TransactionEnvelope) -> str:
    """A fee bump with an arbitrary fee, bypassing the SDK builder's own checks."""
    envelope = FeeBumpTransactionEnvelope(FeeBumpTransaction(FEE_KP.public_key, fee, inner), TESTNET)
    envelope.sign(FEE_KP)
    return envelope.to_xdr()


# ---------------------------------------------------------------------------
# Decoding and the explanation
# ---------------------------------------------------------------------------


def test_explains_a_signed_fee_bump():
    envelope = build_fee_bump()
    result = explain_fee_bump(envelope.to_xdr(), "testnet")

    assert result.network == "testnet"
    assert result.fee_source == FEE_KP.public_key
    assert result.inner_source == INNER_KP.public_key
    assert result.fee_bump_hash == envelope.hash_hex()
    assert result.inner_transaction_hash == envelope.transaction.inner_transaction_envelope.hash_hex()
    assert result.explorer_url == f"https://stellar.expert/explorer/testnet/tx/{envelope.hash_hex()}"
    assert result.operation_count == 2
    assert result.inner_sequence == "1235"
    # 500 stroops per operation × (2 inner operations + 1 for the fee bump).
    assert result.max_fee_stroops == "1500"
    assert result.max_fee_xlm == "0.0001500"
    assert result.inner_fee_stroops == "200"
    assert result.minimum_fee_stroops == "300"
    assert result.meets_minimum_fee is True
    assert result.meets_inner_fee_rate is True
    assert result.fee_source_signature == "verified"
    assert result.inner_source_signature == "verified"
    assert result.soroban_resource_fee_stroops == "0"
    assert not result.fee_source_is_muxed


def test_explanation_is_plain_language_and_non_custodial():
    text = " ".join(explain_fee_bump(build_fee_bump().to_xdr(), "testnet").explanation)
    assert "paid in XLM" in text
    assert "not charged" in text
    assert "never needs the inner account's secret key" in text


def test_hashes_and_signatures_are_network_bound():
    xdr = build_fee_bump(TESTNET).to_xdr()
    on_public = explain_fee_bump(xdr, "public")
    on_testnet = explain_fee_bump(xdr, "testnet")

    assert on_public.fee_bump_hash != on_testnet.fee_bump_hash
    assert on_public.inner_transaction_hash != on_testnet.inner_transaction_hash
    assert on_public.explorer_url.startswith("https://stellar.expert/explorer/public/tx/")
    # Signed for testnet, so the public network would reject both signatures.
    assert on_public.fee_source_signature == "other_network"
    assert on_public.inner_source_signature == "other_network"
    assert any("wrong network" in line for line in on_public.explanation)


def test_mainnet_alias_and_default_network(monkeypatch):
    xdr = build_fee_bump(PUBLIC).to_xdr()
    assert explain_fee_bump(xdr, "mainnet").network == "public"
    assert explain_fee_bump(xdr, "mainnet").fee_source_signature == "verified"

    monkeypatch.setattr(stellar_fee_bump, "STELLAR_NETWORK", "testnet")
    assert explain_fee_bump(xdr).network == "testnet"


@pytest.mark.parametrize(
    "configured,expected",
    [
        ("testnet", "testnet"),
        ("public", "public"),
        ("mainnet", "public"),
        (" Mainnet ", "public"),
        ("futurenet", "testnet"),
    ],
)
def test_canonical_network(configured, expected):
    assert canonical_network(configured) == expected


def test_unsigned_fee_bump_reports_missing_signatures():
    result = explain_fee_bump(build_fee_bump(sign=False, sign_inner=False).to_xdr(), "testnet")
    assert result.fee_source_signature == "not_found"
    assert result.inner_source_signature == "not_found"


def test_tampered_signature_is_invalid():
    envelope = build_fee_bump()
    envelope.signatures[0].signature = bytes(64)
    assert explain_fee_bump(envelope.to_xdr(), "testnet").fee_source_signature == "invalid"


def test_fee_below_minimum_is_flagged():
    # Two operations need at least 100 × 3 = 300 stroops.
    result = explain_fee_bump(raw_fee_bump(250, build_inner(base_fee=100)), "testnet")
    assert result.meets_minimum_fee is False
    assert result.minimum_fee_stroops == "300"
    assert any("too low" in line for line in result.explanation)


def test_fee_rate_lower_than_inner_is_flagged():
    # Inner pays 1000 per operation; 2000 over (1 + 1) operations exactly matches it.
    tie = explain_fee_bump(raw_fee_bump(2000, build_inner(operations=1, base_fee=1000)), "testnet")
    assert tie.meets_inner_fee_rate is True
    # 1999 / 2 < 1000 / 1, so the fee bump would lower the rate.
    lower = explain_fee_bump(raw_fee_bump(1999, build_inner(operations=1, base_fee=1000)), "testnet")
    assert lower.meets_minimum_fee is True
    assert lower.meets_inner_fee_rate is False
    assert any("may only raise the fee rate" in line for line in lower.explanation)


def test_surge_base_fee_raises_the_minimum():
    result = explain_fee_bump(build_fee_bump(base_fee=500).to_xdr(), "testnet", base_fee_stroops=1000)
    assert result.minimum_fee_stroops == "3000"
    assert result.meets_minimum_fee is False


def test_soroban_resource_fee_is_excluded_from_fee_rate_rules():
    envelope = build_fee_bump(base_fee=200, operations=1, resource_fee=5000)
    result = explain_fee_bump(envelope.to_xdr(), "testnet")
    assert result.soroban_resource_fee_stroops == "5000"
    assert result.max_fee_stroops == "5400"
    assert result.inner_fee_stroops == "5100"
    assert result.minimum_fee_stroops == "5200"
    assert result.meets_minimum_fee is True
    assert result.meets_inner_fee_rate is True


def test_muxed_fee_source_reports_the_base_account():
    result = explain_fee_bump(build_fee_bump(fee_source=MuxedAccount(FEE_KP.public_key, 42)).to_xdr(), "testnet")
    assert result.fee_source == FEE_KP.public_key
    assert result.fee_source_is_muxed is True
    assert result.fee_source_signature == "verified"


def test_same_account_fee_bump_is_described():
    inner = build_inner()
    envelope = TransactionBuilder.build_fee_bump_transaction(INNER_KP.public_key, 500, inner, TESTNET)
    result = explain_fee_bump(envelope.to_xdr(), "testnet")
    assert any("same account" in line for line in result.explanation)


def test_whitespace_wrapped_xdr_is_accepted():
    xdr = build_fee_bump().to_xdr()
    wrapped = "\n".join(xdr[i : i + 64] for i in range(0, len(xdr), 64))
    assert explain_fee_bump(f"  {wrapped}\n", "testnet").fee_bump_hash == build_fee_bump().hash_hex()


def test_ordinary_transaction_is_rejected_with_guidance():
    with pytest.raises(FeeBumpDecodeError, match="ordinary transaction envelope") as excinfo:
        explain_fee_bump(build_inner().to_xdr(), "testnet")
    assert "AAAABQ" in excinfo.value.hint


@pytest.mark.parametrize(
    "envelope_xdr,message",
    [
        ("", "No transaction envelope"),
        ("   \n ", "No transaction envelope"),
        ("not base64 at all!", "not base64"),
        ("<script>alert(1)</script>", "not base64"),
        ("AAAABQAAAAA=", "could not be decoded"),
        ("A" * 200_001, "too large"),
    ],
)
def test_malformed_input_is_rejected(envelope_xdr, message):
    with pytest.raises(FeeBumpDecodeError, match=message):
        explain_fee_bump(envelope_xdr, "testnet")


def test_truncated_envelope_is_rejected():
    xdr = build_fee_bump().to_xdr()
    raw = base64.b64decode(xdr)
    with pytest.raises(FeeBumpDecodeError, match="could not be decoded"):
        explain_fee_bump(base64.b64encode(raw[:-20]).decode(), "testnet")


def test_corrupted_envelopes_only_ever_raise_decode_errors():
    """Regression: hostile bytes must surface as a 400, never a 500."""
    raw = bytearray(base64.b64decode(build_fee_bump().to_xdr()))
    rng = random.Random(633)
    for _ in range(300):
        mutated = bytearray(raw)
        for _ in range(rng.randint(1, 8)):
            mutated[rng.randrange(len(mutated))] = rng.randrange(256)
        try:
            explain_fee_bump(base64.b64encode(bytes(mutated)).decode(), "testnet")
        except FeeBumpDecodeError:
            pass


def test_format_xlm():
    assert format_xlm(100) == "0.0000100"
    assert format_xlm(10_000_000) == "1.0000000"


# ---------------------------------------------------------------------------
# HTTP endpoint
# ---------------------------------------------------------------------------


def test_endpoint_explains_envelope():
    body = FeeBumpExplainRequest(envelope_xdr=build_fee_bump().to_xdr(), network="testnet")
    result = run(explain_fee_bump_endpoint(body))
    assert result.max_fee_stroops == "1500"
    assert "nothing was signed or submitted" in result.disclaimer


def test_endpoint_maps_decode_errors_to_400():
    body = FeeBumpExplainRequest(envelope_xdr=build_inner().to_xdr(), network="testnet")
    with pytest.raises(HTTPException) as excinfo:
        run(explain_fee_bump_endpoint(body))
    assert excinfo.value.status_code == 400
    assert "not a fee bump" in excinfo.value.detail
    assert excinfo.value.hint


def test_endpoint_refuses_secret_keys_without_echoing_them():
    secret = INNER_KP.secret
    with pytest.raises(HTTPException) as excinfo:
        run(explain_fee_bump_endpoint(FeeBumpExplainRequest(envelope_xdr=secret)))
    assert excinfo.value.status_code == 400
    assert "secret key" in excinfo.value.detail
    assert secret not in excinfo.value.detail
    assert secret not in excinfo.value.hint


@pytest.mark.parametrize(
    "overrides",
    [
        {"base_fee_stroops": 99},
        {"base_fee_stroops": 1_000_000_001},
        {"network": "futurenet"},
        {"envelope_xdr": ""},
    ],
)
def test_request_validation(overrides):
    with pytest.raises(ValidationError):
        FeeBumpExplainRequest(**{"envelope_xdr": "AAAABQ", **overrides})


def test_stellar_info_advertises_fee_bump_features():
    features = run(stellar_info())["features"]
    assert "fee-bump-explainer" in features
    assert "chat-fee-bump-explainer" in features


# ---------------------------------------------------------------------------
# Chat integration
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "prompt",
    [
        "What is a fee bump transaction on Stellar?",
        "Can someone fee-bump my stuck payment?",
        "How does feebump work?",
        "Can a sponsor pay the fee? Is that fee sponsorship?",
        "Can the course creator sponsor my transaction fees?",
        "How do I bump the fee on my transaction?",
    ],
)
def test_fee_bump_questions_are_detected(prompt):
    assert is_fee_bump_question(prompt)


@pytest.mark.parametrize(
    "prompt",
    [
        "What is a Stellar trustline?",
        "How many stroops are in 12.5 USDC?",
        "What is the fee for this course?",
        "Explain the hadith about sponsoring an orphan",
        "",
    ],
)
def test_unrelated_prompts_get_no_context(prompt):
    assert build_chat_fee_bump_context(prompt) is None


def test_concept_question_gets_facts_and_guardrails_only():
    context = build_chat_fee_bump_context("What is a fee bump?")
    assert context is not None
    assert context.startswith(FEE_BUMP_CONCEPTS)
    assert "DECODED FEE-BUMP ENVELOPE" not in context
    assert "Never ask for, accept, or repeat a secret key or seed phrase" in context
    assert "cannot" in context


def test_pasted_envelope_is_decoded_for_the_service_network(monkeypatch):
    monkeypatch.setattr(stellar_fee_bump, "STELLAR_NETWORK", "testnet")
    envelope = build_fee_bump()
    context = build_chat_fee_bump_context(f"Who pays for this?\n{envelope.to_xdr()}")
    assert context is not None
    assert "DECODED FEE-BUMP ENVELOPE" in context
    assert envelope.hash_hex() in context
    assert f"Fee source (pays): {FEE_KP.public_key}" in context
    assert "1500 stroops (0.0001500 XLM)" in context


def test_envelope_alone_triggers_the_explainer(monkeypatch):
    monkeypatch.setattr(stellar_fee_bump, "STELLAR_NETWORK", "testnet")
    context = build_chat_fee_bump_context(f"what does this do {build_fee_bump().to_xdr()}")
    assert context is not None
    assert "DECODED FEE-BUMP ENVELOPE" in context


def test_chat_on_public_service_flags_testnet_envelope(monkeypatch):
    monkeypatch.setattr(stellar_fee_bump, "STELLAR_NETWORK", "public")
    context = build_chat_fee_bump_context(f"Explain this fee bump {build_fee_bump(TESTNET).to_xdr()}")
    assert context is not None
    assert "decoded for the public network" in context
    assert "Fee source signature: other_network" in context


def test_ordinary_envelope_in_fee_bump_question_is_explained_not_guessed():
    context = build_chat_fee_bump_context(f"Can you fee bump this? {build_inner().to_xdr()}")
    assert context is not None
    assert "FEE-BUMP ENVELOPE NOT DECODED" in context
    assert "ordinary transaction envelope" in context
    assert "do not guess" in context


def test_ordinary_envelope_without_fee_bump_question_is_ignored():
    assert build_chat_fee_bump_context(f"What is this? {build_inner().to_xdr()}") is None


def test_corrupt_envelope_in_chat_degrades_gracefully():
    context = build_chat_fee_bump_context("fee bump AAAABQ" + "A" * 80 + "!")
    assert context is not None
    assert "FEE-BUMP ENVELOPE NOT DECODED" in context


def test_secret_key_in_chat_is_refused_and_not_echoed():
    secret = FEE_KP.secret
    context = build_chat_fee_bump_context(f"Here is my key {secret}, please fee bump {build_fee_bump().to_xdr()}")
    assert context is not None
    assert SECRET_KEY_WARNING in context
    assert secret not in context
    assert "DECODED FEE-BUMP ENVELOPE" not in context
