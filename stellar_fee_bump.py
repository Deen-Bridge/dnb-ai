"""Explain Stellar fee-bump transactions (CAP-15) without touching the network.

A fee bump wraps an already-signed transaction in an outer envelope whose fee
is paid by a different account, the *fee source*. Deen Bridge users meet it
when a sponsor covers their network fees, or when a transaction stuck on a
low fee is resubmitted with a higher one.

This module decodes a fee-bump envelope the caller supplies and explains it in
plain terms: who pays, whose transaction is being wrapped, the fee ceiling in
stroops and XLM, and whether the fee satisfies the protocol's fee-bump rules.
Everything is local and read-only. Nothing is fetched from Horizon, nothing is
signed or submitted, and input that looks like a secret key is refused before
any decoding happens.
"""

import logging
import re
from decimal import Decimal
from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel, Field
from stellar_sdk import Keypair, Network
from stellar_sdk.decorated_signature import DecoratedSignature
from stellar_sdk.exceptions import BadSignatureError
from stellar_sdk.fee_bump_transaction_envelope import FeeBumpTransactionEnvelope
from stellar_sdk.transaction_builder import MIN_BASE_FEE
from stellar_sdk.transaction_envelope import TransactionEnvelope

from errors import APIException
from stellar import SECRET_KEY_WARNING, STELLAR_NETWORK, contains_secret_key

logger = logging.getLogger(__name__)

router = APIRouter(tags=["stellar"])

CanonicalNetwork = Literal["testnet", "public"]

# Transaction hashes, and therefore signatures, commit to the network
# passphrase. The same envelope bytes hash differently on each network, which
# is what lets us tell a caller their envelope was signed for the other one.
NETWORK_PASSPHRASES: dict[CanonicalNetwork, str] = {
    "testnet": Network.TESTNET_NETWORK_PASSPHRASE,
    "public": Network.PUBLIC_NETWORK_PASSPHRASE,
}

STROOPS_PER_XLM = Decimal("10000000")

# Comfortably above any envelope the network will accept, while still keeping
# a pasted megabyte of junk away from the XDR decoder.
MAX_ENVELOPE_XDR_CHARS = 200_000

# 100 XLM per operation: far beyond any surge price, and it keeps the request
# from carrying absurd numbers into the explanation.
MAX_BASE_FEE_STROOPS = 1_000_000_000

_BASE64_XDR = re.compile(r"[A-Za-z0-9+/]+={0,2}")
_WHITESPACE = re.compile(r"\s+")

SignatureStatus = Literal["verified", "other_network", "invalid", "not_found"]

FEE_BUMP_DISCLAIMER = (
    "This explanation decodes the envelope you supplied; it does not check account balances, "
    "sequence numbers, or the live network fee, and nothing was signed or submitted. "
    "Fees are dynamic, so confirm in your wallet before signing."
)


def canonical_network(network: str | None = None) -> CanonicalNetwork:
    """Map a configured or requested network name onto a known passphrase key.

    ``mainnet`` is an alias for ``public``; anything unrecognised falls back to
    testnet, mirroring ``stellar.horizon_url``.
    """
    name = (network or STELLAR_NETWORK).strip().lower()
    return "public" if name in ("public", "mainnet") else "testnet"


def format_xlm(stroops: int) -> str:
    """Format a stroop amount as XLM with Stellar's seven decimal places."""
    return f"{Decimal(stroops) / STROOPS_PER_XLM:.7f}"


def explorer_tx_url(tx_hash: str, network: CanonicalNetwork) -> str:
    return f"https://stellar.expert/explorer/{network}/tx/{tx_hash}"


class FeeBumpDecodeError(ValueError):
    """The input is not a decodable fee-bump envelope; *hint* says what to do."""

    def __init__(self, message: str, hint: str) -> None:
        super().__init__(message)
        self.hint = hint


def decode_fee_bump_envelope(envelope_xdr: str, network: CanonicalNetwork) -> FeeBumpTransactionEnvelope:
    """Parse base64 XDR into a fee-bump envelope bound to *network*.

    Whitespace is stripped first, since envelopes copied out of wallets and
    terminals often arrive wrapped across lines.
    """
    cleaned = _WHITESPACE.sub("", envelope_xdr or "")
    if not cleaned:
        raise FeeBumpDecodeError(
            "No transaction envelope was provided.",
            hint="Paste the base64 XDR of the fee-bump transaction envelope.",
        )
    if len(cleaned) > MAX_ENVELOPE_XDR_CHARS:
        raise FeeBumpDecodeError(
            "The transaction envelope is too large to be a valid Stellar transaction.",
            hint=f"Envelopes are limited to {MAX_ENVELOPE_XDR_CHARS:,} base64 characters here.",
        )
    if not _BASE64_XDR.fullmatch(cleaned):
        raise FeeBumpDecodeError(
            "The transaction envelope is not base64-encoded XDR.",
            hint="Copy the envelope XDR exactly as your wallet or SDK exported it.",
        )

    passphrase = NETWORK_PASSPHRASES[network]
    try:
        if FeeBumpTransactionEnvelope.is_fee_bump_transaction_envelope(cleaned):
            return FeeBumpTransactionEnvelope.from_xdr(cleaned, passphrase)
        # Decode it anyway so the caller learns *why* it was rejected: a valid
        # ordinary transaction deserves a different answer from random bytes.
        TransactionEnvelope.from_xdr(cleaned, passphrase)
    except (ValueError, EOFError):
        raise FeeBumpDecodeError(
            "The transaction envelope could not be decoded.",
            hint="The XDR may be truncated or corrupted. Export it again from your wallet or SDK.",
        ) from None
    raise FeeBumpDecodeError(
        "This is an ordinary transaction envelope, not a fee bump.",
        hint=(
            "A fee bump wraps an already-signed transaction with a new fee paid by another account. "
            "Paste the outer fee-bump envelope (its XDR starts with 'AAAABQ')."
        ),
    )


def _signature_verifies(keypair: Keypair, tx_hash: bytes, signature: bytes) -> bool:
    try:
        keypair.verify(tx_hash, signature)
    except BadSignatureError:
        return False
    return True


def signature_status(
    account_id: str,
    signatures: list[DecoratedSignature],
    hashes: dict[CanonicalNetwork, bytes],
    network: CanonicalNetwork,
) -> SignatureStatus:
    """Whether *account_id*'s own key signed the transaction for *network*.

    Only the account's master key is checked. Multisig accounts and
    pre-authorized signers are legitimate, so ``not_found`` means "no direct
    signature from this key", not "invalid transaction".
    """
    keypair = Keypair.from_public_key(account_id)
    hint = keypair.signature_hint()
    candidates = [sig.signature for sig in signatures if sig.signature_hint == hint]
    if not candidates:
        return "not_found"

    def verifies_for(target: CanonicalNetwork) -> bool:
        return any(_signature_verifies(keypair, hashes[target], sig) for sig in candidates)

    if verifies_for(network):
        return "verified"
    if any(verifies_for(other) for other in hashes if other != network):
        return "other_network"
    return "invalid"


class FeeBumpExplainRequest(BaseModel):
    envelope_xdr: str = Field(
        ...,
        min_length=1,
        max_length=MAX_ENVELOPE_XDR_CHARS,
        description="Base64 XDR of a signed or unsigned fee-bump transaction envelope. Never a secret key.",
    )
    network: Literal["testnet", "public", "mainnet"] | None = Field(
        None,
        description="Network the envelope was built for. Defaults to this service's configured network.",
    )
    base_fee_stroops: int = Field(
        MIN_BASE_FEE,
        ge=MIN_BASE_FEE,
        le=MAX_BASE_FEE_STROOPS,
        description="Per-operation base fee to check against, in stroops. 100 unless the network is surge pricing.",
    )


class FeeBumpExplanation(BaseModel):
    network: CanonicalNetwork
    fee_bump_hash: str = Field(..., description="Hash of the outer fee-bump transaction on this network")
    inner_transaction_hash: str = Field(..., description="Hash of the wrapped transaction on this network")
    explorer_url: str = Field(..., description="stellar.expert link; resolves only once the fee bump is submitted")
    fee_source: str = Field(..., description="Account (G...) that pays the fee")
    fee_source_is_muxed: bool
    inner_source: str = Field(..., description="Account (G...) whose transaction is wrapped")
    inner_source_is_muxed: bool
    inner_sequence: str
    operation_count: int
    max_fee_stroops: str = Field(..., description="Most the fee source can be charged, in stroops")
    max_fee_xlm: str
    inner_fee_stroops: str = Field(..., description="Fee written into the inner transaction; not charged")
    soroban_resource_fee_stroops: str
    base_fee_stroops: int
    minimum_fee_stroops: str = Field(..., description="Smallest fee-bump fee the network accepts at this base fee")
    meets_minimum_fee: bool
    meets_inner_fee_rate: bool = Field(
        ..., description="Fee-bump fee per operation is at least the inner transaction's fee per operation"
    )
    fee_source_signature: SignatureStatus
    inner_source_signature: SignatureStatus
    explanation: list[str]
    disclaimer: str = FEE_BUMP_DISCLAIMER


_SIGNATURE_WORDING: dict[SignatureStatus, str] = {
    "verified": "has a valid signature from {who} for {network}",
    "other_network": (
        "has a signature from {who} that is only valid on {other} — it was signed for the wrong network "
        "and {network} will reject it"
    ),
    "invalid": "has a signature that claims to be from {who} but does not verify on any known network",
    "not_found": "has no signature from {who}'s own key (it may be unsigned, or use other signers)",
}


def _describe_signature(status: SignatureStatus, subject: str, who: str, network: CanonicalNetwork) -> str:
    other = "public" if network == "testnet" else "testnet"
    return f"The {subject} " + _SIGNATURE_WORDING[status].format(who=who, network=network, other=other) + "."


def explain_fee_bump(
    envelope_xdr: str,
    network: str | None = None,
    base_fee_stroops: int = MIN_BASE_FEE,
) -> FeeBumpExplanation:
    """Decode *envelope_xdr* and explain the fee bump it contains.

    Pure and offline: no Horizon reads, no signing, no submission. Raises
    FeeBumpDecodeError when the input is not a fee-bump envelope.
    """
    target = canonical_network(network)
    envelope = decode_fee_bump_envelope(envelope_xdr, target)
    fee_bump = envelope.transaction
    inner_envelope = fee_bump.inner_transaction_envelope
    inner = inner_envelope.transaction

    # Rebinding the decoded transactions to every passphrase is cheap and
    # gives us each network's hash for the signature checks.
    outer_hashes: dict[CanonicalNetwork, bytes] = {
        name: FeeBumpTransactionEnvelope(fee_bump, passphrase).hash()
        for name, passphrase in NETWORK_PASSPHRASES.items()
    }
    inner_hashes: dict[CanonicalNetwork, bytes] = {
        name: TransactionEnvelope(inner, passphrase).hash() for name, passphrase in NETWORK_PASSPHRASES.items()
    }

    operation_count = len(inner.operations)
    # Soroban resource fees pass through untouched; CAP-15's fee-rate rules
    # apply only to the inclusion fee that remains.
    resource_fee = inner.soroban_data.resource_fee.int64 if inner.soroban_data else 0
    outer_inclusion_fee = fee_bump.fee - resource_fee
    inner_inclusion_fee = inner.fee - resource_fee
    minimum_fee = base_fee_stroops * (operation_count + 1) + resource_fee
    meets_minimum_fee = fee_bump.fee >= minimum_fee
    # outer / (ops + 1) >= inner / ops, cross-multiplied to stay in integers.
    meets_inner_fee_rate = outer_inclusion_fee * operation_count >= inner_inclusion_fee * (operation_count + 1)

    fee_source = fee_bump.fee_source.account_id
    inner_source = inner.source.account_id
    fee_source_signature = signature_status(fee_source, envelope.signatures, outer_hashes, target)
    inner_source_signature = signature_status(inner_source, inner_envelope.signatures, inner_hashes, target)

    explanation = [
        f"This is a fee-bump transaction for the Stellar {target} network. "
        f"Account {fee_source} pays the fee for a transaction from {inner_source}.",
        f"The fee source can be charged at most {fee_bump.fee} stroops ({format_xlm(fee_bump.fee)} XLM). "
        "Fees are always paid in XLM, and the network usually charges less than this maximum.",
        f"The inner transaction's own fee of {inner.fee} stroops is not charged; "
        "only the fee-bump fee is paid, and it is paid by the fee source.",
        f"The inner transaction keeps its source account, sequence number {inner.sequence}, "
        f"and {operation_count} operation(s). Wrapping it does not change what it does.",
    ]
    if fee_source == inner_source:
        explanation.append("The fee source and the inner source are the same account, so this simply raises its fee.")
    if operation_count == 0:
        explanation.append("The inner transaction has no operations, so the network will reject it.")
    if resource_fee:
        explanation.append(
            f"It includes a Soroban resource fee of {resource_fee} stroops, carried over from the inner transaction."
        )

    if meets_minimum_fee:
        explanation.append(
            f"At a base fee of {base_fee_stroops} stroops, the minimum fee-bump fee is {minimum_fee} stroops "
            f"(base fee × {operation_count + 1}, one more than the operation count), and this fee meets it."
        )
    else:
        explanation.append(
            f"At a base fee of {base_fee_stroops} stroops, the minimum fee-bump fee is {minimum_fee} stroops "
            f"(base fee × {operation_count + 1}), so this fee of {fee_bump.fee} stroops is too low and the "
            "network will reject it."
        )
    if not meets_inner_fee_rate:
        explanation.append(
            "Its fee per operation is lower than the inner transaction's, which the protocol forbids: "
            "a fee bump may only raise the fee rate."
        )

    explanation.append(_describe_signature(fee_source_signature, "fee bump", "the fee source", target))
    explanation.append(_describe_signature(inner_source_signature, "inner transaction", "its source account", target))
    explanation.append(
        "The fee source signs only the outer envelope and never needs the inner account's secret key; "
        "each party signs in its own wallet."
    )

    return FeeBumpExplanation(
        network=target,
        fee_bump_hash=outer_hashes[target].hex(),
        inner_transaction_hash=inner_hashes[target].hex(),
        explorer_url=explorer_tx_url(outer_hashes[target].hex(), target),
        fee_source=fee_source,
        fee_source_is_muxed=fee_bump.fee_source.account_muxed_id is not None,
        inner_source=inner_source,
        inner_source_is_muxed=inner.source.account_muxed_id is not None,
        inner_sequence=str(inner.sequence),
        operation_count=operation_count,
        max_fee_stroops=str(fee_bump.fee),
        max_fee_xlm=format_xlm(fee_bump.fee),
        inner_fee_stroops=str(inner.fee),
        soroban_resource_fee_stroops=str(resource_fee),
        base_fee_stroops=base_fee_stroops,
        minimum_fee_stroops=str(minimum_fee),
        meets_minimum_fee=meets_minimum_fee,
        meets_inner_fee_rate=meets_inner_fee_rate,
        fee_source_signature=fee_source_signature,
        inner_source_signature=inner_source_signature,
        explanation=explanation,
    )


@router.post("/stellar/fee-bump/explain", response_model=FeeBumpExplanation)
async def explain_fee_bump_endpoint(body: FeeBumpExplainRequest) -> FeeBumpExplanation:
    """Explain a fee-bump transaction envelope. Offline: nothing is signed or submitted."""
    if contains_secret_key(body.envelope_xdr):
        logger.warning("Secret-key-shaped string sent to the fee-bump explainer; refusing it")
        raise APIException(
            status_code=400,
            detail="This looks like a Stellar secret key, not a transaction envelope. It was not used or stored.",
            hint=(
                "Never share a secret key. Treat it as compromised and move your funds to a new wallet. "
                "This endpoint only needs the base64 XDR of a fee-bump transaction envelope."
            ),
        )
    try:
        return explain_fee_bump(body.envelope_xdr, body.network, body.base_fee_stroops)
    except FeeBumpDecodeError as exc:
        # The envelope itself is never logged: it is user-supplied and may be large.
        logger.info("Fee-bump explain rejected input: %s", exc)
        raise APIException(status_code=400, detail=str(exc), hint=exc.hint) from None


# ---------------------------------------------------------------------------
# Chat integration
# ---------------------------------------------------------------------------

_FEE_BUMP_QUESTION = re.compile(
    r"\bfee[\s-]?bump"
    r"|\bbump(?:s|ed|ing)?\s+(?:up\s+)?(?:the\s+|a\s+|my\s+)?(?:transaction\s+|network\s+)?fees?\b"
    r"|\bfee[\s-]?sponsor"
    r"|\bsponsor(?:s|ed|ing)?\s+(?:the\s+|a\s+|my\s+)?(?:transaction\s+|network\s+)?fees?\b",
    re.I,
)

# An envelope's first four bytes are its type, so the base64 prefix gives it
# away: "AAAABQ" is a fee bump (type 5), "AAAAAg" and "AAAAAA" are ordinary
# transactions (types 2 and 0). A fee-bump prefix alone marks a message as a
# fee-bump question; an ordinary one is only decoded when the user asked
# about fee bumps, so they can be told it is the wrong kind of envelope.
_ENVELOPE = r"(?<![A-Za-z0-9+/])AAAA{prefix}[A-Za-z0-9+/]{{60,}}={{0,2}}"
_FEE_BUMP_ENVELOPE = re.compile(_ENVELOPE.format(prefix="BQ"))
_ANY_ENVELOPE = re.compile(_ENVELOPE.format(prefix="(?:BQ|Ag|AA)"))

FEE_BUMP_CONCEPTS = (
    "STELLAR FEE BUMP EXPLAINER (offline facts; nothing was signed or submitted):\n"
    "- A fee-bump transaction (CAP-15, protocol 13+) wraps an already-signed transaction in an outer "
    "envelope whose fee is paid by a different account, the fee source.\n"
    "- The inner transaction is unchanged: same source account, sequence number, operations, and "
    "signatures. Only the fee and who pays it change.\n"
    "- The fee source signs only the outer envelope. Nobody ever needs another person's secret key; "
    "each party signs in their own wallet.\n"
    "- Fees are paid in XLM, never USDC. The fee-bump fee is a maximum; the network usually charges "
    "less. The inner transaction's own fee is not charged.\n"
    "- The fee-bump fee must be at least the base fee (100 stroops minimum) × (inner operations + 1), "
    "and its fee per operation must be at least the inner transaction's.\n"
    "- Common uses: a sponsor (for example a course creator or app) covering a learner's network fee, "
    "or resubmitting a transaction stuck on a fee that was too low.\n"
    "- A fee bump has its own hash, distinct from the inner transaction's hash. Hashes and signatures "
    "are bound to one network, so a testnet envelope is invalid on the public network and vice versa."
)

FEE_BUMP_GUARDRAILS = (
    "Explain fee bumps using these facts. Never ask for, accept, or repeat a secret key or seed phrase. "
    "Do not claim that Deen Bridge or this assistant can sign, submit, or pay for a transaction — it "
    "cannot. Do not quote a live network fee; fees are dynamic, so tell the user to confirm in their wallet."
)


def is_fee_bump_question(text: str) -> bool:
    return bool(_FEE_BUMP_QUESTION.search(text or "") or _FEE_BUMP_ENVELOPE.search(text or ""))


def build_fee_bump_prompt_block(explanation: FeeBumpExplanation) -> str:
    """Render a decoded envelope as grounding for the model."""
    lines = [
        "",
        f"DECODED FEE-BUMP ENVELOPE (supplied by the user; decoded for the {explanation.network} network):",
        f"- Fee bump hash: {explanation.fee_bump_hash}",
        f"- Inner transaction hash: {explanation.inner_transaction_hash}",
        f"- Explorer (only if submitted): {explanation.explorer_url}",
        f"- Fee source (pays): {explanation.fee_source}",
        f"- Inner source: {explanation.inner_source}",
        f"- Operations in inner transaction: {explanation.operation_count}",
        f"- Max fee: {explanation.max_fee_stroops} stroops ({explanation.max_fee_xlm} XLM)",
        f"- Inner fee (not charged): {explanation.inner_fee_stroops} stroops",
        f"- Minimum fee at {explanation.base_fee_stroops} stroops base fee: {explanation.minimum_fee_stroops} stroops",
        f"- Meets minimum fee: {'yes' if explanation.meets_minimum_fee else 'no'}",
        f"- Meets inner fee rate: {'yes' if explanation.meets_inner_fee_rate else 'no'}",
        f"- Fee source signature: {explanation.fee_source_signature}",
        f"- Inner source signature: {explanation.inner_source_signature}",
        "",
        "Plain-language findings:",
        *(f"- {line}" for line in explanation.explanation),
        "",
        "Quote these decoded figures exactly. Do not infer balances, sequence validity, or whether "
        "the transaction succeeded — none of that was checked.",
    ]
    return "\n".join(lines)


def build_chat_fee_bump_context(prompt: str) -> str | None:
    """Grounding for fee-bump questions in chat, or None for anything else.

    Detection is offline. When the message carries a fee-bump envelope it is
    decoded locally for the service's network; otherwise the model gets the
    concept facts only.
    """
    text = prompt or ""
    if not is_fee_bump_question(text):
        return None

    if contains_secret_key(text):
        logger.warning("Secret-key-shaped string in a fee-bump chat turn; refusing to decode")
        return FEE_BUMP_CONCEPTS + SECRET_KEY_WARNING

    blocks = [FEE_BUMP_CONCEPTS]
    match = _FEE_BUMP_ENVELOPE.search(text) or _ANY_ENVELOPE.search(text)
    if match:
        try:
            blocks.append(build_fee_bump_prompt_block(explain_fee_bump(match.group(0))))
        except FeeBumpDecodeError as exc:
            logger.info("Fee-bump envelope in chat could not be decoded: %s", exc)
            blocks.append(
                f"\nFEE-BUMP ENVELOPE NOT DECODED: {exc} {exc.hint} "
                "Tell the user plainly and do not guess what the envelope contains."
            )
    blocks.append(FEE_BUMP_GUARDRAILS)
    return "\n\n".join(blocks)
