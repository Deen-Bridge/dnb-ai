"""Guard against Stellar network passphrase mismatches, without touching the network.

Every Stellar network is named by a passphrase, and a transaction's hash, which
is what every signature covers, is computed over that passphrase. Nothing in
an envelope's bytes says which network it was built for. A wallet left on the
public network, an RPC provider that serves testnet, or a passphrase retyped
with a stray capital letter all produce envelopes that decode perfectly and are
still rejected by the network the user meant to use.

This module catches those mismatches before anything is signed or submitted.
It compares a passphrase reported by a wallet, by Horizon (``network_passphrase``
on its root endpoint) or by Soroban RPC (``getNetwork``) with the network Deen
Bridge expects, and works out which network a signed envelope's signatures are
actually valid on.

Everything is local and read-only. Nothing is fetched from Horizon or Soroban
RPC, nothing is signed or submitted, and input that looks like a secret key is
refused before any decoding happens.
"""

import logging
import re
from typing import Any, Literal

from fastapi import APIRouter
from pydantic import BaseModel, Field, model_validator
from stellar_sdk import Keypair, Network
from stellar_sdk.decorated_signature import DecoratedSignature
from stellar_sdk.fee_bump_transaction_envelope import FeeBumpTransactionEnvelope
from stellar_sdk.transaction import Transaction
from stellar_sdk.transaction_envelope import TransactionEnvelope

from errors import APIException
from stellar import contains_secret_key
from stellar_fee_bump import (
    MAX_ENVELOPE_XDR_CHARS,
    NETWORK_PASSPHRASES,
    CanonicalNetwork,
    _signature_verifies,
    canonical_network,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["stellar"])

# Deen Bridge only runs on testnet and public, but the other well-known
# passphrases are worth recognising: "your wallet is on futurenet" is a far
# more useful answer than "unknown network".
KNOWN_PASSPHRASES: dict[str, str] = {
    **NETWORK_PASSPHRASES,
    "futurenet": Network.FUTURENET_NETWORK_PASSPHRASE,
    "sandbox": Network.SANDBOX_NETWORK_PASSPHRASE,
    "standalone": Network.STANDALONE_NETWORK_PASSPHRASE,
}

# The longest well-known passphrase is under 50 characters.
MAX_PASSPHRASE_CHARS = 128

_BASE64_XDR = re.compile(r"[A-Za-z0-9+/]+={0,2}")
_WHITESPACE = re.compile(r"\s+")

PassphraseStatus = Literal["match", "other_network", "near_miss", "unknown"]
SignatureVerdict = Literal["verified", "other_network", "invalid", "unknown_signer"]
SignatureLayer = Literal["transaction", "fee_bump", "inner_transaction"]
Severity = Literal["error", "warning", "info"]
FindingCode = Literal[
    "passphrase_mismatch",
    "passphrase_near_miss",
    "unknown_passphrase",
    "signed_for_other_network",
    "invalid_signature",
    "unverifiable_signature",
    "unsigned_envelope",
]

_LAYER_NAMES: dict[SignatureLayer, str] = {
    "transaction": "transaction",
    "fee_bump": "fee bump",
    "inner_transaction": "inner transaction",
}

NETWORK_GUARD_DISCLAIMER = (
    "This compares the passphrase and envelope you supplied with the network Deen Bridge expects; nothing was "
    "fetched from Horizon or Soroban RPC, and nothing was signed or submitted. A matching network does not mean "
    "a transaction will succeed: balances, sequence numbers and fees were not checked."
)


class NetworkGuardError(ValueError):
    """The input cannot be checked; *hint* says what to do."""

    def __init__(self, message: str, hint: str) -> None:
        super().__init__(message)
        self.hint = hint


# ---------------------------------------------------------------------------
# Passphrases
# ---------------------------------------------------------------------------


class PassphraseCheck(BaseModel):
    status: PassphraseStatus = Field(
        ..., description="match, other_network, near_miss (differs only in case or spacing) or unknown"
    )
    reported_network: str | None = Field(
        None, description="The well-known network the passphrase names, or resembles for a near miss"
    )


def _loosely(text: str) -> str:
    return " ".join(text.split()).casefold()


def check_passphrase(reported: str | None, network: CanonicalNetwork) -> PassphraseCheck | None:
    """Classify a reported passphrase against *network*; None when nothing was reported.

    Surrounding whitespace is a copy-and-paste artifact and is ignored. Anything
    else is compared byte for byte, exactly as the network hashes it, so a
    passphrase that differs only in letter case or inner spacing is a
    ``near_miss``: recognisable to a person, but a different network to Stellar.
    """
    cleaned = (reported or "").strip()
    if not cleaned:
        return None
    for name, passphrase in KNOWN_PASSPHRASES.items():
        if cleaned == passphrase:
            return PassphraseCheck(status="match" if name == network else "other_network", reported_network=name)
    loose = _loosely(cleaned)
    resembles = next((name for name, passphrase in KNOWN_PASSPHRASES.items() if _loosely(passphrase) == loose), None)
    if resembles is not None:
        return PassphraseCheck(status="near_miss", reported_network=resembles)
    return PassphraseCheck(status="unknown")


def describe_reported_network(check: PassphraseCheck) -> str:
    """Name what a mismatched passphrase points at, without echoing caller text."""
    if check.status == "near_miss":
        return f"a passphrase that differs from the {check.reported_network} one only in letter case or spacing"
    if check.reported_network in NETWORK_PASSPHRASES:
        return f"the {check.reported_network} network"
    if check.reported_network is not None:
        return f"the {check.reported_network} network, which Deen Bridge does not use"
    return "a network that is neither testnet nor public"


# ---------------------------------------------------------------------------
# Envelopes
# ---------------------------------------------------------------------------


class SignatureCheck(BaseModel):
    layer: SignatureLayer = Field(..., description="Which part of the envelope carries the signature")
    signer: str | None = Field(None, description="The envelope account (G...) whose key made it, when known")
    verdict: SignatureVerdict
    valid_on: list[str] = Field(default_factory=list, description="Well-known networks the signature verifies on")


class EnvelopeCheck(BaseModel):
    kind: Literal["transaction", "fee_bump"]
    transaction_hash: str = Field(..., description="Hash of the envelope on the expected network")
    signatures: list[SignatureCheck]
    signed_for: list[str] = Field(..., description="Every well-known network at least one signature is valid on")


def decode_envelope(envelope_xdr: str, network: CanonicalNetwork) -> TransactionEnvelope | FeeBumpTransactionEnvelope:
    """Parse base64 XDR into an ordinary or fee-bump envelope bound to *network*."""
    cleaned = _WHITESPACE.sub("", envelope_xdr or "")
    if not cleaned:
        raise NetworkGuardError(
            "The transaction envelope is empty.", hint="Paste the base64 XDR your wallet or SDK exported."
        )
    if len(cleaned) > MAX_ENVELOPE_XDR_CHARS:
        raise NetworkGuardError(
            "The transaction envelope is too large to be a valid Stellar transaction.",
            hint=f"Envelopes are limited to {MAX_ENVELOPE_XDR_CHARS:,} base64 characters here.",
        )
    if not _BASE64_XDR.fullmatch(cleaned):
        raise NetworkGuardError(
            "The transaction envelope is not base64-encoded XDR.",
            hint="Copy the envelope XDR exactly as your wallet or SDK exported it.",
        )
    passphrase = NETWORK_PASSPHRASES[network]
    try:
        if FeeBumpTransactionEnvelope.is_fee_bump_transaction_envelope(cleaned):
            return FeeBumpTransactionEnvelope.from_xdr(cleaned, passphrase)
        return TransactionEnvelope.from_xdr(cleaned, passphrase)
    except (ValueError, EOFError):
        raise NetworkGuardError(
            "The transaction envelope could not be decoded.",
            hint="The XDR may be truncated or corrupted. Export it again from your wallet or SDK.",
        ) from None


def _transaction_signers(transaction: Transaction) -> list[str]:
    """Accounts that may sign a transaction directly: its source and any operation sources."""
    signers = [transaction.source.account_id]
    signers += [operation.source.account_id for operation in transaction.operations if operation.source is not None]
    return list(dict.fromkeys(signers))


def check_signatures(
    layer: SignatureLayer,
    signatures: list[DecoratedSignature],
    signers: list[str],
    hashes: dict[str, bytes],
    network: CanonicalNetwork,
) -> list[SignatureCheck]:
    """Work out which network each signature in one layer of an envelope is valid on.

    Only the keys of the envelope's own accounts can be tried. A signature from
    any other key (a multisig co-signer, say) is legitimate but unverifiable
    here, and is reported as ``unknown_signer``.
    """
    by_hint: dict[bytes, list[Keypair]] = {}
    for account in signers:
        keypair = Keypair.from_public_key(account)
        by_hint.setdefault(keypair.signature_hint(), []).append(keypair)

    checks = []
    for decorated in signatures:
        candidates = by_hint.get(decorated.signature_hint, [])
        if not candidates:
            checks.append(SignatureCheck(layer=layer, verdict="unknown_signer"))
            continue
        # Four-byte hints can collide, so try every key that shares one.
        signer, valid_on = candidates[0].public_key, []
        for keypair in candidates:
            valid_on = [
                name for name, tx_hash in hashes.items() if _signature_verifies(keypair, tx_hash, decorated.signature)
            ]
            if valid_on:
                signer = keypair.public_key
                break
        if not valid_on:
            verdict: SignatureVerdict = "invalid"
        elif network in valid_on:
            verdict = "verified"
        else:
            verdict = "other_network"
        checks.append(SignatureCheck(layer=layer, signer=signer, verdict=verdict, valid_on=valid_on))
    return checks


def check_envelope(envelope_xdr: str, network: CanonicalNetwork) -> EnvelopeCheck:
    """Decode an envelope and check every signature against every well-known network."""
    envelope = decode_envelope(envelope_xdr, network)

    if isinstance(envelope, FeeBumpTransactionEnvelope):
        fee_bump = envelope.transaction
        inner_envelope = fee_bump.inner_transaction_envelope
        inner = inner_envelope.transaction
        outer_hashes = {
            name: FeeBumpTransactionEnvelope(fee_bump, passphrase).hash()
            for name, passphrase in KNOWN_PASSPHRASES.items()
        }
        inner_hashes = {
            name: TransactionEnvelope(inner, passphrase).hash() for name, passphrase in KNOWN_PASSPHRASES.items()
        }
        signatures = check_signatures(
            "fee_bump", envelope.signatures, [fee_bump.fee_source.account_id], outer_hashes, network
        )
        signatures += check_signatures(
            "inner_transaction", inner_envelope.signatures, _transaction_signers(inner), inner_hashes, network
        )
        kind: Literal["transaction", "fee_bump"] = "fee_bump"
        transaction_hash = outer_hashes[network]
    else:
        transaction = envelope.transaction
        hashes = {
            name: TransactionEnvelope(transaction, passphrase).hash() for name, passphrase in KNOWN_PASSPHRASES.items()
        }
        signatures = check_signatures(
            "transaction", envelope.signatures, _transaction_signers(transaction), hashes, network
        )
        kind = "transaction"
        transaction_hash = hashes[network]

    signed_for = [name for name in KNOWN_PASSPHRASES if any(name in check.valid_on for check in signatures)]
    return EnvelopeCheck(
        kind=kind, transaction_hash=transaction_hash.hex(), signatures=signatures, signed_for=signed_for
    )


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------


class GuardFinding(BaseModel):
    code: FindingCode
    severity: Severity = Field(..., description="error: do not sign or submit; warning: check first; info: note")
    message: str


class NetworkGuardReport(BaseModel):
    network: CanonicalNetwork
    expected_passphrase: str = Field(..., description="The exact passphrase of the network Deen Bridge expects")
    passphrase: PassphraseCheck | None = Field(None, description="Absent when no passphrase was supplied")
    envelope: EnvelopeCheck | None = Field(None, description="Absent when no envelope was supplied")
    mismatch_detected: bool = Field(..., description="True when any check found an error")
    network_confirmed: bool = Field(
        ..., description="True when the passphrase matched or a signature verified on the expected network"
    )
    findings: list[GuardFinding]
    explanation: list[str]
    disclaimer: str = NETWORK_GUARD_DISCLAIMER


def _passphrase_findings(check: PassphraseCheck, network: CanonicalNetwork) -> list[GuardFinding]:
    if check.status == "match":
        return []
    if check.status == "near_miss":
        message = (
            f"The reported passphrase looks like the {check.reported_network} one but differs in letter case or "
            "spacing. Passphrases are compared byte for byte, so anything signed with it is valid on no real network."
        )
        if check.reported_network != network:
            message += f" Deen Bridge expects {network} here."
        return [GuardFinding(code="passphrase_near_miss", severity="error", message=message)]
    if check.status == "other_network":
        return [
            GuardFinding(
                code="passphrase_mismatch",
                severity="error",
                message=(
                    f"The reported passphrase is for {describe_reported_network(check)}, but Deen Bridge expects "
                    f"{network}. Anything signed on it will be rejected on {network}; switch the wallet or RPC "
                    f"provider to {network} first."
                ),
            )
        ]
    return [
        GuardFinding(
            code="unknown_passphrase",
            severity="error",
            message=(
                f"The reported passphrase belongs to {describe_reported_network(check)}, and Deen Bridge expects "
                f"{network}. Check the network setting of the wallet or RPC provider that reported it."
            ),
        )
    ]


def _envelope_findings(check: EnvelopeCheck, network: CanonicalNetwork) -> list[GuardFinding]:
    findings = []
    if not check.signatures:
        findings.append(
            GuardFinding(
                code="unsigned_envelope",
                severity="info",
                message=(
                    "The envelope carries no signatures, and its bytes do not name a network, so it cannot show "
                    "which network it was built for. Check the wallet's network before signing it."
                ),
            )
        )
    for signature in check.signatures:
        where = _LAYER_NAMES[signature.layer]
        if signature.verdict == "other_network":
            findings.append(
                GuardFinding(
                    code="signed_for_other_network",
                    severity="error",
                    message=(
                        f"The {where} signature from {signature.signer} is valid only on "
                        f"{', '.join(signature.valid_on)}, not {network}. The envelope was signed for the wrong "
                        f"network and {network} will reject it; sign it again with the wallet set to {network}."
                    ),
                )
            )
        elif signature.verdict == "invalid":
            findings.append(
                GuardFinding(
                    code="invalid_signature",
                    severity="error",
                    message=(
                        f"The {where} has a signature that claims to be from {signature.signer} but verifies on no "
                        "well-known network. It may be corrupted, or made for a private network."
                    ),
                )
            )

    unknown = [signature for signature in check.signatures if signature.verdict == "unknown_signer"]
    for layer in dict.fromkeys(signature.layer for signature in unknown):
        count = sum(1 for signature in unknown if signature.layer == layer)
        findings.append(
            GuardFinding(
                code="unverifiable_signature",
                severity="info",
                message=(
                    f"{count} {_LAYER_NAMES[layer]} signature(s) came from keys other than the envelope's own "
                    "accounts, such as multisig co-signers, so their network could not be checked."
                ),
            )
        )
    return findings


def _explain(
    network: CanonicalNetwork,
    passphrase: PassphraseCheck | None,
    envelope: EnvelopeCheck | None,
    mismatch: bool,
    confirmed: bool,
) -> list[str]:
    explanation = [
        f'Deen Bridge expects the Stellar {network} network, whose passphrase is "{NETWORK_PASSPHRASES[network]}".'
    ]
    if passphrase is not None and passphrase.status == "match":
        explanation.append(f"The reported passphrase matches {network} exactly.")
    if envelope is not None:
        kind = "fee-bump transaction" if envelope.kind == "fee_bump" else "transaction"
        explanation.append(
            f"The envelope is a {kind} whose hash on {network} is {envelope.transaction_hash}. "
            "Each signature was checked against the transaction hash of every well-known network."
        )
        if envelope.signed_for:
            explanation.append(f"Its signatures are valid on: {', '.join(envelope.signed_for)}.")
    if mismatch:
        explanation.append(
            "A network mismatch was found. Do not sign or submit anything until the wallet, the RPC provider and "
            f"Deen Bridge all agree on {network}."
        )
    elif confirmed:
        explanation.append(f"Everything supplied agrees with {network}.")
    else:
        explanation.append(
            "Nothing supplied could confirm the network. Compare the wallet's passphrase with the one above "
            "before signing."
        )
    return explanation


def guard_network(
    network: str | None = None,
    network_passphrase: str | None = None,
    envelope_xdr: str | None = None,
) -> NetworkGuardReport:
    """Check a reported passphrase and/or a signed envelope against the expected network.

    Pure and offline: no Horizon or RPC reads, no signing, no submission.
    Mismatches are reported as findings; NetworkGuardError is raised only when
    the input itself cannot be checked.
    """
    target = canonical_network(network)
    has_envelope = bool(envelope_xdr and envelope_xdr.strip())
    passphrase = check_passphrase(network_passphrase, target)
    if passphrase is None and not has_envelope:
        raise NetworkGuardError(
            "Nothing to check.",
            hint="Provide the passphrase your wallet or RPC provider reports, a transaction envelope, or both.",
        )
    envelope = check_envelope(envelope_xdr or "", target) if has_envelope else None

    findings: list[GuardFinding] = []
    if passphrase is not None:
        findings += _passphrase_findings(passphrase, target)
    if envelope is not None:
        findings += _envelope_findings(envelope, target)

    mismatch = any(finding.severity == "error" for finding in findings)
    confirmed = (passphrase is not None and passphrase.status == "match") or (
        envelope is not None and target in envelope.signed_for
    )
    return NetworkGuardReport(
        network=target,
        expected_passphrase=NETWORK_PASSPHRASES[target],
        passphrase=passphrase,
        envelope=envelope,
        mismatch_detected=mismatch,
        network_confirmed=confirmed,
        findings=findings,
        explanation=_explain(target, passphrase, envelope, mismatch, confirmed),
    )


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------


class NetworkGuardRequest(BaseModel):
    network: Literal["testnet", "public", "mainnet"] | None = Field(
        None, description="Network you intend to use. Defaults to this service's configured network."
    )
    network_passphrase: str | None = Field(
        None,
        max_length=MAX_PASSPHRASE_CHARS,
        description="Passphrase reported by the wallet, Horizon's root endpoint or Soroban RPC's getNetwork",
    )
    envelope_xdr: str | None = Field(
        None,
        max_length=MAX_ENVELOPE_XDR_CHARS,
        description="Base64 XDR of a transaction or fee-bump envelope. Never a secret key.",
    )

    @model_validator(mode="before")
    @classmethod
    def accept_sdk_names(cls, data: Any) -> Any:
        # Freighter and the JS SDK say networkPassphrase, getNetwork says
        # passphrase, and envelopes are often posted as envelopeXdr.
        if isinstance(data, dict):
            data = dict(data)
            for other_name, field_name in (
                ("networkPassphrase", "network_passphrase"),
                ("passphrase", "network_passphrase"),
                ("envelopeXdr", "envelope_xdr"),
            ):
                if other_name in data:
                    data.setdefault(field_name, data.pop(other_name))
        return data


@router.post("/stellar/network/check", response_model=NetworkGuardReport)
async def check_network_endpoint(body: NetworkGuardRequest) -> NetworkGuardReport:
    """Check a passphrase or signed envelope against the expected network. Offline: nothing is fetched or signed."""
    if contains_secret_key(body.network_passphrase, body.envelope_xdr):
        logger.warning("Secret-key-shaped string sent to the network passphrase guard; refusing it")
        raise APIException(
            status_code=400,
            detail="This looks like a Stellar secret key, not a passphrase or envelope. It was not used or stored.",
            hint=(
                "Never share a secret key. Treat it as compromised and move your funds to a new wallet. "
                "This endpoint only needs the network passphrase or a transaction envelope."
            ),
        )
    try:
        return guard_network(body.network, body.network_passphrase, body.envelope_xdr)
    except NetworkGuardError as exc:
        # Neither the passphrase nor the envelope is logged: both are caller-supplied text.
        logger.info("Network passphrase guard rejected input: %s", exc)
        raise APIException(status_code=400, detail=str(exc), hint=exc.hint) from None
