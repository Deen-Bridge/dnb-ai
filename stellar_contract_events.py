"""Decode Soroban contract event topics without touching the network.

Every Soroban contract event carries up to four *topics* and a single *value*,
each an XDR-encoded ``SCVal``. Soroban RPC's ``getEvents`` returns them as
base64 strings that mean nothing until decoded. Token contracts that follow
SEP-41, including the Stellar Asset Contract (SAC) that wraps USDC, name the
event in the first topic and list the addresses involved in the rest.

This module decodes the topics and value a caller supplies into typed,
JSON-safe values and, for SEP-41 token events, says what happened: who moved
how much to whom. When an event names a classic asset, the SAC contract ID for
that asset is derived from the network passphrase, so a genuine USDC transfer
can be told apart from a lookalike event emitted by some other contract.

Everything is local and read-only. Nothing is fetched from Soroban RPC or
Horizon, nothing is signed or submitted, and input that looks like a secret
key is refused before any decoding happens.
"""

import logging
import re
from typing import Annotated, Any, Literal, cast

from fastapi import APIRouter
from pydantic import BaseModel, Field, StringConstraints, model_validator
from stellar_sdk import Address, Asset, scval, xdr
from stellar_sdk.strkey import StrKey

from errors import APIException
from stellar import contains_secret_key, redact_secret_keys, usdc_issuer
from stellar_fee_bump import NETWORK_PASSPHRASES, CanonicalNetwork, canonical_network, format_xlm

logger = logging.getLogger(__name__)

router = APIRouter(tags=["stellar"])

# The Soroban host refuses to emit an event with more than four topics.
MAX_EVENT_TOPICS = 4

# Far larger than any single event value the network will emit, while still
# keeping a pasted wall of junk away from the XDR decoder.
MAX_SCVAL_XDR_CHARS = 64_000

# Real token events nest a level or two. The SDK has its own, much deeper,
# parse limit; this one keeps the decoded response small and readable.
MAX_SCVAL_DEPTH = 32

_BASE64_XDR = re.compile(r"[A-Za-z0-9+/]+={0,2}")
_WHITESPACE = re.compile(r"\s+")

# Integer-like SCVal types and how to read each one. Every integer is rendered
# as a decimal string, because i128 and u256 amounts do not survive a trip
# through a JavaScript number.
_INTEGER_READERS = {
    xdr.SCValType.SCV_U32: scval.from_uint32,
    xdr.SCValType.SCV_I32: scval.from_int32,
    xdr.SCValType.SCV_U64: scval.from_uint64,
    xdr.SCValType.SCV_I64: scval.from_int64,
    xdr.SCValType.SCV_TIMEPOINT: scval.from_timepoint,
    xdr.SCValType.SCV_DURATION: scval.from_duration,
    xdr.SCValType.SCV_U128: scval.from_uint128,
    xdr.SCValType.SCV_I128: scval.from_int128,
    xdr.SCValType.SCV_U256: scval.from_uint256,
    xdr.SCValType.SCV_I256: scval.from_int256,
}

CONTRACT_EVENT_DISCLAIMER = (
    "This decodes the event data you supplied; nothing was fetched from Soroban RPC or Horizon, and nothing "
    "was signed or submitted. Any contract can emit any event, so an event only proves something when it "
    "came from the contract you expect, in a successful transaction. Check both on an explorer."
)


class ContractEventDecodeError(ValueError):
    """The input is not a decodable contract event; *hint* says what to do."""

    def __init__(self, message: str, hint: str) -> None:
        super().__init__(message)
        self.hint = hint


class ScValue(BaseModel):
    """One decoded SCVal.

    ``type`` is the XDR type without its ``SCV_`` prefix (``symbol``,
    ``address``, ``i128``...). Scalars land in ``value``; vectors in ``items``
    and maps in ``entries``.
    """

    type: str
    value: str | bool | None = None
    items: list["ScValue"] | None = None
    entries: list["ScMapEntry"] | None = None


class ScMapEntry(BaseModel):
    key: ScValue
    value: ScValue


ScValue.model_rebuild()


def parse_scval_xdr(scval_xdr: str, label: str) -> xdr.SCVal:
    """Parse one base64 SCVal. *label* names it in errors, e.g. "Topic 2"."""
    cleaned = _WHITESPACE.sub("", scval_xdr or "")
    if not cleaned:
        raise ContractEventDecodeError(
            f"{label} is empty.",
            hint="Paste each topic exactly as Soroban RPC returned it: a base64 XDR SCVal.",
        )
    if len(cleaned) > MAX_SCVAL_XDR_CHARS:
        raise ContractEventDecodeError(
            f"{label} is too large to be a contract event SCVal.",
            hint=f"Each topic and value is limited to {MAX_SCVAL_XDR_CHARS:,} base64 characters here.",
        )
    if not _BASE64_XDR.fullmatch(cleaned):
        raise ContractEventDecodeError(
            f"{label} is not base64-encoded XDR.",
            hint="Copy the topic or value exactly as Soroban RPC's getEvents returned it.",
        )
    try:
        return xdr.SCVal.from_xdr(cleaned)
    except (ValueError, EOFError, RecursionError):
        # ValueError covers bad base64, unknown enum values, trailing bytes and
        # the SDK's own nesting limit; EOFError covers truncation.
        raise ContractEventDecodeError(
            f"{label} could not be decoded as an SCVal.",
            hint="The XDR may be truncated or corrupted. Fetch the event again from Soroban RPC.",
        ) from None


def _describe_error(error: xdr.SCError) -> str:
    kind = error.type.name.removeprefix("SCE_").lower()
    if error.type == xdr.SCErrorType.SCE_CONTRACT and error.contract_code is not None:
        return f"{kind}:{error.contract_code.uint32}"
    code = error.code.name.removeprefix("SCEC_").lower() if error.code is not None else "unknown"
    return f"{kind}:{code}"


class _ScValDecoder:
    """Turns SCVals into ScValue trees, remembering whether it redacted anything.

    Event strings are public on-chain, but a contract can still carry a secret
    key in one, and this service never repeats one back.
    """

    def __init__(self) -> None:
        self.redacted_secret_key = False

    def decode(self, sc_val: xdr.SCVal, depth: int = 0) -> ScValue:
        if depth > MAX_SCVAL_DEPTH:
            raise ContractEventDecodeError(
                f"The event data is nested more than {MAX_SCVAL_DEPTH} levels deep.",
                hint="Token and application events are shallow; this does not look like a real contract event.",
            )

        kind = sc_val.type
        label = kind.name.removeprefix("SCV_").lower()

        if kind in _INTEGER_READERS:
            return ScValue(type=label, value=str(_INTEGER_READERS[kind](sc_val)))
        if kind == xdr.SCValType.SCV_BOOL:
            return ScValue(type=label, value=sc_val.b)
        if kind == xdr.SCValType.SCV_BYTES:
            return ScValue(type=label, value=scval.from_bytes(sc_val).hex())
        if kind == xdr.SCValType.SCV_STRING:
            return ScValue(type=label, value=self._text(scval.from_string(sc_val)))
        if kind == xdr.SCValType.SCV_SYMBOL and sc_val.sym is not None:
            # Read the raw bytes rather than scval.from_symbol, which raises on
            # the non-UTF-8 bytes a hostile envelope can carry.
            return ScValue(type=label, value=self._text(sc_val.sym.sc_symbol))
        if kind == xdr.SCValType.SCV_ADDRESS:
            return ScValue(type=label, value=self._address(sc_val))
        if kind == xdr.SCValType.SCV_ERROR and sc_val.error is not None:
            return ScValue(type=label, value=_describe_error(sc_val.error))
        if kind == xdr.SCValType.SCV_VEC:
            items = sc_val.vec.sc_vec if sc_val.vec is not None else []
            return ScValue(type=label, items=[self.decode(item, depth + 1) for item in items])
        if kind == xdr.SCValType.SCV_MAP:
            entries = sc_val.map.sc_map if sc_val.map is not None else []
            return ScValue(
                type=label,
                entries=[
                    ScMapEntry(key=self.decode(entry.key, depth + 1), value=self.decode(entry.val, depth + 1))
                    for entry in entries
                ],
            )
        # void needs no value; ledger keys and contract instances never appear
        # in events, so naming the type is enough.
        return ScValue(type=label)

    def _text(self, raw: bytes) -> str:
        text = raw.decode("utf-8", errors="replace")
        redacted = redact_secret_keys(text)
        if redacted != text:
            self.redacted_secret_key = True
        return redacted

    @staticmethod
    def _address(sc_val: xdr.SCVal) -> str:
        try:
            return Address.from_xdr_sc_val(sc_val).address
        except ValueError:
            raise ContractEventDecodeError(
                "The event contains an address the decoder does not recognise.",
                hint="The XDR may be corrupted, or use an address type newer than this service supports.",
            ) from None


def decode_scval(scval_xdr: str, label: str = "The value") -> ScValue:
    """Decode a single base64 SCVal into a JSON-safe ScValue."""
    return _ScValDecoder().decode(parse_scval_xdr(scval_xdr, label))


# ---------------------------------------------------------------------------
# SEP-41 token events
# ---------------------------------------------------------------------------

TokenEventKind = Literal["transfer", "mint", "burn", "clawback", "approve", "set_admin", "set_authorized", "fee"]

# Address roles that follow the event name, per accepted layout. Protocol 23
# (CAP-67) dropped the admin topic from SAC mint, clawback and set_authorized
# events; older events, and many custom SEP-41 tokens, still carry it. The SAC
# also appends the asset as a final string topic, handled separately.
_TOKEN_TOPIC_LAYOUTS: dict[TokenEventKind, tuple[tuple[str, ...], ...]] = {
    "transfer": (("from", "to"),),
    "approve": (("from", "spender"),),
    "mint": (("to",), ("admin", "to")),
    "burn": (("from",),),
    "clawback": (("from",), ("admin", "from")),
    "set_admin": (("admin",),),
    "set_authorized": (("id",), ("admin", "id")),
    "fee": (("from",),),
}

AssetContractStatus = Literal["verified", "other_network", "mismatch", "not_checked", "invalid_asset"]


class TokenEvent(BaseModel):
    kind: TokenEventKind
    addresses: dict[str, str] = Field(..., description="Role (from, to, spender, admin, ...) to G/C/M address")
    amount: str | None = Field(None, description="Amount in the token's smallest unit, as a decimal string")
    amount_display: str | None = Field(
        None, description="Amount with seven decimals; only for Stellar Asset Contract events, whose scale is fixed"
    )
    asset: str | None = Field(None, description="SEP-11 asset named by a Stellar Asset Contract event")
    to_muxed_id: str | None = Field(None, description="Muxed destination ID carried in the value (CAP-67)")
    live_until_ledger: int | None = Field(None, description="Ledger an approval expires after")
    authorized: bool | None = None


def _symbol_text(value: ScValue) -> str | None:
    return value.value if value.type == "symbol" and isinstance(value.value, str) else None


def _token_value_fields(kind: TokenEventKind, value: ScValue) -> dict[str, Any] | None:
    """Pull the typed fields out of a token event's value, or None if its shape is wrong."""
    if kind in ("transfer", "mint", "burn", "clawback", "fee"):
        if value.type == "i128":
            return {"amount": value.value}
        # CAP-67: a transfer to a muxed account carries {amount, to_muxed_id}.
        if kind == "transfer" and value.type == "map" and value.entries is not None:
            fields = {_symbol_text(entry.key): entry.value for entry in value.entries}
            amount = fields.get("amount")
            muxed = fields.get("to_muxed_id")
            if amount is not None and amount.type == "i128" and muxed is not None:
                return {"amount": amount.value, "to_muxed_id": None if muxed.value is None else str(muxed.value)}
        return None
    if kind == "approve":
        items = value.items or []
        if value.type == "vec" and len(items) == 2 and (items[0].type, items[1].type) == ("i128", "u32"):
            return {"amount": items[0].value, "live_until_ledger": int(str(items[1].value))}
        return None
    if kind == "set_admin":
        return {"new_admin": value.value} if value.type == "address" else None
    # set_authorized
    return {"authorized": value.value} if value.type == "bool" else None


def interpret_token_event(topics: list[ScValue], value: ScValue | None) -> TokenEvent | None:
    """Read *topics* (and *value*, if given) as a SEP-41 token event.

    Returns None when the event name is not a SEP-41 one, or when the topic or
    value types do not match the standard layout. That is never an error: most
    contracts emit their own events.
    """
    name = _symbol_text(topics[0]) if topics else None
    if name not in _TOKEN_TOPIC_LAYOUTS:
        return None
    kind = cast(TokenEventKind, name)

    rest = topics[1:]
    asset = None
    if rest and rest[-1].type == "string":
        asset = str(rest[-1].value)
        rest = rest[:-1]
    if any(topic.type != "address" for topic in rest):
        return None
    layout = next((roles for roles in _TOKEN_TOPIC_LAYOUTS[kind] if len(roles) == len(rest)), None)
    if layout is None:
        return None
    addresses = {role: str(topic.value) for role, topic in zip(layout, rest, strict=True)}

    fields: dict[str, Any] = {}
    if value is not None:
        parsed = _token_value_fields(kind, value)
        if parsed is None:
            return None
        fields = parsed
    new_admin = fields.pop("new_admin", None)
    if new_admin is not None:
        addresses["new_admin"] = str(new_admin)

    # Stellar Asset Contracts always use seven decimals. A custom token's
    # decimals live in its contract, not the event, so its amount stays raw.
    amount = fields.get("amount")
    is_sac_asset = asset is not None and parse_sep11_asset(asset) is not None
    return TokenEvent(
        kind=kind,
        addresses=addresses,
        asset=asset,
        amount_display=format_xlm(int(str(amount))) if amount is not None and is_sac_asset else None,
        **fields,
    )


def parse_sep11_asset(asset: str) -> Asset | None:
    """``native`` or ``CODE:ISSUER`` as an Asset, or None if it is neither."""
    if asset == "native":
        return Asset.native()
    code, separator, issuer = asset.partition(":")
    if not separator:
        return None
    try:
        return Asset(code, issuer)
    except ValueError:
        return None


def asset_contract_ids(asset: Asset) -> dict[CanonicalNetwork, str]:
    """The asset's Stellar Asset Contract ID on each network.

    The ID hashes the network passphrase, so the same asset has a different
    contract on testnet and on the public network.
    """
    return {name: asset.contract_id(passphrase) for name, passphrase in NETWORK_PASSPHRASES.items()}


def asset_contract_status(
    contract_id: str | None, ids: dict[CanonicalNetwork, str], network: CanonicalNetwork
) -> AssetContractStatus:
    if contract_id is None:
        return "not_checked"
    if contract_id == ids[network]:
        return "verified"
    if contract_id in ids.values():
        return "other_network"
    return "mismatch"


# ---------------------------------------------------------------------------
# Whole-event decoding
# ---------------------------------------------------------------------------


class ContractEventDecodeRequest(BaseModel):
    """One contract event. A Soroban RPC getEvents entry can be posted unchanged."""

    topics: list[Annotated[str, StringConstraints(max_length=MAX_SCVAL_XDR_CHARS)]] = Field(
        ...,
        min_length=1,
        max_length=MAX_EVENT_TOPICS,
        description="Base64 XDR SCVal for each topic, in order (getEvents calls this 'topic'). Never a secret key.",
    )
    value: str | None = Field(
        None,
        max_length=MAX_SCVAL_XDR_CHARS,
        description="Base64 XDR SCVal of the event value. Optional: topics alone still name the event.",
    )
    contract_id: str | None = Field(
        None,
        max_length=64,
        description="Contract (C...) that emitted the event (getEvents' 'contractId'), used to check SAC events.",
    )
    network: Literal["testnet", "public", "mainnet"] | None = Field(
        None,
        description="Network the event was emitted on. Defaults to this service's configured network.",
    )

    @model_validator(mode="before")
    @classmethod
    def accept_get_events_names(cls, data: Any) -> Any:
        # getEvents spells these 'topic' and 'contractId'. Mapped here rather
        # than with field aliases, which FastAPI's body handling does not keep.
        if isinstance(data, dict):
            data = dict(data)
            for rpc_name, field_name in (("topic", "topics"), ("contractId", "contract_id")):
                if rpc_name in data:
                    data.setdefault(field_name, data.pop(rpc_name))
        return data


class ContractEventDecoding(BaseModel):
    network: CanonicalNetwork
    contract_id: str | None
    event_name: str | None = Field(..., description="First topic, when it is a symbol (the SEP-41 convention)")
    topics: list[ScValue]
    value: ScValue | None
    token_event: TokenEvent | None
    asset_contract_id: str | None = Field(
        None, description="Stellar Asset Contract ID for the named asset on this network"
    )
    asset_contract_status: AssetContractStatus | None = None
    is_platform_usdc: bool = Field(..., description="The event names the USDC asset Deen Bridge settles in")
    explanation: list[str]
    disclaimer: str = CONTRACT_EVENT_DISCLAIMER


def _amount_phrase(event: TokenEvent) -> str:
    if event.amount is None:
        return "an amount not supplied (no event value was given)"
    if event.amount_display is not None and event.asset is not None:
        unit = "XLM" if event.asset == "native" else event.asset.partition(":")[0]
        return f"{event.amount_display} {unit} ({event.amount} base units)"
    return f"{event.amount} base units (a custom token's decimals are set by its contract, not the event)"


def _describe_token_event(event: TokenEvent) -> str:
    who = event.addresses
    amount = _amount_phrase(event)
    if event.kind == "transfer":
        line = f"A SEP-41 transfer of {amount} from {who['from']} to {who['to']}."
        if event.to_muxed_id is not None:
            line += f" The destination is muxed with ID {event.to_muxed_id}."
        return line
    if event.kind == "mint":
        return f"A SEP-41 mint of {amount} to {who['to']}."
    if event.kind == "burn":
        return f"A SEP-41 burn of {amount} from {who['from']}."
    if event.kind == "clawback":
        return f"A clawback of {amount} from {who['from']} by the asset issuer or admin."
    if event.kind == "fee":
        return f"A network fee of {amount} charged to {who['from']}."
    if event.kind == "approve":
        expiry = f", valid until ledger {event.live_until_ledger}" if event.live_until_ledger is not None else ""
        return f"{who['from']} approved {who['spender']} to spend up to {amount}{expiry}."
    if event.kind == "set_admin":
        new_admin = who.get("new_admin", "an admin not supplied (no event value was given)")
        return f"The token admin {who['admin']} handed admin rights to {new_admin}."
    state = {True: "authorized", False: "deauthorized", None: "had its authorization changed"}[event.authorized]
    return f"Account {who['id']} was {state} to hold this token."


_ASSET_STATUS_WORDING: dict[AssetContractStatus, str] = {
    "verified": "The emitting contract {contract} is the genuine Stellar Asset Contract for {asset} on {network}.",
    "other_network": (
        "The emitting contract {contract} is the Stellar Asset Contract for {asset} on {other}, not {network}. "
        "Check which network this event came from."
    ),
    "mismatch": (
        "Warning: the emitting contract {contract} is NOT the Stellar Asset Contract for {asset} on {network} "
        "(that is {expected}). Any contract can emit an event naming {asset}; this one did not move the real asset."
    ),
    "not_checked": (
        "No contract ID was supplied, so it was not checked. The genuine Stellar Asset Contract for {asset} "
        "on {network} is {expected}; only events from that contract move the real asset."
    ),
    "invalid_asset": (
        "The final topic '{asset}' is not a valid Stellar asset, so no Stellar Asset Contract emitted this event."
    ),
}


def _validate_contract_id(contract_id: str | None) -> str | None:
    if contract_id is None or not contract_id.strip():
        return None
    cleaned = contract_id.strip()
    if not StrKey.is_valid_contract(cleaned):
        raise ContractEventDecodeError(
            "The contract ID is not a valid Soroban contract address.",
            hint="Contract IDs start with 'C' and are 56 characters long, as getEvents returns them in contractId.",
        )
    return cleaned


def decode_contract_event(
    topics: list[str],
    value: str | None = None,
    contract_id: str | None = None,
    network: str | None = None,
) -> ContractEventDecoding:
    """Decode an event's topics and value and explain what it records.

    Pure and offline: no RPC or Horizon reads, no signing, no submission.
    Raises ContractEventDecodeError when the input cannot be decoded.
    """
    target = canonical_network(network)
    if not topics:
        raise ContractEventDecodeError(
            "No topics were provided.",
            hint="Paste the event's topic list from Soroban RPC getEvents; every contract event has at least one.",
        )
    if len(topics) > MAX_EVENT_TOPICS:
        raise ContractEventDecodeError(
            f"A contract event has at most {MAX_EVENT_TOPICS} topics; {len(topics)} were provided.",
            hint="Check that the event value was not pasted into the topic list.",
        )
    emitter = _validate_contract_id(contract_id)

    decoder = _ScValDecoder()
    decoded_topics = [
        decoder.decode(parse_scval_xdr(topic, f"Topic {index}")) for index, topic in enumerate(topics, start=1)
    ]
    decoded_value = decoder.decode(parse_scval_xdr(value, "The value")) if value and value.strip() else None

    event_name = _symbol_text(decoded_topics[0])
    token_event = interpret_token_event(decoded_topics, decoded_value)

    explanation = [f"Decoded {len(decoded_topics)} topic(s) for the Stellar {target} network."]
    if event_name is not None:
        explanation.append(f"The event is named '{event_name}'.")
    else:
        explanation.append(
            f"The first topic is a {decoded_topics[0].type}, not a symbol, so the event does not follow the "
            "SEP-41 naming convention."
        )

    if token_event is not None:
        explanation.append(_describe_token_event(token_event))
    elif event_name in _TOKEN_TOPIC_LAYOUTS:
        explanation.append(
            f"'{event_name}' is a SEP-41 token event name, but the topic or value types do not match the standard "
            "layout, so it was not read as a token event."
        )

    asset_id: str | None = None
    status: AssetContractStatus | None = None
    is_platform_usdc = False
    if token_event is not None and token_event.asset is not None:
        asset = parse_sep11_asset(token_event.asset)
        is_platform_usdc = token_event.asset == f"USDC:{usdc_issuer(target)}"
        if is_platform_usdc:
            explanation.append("The event names the USDC asset Deen Bridge uses for course and book purchases.")
        if asset is None:
            status = "invalid_asset"
        else:
            ids = asset_contract_ids(asset)
            asset_id = ids[target]
            status = asset_contract_status(emitter, ids, target)
        explanation.append(
            _ASSET_STATUS_WORDING[status].format(
                contract=emitter,
                asset=token_event.asset,
                network=target,
                other="public" if target == "testnet" else "testnet",
                expected=asset_id,
            )
        )

    if decoder.redacted_secret_key:
        explanation.append(
            "A string in this event looked like a Stellar secret key and was redacted. Whoever owns that key "
            "should treat it as compromised."
        )

    return ContractEventDecoding(
        network=target,
        contract_id=emitter,
        event_name=event_name,
        topics=decoded_topics,
        value=decoded_value,
        token_event=token_event,
        asset_contract_id=asset_id,
        asset_contract_status=status,
        is_platform_usdc=is_platform_usdc,
        explanation=explanation,
    )


@router.post("/stellar/contract-events/decode", response_model=ContractEventDecoding)
async def decode_contract_event_endpoint(body: ContractEventDecodeRequest) -> ContractEventDecoding:
    """Decode a Soroban contract event's topics and value. Offline: nothing is fetched, signed or submitted."""
    if contains_secret_key(*body.topics, body.value, body.contract_id):
        logger.warning("Secret-key-shaped string sent to the contract event decoder; refusing it")
        raise APIException(
            status_code=400,
            detail="This looks like a Stellar secret key, not contract event data. It was not used or stored.",
            hint=(
                "Never share a secret key. Treat it as compromised and move your funds to a new wallet. "
                "This endpoint only needs the base64 topics and value from Soroban RPC getEvents."
            ),
        )
    try:
        return decode_contract_event(body.topics, body.value, body.contract_id, body.network)
    except ContractEventDecodeError as exc:
        # The event data itself is never logged: it is user-supplied and may be large.
        logger.info("Contract event decode rejected input: %s", exc)
        raise APIException(status_code=400, detail=str(exc), hint=exc.hint) from None
