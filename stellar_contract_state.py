"""Check a Soroban contract state read for consistency, without touching the network.

Contract state lives in ledger entries: one *instance* entry per contract,
any number of persistent and temporary data entries keyed by an ``SCVal``, and
the Wasm code the instance runs. Soroban RPC's ``getLedgerEntries`` returns
them as base64 XDR together with the ledger each was last modified in, the
ledger its TTL runs out after, and the latest ledger the node had seen.

Nothing in that response proves itself. A buggy or hostile RPC node can pair a
key with someone else's value, hand back an entry from a different contract,
serve an archived entry as if it were live, or, behind a load balancer, answer
one read from a node that is several ledgers behind the last. This module takes
the responses a caller already has and checks them against the XDR they carry
and against each other, so an app can tell a trustworthy read from one that
only looks plausible.

Everything is local and read-only. Nothing is fetched from Soroban RPC or
Horizon, nothing is signed or submitted, and input that looks like a secret key
is refused before any decoding happens.
"""

import logging
import re
from dataclasses import dataclass
from itertools import combinations
from typing import Annotated, Any, Literal

from fastapi import APIRouter
from pydantic import BaseModel, Field, StringConstraints, model_validator
from stellar_sdk import Address, Asset, xdr
from stellar_sdk.strkey import StrKey

from errors import APIException
from stellar import contains_secret_key, usdc_issuer
from stellar_contract_events import ContractEventDecodeError, ScValue, _ScValDecoder
from stellar_fee_bump import NETWORK_PASSPHRASES, CanonicalNetwork, canonical_network
from stellar_network_guard import check_passphrase, describe_reported_network

logger = logging.getLogger(__name__)

router = APIRouter(tags=["stellar"])

# Soroban RPC refuses a getLedgerEntries call with more keys than this.
MAX_LEDGER_KEYS = 200

# Comparing a handful of reads is enough to catch a lagging or disagreeing
# node; more than that is a crawl, not a consistency check.
MAX_STATE_READS = 5

# Ledger keys are small: a contract address plus one SCVal.
MAX_KEY_XDR_CHARS = 64_000

# A contract code entry carries up to 128 KiB of Wasm, about 175,000 base64
# characters once framed. Data entries are capped at 64 KiB by the network.
MAX_ENTRY_XDR_CHARS = 180_000

# One request may hold several reads of several entries; this keeps the whole
# body to a size the XDR decoder can get through quickly.
MAX_TOTAL_XDR_CHARS = 2_000_000

UINT32_MAX = 2**32 - 1

_BASE64_XDR = re.compile(r"[A-Za-z0-9+/]+={0,2}")
_WHITESPACE = re.compile(r"\s+")

CONTRACT_STATE_DISCLAIMER = (
    "This checks the getLedgerEntries responses you supplied against the XDR they contain and against each "
    "other; nothing was fetched from Soroban RPC or Horizon, and nothing was signed or submitted. A consistent "
    "read is not proof the data is current: a node can be self-consistent and still behind the network, so "
    "compare latestLedger with an explorer or a second RPC provider."
)


class ContractStateCheckError(ValueError):
    """The input is not a checkable contract state read; *hint* says what to do."""

    def __init__(self, message: str, hint: str) -> None:
        super().__init__(message)
        self.hint = hint


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


def _rename_rpc_fields(data: Any, names: tuple[tuple[str, str], ...]) -> Any:
    # Soroban RPC uses camelCase. Mapped here rather than with field aliases,
    # which FastAPI's body handling does not keep.
    if isinstance(data, dict):
        data = dict(data)
        for rpc_name, field_name in names:
            if rpc_name in data:
                data.setdefault(field_name, data.pop(rpc_name))
    return data


class LedgerEntryResult(BaseModel):
    """One entry of a getLedgerEntries result, as Soroban RPC returns it."""

    key: str = Field(..., max_length=MAX_KEY_XDR_CHARS, description="Base64 XDR LedgerKey")
    data_xdr: str = Field(
        ...,
        max_length=MAX_ENTRY_XDR_CHARS,
        description="Base64 XDR LedgerEntryData (getLedgerEntries calls this 'xdr')",
    )
    last_modified_ledger: int = Field(..., ge=0, le=UINT32_MAX, description="lastModifiedLedgerSeq")
    live_until_ledger: int | None = Field(None, ge=0, le=UINT32_MAX, description="liveUntilLedgerSeq")

    @model_validator(mode="before")
    @classmethod
    def accept_rpc_names(cls, data: Any) -> Any:
        return _rename_rpc_fields(
            data,
            (
                ("xdr", "data_xdr"),
                ("lastModifiedLedgerSeq", "last_modified_ledger"),
                ("liveUntilLedgerSeq", "live_until_ledger"),
            ),
        )


class StateRead(BaseModel):
    """One getLedgerEntries result. The JSON-RPC envelope may be posted as-is."""

    latest_ledger: int = Field(..., ge=0, le=UINT32_MAX, description="latestLedger the node reported")
    entries: list[LedgerEntryResult] = Field(default_factory=list, max_length=MAX_LEDGER_KEYS)

    @model_validator(mode="before")
    @classmethod
    def accept_rpc_shape(cls, data: Any) -> Any:
        if isinstance(data, dict) and isinstance(data.get("result"), dict) and "latestLedger" not in data:
            data = data["result"]
        data = _rename_rpc_fields(data, (("latestLedger", "latest_ledger"),))
        # RPC returns null rather than [] when none of the keys exist.
        if isinstance(data, dict) and data.get("entries") is None:
            data["entries"] = []
        return data


class ContractStateCheckRequest(BaseModel):
    """The contract being read, what was asked for, and what came back."""

    contract_id: str = Field(..., max_length=64, description="Contract (C...) whose state was read")
    keys: list[Annotated[str, StringConstraints(max_length=MAX_KEY_XDR_CHARS)]] | None = Field(
        None,
        max_length=MAX_LEDGER_KEYS,
        description="Base64 LedgerKeys sent to getLedgerEntries. Enables missing- and extra-entry checks.",
    )
    reads: list[StateRead] = Field(
        ...,
        min_length=1,
        max_length=MAX_STATE_READS,
        description="One or more getLedgerEntries results for the same keys, in the order they were fetched",
    )
    network: Literal["testnet", "public", "mainnet"] | None = Field(
        None, description="Network the reads should come from. Defaults to this service's configured network."
    )
    network_passphrase: str | None = Field(
        None, max_length=128, description="Passphrase the RPC server reported via getNetwork, to confirm the network"
    )

    @model_validator(mode="before")
    @classmethod
    def accept_rpc_names(cls, data: Any) -> Any:
        return _rename_rpc_fields(data, (("contractId", "contract_id"), ("networkPassphrase", "network_passphrase")))

    @model_validator(mode="after")
    def limit_total_size(self) -> "ContractStateCheckRequest":
        total = sum(len(key) for key in self.keys or [])
        total += sum(len(entry.key) + len(entry.data_xdr) for read in self.reads for entry in read.entries)
        if total > MAX_TOTAL_XDR_CHARS:
            raise ValueError(f"The reads hold more than {MAX_TOTAL_XDR_CHARS:,} characters of XDR in total.")
        return self

    def all_strings(self) -> list[str | None]:
        """Every caller-supplied string, for the secret-key screen."""
        strings: list[str | None] = [self.contract_id, self.network_passphrase, *(self.keys or [])]
        for read in self.reads:
            for entry in read.entries:
                strings += [entry.key, entry.data_xdr]
        return strings


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

EntryKind = Literal["instance", "persistent", "temporary", "code"]
EntryStatus = Literal["live", "archived", "expired", "unknown"]
Severity = Literal["error", "warning", "info"]
FindingCode = Literal[
    "network_mismatch",
    "other_network_contract",
    "foreign_key",
    "not_contract_state",
    "duplicate_entry",
    "key_data_mismatch",
    "foreign_contract",
    "unrequested_entry",
    "modified_after_latest",
    "ttl_before_modification",
    "missing_ttl",
    "entry_not_live",
    "missing_entry",
    "instance_missing",
    "unrelated_code",
    "ledger_regressed",
    "mixed_snapshots",
    "conflicting_versions",
    "stale_version",
    "ttl_regressed",
    "entry_disappeared",
]


class StateFinding(BaseModel):
    code: FindingCode
    severity: Severity = Field(..., description="error: do not trust the read; warning: usable with care")
    message: str
    key: str | None = Field(None, description="Readable label of the ledger key concerned")
    read: int | None = Field(None, description="1-based index of the read concerned")


class StateEntrySummary(BaseModel):
    """The freshest version of one contract state entry across all reads."""

    label: str
    kind: EntryKind
    key_xdr: str = Field(..., description="Canonical base64 LedgerKey")
    key: ScValue | None = Field(None, description="Decoded data key; absent for code entries")
    value: ScValue | None = Field(None, description="Decoded value; instance storage for the instance entry")
    executable: Literal["wasm", "stellar_asset"] | None = Field(None, description="What the instance runs")
    wasm_hash: str | None = Field(None, description="Hex Wasm hash the instance runs, or a code entry's hash")
    last_modified_ledger: int
    live_until_ledger: int | None
    status: EntryStatus = Field(..., description="Whether the entry was live at the ledger it was read")
    read: int = Field(..., description="1-based index of the read this version came from")


class ContractStateReport(BaseModel):
    network: CanonicalNetwork
    contract_id: str
    consistent: bool = Field(..., description="True when no check found an error")
    latest_ledger: int = Field(..., description="Highest latestLedger across the reads")
    read_count: int
    entries: list[StateEntrySummary]
    findings: list[StateFinding]
    is_platform_usdc: bool = Field(..., description="The contract is the USDC Deen Bridge settles in, on this network")
    explanation: list[str]
    disclaimer: str = CONTRACT_STATE_DISCLAIMER


# ---------------------------------------------------------------------------
# XDR helpers
# ---------------------------------------------------------------------------


def _parse_xdr(xdr_class: Any, text: str, label: str, max_chars: int) -> Any:
    """Parse one base64 XDR value of *xdr_class*. *label* names it in errors."""
    cleaned = _WHITESPACE.sub("", text or "")
    if not cleaned:
        raise ContractStateCheckError(
            f"{label} is empty.", hint="Paste each key and xdr field exactly as Soroban RPC returned it."
        )
    if len(cleaned) > max_chars:
        raise ContractStateCheckError(
            f"{label} is too large to be a ledger entry.",
            hint=f"Keys are limited to {MAX_KEY_XDR_CHARS:,} and entries to {MAX_ENTRY_XDR_CHARS:,} base64 characters.",
        )
    if not _BASE64_XDR.fullmatch(cleaned):
        raise ContractStateCheckError(
            f"{label} is not base64-encoded XDR.",
            hint="Copy the key and xdr fields exactly as getLedgerEntries returned them.",
        )
    try:
        return xdr_class.from_xdr(cleaned)
    except (ValueError, EOFError, RecursionError):
        raise ContractStateCheckError(
            f"{label} could not be decoded.",
            hint="The XDR may be truncated or corrupted. Fetch the entries again from Soroban RPC.",
        ) from None


def _address_of(sc_address: xdr.SCAddress) -> str:
    try:
        return Address.from_xdr_sc_address(sc_address).address
    except ValueError:
        raise ContractStateCheckError(
            "A ledger key names an address the checker does not recognise.",
            hint="The XDR may be corrupted, or use an address type newer than this service supports.",
        ) from None


def ledger_key_for(data: xdr.LedgerEntryData) -> xdr.LedgerKey | None:
    """The ledger key an entry's data belongs under, or None if it is not contract state."""
    if data.type == xdr.LedgerEntryType.CONTRACT_DATA and data.contract_data is not None:
        entry = data.contract_data
        return xdr.LedgerKey(
            xdr.LedgerEntryType.CONTRACT_DATA,
            contract_data=xdr.LedgerKeyContractData(entry.contract, entry.key, entry.durability),
        )
    if data.type == xdr.LedgerEntryType.CONTRACT_CODE and data.contract_code is not None:
        return xdr.LedgerKey(
            xdr.LedgerEntryType.CONTRACT_CODE, contract_code=xdr.LedgerKeyContractCode(data.contract_code.hash)
        )
    return None


def _is_instance_key(sc_val: xdr.SCVal) -> bool:
    return sc_val.type == xdr.SCValType.SCV_LEDGER_KEY_CONTRACT_INSTANCE


def contract_instance_key(contract_id: str) -> xdr.LedgerKey:
    """The ledger key of a contract's instance entry: where its executable and instance storage live."""
    return xdr.LedgerKey(
        xdr.LedgerEntryType.CONTRACT_DATA,
        contract_data=xdr.LedgerKeyContractData(
            Address(contract_id).to_xdr_sc_address(),
            xdr.SCVal(xdr.SCValType.SCV_LEDGER_KEY_CONTRACT_INSTANCE),
            xdr.ContractDataDurability.PERSISTENT,
        ),
    )


def entry_kind(data: xdr.LedgerEntryData) -> EntryKind | None:
    if data.type == xdr.LedgerEntryType.CONTRACT_CODE:
        return "code"
    if data.type != xdr.LedgerEntryType.CONTRACT_DATA or data.contract_data is None:
        return None
    if _is_instance_key(data.contract_data.key):
        return "instance"
    return "temporary" if data.contract_data.durability == xdr.ContractDataDurability.TEMPORARY else "persistent"


def entry_status(kind: EntryKind, live_until: int | None, at_ledger: int) -> EntryStatus:
    """An entry is live through its liveUntilLedgerSeq, inclusive.

    Past it, persistent state (instances, code, persistent data) is archived
    and must be restored before a contract can use it; temporary data is gone.
    """
    if live_until is None:
        return "unknown"
    if live_until >= at_ledger:
        return "live"
    return "expired" if kind == "temporary" else "archived"


def _short(value: ScValue, limit: int = 80) -> str:
    """A one-line rendering of a decoded SCVal, for finding messages."""
    if value.items is not None:
        text = "[" + ", ".join(_short(item, limit) for item in value.items) + "]"
    elif value.entries is not None:
        text = "{" + ", ".join(f"{_short(e.key, limit)}: {_short(e.value, limit)}" for e in value.entries) + "}"
    elif value.value is not None:
        text = str(value.value)
    else:
        text = value.type
    return text if len(text) <= limit else text[: limit - 1] + "…"


def platform_usdc_contracts() -> dict[CanonicalNetwork, str]:
    """The Stellar Asset Contract of Deen Bridge's USDC on each network.

    The issuer differs between testnet and public, and the contract ID hashes
    the network passphrase, so the two IDs are unrelated.
    """
    return {
        name: Asset("USDC", usdc_issuer(name)).contract_id(passphrase)
        for name, passphrase in NETWORK_PASSPHRASES.items()
    }


# ---------------------------------------------------------------------------
# The checker
# ---------------------------------------------------------------------------


@dataclass
class _ParsedEntry:
    read: int
    latest_ledger: int
    key_xdr: str
    data: xdr.LedgerEntryData
    data_xdr: str
    kind: EntryKind
    last_modified: int
    live_until: int | None


class _ConsistencyChecker:
    """Collects findings for one check. Each method covers one class of problem."""

    def __init__(self, contract_id: str, network: CanonicalNetwork) -> None:
        self.contract_id = contract_id
        self.network = network
        self.findings: list[StateFinding] = []
        # Shared so a secret key in any key or value is redacted and remembered.
        self.decoder = _ScValDecoder()
        self._labels: dict[str, str] = {}

    def flag(
        self, code: FindingCode, severity: Severity, message: str, *, key: str | None = None, read: int | None = None
    ) -> None:
        # Messages often open with a key label such as "persistent entry ...".
        message = message[:1].upper() + message[1:]
        self.findings.append(StateFinding(code=code, severity=severity, message=message, key=key, read=read))

    def label(self, key: xdr.LedgerKey) -> str:
        key_xdr = key.to_xdr()
        if key_xdr not in self._labels:
            self._labels[key_xdr] = self._describe(key)
        return self._labels[key_xdr]

    def _describe(self, key: xdr.LedgerKey) -> str:
        if key.type == xdr.LedgerEntryType.CONTRACT_CODE and key.contract_code is not None:
            return f"contract code {key.contract_code.hash.hash.hex()[:16]}…"
        if key.type == xdr.LedgerEntryType.CONTRACT_DATA and key.contract_data is not None:
            data_key = key.contract_data
            owner = _address_of(data_key.contract)
            prefix = "" if owner == self.contract_id else f"{owner}'s "
            if _is_instance_key(data_key.key):
                return f"{prefix}contract instance"
            durability = data_key.durability.name.lower()
            return f"{prefix}{durability} entry {_short(self.decoder.decode(data_key.key))}"
        return f"{key.type.name.lower()} ledger entry"

    # -- network ------------------------------------------------------------

    def check_network(self, passphrase: str | None) -> bool:
        """Compare the reported passphrase with the target network; return is-platform-USDC."""
        reported = check_passphrase(passphrase, self.network)
        if reported is not None and reported.status != "match":
            self.flag(
                "network_mismatch",
                "error",
                f"The RPC server reports {describe_reported_network(reported)}, but these reads are being checked "
                f"as {self.network}. Its entries describe a different ledger.",
            )

        usdc = platform_usdc_contracts()
        if self.contract_id == usdc[self.network]:
            return True
        if self.contract_id in usdc.values():
            other = next(name for name, contract in usdc.items() if contract == self.contract_id)
            self.flag(
                "other_network_contract",
                "warning",
                f"{self.contract_id} is Deen Bridge's USDC contract on {other}, not {self.network}. On {self.network} "
                f"that USDC lives at {usdc[self.network]}.",
            )
        return False

    # -- the keys the caller asked for --------------------------------------

    def parse_requested_keys(self, keys: list[str]) -> list[str]:
        """Canonical XDR of each distinct contract state key the caller asked for."""
        seen: set[str] = set()
        requested: list[str] = []
        for index, key_text in enumerate(keys, start=1):
            key = _parse_xdr(xdr.LedgerKey, key_text, f"Requested key {index}", MAX_KEY_XDR_CHARS)
            key_xdr = key.to_xdr()
            if key_xdr in seen:
                continue
            seen.add(key_xdr)
            label = self.label(key)
            if key.type not in (xdr.LedgerEntryType.CONTRACT_DATA, xdr.LedgerEntryType.CONTRACT_CODE):
                # Flagged once here; neither its presence nor absence is checked.
                self.flag(
                    "not_contract_state",
                    "warning",
                    f"Requested key {index} is a {label}, not contract state, so it was not checked.",
                    key=label,
                )
                continue
            if key.contract_data is not None and _address_of(key.contract_data.contract) != self.contract_id:
                self.flag(
                    "foreign_key",
                    "error",
                    f"Requested key {index} reads {label}, which belongs to another contract.",
                    key=label,
                )
            requested.append(key_xdr)
        return requested

    # -- one read -----------------------------------------------------------

    def check_read(self, index: int, read: StateRead, requested: list[str] | None) -> dict[str, _ParsedEntry]:
        found: dict[str, _ParsedEntry] = {}
        for position, raw in enumerate(read.entries, start=1):
            where = f"Read {index}, entry {position}"
            key = _parse_xdr(xdr.LedgerKey, raw.key, f"{where} key", MAX_KEY_XDR_CHARS)
            data = _parse_xdr(xdr.LedgerEntryData, raw.data_xdr, f"{where} xdr", MAX_ENTRY_XDR_CHARS)
            key_xdr = key.to_xdr()
            label = self.label(key)

            kind = entry_kind(data)
            if kind is None:
                self.flag(
                    "not_contract_state",
                    "warning",
                    f"{where} is a {data.type.name.lower()} entry, not contract state, so it was not checked.",
                    key=label,
                    read=index,
                )
                continue
            if key_xdr in found:
                self.flag(
                    "duplicate_entry",
                    "error",
                    f"Read {index} returned {label} more than once. A node returns each key at most once.",
                    key=label,
                    read=index,
                )
                continue

            entry = _ParsedEntry(
                read=index,
                latest_ledger=read.latest_ledger,
                key_xdr=key_xdr,
                data=data,
                data_xdr=data.to_xdr(),
                kind=kind,
                last_modified=raw.last_modified_ledger,
                live_until=raw.live_until_ledger,
            )
            found[key_xdr] = entry
            self._check_entry(entry, label, requested)

        if requested is not None:
            self._check_missing(index, requested, found)
        self._check_code_matches_instance(index, found)
        return found

    def _check_entry(self, entry: _ParsedEntry, label: str, requested: list[str] | None) -> None:
        index = entry.read
        expected_key = ledger_key_for(entry.data)
        if expected_key is not None and expected_key.to_xdr() != entry.key_xdr:
            self.flag(
                "key_data_mismatch",
                "error",
                f"The value returned for {label} belongs to {self.label(expected_key)}. The node paired a key with "
                "another entry's data; nothing in this read can be trusted.",
                key=label,
                read=index,
            )
        if entry.data.contract_data is not None:
            owner = _address_of(entry.data.contract_data.contract)
            if owner != self.contract_id:
                self.flag(
                    "foreign_contract",
                    "error",
                    f"{label} is stored by {owner}, not {self.contract_id}.",
                    key=label,
                    read=index,
                )
        if requested is not None and entry.key_xdr not in requested:
            self.flag(
                "unrequested_entry",
                "error",
                f"Read {index} returned {label}, which was not one of the requested keys.",
                key=label,
                read=index,
            )

        if entry.last_modified > entry.latest_ledger:
            self.flag(
                "modified_after_latest",
                "error",
                f"{label} claims a change in ledger {entry.last_modified}, after the read's latest ledger "
                f"{entry.latest_ledger}.",
                key=label,
                read=index,
            )
        if entry.live_until is None:
            self.flag(
                "missing_ttl",
                "warning",
                f"{label} came without liveUntilLedgerSeq, so whether it is still live cannot be checked.",
                key=label,
                read=index,
            )
            return
        if entry.live_until < entry.last_modified:
            self.flag(
                "ttl_before_modification",
                "error",
                f"{label} expires at ledger {entry.live_until}, before it was last modified in ledger "
                f"{entry.last_modified}. No valid entry looks like that.",
                key=label,
                read=index,
            )
            return
        status = entry_status(entry.kind, entry.live_until, entry.latest_ledger)
        if status == "archived":
            self.flag(
                "entry_not_live",
                "warning",
                f"{label} was archived after ledger {entry.live_until}. A contract cannot use it until it is "
                "restored, so this value does not reflect what the contract would see.",
                key=label,
                read=index,
            )
        elif status == "expired":
            self.flag(
                "entry_not_live",
                "warning",
                f"{label} is temporary and expired after ledger {entry.live_until}. It no longer exists for the "
                "contract; treat it as absent.",
                key=label,
                read=index,
            )

    def _check_missing(self, index: int, requested: list[str], found: dict[str, _ParsedEntry]) -> None:
        instance_key = contract_instance_key(self.contract_id).to_xdr()
        for key_xdr in requested:
            if key_xdr in found:
                continue
            label = self._labels[key_xdr]
            if key_xdr == instance_key:
                self.flag(
                    "instance_missing",
                    "warning",
                    f"Read {index} found no contract instance for {self.contract_id} on {self.network}. The contract "
                    "may not exist on this network; check the contract ID and the RPC endpoint.",
                    key=label,
                    read=index,
                )
            else:
                self.flag(
                    "missing_entry",
                    "info",
                    f"Read {index} did not return {label}: it was never written, was removed, or has expired.",
                    key=label,
                    read=index,
                )

    def _check_code_matches_instance(self, index: int, found: dict[str, _ParsedEntry]) -> None:
        instance = next((entry for entry in found.values() if entry.kind == "instance"), None)
        code_entries = [entry for entry in found.values() if entry.kind == "code"]
        if instance is None or not code_entries:
            return
        executable = _executable_of(instance.data)
        if executable is None:
            return
        _, wasm_hash = executable
        reason = f"runs Wasm {wasm_hash[:16]}…" if wasm_hash else "runs the built-in Stellar Asset Contract, not Wasm"
        for code in code_entries:
            if code.data.contract_code is None:
                continue
            code_hash = code.data.contract_code.hash.hash.hex()
            if code_hash == wasm_hash:
                continue
            self.flag(
                "unrelated_code",
                "warning",
                f"Read {index} returned contract code {code_hash[:16]}…, but the instance {reason}. "
                "That code is not what this contract executes.",
                key=self._labels[code.key_xdr],
                read=index,
            )

    # -- several reads ------------------------------------------------------

    def check_across_reads(
        self, reads: list[dict[str, _ParsedEntry]], latest: list[int], requested: list[str] | None
    ) -> None:
        for index in range(1, len(latest)):
            if latest[index] < latest[index - 1]:
                self.flag(
                    "ledger_regressed",
                    "warning",
                    f"Read {index + 1} reports latest ledger {latest[index]}, behind read {index}'s "
                    f"{latest[index - 1]}. It was served by a node that is lagging.",
                    read=index + 1,
                )
        if len(set(latest)) > 1:
            self.flag(
                "mixed_snapshots",
                "info",
                f"The reads were taken at {len(set(latest))} different ledgers ({min(latest)} to {max(latest)}). "
                "Values that must agree with each other should come from one getLedgerEntries call.",
            )

        keys = list(dict.fromkeys(key for found in reads for key in found))
        for key_xdr in keys:
            versions = sorted(
                (found[key_xdr] for found in reads if key_xdr in found),
                key=lambda entry: (entry.latest_ledger, entry.read),
            )
            # One finding per key is enough to distrust it; more would be noise.
            # Absence only means something when every read asked for the key.
            if not self._compare_versions(key_xdr, versions) and requested is not None:
                self._compare_presence(key_xdr, versions, reads, latest)

    def _compare_versions(self, key_xdr: str, versions: list[_ParsedEntry]) -> bool:
        label = self._labels[key_xdr]
        for older, newer in combinations(versions, 2):
            same_ledger = older.latest_ledger == newer.latest_ledger
            if older.last_modified == newer.last_modified and older.data_xdr != newer.data_xdr:
                self.flag(
                    "conflicting_versions",
                    "error",
                    f"Reads {older.read} and {newer.read} both say {label} was last modified in ledger "
                    f"{older.last_modified}, but return different values. At least one node is wrong.",
                    key=label,
                    read=newer.read,
                )
                return True
            if older.last_modified != newer.last_modified and same_ledger:
                self.flag(
                    "conflicting_versions",
                    "error",
                    f"Reads {older.read} and {newer.read} were both at ledger {older.latest_ledger} but disagree on "
                    f"when {label} last changed ({older.last_modified} vs {newer.last_modified}).",
                    key=label,
                    read=newer.read,
                )
                return True
            if newer.last_modified < older.last_modified:
                self.flag(
                    "stale_version",
                    "error",
                    f"Read {newer.read} is at ledger {newer.latest_ledger} yet returns {label} as of ledger "
                    f"{newer.last_modified}, older than the ledger {older.last_modified} version read {older.read} "
                    "already saw. That node missed a write.",
                    key=label,
                    read=newer.read,
                )
                return True
            if older.last_modified < newer.last_modified <= older.latest_ledger:
                self.flag(
                    "stale_version",
                    "error",
                    f"Read {older.read} claims to be at ledger {older.latest_ledger} but returns {label} as of ledger "
                    f"{older.last_modified}, missing the change in ledger {newer.last_modified} that read "
                    f"{newer.read} returns.",
                    key=label,
                    read=older.read,
                )
                return True
            if (
                older.last_modified == newer.last_modified
                and older.live_until is not None
                and newer.live_until is not None
                and newer.live_until < older.live_until
            ):
                self.flag(
                    "ttl_regressed",
                    "warning",
                    f"Read {newer.read} reports {label} live until ledger {newer.live_until}, earlier than the "
                    f"{older.live_until} read {older.read} reported. A TTL only grows; that node is behind.",
                    key=label,
                    read=newer.read,
                )
                return True
        return False

    def _compare_presence(
        self, key_xdr: str, holders: list[_ParsedEntry], reads: list[dict[str, _ParsedEntry]], latest: list[int]
    ) -> None:
        """Check the reads that did not return an entry against those that did."""
        label = self._labels[key_xdr]
        for index, (found, at_ledger) in enumerate(zip(reads, latest, strict=True), start=1):
            if key_xdr in found:
                continue
            for holder in holders:
                was_live = holder.live_until is None or holder.live_until >= at_ledger
                if at_ledger == holder.latest_ledger and was_live:
                    self.flag(
                        "conflicting_versions",
                        "error",
                        f"Read {holder.read} returned {label} at ledger {at_ledger}, but read {index}, at the same "
                        "ledger, did not.",
                        key=label,
                        read=index,
                    )
                    return
                if holder.last_modified <= at_ledger < holder.latest_ledger and was_live:
                    self.flag(
                        "stale_version",
                        "error",
                        f"Read {index} is at ledger {at_ledger}, after {label} was written in ledger "
                        f"{holder.last_modified}, but did not return it.",
                        key=label,
                        read=index,
                    )
                    return
                if at_ledger > holder.latest_ledger:
                    self.flag(
                        "entry_disappeared",
                        "info",
                        f"{label} was present at ledger {holder.latest_ledger} (read {holder.read}) but absent at "
                        f"ledger {at_ledger} (read {index}). It may have been removed or expired in between.",
                        key=label,
                        read=index,
                    )
                    return

    # -- summary ------------------------------------------------------------

    def summarise(self, reads: list[dict[str, _ParsedEntry]]) -> list[StateEntrySummary]:
        freshest: dict[str, _ParsedEntry] = {}
        for found in reads:
            for key_xdr, entry in found.items():
                current = freshest.get(key_xdr)
                if current is None or entry.latest_ledger >= current.latest_ledger:
                    freshest[key_xdr] = entry
        return [self._summary(entry) for entry in freshest.values()]

    def _summary(self, entry: _ParsedEntry) -> StateEntrySummary:
        key: ScValue | None = None
        value: ScValue | None = None
        executable: Literal["wasm", "stellar_asset"] | None = None
        wasm_hash: str | None = None
        if entry.data.contract_code is not None:
            wasm_hash = entry.data.contract_code.hash.hash.hex()
        elif entry.data.contract_data is not None:
            data = entry.data.contract_data
            key = self.decoder.decode(data.key)
            if entry.kind == "instance" and data.val.instance is not None:
                runs = _executable_of(entry.data)
                if runs is not None:
                    executable, wasm_hash = runs
                storage = data.val.instance.storage
                value = self.decoder.decode(xdr.SCVal(xdr.SCValType.SCV_MAP, map=storage)) if storage else None
            else:
                value = self.decoder.decode(data.val)
        return StateEntrySummary(
            label=self._labels[entry.key_xdr],
            kind=entry.kind,
            key_xdr=entry.key_xdr,
            key=key,
            value=value,
            executable=executable,
            wasm_hash=wasm_hash,
            last_modified_ledger=entry.last_modified,
            live_until_ledger=entry.live_until,
            status=entry_status(entry.kind, entry.live_until, entry.latest_ledger),
            read=entry.read,
        )


def _executable_of(data: xdr.LedgerEntryData) -> tuple[Literal["wasm", "stellar_asset"], str | None] | None:
    """What a contract instance entry runs: ("wasm", hash) or ("stellar_asset", None)."""
    if data.contract_data is None or data.contract_data.val.instance is None:
        return None
    executable = data.contract_data.val.instance.executable
    if executable.type == xdr.ContractExecutableType.CONTRACT_EXECUTABLE_WASM and executable.wasm_hash is not None:
        return "wasm", executable.wasm_hash.hash.hex()
    if executable.type == xdr.ContractExecutableType.CONTRACT_EXECUTABLE_STELLAR_ASSET:
        return "stellar_asset", None
    return None


# ---------------------------------------------------------------------------
# Whole-check entry point
# ---------------------------------------------------------------------------


def _validate_contract_id(contract_id: str) -> str:
    cleaned = (contract_id or "").strip()
    if not StrKey.is_valid_contract(cleaned):
        raise ContractStateCheckError(
            "The contract ID is not a valid Soroban contract address.",
            hint="Contract IDs start with 'C' and are 56 characters long.",
        )
    return cleaned


def _explain(
    checker: _ConsistencyChecker, entry_count: int, latest: list[int], is_platform_usdc: bool
) -> tuple[bool, list[str]]:
    errors = sum(finding.severity == "error" for finding in checker.findings)
    warnings = sum(finding.severity == "warning" for finding in checker.findings)
    explanation = [
        f"Checked {entry_count} contract state {'entry' if entry_count == 1 else 'entries'} from {len(latest)} "
        f"read(s) of {checker.contract_id} on the Stellar {checker.network} network, up to ledger {max(latest)}."
    ]
    if errors:
        explanation.append(
            f"Found {errors} error(s): at least part of this read cannot be trusted as {checker.contract_id}'s "
            "state. Read it again, preferably from a second RPC provider."
        )
    elif warnings:
        explanation.append(f"No errors, but {warnings} warning(s) need attention before relying on these values.")
    else:
        explanation.append(
            "No inconsistencies found: every entry matches its key, belongs to this contract and was live when read."
        )
    if is_platform_usdc:
        explanation.append("This is the USDC contract Deen Bridge uses for course and book purchases.")
    if checker.decoder.redacted_secret_key:
        explanation.append(
            "A string in this contract's state looked like a Stellar secret key and was redacted. Whoever owns that "
            "key should treat it as compromised."
        )
    return errors == 0, explanation


def check_contract_state(
    contract_id: str,
    reads: list[StateRead],
    keys: list[str] | None = None,
    network: str | None = None,
    network_passphrase: str | None = None,
) -> ContractStateReport:
    """Check one or more getLedgerEntries results for a contract's state.

    Pure and offline: no RPC or Horizon reads, no signing, no submission.
    Inconsistencies are reported as findings; ContractStateCheckError is raised
    only when the input itself cannot be decoded.
    """
    target = canonical_network(network)
    contract = _validate_contract_id(contract_id)
    if not reads:
        raise ContractStateCheckError(
            "No reads were provided.", hint="Paste at least one getLedgerEntries result for this contract."
        )
    if len(reads) > MAX_STATE_READS:
        raise ContractStateCheckError(
            f"At most {MAX_STATE_READS} reads can be compared at once; {len(reads)} were provided.",
            hint="Compare the most recent reads, or the reads from the providers you are unsure about.",
        )

    checker = _ConsistencyChecker(contract, target)
    try:
        is_platform_usdc = checker.check_network(network_passphrase)
        requested = checker.parse_requested_keys(keys) if keys is not None else None
        parsed = [checker.check_read(index, read, requested) for index, read in enumerate(reads, start=1)]
        latest = [read.latest_ledger for read in reads]
        if len(reads) > 1:
            checker.check_across_reads(parsed, latest, requested)
        entries = checker.summarise(parsed)
    except ContractEventDecodeError as exc:
        # Raised while decoding an SCVal inside a key or value.
        raise ContractStateCheckError(str(exc), exc.hint) from None

    consistent, explanation = _explain(checker, len(entries), latest, is_platform_usdc)
    return ContractStateReport(
        network=target,
        contract_id=contract,
        consistent=consistent,
        latest_ledger=max(latest),
        read_count=len(reads),
        entries=entries,
        findings=checker.findings,
        is_platform_usdc=is_platform_usdc,
        explanation=explanation,
    )


@router.post("/stellar/contract-state/check", response_model=ContractStateReport)
async def check_contract_state_endpoint(body: ContractStateCheckRequest) -> ContractStateReport:
    """Check getLedgerEntries results for consistency. Offline: nothing is fetched, signed or submitted."""
    if contains_secret_key(*body.all_strings()):
        logger.warning("Secret-key-shaped string sent to the contract state checker; refusing it")
        raise APIException(
            status_code=400,
            detail="This looks like a Stellar secret key, not contract state. It was not used or stored.",
            hint=(
                "Never share a secret key. Treat it as compromised and move your funds to a new wallet. "
                "This endpoint only needs the contract ID and the getLedgerEntries results."
            ),
        )
    try:
        return check_contract_state(body.contract_id, body.reads, body.keys, body.network, body.network_passphrase)
    except ContractStateCheckError as exc:
        # The ledger entries themselves are never logged: they are user-supplied and may be large.
        logger.info("Contract state check rejected input: %s", exc)
        raise APIException(status_code=400, detail=str(exc), hint=exc.hint) from None
