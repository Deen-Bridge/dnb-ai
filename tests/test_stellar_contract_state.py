"""Offline tests for the Soroban contract state read consistency checks.

Every ledger entry is built here from fixed contract IDs and publicly known
test seeds, then shaped like a Soroban RPC getLedgerEntries result, so nothing
touches a network or a real account.
"""

import asyncio
import base64
import random
import struct

import pytest
from fastapi import HTTPException
from pydantic import ValidationError
from stellar_sdk import Address, Keypair, Network, StrKey, scval, xdr

import stellar_fee_bump
from stellar import stellar_info
from stellar_contract_state import (
    CONTRACT_STATE_DISCLAIMER,
    MAX_LEDGER_KEYS,
    MAX_STATE_READS,
    MAX_TOTAL_XDR_CHARS,
    ContractStateCheckError,
    ContractStateCheckRequest,
    check_contract_state,
    check_contract_state_endpoint,
    contract_instance_key,
    entry_status,
    ledger_key_for,
    platform_usdc_contracts,
)

CONTRACT = Address.from_raw_contract(bytes([7]) * 32).address
OTHER_CONTRACT = Address.from_raw_contract(bytes([8]) * 32).address

USDC_PUBLIC_SAC = "CCW67TSZV3SSS2HXMBQ5JFGCKJNXKZM7UQUWUZPUTHXSTZLEO7SJMI75"
USDC_TESTNET_SAC = "CBIELTK6YBZJU5UP2WWQEUCYKLPU6AUNZ2BQ4WWFEIE3USCIHMXQDAMA"

ALICE_KP = Keypair.from_raw_ed25519_seed(bytes(32))
ALICE = ALICE_KP.public_key
BOB = Keypair.from_raw_ed25519_seed(bytes([1]) * 32).public_key

WASM = b"\x00asm\x01\x00\x00\x00"
WASM_HASH = bytes([0xAB]) * 32

PERSISTENT = xdr.ContractDataDurability.PERSISTENT
TEMPORARY = xdr.ContractDataDurability.TEMPORARY


def run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def data_entry(key: xdr.SCVal, val: xdr.SCVal, contract=CONTRACT, durability=PERSISTENT) -> xdr.LedgerEntryData:
    return xdr.LedgerEntryData(
        xdr.LedgerEntryType.CONTRACT_DATA,
        contract_data=xdr.ContractDataEntry(
            xdr.ExtensionPoint(0), Address(contract).to_xdr_sc_address(), key, durability, val
        ),
    )


def balance_entry(holder=ALICE, amount=5, contract=CONTRACT, durability=PERSISTENT) -> xdr.LedgerEntryData:
    key = scval.to_vec([scval.to_symbol("Balance"), scval.to_address(holder)])
    return data_entry(key, scval.to_int128(amount), contract, durability)


def instance_entry(contract=CONTRACT, wasm_hash: bytes | None = WASM_HASH, storage=None) -> xdr.LedgerEntryData:
    if wasm_hash is None:
        executable = xdr.ContractExecutable(xdr.ContractExecutableType.CONTRACT_EXECUTABLE_STELLAR_ASSET)
    else:
        executable = xdr.ContractExecutable(
            xdr.ContractExecutableType.CONTRACT_EXECUTABLE_WASM, wasm_hash=xdr.Hash(wasm_hash)
        )
    instance = xdr.SCContractInstance(executable, xdr.SCMap(storage) if storage is not None else None)
    return data_entry(
        xdr.SCVal(xdr.SCValType.SCV_LEDGER_KEY_CONTRACT_INSTANCE),
        xdr.SCVal(xdr.SCValType.SCV_CONTRACT_INSTANCE, instance=instance),
        contract,
    )


def code_entry(code_hash: bytes = WASM_HASH) -> xdr.LedgerEntryData:
    return xdr.LedgerEntryData(
        xdr.LedgerEntryType.CONTRACT_CODE,
        contract_code=xdr.ContractCodeEntry(xdr.ContractCodeEntryExt(0), xdr.Hash(code_hash), WASM),
    )


def account_entry() -> xdr.LedgerEntryData:
    account = xdr.AccountEntry(
        account_id=Keypair.from_public_key(ALICE).xdr_account_id(),
        balance=xdr.Int64(10_000_000),
        seq_num=xdr.SequenceNumber(xdr.Int64(1)),
        num_sub_entries=xdr.Uint32(0),
        inflation_dest=None,
        flags=xdr.Uint32(0),
        home_domain=xdr.String32(b""),
        thresholds=xdr.Thresholds(bytes([1, 0, 0, 0])),
        signers=[],
        ext=xdr.AccountEntryExt(0),
    )
    return xdr.LedgerEntryData(xdr.LedgerEntryType.ACCOUNT, account=account)


def key_of(data: xdr.LedgerEntryData) -> str:
    key = ledger_key_for(data)
    assert key is not None
    return key.to_xdr()


def entry(data: xdr.LedgerEntryData, modified=900, live_until: int | None = 5000, key: str | None = None) -> dict:
    """One getLedgerEntries entry, spelled the way Soroban RPC spells it."""
    shaped = {"key": key or key_of(data), "xdr": data.to_xdr(), "lastModifiedLedgerSeq": modified}
    if live_until is not None:
        shaped["liveUntilLedgerSeq"] = live_until
    return shaped


def read(latest: int, *entries: dict) -> dict:
    return {"latestLedger": latest, "entries": list(entries)}


def check(*reads: dict, keys=None, contract=CONTRACT, network="testnet", passphrase=None):
    """Run the check the way the endpoint does, through the RPC field names."""
    body = ContractStateCheckRequest.model_validate(
        {
            "contractId": contract,
            "keys": keys,
            "reads": list(reads),
            "network": network,
            "networkPassphrase": passphrase,
        }
    )
    return check_contract_state(body.contract_id, body.reads, body.keys, body.network, body.network_passphrase)


def codes(report) -> list[str]:
    return [finding.code for finding in report.findings]


def finding(report, code: str):
    return next(item for item in report.findings if item.code == code)


INSTANCE = instance_entry()
BALANCE = balance_entry()
CLEAN_READ = read(1000, entry(INSTANCE, modified=10), entry(BALANCE), entry(code_entry(), modified=10))
CLEAN_KEYS = [key_of(INSTANCE), key_of(BALANCE), key_of(code_entry())]


# ---------------------------------------------------------------------------
# A clean read
# ---------------------------------------------------------------------------


def test_clean_read_is_consistent_and_decoded():
    report = check(CLEAN_READ, keys=CLEAN_KEYS)

    assert report.consistent is True
    assert report.findings == []
    assert report.network == "testnet"
    assert report.latest_ledger == 1000
    assert report.read_count == 1
    assert report.disclaimer == CONTRACT_STATE_DISCLAIMER
    assert "No inconsistencies found" in " ".join(report.explanation)

    instance, balance, code = report.entries
    assert (instance.kind, instance.executable, instance.wasm_hash) == ("instance", "wasm", WASM_HASH.hex())
    assert balance.kind == "persistent" and balance.status == "live"
    assert balance.value is not None and balance.value.model_dump(exclude_none=True) == {"type": "i128", "value": "5"}
    assert balance.label == f"persistent entry [Balance, {ALICE}]"
    assert code.kind == "code" and code.wasm_hash == WASM_HASH.hex()


def test_instance_storage_is_decoded():
    storage = [xdr.SCMapEntry(scval.to_symbol("Admin"), scval.to_address(BOB))]
    report = check(read(1000, entry(instance_entry(storage=storage))))
    value = report.entries[0].value
    assert value is not None and value.entries is not None
    assert (value.entries[0].key.value, value.entries[0].value.value) == ("Admin", BOB)


def test_rpc_envelope_and_null_entries_are_accepted():
    envelope = {"jsonrpc": "2.0", "id": 8675309, "result": CLEAN_READ}
    assert check(envelope, keys=CLEAN_KEYS).consistent is True

    empty = check({"jsonrpc": "2.0", "id": 1, "result": {"latestLedger": 1000, "entries": None}})
    assert empty.entries == [] and empty.consistent is True


def test_instance_key_matches_the_xdr_layout():
    """Regression: the instance key is built by hand here, independent of the SDK's key builder."""
    raw = (
        struct.pack(">II", 6, 1)  # LedgerEntryType.CONTRACT_DATA, SCAddressType.CONTRACT
        + StrKey.decode_contract(USDC_TESTNET_SAC)
        + struct.pack(">II", 20, 1)  # SCV_LEDGER_KEY_CONTRACT_INSTANCE, PERSISTENT
    )
    assert contract_instance_key(USDC_TESTNET_SAC).to_xdr() == base64.b64encode(raw).decode()
    assert key_of(instance_entry(USDC_TESTNET_SAC, wasm_hash=None)) == contract_instance_key(USDC_TESTNET_SAC).to_xdr()


@pytest.mark.parametrize(
    "kind,live_until,at_ledger,expected",
    [
        ("persistent", 1000, 1000, "live"),  # live through liveUntilLedgerSeq, inclusive
        ("persistent", 999, 1000, "archived"),
        ("instance", 999, 1000, "archived"),
        ("code", 999, 1000, "archived"),
        ("temporary", 999, 1000, "expired"),
        ("temporary", None, 1000, "unknown"),
    ],
)
def test_entry_status(kind, live_until, at_ledger, expected):
    assert entry_status(kind, live_until, at_ledger) == expected


# ---------------------------------------------------------------------------
# Checks within one read
# ---------------------------------------------------------------------------


def test_value_returned_under_the_wrong_key_is_an_error():
    bob_balance = balance_entry(BOB, amount=1_000_000)
    swapped = read(1000, entry(bob_balance, key=key_of(BALANCE)))
    report = check(swapped, keys=[key_of(BALANCE)])
    assert report.consistent is False
    mismatch = finding(report, "key_data_mismatch")
    assert mismatch.severity == "error"
    assert f"belongs to persistent entry [Balance, {BOB}]" in mismatch.message


def test_durability_is_part_of_the_key():
    temporary = balance_entry(durability=TEMPORARY)
    report = check(read(1000, entry(temporary, key=key_of(BALANCE))))
    assert "key_data_mismatch" in codes(report)


def test_entry_from_another_contract_is_an_error():
    report = check(read(1000, entry(balance_entry(contract=OTHER_CONTRACT))))
    assert codes(report) == ["foreign_contract"]
    assert OTHER_CONTRACT in report.findings[0].message
    assert report.findings[0].key.startswith(f"{OTHER_CONTRACT}'s persistent entry")


def test_requesting_another_contracts_key_is_flagged():
    report = check(read(1000), keys=[key_of(balance_entry(contract=OTHER_CONTRACT))])
    assert "foreign_key" in codes(report)
    assert report.consistent is False


def test_duplicate_entry_is_an_error():
    report = check(read(1000, entry(BALANCE), entry(BALANCE)))
    assert codes(report) == ["duplicate_entry"]
    assert len(report.entries) == 1


def test_unrequested_entry_is_an_error():
    report = check(read(1000, entry(INSTANCE), entry(BALANCE)), keys=[key_of(INSTANCE)])
    assert codes(report) == ["unrequested_entry"]


def test_requested_keys_are_deduplicated():
    report = check(read(1000, entry(BALANCE)), keys=[key_of(BALANCE), key_of(BALANCE)])
    assert report.consistent is True and report.findings == []


def test_modification_after_the_latest_ledger_is_an_error():
    report = check(read(1000, entry(BALANCE, modified=1001)))
    assert "modified_after_latest" in codes(report)


def test_modification_at_the_latest_ledger_is_fine():
    assert check(read(1000, entry(BALANCE, modified=1000))).findings == []


def test_ttl_before_last_modification_is_an_error():
    report = check(read(1000, entry(BALANCE, modified=900, live_until=899)))
    assert codes(report) == ["ttl_before_modification"]


def test_missing_ttl_is_a_warning_only():
    report = check(read(1000, entry(BALANCE, live_until=None)))
    assert codes(report) == ["missing_ttl"]
    assert report.consistent is True
    assert report.entries[0].status == "unknown"
    assert "1 warning(s)" in " ".join(report.explanation)


def test_archived_persistent_entry_is_flagged():
    report = check(read(1000, entry(BALANCE, live_until=999)))
    assert report.entries[0].status == "archived"
    assert "until it is restored" in finding(report, "entry_not_live").message
    assert report.consistent is True


def test_expired_temporary_entry_is_flagged():
    report = check(read(1000, entry(balance_entry(durability=TEMPORARY), live_until=950)))
    assert report.entries[0].status == "expired"
    assert "treat it as absent" in finding(report, "entry_not_live").message


def test_missing_entry_is_informational():
    report = check(read(1000, entry(INSTANCE)), keys=[key_of(INSTANCE), key_of(BALANCE)])
    assert codes(report) == ["missing_entry"]
    assert report.findings[0].severity == "info"
    assert report.consistent is True


def test_missing_instance_suggests_the_wrong_network():
    report = check(read(1000), keys=[contract_instance_key(CONTRACT).to_xdr()])
    missing = finding(report, "instance_missing")
    assert missing.severity == "warning"
    assert "may not exist on this network" in missing.message


def test_code_the_instance_does_not_run_is_flagged():
    report = check(read(1000, entry(INSTANCE), entry(code_entry(bytes([0xCD]) * 32))))
    assert codes(report) == ["unrelated_code"]
    assert WASM_HASH.hex()[:16] in report.findings[0].message


def test_code_entry_next_to_an_asset_contract_is_flagged():
    report = check(read(1000, entry(instance_entry(wasm_hash=None)), entry(code_entry())))
    assert "built-in Stellar Asset Contract" in finding(report, "unrelated_code").message
    assert report.entries[0].executable == "stellar_asset" and report.entries[0].wasm_hash is None


def test_non_contract_entries_are_skipped_with_a_warning():
    account = account_entry()
    account_key = xdr.LedgerKey(
        xdr.LedgerEntryType.ACCOUNT,
        account=xdr.LedgerKeyAccount(Keypair.from_public_key(ALICE).xdr_account_id()),
    ).to_xdr()
    report = check(read(1000, entry(account, key=account_key)), keys=[account_key])
    assert codes(report) == ["not_contract_state", "not_contract_state"]
    assert report.entries == []


# ---------------------------------------------------------------------------
# Network awareness
# ---------------------------------------------------------------------------


def test_matching_network_passphrase_passes():
    report = check(CLEAN_READ, passphrase=Network.TESTNET_NETWORK_PASSPHRASE)
    assert report.findings == []


def test_passphrase_from_the_other_network_is_an_error():
    report = check(CLEAN_READ, passphrase=Network.PUBLIC_NETWORK_PASSPHRASE)
    mismatch = finding(report, "network_mismatch")
    assert "reports the public network" in mismatch.message
    assert report.consistent is False


def test_passphrase_differing_only_in_case_is_an_error():
    report = check(CLEAN_READ, passphrase=Network.TESTNET_NETWORK_PASSPHRASE.lower())
    assert "only in letter case or spacing" in finding(report, "network_mismatch").message
    assert report.consistent is False


def test_well_known_unsupported_network_is_named():
    report = check(CLEAN_READ, passphrase=Network.FUTURENET_NETWORK_PASSPHRASE)
    assert "reports the futurenet network" in finding(report, "network_mismatch").message


def test_unknown_passphrase_is_not_echoed():
    passphrase = "Standalone Network ; February 2017 <script>"
    report = check(CLEAN_READ, passphrase=passphrase)
    assert "neither testnet nor public" in finding(report, "network_mismatch").message
    assert passphrase not in report.model_dump_json()


def test_platform_usdc_contract_is_recognised_per_network():
    assert platform_usdc_contracts() == {"testnet": USDC_TESTNET_SAC, "public": USDC_PUBLIC_SAC}

    usdc_read = read(1000, entry(instance_entry(USDC_TESTNET_SAC, wasm_hash=None)))
    report = check(usdc_read, contract=USDC_TESTNET_SAC)
    assert report.is_platform_usdc is True
    assert "Deen Bridge uses for course and book purchases" in " ".join(report.explanation)


def test_usdc_contract_from_the_other_network_is_flagged():
    report = check(read(1000), contract=USDC_PUBLIC_SAC, network="testnet")
    assert report.is_platform_usdc is False
    message = finding(report, "other_network_contract").message
    assert "USDC contract on public, not testnet" in message
    assert USDC_TESTNET_SAC in message


def test_default_network_follows_service_config(monkeypatch):
    monkeypatch.setattr(stellar_fee_bump, "STELLAR_NETWORK", "mainnet")
    body = ContractStateCheckRequest.model_validate({"contractId": USDC_PUBLIC_SAC, "reads": [read(1000)]})
    report = check_contract_state(body.contract_id, body.reads)
    assert report.network == "public"
    assert report.is_platform_usdc is True


# ---------------------------------------------------------------------------
# Checks across reads
# ---------------------------------------------------------------------------


def test_identical_reads_are_consistent():
    report = check(CLEAN_READ, CLEAN_READ, keys=CLEAN_KEYS)
    assert report.findings == []


def test_ledger_going_backwards_is_flagged():
    report = check(read(1000, entry(BALANCE)), read(990, entry(BALANCE)))
    assert codes(report) == ["ledger_regressed", "mixed_snapshots"]
    assert finding(report, "ledger_regressed").read == 2
    assert report.consistent is True


def test_newer_value_from_a_newer_ledger_is_fine():
    report = check(
        read(1000, entry(BALANCE, modified=900)),
        read(1010, entry(balance_entry(amount=7), modified=1005)),
    )
    assert codes(report) == ["mixed_snapshots"]
    assert report.entries[0].value is not None and report.entries[0].value.value == "7"
    assert report.entries[0].read == 2


def test_two_values_for_one_version_conflict():
    report = check(
        read(1000, entry(BALANCE, modified=900)),
        read(1000, entry(balance_entry(amount=6), modified=900)),
    )
    assert codes(report) == ["conflicting_versions"]
    assert "At least one node is wrong" in report.findings[0].message
    assert report.consistent is False


def test_same_ledger_different_modification_conflicts():
    report = check(read(1000, entry(BALANCE, modified=900)), read(1000, entry(BALANCE, modified=950)))
    assert codes(report) == ["conflicting_versions"]


def test_newer_read_with_an_older_version_is_stale():
    report = check(
        read(1000, entry(balance_entry(amount=6), modified=950)),
        read(1010, entry(BALANCE, modified=900)),
    )
    stale = finding(report, "stale_version")
    assert stale.read == 2
    assert "missed a write" in stale.message


def test_read_that_missed_a_write_inside_its_own_window_is_stale():
    # Read 1 claims ledger 1000 but shows the ledger-900 value; read 2 shows a
    # change made in ledger 950, which read 1 should already have seen.
    report = check(
        read(1000, entry(BALANCE, modified=900)),
        read(1010, entry(balance_entry(amount=6), modified=950)),
    )
    stale = finding(report, "stale_version")
    assert stale.read == 1
    assert "missing the change in ledger 950" in stale.message


def test_shrinking_ttl_is_flagged():
    report = check(read(1000, entry(BALANCE, live_until=6000)), read(1010, entry(BALANCE, live_until=5000)))
    assert "ttl_regressed" in codes(report)


def test_ttl_extension_is_fine():
    report = check(read(1000, entry(BALANCE, live_until=5000)), read(1010, entry(BALANCE, live_until=9000)))
    assert codes(report) == ["mixed_snapshots"]


def test_entry_missing_at_the_same_ledger_conflicts():
    report = check(read(1000, entry(BALANCE)), read(1000), keys=[key_of(BALANCE)])
    conflict = finding(report, "conflicting_versions")
    assert conflict.read == 2
    assert "at the same ledger" in conflict.message


def test_entry_missing_after_it_was_written_is_stale():
    report = check(read(1000, entry(BALANCE, modified=900)), read(950), keys=[key_of(BALANCE)])
    assert finding(report, "stale_version").read == 2


def test_entry_missing_before_it_was_written_is_fine():
    report = check(read(1000, entry(BALANCE, modified=900)), read(850), keys=[key_of(BALANCE)])
    assert "stale_version" not in codes(report)
    assert report.consistent is True


def test_entry_removed_later_is_informational():
    report = check(read(1000, entry(BALANCE)), read(1100), keys=[key_of(BALANCE)])
    assert finding(report, "entry_disappeared").severity == "info"
    assert report.consistent is True


def test_absence_is_not_judged_without_the_requested_keys():
    report = check(read(1000, entry(BALANCE)), read(1000))
    assert report.findings == []


# ---------------------------------------------------------------------------
# Secret keys
# ---------------------------------------------------------------------------


def test_secret_key_in_contract_state_is_redacted():
    secret = ALICE_KP.secret
    note = data_entry(scval.to_symbol("Note"), scval.to_string(f"backup: {secret}"))
    report = check(read(1000, entry(note)))
    assert secret not in report.model_dump_json()
    assert "REDACTED" in str(report.entries[0].value.value)
    assert any("treat it as compromised" in line for line in report.explanation)


def test_secret_key_in_a_ledger_key_is_redacted_from_labels():
    secret = ALICE_KP.secret
    report = check(read(1000, entry(data_entry(scval.to_string(secret), scval.to_bool(True)))))
    assert secret not in report.model_dump_json()


@pytest.mark.parametrize("field", ["contract_id", "keys", "network_passphrase", "xdr"])
def test_endpoint_refuses_secret_keys_without_echoing_them(field):
    secret = ALICE_KP.secret
    body: dict = {"contract_id": CONTRACT, "reads": [read(1000)]}
    if field == "keys":
        body["keys"] = [secret]
    elif field == "xdr":
        body["reads"] = [read(1000, {"key": key_of(BALANCE), "xdr": secret, "lastModifiedLedgerSeq": 1})]
    else:
        body[field] = secret
    with pytest.raises(HTTPException) as excinfo:
        run(check_contract_state_endpoint(ContractStateCheckRequest.model_validate(body)))
    assert excinfo.value.status_code == 400
    assert "secret key" in excinfo.value.detail
    assert secret not in excinfo.value.detail
    assert secret not in excinfo.value.hint


# ---------------------------------------------------------------------------
# Malformed input
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("contract_id", [ALICE, "C123", "", "c" + CONTRACT[1:]])
def test_invalid_contract_id_is_rejected(contract_id):
    with pytest.raises(ContractStateCheckError, match="not a valid Soroban contract address") as excinfo:
        check(read(1000), contract=contract_id)
    assert excinfo.value.hint


@pytest.mark.parametrize(
    "bad_entry,message",
    [
        ({"key": "", "xdr": BALANCE.to_xdr()}, "Read 1, entry 1 key is empty"),
        ({"key": key_of(BALANCE), "xdr": "  \n"}, "Read 1, entry 1 xdr is empty"),
        ({"key": "not base64!", "xdr": BALANCE.to_xdr()}, "key is not base64"),
        ({"key": key_of(BALANCE), "xdr": "<script>alert(1)</script>"}, "xdr is not base64"),
        ({"key": key_of(BALANCE)[:12], "xdr": BALANCE.to_xdr()}, "key could not be decoded"),
        (
            {"key": key_of(BALANCE), "xdr": base64.b64encode(BALANCE.to_xdr_bytes() + bytes(4)).decode()},
            "xdr could not be decoded",
        ),
    ],
)
def test_malformed_entries_are_rejected(bad_entry, message):
    with pytest.raises(ContractStateCheckError, match=message) as excinfo:
        check(read(1000, {**bad_entry, "lastModifiedLedgerSeq": 1}))
    assert excinfo.value.hint


def test_malformed_requested_key_is_named():
    with pytest.raises(ContractStateCheckError, match="Requested key 2 could not be decoded"):
        check(read(1000), keys=[key_of(BALANCE), "AAAABg=="])


def test_whitespace_wrapped_xdr_is_accepted():
    wrapped = "\n".join(BALANCE.to_xdr()[i : i + 20] for i in range(0, len(BALANCE.to_xdr()), 20))
    assert check(read(1000, {**entry(BALANCE), "xdr": f"  {wrapped}\n"})).findings == []


def test_corrupted_entries_only_ever_raise_check_errors():
    """Regression: hostile bytes must surface as a 400, never a 500."""
    rng = random.Random(648)
    samples = [INSTANCE, BALANCE, code_entry(), instance_entry(wasm_hash=None), account_entry()]
    for data in samples:
        for _ in range(60):
            raw = bytearray(base64.b64decode(data.to_xdr()))
            for _ in range(rng.randint(1, 4)):
                raw[rng.randrange(len(raw))] = rng.randrange(256)
            mutated = base64.b64encode(bytes(raw)).decode()
            try:
                check(read(1000, {"key": mutated, "xdr": mutated, "lastModifiedLedgerSeq": 1}))
            except ContractStateCheckError:
                pass


def test_too_many_reads_is_rejected_by_the_function():
    reads = [ContractStateCheckRequest.model_validate({"contract_id": CONTRACT, "reads": [read(1)]}).reads[0]]
    with pytest.raises(ContractStateCheckError, match=f"At most {MAX_STATE_READS} reads"):
        check_contract_state(CONTRACT, reads * (MAX_STATE_READS + 1))
    with pytest.raises(ContractStateCheckError, match="No reads"):
        check_contract_state(CONTRACT, [])


# ---------------------------------------------------------------------------
# HTTP endpoint and request model
# ---------------------------------------------------------------------------


def test_endpoint_returns_the_report():
    body = ContractStateCheckRequest.model_validate({"contractId": CONTRACT, "keys": CLEAN_KEYS, "reads": [CLEAN_READ]})
    report = run(check_contract_state_endpoint(body))
    assert report.consistent is True
    assert "nothing was signed or submitted" in report.disclaimer


def test_endpoint_maps_check_errors_to_400():
    body = ContractStateCheckRequest.model_validate(
        {"contract_id": CONTRACT, "reads": [read(1000, {"key": "not base64!", "xdr": "x", "lastModifiedLedgerSeq": 1})]}
    )
    with pytest.raises(HTTPException) as excinfo:
        run(check_contract_state_endpoint(body))
    assert excinfo.value.status_code == 400
    assert "not base64" in excinfo.value.detail
    assert excinfo.value.hint


def test_inconsistent_reads_are_still_a_200_report():
    """Inconsistency is the answer, not a request error."""
    body = ContractStateCheckRequest.model_validate(
        {"contract_id": CONTRACT, "reads": [read(1000, entry(balance_entry(contract=OTHER_CONTRACT)))]}
    )
    report = run(check_contract_state_endpoint(body))
    assert report.consistent is False
    assert "cannot be trusted" in " ".join(report.explanation)


@pytest.mark.parametrize(
    "overrides",
    [
        {"reads": []},
        {"reads": [read(1)] * (MAX_STATE_READS + 1)},
        {"keys": [key_of(BALANCE)] * (MAX_LEDGER_KEYS + 1)},
        {"network": "futurenet"},
        {"contract_id": "C" * 65},
        {"network_passphrase": "x" * 129},
        {"reads": [read(-1)]},
        {"reads": [read(2**32)]},
        {"reads": [read(1000, {**entry(BALANCE), "lastModifiedLedgerSeq": -1})]},
        {"reads": [read(1000, {**entry(BALANCE), "xdr": "A" * 180_001})]},
        {"reads": [read(1000, *[entry(BALANCE)] * (MAX_LEDGER_KEYS + 1))]},
    ],
)
def test_request_validation(overrides):
    with pytest.raises(ValidationError):
        ContractStateCheckRequest.model_validate({"contract_id": CONTRACT, "reads": [read(1)], **overrides})


def test_total_request_size_is_capped():
    big = {"key": "A" * 60_000, "xdr": "A" * 170_000, "lastModifiedLedgerSeq": 1}
    reads = [read(1000, *[big] * 2)] * MAX_STATE_READS
    assert sum(len(e["key"]) + len(e["xdr"]) for r in reads for e in r["entries"]) > MAX_TOTAL_XDR_CHARS
    with pytest.raises(ValidationError, match="in total"):
        ContractStateCheckRequest.model_validate({"contract_id": CONTRACT, "reads": reads})


def test_request_accepts_both_field_spellings():
    snake = ContractStateCheckRequest.model_validate(
        {
            "contract_id": CONTRACT,
            "network_passphrase": Network.TESTNET_NETWORK_PASSPHRASE,
            "reads": [
                {
                    "latest_ledger": 1000,
                    "entries": [{"key": key_of(BALANCE), "data_xdr": BALANCE.to_xdr(), "last_modified_ledger": 900}],
                }
            ],
        }
    )
    camel = ContractStateCheckRequest.model_validate(
        {
            "contractId": CONTRACT,
            "networkPassphrase": Network.TESTNET_NETWORK_PASSPHRASE,
            "reads": [read(1000, entry(BALANCE, live_until=None))],
        }
    )
    assert snake == camel


def test_stellar_info_advertises_the_checker():
    assert "contract-state-consistency" in run(stellar_info())["features"]
