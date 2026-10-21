"""Deterministic sanitized Horizon and Soroban response fixtures (#580)."""

from __future__ import annotations

import copy
from typing import Any

PUBLIC_TEST_ACCOUNT_G = "G" + "A" * 55
PUBLIC_USDC_ISSUER_G = "G" + "B" * 55
SAMPLE_TX_HASH = "c" * 64

SANITIZED_FIXTURES: dict[str, dict[str, Any]] = {
    "horizon_account_public": {
        "network": "public",
        "id": PUBLIC_TEST_ACCOUNT_G,
        "sequence": "100001",
        "last_modified_ledger": 51200000,
        "last_modified_time": "2026-10-21T08:00:00Z",
        "thresholds": {"low_threshold": 1, "med_threshold": 2, "high_threshold": 2},
        "balances": [
            {"asset_type": "native", "balance": "250.0000000"},
            {
                "asset_type": "credit_alphanum4",
                "asset_code": "USDC",
                "asset_issuer": PUBLIC_USDC_ISSUER_G,
                "balance": "100.0000000",
                "limit": "50000.0000000",
                "is_authorized": True,
            },
        ],
    },
    "horizon_transaction_testnet": {
        "network": "testnet",
        "hash": SAMPLE_TX_HASH,
        "successful": True,
        "ledger": 1200500,
        "created_at": "2026-10-21T08:01:00Z",
        "fee_charged": "100",
        "memo_type": "none",
        "operations": [
            {
                "type": "payment",
                "source_account": PUBLIC_TEST_ACCOUNT_G,
                "to": PUBLIC_USDC_ISSUER_G,
                "asset_type": "native",
                "amount": "10.0000000",
            }
        ],
    },
    "soroban_simulation_success_testnet": {
        "network": "testnet",
        "latest_ledger": 1200501,
        "observed_at": "2026-10-21T08:02:00Z",
        "is_success": True,
        "cpu_instructions": 240000,
        "memory_bytes": 8192,
        "min_resource_fee_stroops": 12500,
    },
    "soroban_simulation_failure_public": {
        "network": "public",
        "latest_ledger": 51200001,
        "observed_at": "2026-10-21T08:03:00Z",
        "is_success": False,
        "cpu_instructions": 40000,
        "memory_bytes": 1024,
        "min_resource_fee_stroops": 0,
        "error_code": "HostError: Error(Contract, #1)",
    },
}


def get_stellar_fixture(name: str) -> dict[str, Any]:
    """Return a deep copy of a named deterministic Stellar fixture."""
    if name not in SANITIZED_FIXTURES:
        raise KeyError(f"Unknown Stellar fixture: {name}")
    return copy.deepcopy(SANITIZED_FIXTURES[name])
