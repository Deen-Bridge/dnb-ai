"""Offline verification of the Stellar production freshness contract."""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stellar_contract_state import evaluate_provider_freshness, validate_contract_request  # noqa: E402


def run() -> dict[str, object]:
    now = datetime(2026, 11, 22, 12, 0, tzinfo=UTC)
    issuer = "G" + "A" * 55
    checks = 0
    failures = 0

    for network in ("testnet", "public"):
        checks += 1
        failures += (
            validate_contract_request(
                network=network,
                configured_network=network,
                asset_code="USDC",
                asset_issuer=issuer,
                expected_issuer=issuer,
                public_key_valid=True,
            )
            is not None
        )

    checks += 1
    failures += (
        validate_contract_request(
            network="testnet",
            configured_network="testnet",
            asset_code="USDC",
            asset_issuer="G" + "B" * 55,
            expected_issuer=issuer,
            public_key_valid=True,
        )
        is None
    )

    checks += 1
    failures += (
        validate_contract_request(
            network="unsupported",
            configured_network="testnet",
            asset_code="USDC",
            asset_issuer=issuer,
            expected_issuer=issuer,
            public_key_valid=True,
        )
        is None
    )

    checks += 1
    failures += (
        validate_contract_request(
            network="testnet",
            configured_network="testnet",
            asset_code="USDC",
            asset_issuer=issuer,
            expected_issuer=issuer,
            public_key_valid=False,
        )
        is None
    )

    fresh = evaluate_provider_freshness(
        {
            "network_passphrase": "Public Global Stellar Network ; September 2015",
            "history_latest_ledger": 51200001,
            "history_latest_ledger_closed_at": (now - timedelta(seconds=5)).isoformat(),
        },
        network="public",
        now=now,
    )
    checks += 1
    failures += not fresh.ok

    stale = evaluate_provider_freshness(
        {
            "network_passphrase": "Test SDF Network ; September 2015",
            "history_latest_ledger": 1200501,
            "history_latest_ledger_closed_at": (now - timedelta(seconds=31)).isoformat(),
        },
        network="testnet",
        now=now,
    )
    checks += 1
    failures += stale.reason != "STALE_LEDGER_EVIDENCE"

    mismatched_network = evaluate_provider_freshness(
        {
            "network_passphrase": "Test SDF Network ; September 2015",
            "history_latest_ledger": 51200001,
            "history_latest_ledger_closed_at": (now - timedelta(seconds=5)).isoformat(),
        },
        network="public",
        now=now,
    )
    checks += 1
    failures += mismatched_network.reason != "NETWORK_PASSPHRASE_MISMATCH"
    return {
        "suite": "stellar_production_contract",
        "network": "offline",
        "checks": checks,
        "failures": failures,
    }


if __name__ == "__main__":
    result = run()
    print(json.dumps(result, sort_keys=True))
    if result["failures"]:
        raise SystemExit(1)
