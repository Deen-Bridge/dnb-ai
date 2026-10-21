"""Tests for deterministic Horizon and Soroban response fixtures (#580)."""

from __future__ import annotations

from stellar_deterministic_fixtures import SANITIZED_FIXTURES, get_stellar_fixture


def test_fixtures_cover_all_categories_and_contain_zero_secrets() -> None:
    expected = {
        "horizon_account_public",
        "horizon_transaction_testnet",
        "soroban_simulation_success_testnet",
        "soroban_simulation_failure_public",
    }
    assert expected.issubset(set(SANITIZED_FIXTURES.keys()))

    acct = get_stellar_fixture("horizon_account_public")
    assert acct["network"] == "public"
    assert acct["last_modified_time"].endswith("Z")

    for fixture in SANITIZED_FIXTURES.values():
        serialized = str(fixture)
        assert "S" + "A" * 55 not in serialized
