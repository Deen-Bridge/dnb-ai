"""Tests verifying the deterministic adversarial evaluation suite for Stellar protocol safety (#588)."""

from __future__ import annotations

from stellar_adversarial_eval import (
    ADVERSARIAL_CASES,
    run_stellar_adversarial_eval,
)


def test_adversarial_eval_dataset_coverage() -> None:
    categories = {case.category for case in ADVERSARIAL_CASES}
    assert len(ADVERSARIAL_CASES) >= 6
    assert {
        "fake_usdc_issuer",
        "testnet_mainnet_confusion",
        "unconfirmed_transaction",
        "malicious_memo",
        "protocol_citations",
        "secret_key_safety",
    } <= categories


def test_adversarial_eval_suite_passes_all_cases() -> None:
    report = run_stellar_adversarial_eval()
    assert report.total >= 6
    assert report.failed == 0
    assert report.pass_rate == 1.0
    assert report.passed == report.total

    for _category, stats in report.category_results.items():
        assert stats["failed"] == 0
        assert stats["passed"] >= 1


def test_samples_are_free_of_real_user_data() -> None:
    # Ensure no personal emails, private phone numbers, or real wallet secrets exist in dataset
    for case in ADVERSARIAL_CASES:
        assert "@" not in case.prompt
        # All secret keys must be the synthetic placeholder
        if "secret_key" in case.context:
            assert case.context["secret_key"].startswith("SAAAAA")
