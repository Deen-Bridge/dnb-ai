"""Tests for Stellar multisignature threshold explainer (#564)."""

from __future__ import annotations

from stellar_multisig_explainer import explain_account_multisig

ACCT = "G" + "A" * 55
SIGNER2 = "G" + "B" * 55


def test_summarizes_thresholds_and_detects_disabled_master_key() -> None:
    res = explain_account_multisig(
        {
            "id": ACCT,
            "thresholds": {"low_threshold": 1, "med_threshold": 2, "high_threshold": 3},
            "signers": [
                {"key": ACCT, "weight": 0},
                {"key": SIGNER2, "weight": 2},
            ],
        }
    )
    assert res.low_threshold == 1
    assert res.med_threshold == 2
    assert res.high_threshold == 3
    assert res.master_key_disabled is True
    assert res.total_active_weight == 2
    assert "never collects secret keys" in res.non_custodial_notice
