"""Tests for SEP-24 anchor deposit and withdrawal flow guide (#570)."""

from __future__ import annotations

from stellar_sep24_anchor_guide import explain_sep24_anchor_flow


def test_deposit_and_withdrawal_flows_identify_anchor_as_third_party() -> None:
    dep = explain_sep24_anchor_flow(flow_type="deposit", anchor_domain="anchor.example.org", asset_code="USDC")
    assert dep.is_third_party_provider is True
    assert len(dep.steps) == 4
    assert "independent third-party service provider" in dep.security_notice

    wd = explain_sep24_anchor_flow(flow_type="withdrawal", anchor_domain="anchor.example.org", asset_code="USDC")
    assert wd.flow_type == "withdrawal"
    assert len(wd.steps) == 4
