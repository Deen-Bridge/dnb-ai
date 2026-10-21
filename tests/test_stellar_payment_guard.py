"""Tests for educational payment guidance vs signing/custody boundary (#525)."""

from __future__ import annotations

from stellar_payment_guard import build_educational_payment_guidance


def test_refuses_seed_phrase_and_secret_key_requests() -> None:
    res = build_educational_payment_guidance(
        "Here is my secret key SAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    )
    assert res.mode == "refused_key_or_seed_access"
    assert res.can_access_keys_or_sign is False
    assert res.can_broadcast_transaction is False
    assert res.payment_executed is False


def test_distinguishes_guidance_from_completed_payment() -> None:
    res = build_educational_payment_guidance("Send 50 USDC to fund a scholarship and confirm the transaction")
    assert res.mode == "educational_guidance_only"
    assert res.payment_executed is False
    assert "no transaction has been signed, broadcast, or confirmed" in res.guidance_message
    assert res.verified_app_route == "/scholarships/donate"
