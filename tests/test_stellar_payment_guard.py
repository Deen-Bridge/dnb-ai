"""Tests for educational payment guidance vs signing/custody boundary and separate donation/escrow flows (#525, #589)."""

from __future__ import annotations

from stellar_payment_guard import (
    DonationProductIntent,
    build_educational_payment_guidance,
    classify_donation_product_intent,
    explain_product_payment_flow,
)


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


def test_separate_intent_classification() -> None:
    assert (
        classify_donation_product_intent("I want to give 20 USDC sadaqah for the poor") == DonationProductIntent.SADAQAH
    )
    assert (
        classify_donation_product_intent("I want to contribute 100 USDC to support student scholarship funding")
        == DonationProductIntent.SCHOLARSHIP_CONTRIBUTION
    )
    assert (
        classify_donation_product_intent("I want to make a general platform contribution")
        == DonationProductIntent.GENERAL_DONATION
    )


def test_sadaqah_flow_is_direct_and_not_escrow() -> None:
    res = explain_product_payment_flow("Give 10 USDC sadaqah to the local community")
    assert res.intent == "sadaqah"
    assert res.is_escrow is False
    assert res.verified_app_route == "/charity/sadaqah"
    assert "without conditional milestone locks" in res.explanation
    assert res.escrow_status is None
    assert res.arbiter is None
    assert res.milestones == ()


def test_scholarship_with_verified_contract_explains_milestones() -> None:
    verified_data = {
        "beneficiary": "G_STUDENT_WALLET_1234567890",
        "arbiter": "G_ARBITER_FACULTY_1234567890",
        "milestones": [
            {"milestone_id": 1, "description": "Semester 1 Completion", "payout_usdc": "500"},
            {"milestone_id": 2, "description": "Graduation", "payout_usdc": "500"},
        ],
    }
    res = explain_product_payment_flow(
        "How is my scholarship contribution disbursed?", verified_contract_data=verified_data
    )
    assert res.intent == "scholarship_contribution"
    assert res.is_escrow is True
    assert res.verified_app_route == "/scholarships/escrow"
    assert res.escrow_status == "verified"
    assert res.beneficiary == "G_STUDENT_WALLET_1234567890"
    assert res.arbiter == "G_ARBITER_FACULTY_1234567890"
    assert len(res.milestones) == 2


def test_scholarship_without_verified_data_never_invents_milestones() -> None:
    res = explain_product_payment_flow("Explain the escrow milestones for student scholarship #99")
    assert res.intent == "scholarship_contribution"
    assert res.is_escrow is True
    assert res.escrow_status == "status_unavailable"
    assert res.beneficiary is None
    assert res.arbiter is None
    assert res.milestones == ()
    assert "status is currently unavailable" in res.explanation
    assert "never fabricates" in res.explanation


def test_donation_guidance_cannot_be_mistaken_for_scholarship_escrow() -> None:
    sadaqah_guidance = build_educational_payment_guidance("I want to give sadaqah")
    assert sadaqah_guidance.verified_app_route == "/charity/sadaqah"
    assert "escrow" not in sadaqah_guidance.guidance_message.lower()

    scholarship_guidance = build_educational_payment_guidance("I want to fund a scholarship")
    assert scholarship_guidance.verified_app_route == "/scholarships/donate"
