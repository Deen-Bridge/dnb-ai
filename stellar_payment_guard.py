"""Keep educational Stellar payment guidance strictly separate from signing/custody (#525, #589)."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

FORBIDDEN_CUSTODY_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b(?:seed\s+phrase|secret\s+key|private\s+key|mnemonic)\b", re.IGNORECASE),
    re.compile(r"\bS[A-Z2-7]{55}\b"),
)


class DonationProductIntent(str, Enum):
    """Product intents separating direct charity from milestone escrow."""

    SADAQAH = "sadaqah"
    GENERAL_DONATION = "general_donation"
    SCHOLARSHIP_CONTRIBUTION = "scholarship_contribution"


SADAQAH_KEYWORDS: tuple[str, ...] = (
    "sadaqah",
    "sadaqa",
    "charity",
    "lillah",
    "voluntary charity",
    "direct gift",
)

SCHOLARSHIP_KEYWORDS: tuple[str, ...] = (
    "scholarship",
    "tuition",
    "student",
    "escrow",
    "milestone",
    "grant",
    "academic",
)


def classify_donation_product_intent(user_prompt: str) -> DonationProductIntent:
    """Classify user prompt into sadaqah, general donation, or scholarship contribution."""
    prompt_lower = user_prompt.lower()

    if any(k in prompt_lower for k in SCHOLARSHIP_KEYWORDS):
        return DonationProductIntent.SCHOLARSHIP_CONTRIBUTION
    if any(k in prompt_lower for k in SADAQAH_KEYWORDS):
        return DonationProductIntent.SADAQAH
    return DonationProductIntent.GENERAL_DONATION


@dataclass(frozen=True)
class PaymentGuidanceResponse:
    mode: str
    can_access_keys_or_sign: bool
    can_broadcast_transaction: bool
    payment_executed: bool
    verified_app_route: str
    guidance_message: str
    intent: str = "general_donation"


@dataclass(frozen=True)
class ProductPaymentExplanation:
    """Distinct response reflecting product intent and verified milestone status."""

    intent: str
    is_escrow: bool
    verified_app_route: str
    explanation: str
    escrow_status: str | None = None
    beneficiary: str | None = None
    arbiter: str | None = None
    milestones: tuple[dict[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def explain_product_payment_flow(
    user_prompt: str,
    *,
    verified_contract_data: dict[str, Any] | None = None,
) -> ProductPaymentExplanation:
    """Provide separate explanations for sadaqah donations and scholarship escrow.

    Invariants:
    1. Sadaqah is explained as a direct voluntary gift without escrow or milestones.
    2. Scholarship funding is held in conditional escrow; milestones are explained
       ONLY when verified contract data is provided. Otherwise, status is reported as unavailable.
    3. Never invents a beneficiary, arbiter, milestone, or payment status.
    """
    intent = classify_donation_product_intent(user_prompt)

    if intent == DonationProductIntent.SADAQAH:
        return ProductPaymentExplanation(
            intent=intent.value,
            is_escrow=False,
            verified_app_route="/charity/sadaqah",
            explanation=(
                "Sadaqah is a voluntary, direct charitable contribution on Stellar. "
                "Payments are transferred directly in USDC to verified charitable relief or community programs "
                "without conditional milestone locks or arbiter approval."
            ),
            escrow_status=None,
            beneficiary=(
                str(verified_contract_data["beneficiary"])
                if verified_contract_data and "beneficiary" in verified_contract_data
                else None
            ),
            arbiter=None,
            milestones=(),
        )

    if intent == DonationProductIntent.GENERAL_DONATION:
        return ProductPaymentExplanation(
            intent=intent.value,
            is_escrow=False,
            verified_app_route="/donate",
            explanation=(
                "General donations support platform operations and community initiatives directly. "
                "Funds are settled immediately on the Stellar network and do not pass through "
                "student escrow milestone contracts."
            ),
            escrow_status=None,
            beneficiary=(
                str(verified_contract_data["beneficiary"])
                if verified_contract_data and "beneficiary" in verified_contract_data
                else None
            ),
            arbiter=None,
            milestones=(),
        )

    # Scholarship contribution (Milestone Escrow)
    if verified_contract_data and verified_contract_data.get("milestones"):
        milestone_list = tuple(verified_contract_data["milestones"])
        return ProductPaymentExplanation(
            intent=intent.value,
            is_escrow=True,
            verified_app_route="/scholarships/escrow",
            explanation=(
                f"Scholarship funding is held in a conditional Stellar escrow smart contract with "
                f"{len(milestone_list)} verified milestone(s). Funds are disbursed sequentially as academic "
                f"proofs are approved by the designated arbiter."
            ),
            escrow_status="verified",
            beneficiary=str(verified_contract_data.get("beneficiary", "")),
            arbiter=str(verified_contract_data.get("arbiter", "")),
            milestones=milestone_list,
        )

    # Milestone data not available -> must state status is unavailable, NEVER invent
    return ProductPaymentExplanation(
        intent=intent.value,
        is_escrow=True,
        verified_app_route="/scholarships/escrow",
        explanation=(
            "Scholarship contributions require conditional milestone-based escrow. "
            "Escrow milestone verification status is currently unavailable; "
            "DeenBridge AI never fabricates beneficiaries, arbiters, or milestone conditions."
        ),
        escrow_status="status_unavailable",
        beneficiary=None,
        arbiter=None,
        milestones=(),
    )


def build_educational_payment_guidance(user_prompt: str) -> PaymentGuidanceResponse:
    """Return educational-only Stellar USDC steps while refusing custody, signing, or confirmation claims."""
    if any(p.search(user_prompt) for p in FORBIDDEN_CUSTODY_PATTERNS):
        return PaymentGuidanceResponse(
            mode="refused_key_or_seed_access",
            can_access_keys_or_sign=False,
            can_broadcast_transaction=False,
            payment_executed=False,
            verified_app_route="/wallet/security",
            guidance_message=(
                "DeenBridge AI never requests, views, or stores secret keys or seed phrases. "
                "Please sign transactions exclusively inside your non-custodial Stellar wallet UI."
            ),
            intent="security_refusal",
        )

    intent = classify_donation_product_intent(user_prompt)
    if intent == DonationProductIntent.SCHOLARSHIP_CONTRIBUTION:
        route = "/scholarships/donate"
        msg = (
            "Educational guidance only: no transaction has been signed, broadcast, or confirmed by AI. "
            "To complete or verify a Stellar USDC transfer, open the verified /scholarships/donate wallet screen."
        )
    elif intent == DonationProductIntent.SADAQAH:
        route = "/charity/sadaqah"
        msg = (
            "Educational guidance only: no transaction has been signed, broadcast, or confirmed by AI. "
            "To complete a direct voluntary sadaqah payment, open the verified /charity/sadaqah screen."
        )
    else:
        route = "/donate"
        msg = (
            "Educational guidance only: no transaction has been signed, broadcast, or confirmed by AI. "
            "To complete a general donation, open the verified /donate screen."
        )

    return PaymentGuidanceResponse(
        mode="educational_guidance_only",
        can_access_keys_or_sign=False,
        can_broadcast_transaction=False,
        payment_executed=False,
        verified_app_route=route,
        guidance_message=msg,
        intent=intent.value,
    )
