"""Keep educational Stellar payment guidance strictly separate from signing/custody (#525)."""

from __future__ import annotations

import re
from dataclasses import dataclass

FORBIDDEN_CUSTODY_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b(?:seed\s+phrase|secret\s+key|private\s+key|mnemonic)\b", re.IGNORECASE),
    re.compile(r"\bS[A-Z2-7]{55}\b"),
)


@dataclass(frozen=True)
class PaymentGuidanceResponse:
    mode: str
    can_access_keys_or_sign: bool
    can_broadcast_transaction: bool
    payment_executed: bool
    verified_app_route: str
    guidance_message: str


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
        )

    return PaymentGuidanceResponse(
        mode="educational_guidance_only",
        can_access_keys_or_sign=False,
        can_broadcast_transaction=False,
        payment_executed=False,
        verified_app_route="/scholarships/donate",
        guidance_message=(
            "Educational guidance only: no transaction has been signed, broadcast, or confirmed by AI. "
            "To complete or verify a Stellar USDC transfer, open the verified /scholarships/donate wallet screen."
        ),
    )
