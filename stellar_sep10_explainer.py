"""Explain SEP-10 authentication without collecting secrets (#572)."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

_SECRET_SEED_PATTERN = re.compile(r"\bS[A-Z2-7]{55}\b")


@dataclass(frozen=True)
class Sep10ExplanationResponse:
    """Safe SEP-10 explanation or secret-input rejection."""

    status: str  # "ok" or "rejected_secret_input"
    steps: tuple[str, ...]
    safety_policy: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def explain_sep10_authentication(
    user_input: str | None = None,
) -> Sep10ExplanationResponse:
    """Explain SEP-10 challenge-response while rejecting any secret seed submission."""
    if user_input is not None and _SECRET_SEED_PATTERN.search(user_input):
        return Sep10ExplanationResponse(
            status="rejected_secret_input",
            steps=(),
            safety_policy=(
                "Rejected: Never paste or transmit a Stellar secret seed (S...). "
                "SEP-10 challenge signing must happen locally inside your wallet."
            ),
        )

    return Sep10ExplanationResponse(
        status="ok",
        steps=(
            "1. Challenge Retrieval: Client requests an unsigned SEP-10 challenge transaction (sequence 0) from the service's web_auth_endpoint.",
            "2. Local Wallet Signing: User's wallet verifies the service home_domain and signs the challenge locally without exposing the secret key.",
            "3. Token Exchange: Client submits the signed challenge XDR back to the auth endpoint to receive a time-bounded JWT session token.",
        ),
        safety_policy=(
            "SEP-10 is strictly non-custodial: the service and AI assistant never "
            "request, receive, or store your private key."
        ),
    )
