"""Safety policy and prompt-injection hardening gate (#94)."""

from __future__ import annotations

import re
from dataclasses import dataclass

BLOCKED_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("jailbreak_override", re.compile(r"\b(?:dan\s+mode|developer\s+mode|bypass\s+safety)\b", re.IGNORECASE)),
    ("system_prompt_exfil", re.compile(r"\b(?:print|reveal|dump)\s+(?:your\s+)?system\s+prompt\b", re.IGNORECASE)),
    ("fabricated_fatwa_coercion", re.compile(r"\bdeclare\s+(?:halal|haram)\s+without\s+evidence\b", re.IGNORECASE)),
)


@dataclass(frozen=True)
class SafetyGateDecision:
    allowed: bool
    violated_rule: str | None
    sanitized_prompt: str


def evaluate_safety_policy_gate(user_input: str) -> SafetyGateDecision:
    """Reject prompt-injection and policy-bypass attempts deterministically."""
    cleaned = user_input.replace("\x00", "").strip()
    for rule_name, pattern in BLOCKED_PATTERNS:
        if pattern.search(cleaned):
            return SafetyGateDecision(
                allowed=False,
                violated_rule=rule_name,
                sanitized_prompt="",
            )
    return SafetyGateDecision(
        allowed=True,
        violated_rule=None,
        sanitized_prompt=cleaned,
    )
