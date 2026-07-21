"""Post-generation enforcement of policy obligations."""

import re
from dataclasses import dataclass
from typing import List, Optional

from .input_gate import InputDecision
from .policy import Policy


@dataclass(frozen=True)
class OutputDecision:
    text: str
    stages_fired: List[str]
    category_id: Optional[str] = None
    action: Optional[str] = None


class OutputCheck:
    def __init__(self, policy: Policy):
        self.policy = policy

    def enforce(self, text: str, decision: InputDecision) -> OutputDecision:
        stages = []
        for violating_category in self.policy.categories.values():
            if not violating_category.refusal:
                continue
            for pattern in violating_category.output_patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    stages.append("policy_violation_replaced")
                    return OutputDecision(
                        violating_category.refusal,
                        stages,
                        category_id=violating_category.id,
                        action=violating_category.action,
                    )

        if decision.category_id == "DB-SAFE-001" and not self._has_scholar_referral(text):
            text = f"{text.rstrip()}\n\n{self.policy.scholar_referral_disclaimer}"
            stages.append("scholar_disclaimer_appended")

        stages.append("output_checked")
        return OutputDecision(text, stages)

    @staticmethod
    def _has_scholar_referral(text: str) -> bool:
        return bool(
            re.search(
                r"(?:consult|speak(?:ing)?|ask|contact|refer).{0,45}"
                r"(?:qualified|trusted|local)?\s*(?:islamic\s+)?scholar",
                text,
                re.IGNORECASE,
            )
        )
