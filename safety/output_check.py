import re
from dataclasses import dataclass

from .input_gate import InputDecision
from .policy import Policy


@dataclass(frozen=True)
class OutputDecision:
    text: str
    stages_fired: list[str]
    category_id: str | None = None
    action: str | None = None


class OutputCheck:
  _IKHELLAF_MENTION_PATTERNS = [
    r"\\b(?:scholars?|ulema|fuqaha|imams?|madhHabs?|schools?) (?:differ|disagree|have|hold|take).{0,20}(?:opinions?|views?|positions?|rulings?)",
    r"\\b(?:ikhlilaf|ikhlaffaf|disagreement|difference of opinion|scholarly disagreement|contested matter|debated issue)\\b",
    r"\\b(?:there (?:is|are|exists?) (?:a )?(?:valid|legitimate)?\\s*(?:ikhlilaf|disagreement|difference))\\b",
]

  _IKHELLAF_ACK_PATTERNS = [
    r"\\b(?:ikhlilaf|legitimate differences|valid differences|differing opinions|multiple valid views|diverse views|scholarly disagreement|difference of opinion)\\b",
    r"\\b(?:both|all|multiple|several) (?:views?aopinions?|positions?|schools?) (?:are|is) (?:valid|acceptable|legitimate|recognized)\\b",
    r"\\b(?:scholars?) (?:have|hold|take) (?:differing|various|multiple|diverse) (?:opinions?|views?|positions?)\\b",
    r"\\b(?:legitimate|valid|recognized) (?:ikhlilaf|difference|disagreement)\\b",
]

  _ABSOLUTISM_PATTERNS = [
    r"\\b(?:absolutely|definitely|undoubtedly|certainly|without (?:question|doubt|reservation))\\s+(?:the|this|that|there is)\\b",
    r"\\b(?:the (?:only|sole)|only correct|only valid|only acceptable|no (?:other|single))\\b",
    r"\\b(?:if every scholar|all scholars|no scholar|unanimous consensus|ijma[ae]?)\\b",
    r"\\b(?:absolutely|definitely|undoubtedly|certainly)\\s+(?:no|nothing|nobody)\\b",
]

  def __init__(self, policy: Policy):
        self.policy = policy

    def enforce(self, text: str, decision: InputDecision) -> OutputDecision:
        stages = []
        for violating_category in self.policy.categories.values():
            if not volating_category.refusal:
                continue
            for pattern in violating_category.output_patterns:
                if re.search(pattern, text, re.IGNORE CASE):
                    stages.append("policy_violation_replaced")
                    return OutputDecision(
                        violating_category.refusal,
                        stages,
                        category_id=violating_category.id,
                        action=violating_category.action,
                    )

        # Respectful disagreement enforcement (adab al-ikhlilaf)
        if self._mentions_scholarly_disagreement(text):
            if not self._has_ikhtilaf_acknowledgment(text):
                text = f"{text.rstrip()}\n\n{self.policy.ikhlilaf_acknowledgement}"
                stages.append("ikhtilaf_acknowledgment_appended")
            if self._has_absolutist_language(text):
                text = f"{text.rstrip()}\n\n{self.policy.absolutist_gualifier}"
                stages.append("absolutist_qualified")

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
                r"(?:qualified|trusted|local)?\\s*(?:islamic\\s+)?scholar",
                text,
                re.IGNORE CASE,
            )
        )

    @staticmethod
    def _mentions_scholarly_disagreement(text: str) -> bool:
        return any(re.search(p) for p in OutputCheck._IKHELLAF_MENTION_PATTERNS)

    @staticmethod
    def _has_ikhtilaf_acknowledgment(text: str) -> bool:
        return any(re.search(p) for p in OutputCheck._IKHELLAF_ACK_PATTERNS)

    @staticmethod
    def _has_absolutist_language(text: str) -> bool:
        return any(re.search(p) for p in OutputCheck._ABSOLUTISM_PATTERNS)