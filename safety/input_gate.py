"""Pre-generation classification and policy routing."""

import asyncio
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from .policy import Policy, PolicyCategory


Classifier = Callable[[str, List[str]], Dict[str, object]]


@dataclass(frozen=True)
class InputDecision:
    category_id: Optional[str]
    confidence: float
    action: str
    guidance: str = ""
    refusal: str = ""
    stages_fired: List[str] = field(default_factory=list)


class InputGate:
    def __init__(self, policy: Policy, classifier: Classifier):
        self.policy = policy
        self.classifier = classifier

    @staticmethod
    def _matches(text: str, patterns: List[str]) -> bool:
        return any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns)

    def evaluate(self, prompt: str) -> InputDecision:
        candidates = [
            category
            for category in self.policy.categories.values()
            if self._matches(prompt, category.keywords)
        ]
        if not candidates and self._matches(prompt, self.policy.benign_near_miss_patterns):
            return InputDecision(None, 1.0, "allow", stages_fired=["prefilter_benign"])
        if not candidates:
            return InputDecision(None, 1.0, "allow", stages_fired=["prefilter_clear"])

        try:
            result = self.classifier(prompt, [candidate.id for candidate in candidates])
            category = self._validated_category(result, candidates)
            if category is None:
                return InputDecision(
                    None,
                    float(result["confidence"]),
                    "allow",
                    stages_fired=["prefilter_match", "classifier_clear"],
                )
            action = str(result["action"])
            if action != category.action:
                action = category.action
            return self._decision(
                category,
                action,
                float(result["confidence"]),
                ["prefilter_match", "classifier"],
            )
        except Exception:
            # Ambiguous matches take the strictest documented category failure mode.
            category = max(candidates, key=lambda item: self._severity(item.failure_action))
            return self._decision(
                category,
                category.failure_action,
                0.0,
                ["prefilter_match", "classifier_failed", "policy_fallback"],
            )

    async def evaluate_async(self, prompt: str) -> InputDecision:
        """Evaluate without running the synchronous classifier on the event loop."""
        return await asyncio.to_thread(self.evaluate, prompt)

    def _validated_category(
        self, result: Dict[str, object], candidates: List[PolicyCategory]
    ) -> Optional[PolicyCategory]:
        required = {"category_id", "confidence", "action"}
        if set(result) != required:
            raise ValueError("Classifier response does not match the strict schema")
        confidence = float(result["confidence"])
        if not 0 <= confidence <= 1:
            raise ValueError("Classifier confidence must be between zero and one")
        if result["action"] not in {"allow", "allow_with_guidance", "refuse"}:
            raise ValueError("Classifier returned an invalid action")
        candidate_map = {candidate.id: candidate for candidate in candidates}
        category_id = str(result["category_id"])
        if category_id == "none" and result["action"] == "allow":
            return None
        if category_id not in candidate_map:
            raise ValueError("Classifier returned a category outside the prefilter candidates")
        return candidate_map[category_id]

    @staticmethod
    def _severity(action: str) -> int:
        return {"allow": 0, "allow_with_guidance": 1, "refuse": 2}[action]

    @staticmethod
    def _decision(
        category: PolicyCategory, action: str, confidence: float, stages: List[str]
    ) -> InputDecision:
        return InputDecision(
            category_id=category.id,
            confidence=confidence,
            action=action,
            guidance=category.guidance if action == "allow_with_guidance" else "",
            refusal=category.refusal if action == "refuse" else "",
            stages_fired=stages,
        )
