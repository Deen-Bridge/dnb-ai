"""A single orchestration seam for input and output safety stages."""

from dataclasses import dataclass
from time import perf_counter
from typing import Callable, List, Optional

from .input_gate import InputDecision, InputGate
from .output_check import OutputCheck


@dataclass(frozen=True)
class SafetyResult:
    text: str
    category_id: Optional[str]
    action: str
    confidence: float
    stages_fired: List[str]
    latency_ms: float
    generator_called: bool


class SafetyPipeline:
    def __init__(self, input_gate: InputGate, output_check: OutputCheck):
        self.input_gate = input_gate
        self.output_check = output_check

    def run(self, prompt: str, generator: Callable[[str], str]) -> SafetyResult:
        started = perf_counter()
        decision = self.input_gate.evaluate(prompt)
        stages = list(decision.stages_fired)

        if decision.action == "refuse":
            return self._result(decision.refusal, decision, stages, started, False)

        generation_prompt = prompt
        if decision.action == "allow_with_guidance":
            generation_prompt = f"{decision.guidance}\n\nUser question: {prompt}"
            stages.append("guidance_injected")

        generated = generator(generation_prompt)
        checked = self.output_check.enforce(generated, decision)
        stages.extend(checked.stages_fired)
        return self._result(checked.text, decision, stages, started, True)

    @staticmethod
    def _result(text, decision, stages, started, generator_called):
        return SafetyResult(
            text=text,
            category_id=decision.category_id,
            action=decision.action,
            confidence=decision.confidence,
            stages_fired=stages,
            latency_ms=round((perf_counter() - started) * 1000, 2),
            generator_called=generator_called,
        )
