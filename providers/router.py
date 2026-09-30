import time
from dataclasses import dataclass

from .types import GenerationConfig, LLMProvider, Message, ProviderReply


@dataclass
class ProviderHealth:
    name: str
    failures: int = 0
    successes: int = 0
    state: str = "closed"
    opened_at: float | None = None

    @property
    def success_rate(self) -> float:
        total = self.failures + self.successes
        return self.successes / total if total else 0.0


class ProviderRouter:
    def __init__(
        self,
        providers: list[LLMProvider],
        *,
        failure_threshold: int = 3,
        cooldown_seconds: float = 60.0,
    ) -> None:
        if not providers:
            raise ValueError("At least one provider is required")
        self.providers = providers
        self.failure_threshold = failure_threshold
        self.cooldown_seconds = cooldown_seconds
        self.health = {provider.name: ProviderHealth(provider.name) for provider in providers}

    def status(self) -> list[dict[str, object]]:
        return [
            {
                "provider": health.name,
                "state": health.state,
                "failure_count": health.failures,
                "success_count": health.successes,
                "success_rate": health.success_rate,
            }
            for health in self.health.values()
        ]

    def _available(self, provider: LLMProvider, now: float) -> bool:
        health = self.health[provider.name]
        if health.state != "open":
            return True
        if health.opened_at is not None and now - health.opened_at >= self.cooldown_seconds:
            health.state = "half_open"
            return True
        return False

    async def generate(
        self,
        messages: list[Message],
        *,
        system: str | None,
        config: GenerationConfig,
    ) -> ProviderReply:
        errors: list[Exception] = []
        for provider in self.providers:
            if not self._available(provider, time.monotonic()):
                continue
            health = self.health[provider.name]
            try:
                reply = await provider.generate(messages, system=system, config=config)
            except Exception as exc:  # provider failures are intentionally isolated
                health.failures += 1
                if health.failures >= self.failure_threshold:
                    health.state = "open"
                    health.opened_at = time.monotonic()
                errors.append(exc)
                continue
            health.successes += 1
            health.failures = 0
            health.state = "closed"
            health.opened_at = None
            return reply
        raise RuntimeError("All configured LLM providers failed") from (errors[-1] if errors else None)
