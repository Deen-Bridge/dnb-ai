"""Ordered fallback model configuration with per-provider circuit breaker (#200)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelEndpointConfig:
    provider: str
    model_name: str
    timeout_seconds: float = 10.0
    max_retries: int = 2


@dataclass
class FallbackModelChain:
    endpoints: list[ModelEndpointConfig]
    failure_counts: dict[str, int] = field(default_factory=dict)
    circuit_breaker_threshold: int = 3

    def record_failure(self, model_name: str) -> None:
        self.failure_counts[model_name] = self.failure_counts.get(model_name, 0) + 1

    def record_success(self, model_name: str) -> None:
        self.failure_counts[model_name] = 0

    def is_healthy(self, model_name: str) -> bool:
        return self.failure_counts.get(model_name, 0) < self.circuit_breaker_threshold

    def select_active_endpoint(self) -> ModelEndpointConfig:
        for ep in self.endpoints:
            if self.is_healthy(ep.model_name):
                return ep
        raise RuntimeError("All configured primary and fallback model endpoints are unhealthy.")
