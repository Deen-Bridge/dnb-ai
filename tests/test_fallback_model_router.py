"""Tests for ordered fallback model configuration chain (#200)."""

from __future__ import annotations

import pytest

from fallback_model_router import FallbackModelChain, ModelEndpointConfig


def test_fails_over_when_primary_exceeds_circuit_breaker_threshold() -> None:
    chain = FallbackModelChain(
        endpoints=[
            ModelEndpointConfig("gemini", "gemini-2.5-flash"),
            ModelEndpointConfig("gemini", "gemini-2.5-pro"),
        ],
        circuit_breaker_threshold=2,
    )
    assert chain.select_active_endpoint().model_name == "gemini-2.5-flash"
    chain.record_failure("gemini-2.5-flash")
    chain.record_failure("gemini-2.5-flash")
    assert chain.select_active_endpoint().model_name == "gemini-2.5-pro"


def test_raises_when_all_models_unhealthy() -> None:
    chain = FallbackModelChain(
        endpoints=[ModelEndpointConfig("gemini", "gemini-2.5-flash")],
        circuit_breaker_threshold=1,
    )
    chain.record_failure("gemini-2.5-flash")
    with pytest.raises(RuntimeError):
        chain.select_active_endpoint()
