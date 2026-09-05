"""Structured exceptions and error response helpers with actionable guidance."

from __future__ import annotations

from typing import Any

from fastapi import HTTPException


class APIException(HTTPException):
    """HTTPException enriched with an actionable hint or suggestion."""

    def __init__(
        self,
        status_code: int,
        detail: Any = None,
        hint: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        super.__init__(status_code=status_code, detail=detail, headers=headers)
        self.hint = hint


class FallbackException(APIException):
    """Base exception for fallback model configuration and execution errors."""


class FallbackConfigurationError(FallbackException):
    """Raised when the fallback configuration is invalid or incomplete."""


class ModelUnavailableError(FallbackException):
    """Raised when a primary model is unavailable and fallback is triggered."""
    def __init__(self, model: str, status_code: int = 503, hint: str | None = None):
        detail = f"Model '{model}' is unavailable."
        if hint is None:
            hint = "The system is attempting to failover to a fallback model."
        super.__init__(status_code=status_code, detail=detail, hint=hint)


class NoAvailableModelError(FallbackException):
    """Raised when all configured models in the fallback chain are unavailable."""
    def __init__(self, model_type: str, status_code: int = 503):
        detail = f"No available model for model_type '{model_type}' after exhausting all fallbacks."
        hint = "Check the health of all models and the fallback chain configuration."
        super.__init__(status_code=status_code, detail=detail, hint=hint)


class HealthCheckFailedError(FallbackException):
    """Raised when a health check for a model fails."""
    def __init__(self, model: str, reason: str, status_code: int = 503):
        detail = f"Health check failed for model '{model}': {reason}"
        hint = "Review the model endpoint and its health check configuration."
        super.__init__(status_code=status_code, detail=detail, hint=hint)


class CircuitOpenError(FallbackException):
    """Raised when a circuit breaker is open for a specific model."""
    def __init__(self, model: str, retry_after: float, status_code: int = 503):
        detail = f"Circuit is open for model '{model}'; requests are not being forwarded."
        hint = f"Retry after {retry_after:.1f} seconds or use an alternative model."
        super.__init__(status_code=status_code, detail=detail, hint=hint)