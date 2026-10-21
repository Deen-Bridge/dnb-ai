"""OpenAPI tag catalog, endpoint request/response examples, and standard error schema models (#18)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class OpenAPIErrorModel:
    code: str
    message: str
    retryable: bool
    request_id: str | None = None


OPENAPI_TAGS_METADATA: list[dict[str, str]] = [
    {
        "name": "chat",
        "description": "Grounded Islamic Q&A, citation verification, and multi-madhhab fiqh guidance.",
    },
    {
        "name": "languages",
        "description": "Arabic OCR, calligraphy recognition, Swahili/Hausa/Malay processing, and dialect handling.",
    },
    {
        "name": "stellar",
        "description": "Deterministic Stellar USDC amount/fee calculation and educational scholarship guidance.",
    },
    {
        "name": "health",
        "description": "Service liveness, readiness, and model provider circuit-breaker diagnostics.",
    },
]

STANDARD_ERROR_EXAMPLES: dict[int, dict[str, Any]] = {
    400: {
        "description": "Validation or malformed input error",
        "content": {
            "application/json": {
                "example": {
                    "code": "invalid_request",
                    "message": "Input text exceeds maximum allowed length.",
                    "retryable": False,
                }
            }
        },
    },
    429: {
        "description": "Rate limit exceeded",
        "content": {
            "application/json": {
                "example": {
                    "code": "rate_limited",
                    "message": "Too many requests; please retry after backoff.",
                    "retryable": True,
                }
            }
        },
    },
    503: {
        "description": "Upstream model provider unavailable",
        "content": {
            "application/json": {
                "example": {
                    "code": "provider_unavailable",
                    "message": "AI provider timed out; safe fallback returned.",
                    "retryable": True,
                }
            }
        },
    },
}


def get_openapi_tags_and_error_responses() -> tuple[list[dict[str, str]], dict[int, dict[str, Any]]]:
    """Return canonical OpenAPI tags metadata and standardized HTTP error responses."""
    return list(OPENAPI_TAGS_METADATA), dict(STANDARD_ERROR_EXAMPLES)
