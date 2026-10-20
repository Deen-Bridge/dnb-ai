"""Shared slowapi limiter and client-key helpers.

Kept outside ``main.py`` so router modules can import the same limiter
without importing the application module (which would be circular).

Limits are in-process and per instance. A shared backend (for example Redis)
is required before running more than one replica.
"""

from __future__ import annotations

import inspect
import os
from collections.abc import Callable
from functools import wraps
from typing import Any, get_type_hints

from fastapi import Request
from slowapi import Limiter
from slowapi.util import get_remote_address

# Only honor X-Forwarded-For when the deployment sits behind a proxy we
# control. Otherwise any client can rotate the header to mint a fresh bucket.
TRUST_PROXY_HEADERS = os.getenv("TRUST_PROXY_HEADERS", "false").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
RATE_LIMIT_ENABLED = os.getenv("RATE_LIMIT_ENABLED", "true").lower() not in {
    "0",
    "false",
    "off",
}


def client_key(request: Request) -> str:
    """Return the slowapi bucket key for a request.

    Order: API key, then the trusted proxy's client address, then the direct
    peer. The rightmost ``X-Forwarded-For`` hop is used because the trusted
    proxy appends it; the leftmost value is client-controlled and would let a
    caller rotate buckets at will.
    """

    api_key = request.headers.get("X-API-Key")
    if api_key:
        return f"key:{api_key}"

    if TRUST_PROXY_HEADERS:
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            candidate = forwarded_for.split(",")[-1].strip()
            if candidate:
                return candidate

    return get_remote_address(request)


limiter = Limiter(key_func=client_key, enabled=RATE_LIMIT_ENABLED)


def limit(limit_value: str | Callable[..., str]) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Apply a slowapi limit while preserving resolved type annotations.

    Router modules use ``from __future__ import annotations``, so their
    annotations are strings resolved by FastAPI against the endpoint function's
    ``__globals__``. slowapi's wrapper lives in ``slowapi.extension`` and would
    leave ``UploadFile``/Pydantic forward references unresolved. Resolve the
    endpoint's hints in its own module first and expose that signature on the
    wrapper, which is what FastAPI inspects.
    """

    slow_decorator = limiter.limit(limit_value)

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        limited = slow_decorator(func)

        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Unit tests call handlers directly with a Pydantic body and no
            # Request. Only enforce the HTTP limit when a real Request is
            # present; FastAPI always supplies one for live calls.
            has_request = any(isinstance(arg, Request) for arg in args) or any(
                isinstance(value, Request) for value in kwargs.values()
            )
            if not has_request:
                # Direct unit calls pass only the Pydantic body; shift it past
                # the leading HTTP request parameter (handlers never read it).
                parameters = list(inspect.signature(func).parameters)
                if parameters and parameters[0] == "request":
                    return await func(None, *args, **kwargs)  # type: ignore[arg-type]
                return await func(*args, **kwargs)
            return await limited(*args, **kwargs)

        signature = inspect.signature(func)
        try:
            hints = get_type_hints(func, include_extras=True)
        except Exception:  # pragma: no cover - fall back to raw annotations
            hints = {}
        wrapper.__signature__ = signature.replace(  # type: ignore[attr-defined]
            parameters=[
                parameter.replace(annotation=hints.get(parameter.name, parameter.annotation))
                for parameter in signature.parameters.values()
            ],
            return_annotation=hints.get("return", signature.return_annotation),
        )

        return wrapper

    return decorator


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


# Tiered defaults for endpoints that are expensive but not as hot as /chat.
# Windows stay at 60 seconds so the RateLimitExceeded handler's Retry-After
# value remains accurate.
LLM_RATE_LIMIT_MAX = _env_int("LLM_RATE_LIMIT_MAX", 10)
LLM_RATE_LIMIT_WINDOW_SECONDS = _env_int("LLM_RATE_LIMIT_WINDOW_SECONDS", 60)
ANALYSIS_RATE_LIMIT_MAX = _env_int("ANALYSIS_RATE_LIMIT_MAX", 30)
ANALYSIS_RATE_LIMIT_WINDOW_SECONDS = _env_int("ANALYSIS_RATE_LIMIT_WINDOW_SECONDS", 60)
MEDIA_RATE_LIMIT_MAX = _env_int("MEDIA_RATE_LIMIT_MAX", 10)
MEDIA_RATE_LIMIT_WINDOW_SECONDS = _env_int("MEDIA_RATE_LIMIT_WINDOW_SECONDS", 60)


def llm_rate_limit() -> str:
    """slowapi limit string for direct LLM-generation endpoints."""

    return f"{LLM_RATE_LIMIT_MAX}/{LLM_RATE_LIMIT_WINDOW_SECONDS} seconds"


def analysis_rate_limit() -> str:
    """slowapi limit string for external/corpus analysis endpoints."""

    return f"{ANALYSIS_RATE_LIMIT_MAX}/{ANALYSIS_RATE_LIMIT_WINDOW_SECONDS} seconds"


def media_rate_limit() -> str:
    """slowapi limit string for upload/media-processing endpoints."""

    return f"{MEDIA_RATE_LIMIT_MAX}/{MEDIA_RATE_LIMIT_WINDOW_SECONDS} seconds"
