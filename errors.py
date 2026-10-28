"""Application exceptions and the shared HTTP error response convention.

Raise :class:`DomainError` (or its :class:`APIException` convenience subclass)
for expected failures. FastAPI serializes these and framework HTTP errors using
the same response shape: ``{"code": ..., "detail": ..., "hint": ...}``.
Keep details and hints safe for clients; unexpected exceptions are logged and
returned as a generic 500 response by the handler in ``main.py``.
"""

from __future__ import annotations

from typing import Any

from fastapi import HTTPException


class DomainError(HTTPException):
    """Expected application failure with an HTTP status and stable error code."""

    def __init__(
        self,
        status_code: int,
        detail: Any = None,
        *,
        code: str = "domain_error",
        hint: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(status_code=status_code, detail=detail, headers=headers)
        self.code = code
        self.hint = hint


class APIException(DomainError):
    """Backward-compatible name for an expected API/domain failure."""

    def __init__(
        self,
        status_code: int,
        detail: Any = None,
        hint: str | None = None,
        headers: dict[str, str] | None = None,
        code: str = "api_error",
    ) -> None:
        super().__init__(status_code, detail, code=code, hint=hint, headers=headers)


def error_payload(
    detail: Any,
    *,
    code: str = "http_error",
    hint: str | None = None,
) -> dict[str, Any]:
    """Build the stable error body while omitting optional fields when absent."""
    payload: dict[str, Any] = {"code": code, "detail": detail}
    if hint:
        payload["hint"] = hint
    return payload
