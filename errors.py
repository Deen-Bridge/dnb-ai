"""Structured exceptions and error response helpers with actionable guidance."""

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
        super().__init__(status_code=status_code, detail=detail, headers=headers)
        self.hint = hint
