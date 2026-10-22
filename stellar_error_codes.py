"""Stable error codes and common error contract for Stellar network data tools (#587)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

from errors import APIException
from privacy_retention import redact_log_payload


class StellarErrorClass(str, Enum):
    """The five canonical error classes for Stellar network data tools."""

    INVALID_INPUT = "invalid_input"
    UNAVAILABLE_PROVIDER = "unavailable_provider"
    NOT_FOUND = "not_found"
    STALE_DATA = "stale_data"
    UNSUPPORTED_NETWORK = "unsupported_network"


# Stable internal error codes
CODE_INVALID_PUBLIC_KEY = "STELLAR_INVALID_PUBLIC_KEY"
CODE_SECRET_KEY_FORBIDDEN = "STELLAR_SECRET_KEY_FORBIDDEN"
CODE_MALFORMED_TX_HASH = "STELLAR_MALFORMED_TX_HASH"
CODE_INVALID_PARAMETER = "STELLAR_INVALID_PARAMETER"

CODE_PROVIDER_TIMEOUT = "STELLAR_PROVIDER_TIMEOUT"
CODE_PROVIDER_UNAVAILABLE = "STELLAR_PROVIDER_UNAVAILABLE"
CODE_PROVIDER_BAD_GATEWAY = "STELLAR_PROVIDER_BAD_GATEWAY"
CODE_RATE_LIMIT_EXCEEDED = "STELLAR_RATE_LIMIT_EXCEEDED"

CODE_ACCOUNT_NOT_FOUND = "STELLAR_ACCOUNT_NOT_FOUND"
CODE_TX_NOT_FOUND = "STELLAR_TX_NOT_FOUND"
CODE_RESOURCE_NOT_FOUND = "STELLAR_RESOURCE_NOT_FOUND"

CODE_LEDGER_STALE = "STELLAR_LEDGER_STALE"
CODE_CACHE_EXPIRED = "STELLAR_CACHE_EXPIRED"

CODE_UNSUPPORTED_NETWORK = "STELLAR_UNSUPPORTED_NETWORK"
CODE_CONTRADICTORY_NETWORK = "STELLAR_CONTRADICTORY_NETWORK"


@dataclass(frozen=True)
class ErrorPolicy:
    """Defines retryability, recommended user action, and default status."""

    retryable: bool
    user_action: str  # "change_request" | "retry_later" | "refresh_data" | "verify_network"
    http_status: int
    default_code: str
    default_message: str


ERROR_POLICIES: dict[StellarErrorClass, ErrorPolicy] = {
    StellarErrorClass.INVALID_INPUT: ErrorPolicy(
        retryable=False,
        user_action="change_request",
        http_status=400,
        default_code=CODE_INVALID_PARAMETER,
        default_message="The provided Stellar parameters are invalid. Please check the format and try again.",
    ),
    StellarErrorClass.UNAVAILABLE_PROVIDER: ErrorPolicy(
        retryable=True,
        user_action="retry_later",
        http_status=503,
        default_code=CODE_PROVIDER_UNAVAILABLE,
        default_message="The Stellar network provider is temporarily unavailable. Please retry after a brief delay.",
    ),
    StellarErrorClass.NOT_FOUND: ErrorPolicy(
        retryable=False,
        user_action="change_request",
        http_status=404,
        default_code=CODE_RESOURCE_NOT_FOUND,
        default_message="The requested Stellar account or transaction was not found on this network.",
    ),
    StellarErrorClass.STALE_DATA: ErrorPolicy(
        retryable=True,
        user_action="refresh_data",
        http_status=409,
        default_code=CODE_LEDGER_STALE,
        default_message="The cached Stellar ledger data is stale. Please request a fresh synchronization.",
    ),
    StellarErrorClass.UNSUPPORTED_NETWORK: ErrorPolicy(
        retryable=False,
        user_action="change_request",
        http_status=400,
        default_code=CODE_UNSUPPORTED_NETWORK,
        default_message="The requested Stellar network is unsupported. Supported networks are 'testnet' and 'public'.",
    ),
}


@dataclass(frozen=True)
class StellarToolErrorEnvelope:
    """Common error envelope across all Stellar network data tools."""

    error_code: str
    error_class: str
    message: str
    retryable: bool
    user_action: str
    http_status: int
    diagnostics: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_user_payload(self) -> dict[str, Any]:
        """User-safe error payload with internal server diagnostics omitted."""
        return {
            "error_code": self.error_code,
            "error_class": self.error_class,
            "message": self.message,
            "retryable": self.retryable,
            "user_action": self.user_action,
        }


def build_stellar_tool_error(
    error_class: StellarErrorClass | str,
    *,
    error_code: str | None = None,
    message: str | None = None,
    diagnostics: dict[str, Any] | None = None,
    http_status: int | None = None,
) -> StellarToolErrorEnvelope:
    """Build a validated StellarToolErrorEnvelope with server-side diagnostics sanitized."""
    cls_enum = StellarErrorClass(error_class) if isinstance(error_class, str) else error_class
    policy = ERROR_POLICIES[cls_enum]

    # Sanitize diagnostics so raw addresses, hashes, or secret keys never persist in logs/traces
    clean_diagnostics = redact_log_payload(diagnostics) if diagnostics else None

    return StellarToolErrorEnvelope(
        error_code=error_code or policy.default_code,
        error_class=cls_enum.value,
        message=message or policy.default_message,
        retryable=policy.retryable,
        user_action=policy.user_action,
        http_status=http_status or policy.http_status,
        diagnostics=clean_diagnostics,
    )


def map_transport_exception_to_error_envelope(
    exc: Exception | str,
    *,
    network: str = "testnet",
    context: str | None = None,
    diagnostics: dict[str, Any] | None = None,
) -> StellarToolErrorEnvelope:
    """Map transport-specific exceptions to stable internal codes and safe user messages."""
    raw_diag = dict(diagnostics or {})
    raw_diag["network"] = network
    if context:
        raw_diag["context"] = context

    exc_str = str(exc)
    exc_type = type(exc).__name__ if isinstance(exc, Exception) else "str"
    raw_diag["exception_type"] = exc_type
    raw_diag["exception_message"] = exc_str

    # 1. Secret key detection -> strictly invalid input
    if "S" in exc_str and len(exc_str) >= 56 and "secret" in exc_str.lower():
        return build_stellar_tool_error(
            StellarErrorClass.INVALID_INPUT,
            error_code=CODE_SECRET_KEY_FORBIDDEN,
            message="Secret keys are strictly prohibited in network data lookups.",
            diagnostics=raw_diag,
        )

    # 2. Unsupported network
    if network.strip().lower() not in {"testnet", "public"}:
        return build_stellar_tool_error(
            StellarErrorClass.UNSUPPORTED_NETWORK,
            error_code=CODE_UNSUPPORTED_NETWORK,
            message=f"Unsupported Stellar network '{network}'. Allowed networks: 'testnet', 'public'.",
            diagnostics=raw_diag,
        )

    # 3. Not Found (404, NotFoundError)
    if "notfound" in exc_type.lower() or "not found" in exc_str.lower() or "404" in exc_str:
        return build_stellar_tool_error(
            StellarErrorClass.NOT_FOUND,
            error_code=CODE_RESOURCE_NOT_FOUND,
            message=f"Requested resource was not found on Stellar {network}.",
            diagnostics=raw_diag,
        )

    # 4. Timeout -> unavailable provider
    if "timeout" in exc_type.lower() or "timeout" in exc_str.lower() or "timed out" in exc_str.lower():
        return build_stellar_tool_error(
            StellarErrorClass.UNAVAILABLE_PROVIDER,
            error_code=CODE_PROVIDER_TIMEOUT,
            message="The external Stellar Horizon/RPC provider request timed out. Please retry.",
            diagnostics=raw_diag,
        )

    # 5. Connect error / 502 / 503 -> unavailable provider
    if any(
        k in exc_type.lower() or k in exc_str.lower()
        for k in ("connecterror", "connection refused", "502", "503", "bad gateway")
    ):
        return build_stellar_tool_error(
            StellarErrorClass.UNAVAILABLE_PROVIDER,
            error_code=CODE_PROVIDER_UNAVAILABLE,
            message="Unable to reach the Stellar Horizon/RPC provider. Please retry later.",
            diagnostics=raw_diag,
        )

    # 6. Stale data
    if "stale" in exc_str.lower() or "expired" in exc_str.lower():
        return build_stellar_tool_error(
            StellarErrorClass.STALE_DATA,
            error_code=CODE_LEDGER_STALE,
            message="Stellar ledger data has expired and is no longer fresh. Please request fresh data.",
            diagnostics=raw_diag,
        )

    # Default to invalid input if ValueError or BadArgument, else provider error
    if isinstance(exc, (ValueError, KeyError)) or "invalid" in exc_str.lower():
        return build_stellar_tool_error(
            StellarErrorClass.INVALID_INPUT,
            error_code=CODE_INVALID_PARAMETER,
            message="Invalid parameters supplied to Stellar network tool.",
            diagnostics=raw_diag,
        )

    return build_stellar_tool_error(
        StellarErrorClass.UNAVAILABLE_PROVIDER,
        error_code=CODE_PROVIDER_UNAVAILABLE,
        message="Stellar provider error occurred during read operation.",
        diagnostics=raw_diag,
    )


def to_api_exception(envelope: StellarToolErrorEnvelope, hint: str | None = None) -> APIException:
    """Convert a StellarToolErrorEnvelope into an APIException for FastAPI dispatch."""
    user_payload = envelope.to_user_payload()
    action_hint = hint or f"Action: {envelope.user_action}. Retryable: {envelope.retryable}."
    return APIException(
        status_code=envelope.http_status,
        detail=user_payload,
        hint=action_hint,
    )
