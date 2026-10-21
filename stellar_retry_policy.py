"""Bounded retry and timeout policy for read-only Stellar network requests (#575)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class ReadOnlyTimeoutPolicy:
    """Explicit connection/read timeout and bounded backoff configuration."""

    connect_timeout_seconds: float = 2.0
    read_timeout_seconds: float = 4.0
    max_retries: int = 2
    base_backoff_seconds: float = 0.25
    max_backoff_seconds: float = 2.0

    def compute_backoff(self, attempt: int, jitter_fraction: float = 0.0) -> float:
        raw: float = min(
            self.max_backoff_seconds,
            self.base_backoff_seconds * float(2 ** max(0, attempt - 1)),
        )
        return raw * (1.0 + max(0.0, min(0.5, jitter_fraction)))


@dataclass(frozen=True)
class BoundedReadExecutionResult:
    """Outcome of executing a read-only provider request under bounded retries."""

    ok: bool
    attempts_used: int
    data: Any | None
    error_code: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def execute_bounded_readonly_request(
    *,
    is_idempotent_read: bool,
    policy: ReadOnlyTimeoutPolicy,
    operation: Callable[[float, float], Any],
) -> BoundedReadExecutionResult:
    """Execute an idempotent read operation with bounded retries on transient failure."""
    if not is_idempotent_read:
        raise ValueError("Retries are restricted strictly to idempotent read requests")

    max_attempts = 1 + max(0, policy.max_retries)
    attempts = 0
    for _ in range(max_attempts):
        attempts += 1
        try:
            res = operation(policy.connect_timeout_seconds, policy.read_timeout_seconds)
            return BoundedReadExecutionResult(ok=True, attempts_used=attempts, data=res)
        except TimeoutError:
            last_code = "READ_TIMEOUT_EXCEEDED"
        except ConnectionError:
            last_code = "TRANSIENT_CONNECTION_ERROR"
        except Exception:
            return BoundedReadExecutionResult(
                ok=False,
                attempts_used=attempts,
                data=None,
                error_code="NON_RETRYABLE_PROVIDER_ERROR",
            )

    return BoundedReadExecutionResult(
        ok=False,
        attempts_used=attempts,
        data=None,
        error_code=last_code,
    )
