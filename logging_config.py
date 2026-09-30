"""Structured JSON logging with per-request correlation ids (#11).

Three things happen here, and they are deliberately in one small
dependency-free module so it can be imported before anything else configures
logging:

* **JSON output.** :class:`JsonFormatter` renders every record as one line of
  valid JSON — ``timestamp``, ``level``, ``logger``, ``message``, plus whatever
  fields the call site passed via ``extra=``. Render's log search (and any
  aggregator added later) can then filter by field instead of grepping prose.
* **Request correlation.** A ``contextvars`` slot holds one id per request.
  :class:`RequestIdFilter` stamps it onto every record emitted anywhere during
  that request — including from modules that know nothing about HTTP — so a
  handler log line, a model-call line and the eventual traceback all carry the
  same ``request_id``. The same value goes out on the ``X-Request-ID``
  response header and is reused as the telemetry trace id, so logs, traces and
  headers correlate on one id rather than two.
* **Prompt redaction.** These are religious questions: sensitive personal
  content. Nothing here logs prompt text unless ``LOG_PROMPTS=true`` is set
  explicitly; :func:`prompt_debug_fields` is the only sanctioned way to attach
  it, and it returns an empty mapping by default.

Configuration (all optional):

``LOG_LEVEL``
    Root log level. Default ``INFO``.
``LOG_PROMPTS``
    ``true`` to include raw prompt text in log records. Default off.
``LOG_JSON``
    ``false`` to fall back to plain text, for a friendlier local console.
    Default on.
``LOG_ACCESS``
    ``true`` to keep uvicorn's own access log. Default off, because the
    ``request completed`` record below already carries method, path, status and
    duration — keeping both double-logs every request.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import sys
import time
import uuid
from contextvars import ContextVar, Token
from datetime import UTC, datetime
from typing import Any

# The header this service reads an inbound id from and echoes back on every
# response.
REQUEST_ID_HEADER = "X-Request-ID"

# An id arrives from outside the process, is echoed into a response header, and
# is written into logs. Constrain it so a caller cannot inject header
# delimiters or flood a log line: anything else is ignored in favour of a
# freshly minted id.
_SAFE_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{1,128}$")

_request_id: ContextVar[str | None] = ContextVar("request_id", default=None)

# Placeholder for records emitted outside any request (startup, background
# tasks). A constant beats omitting the key, so every line has the same shape.
NO_REQUEST_ID = "-"

# Attributes the logging module puts on every LogRecord. Anything else was
# supplied by a call site via ``extra=`` and belongs in the JSON payload.
_STANDARD_RECORD_FIELDS = frozenset(
    {
        "args",
        "asctime",
        "created",
        "exc_info",
        "exc_text",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "message",
        "module",
        "msecs",
        "msg",
        "name",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "taskName",
        "thread",
        "threadName",
        # Injected by RequestIdFilter; promoted to a top-level key instead.
        "request_id",
    }
)


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


# ---------------------------------------------------------------------------
# Request id
# ---------------------------------------------------------------------------
def new_request_id() -> str:
    """Mint an id for a request that did not arrive with one."""
    return uuid.uuid4().hex


def sanitize_request_id(value: str | None) -> str | None:
    """Return a caller-supplied id if it is safe to echo and log, else None."""
    if value is None:
        return None
    candidate = value.strip()
    return candidate if _SAFE_REQUEST_ID.match(candidate) else None


def get_request_id() -> str | None:
    """The id of the request being served, or None outside a request."""
    return _request_id.get()


def set_request_id(value: str) -> Token[str | None]:
    return _request_id.set(value)


def reset_request_id(token: Token[str | None]) -> None:
    _request_id.reset(token)


class RequestIdFilter(logging.Filter):
    """Stamp the current request's id onto every record that lacks one."""

    def filter(self, record: logging.LogRecord) -> bool:
        if not getattr(record, "request_id", None):
            record.request_id = _request_id.get() or NO_REQUEST_ID
        return True


# ---------------------------------------------------------------------------
# Prompt redaction
# ---------------------------------------------------------------------------
def log_prompts_enabled() -> bool:
    """True only when an operator opted in to raw-prompt logging."""
    return _env_flag("LOG_PROMPTS", False)


def prompt_debug_fields(prompt: str | None, key: str = "prompt") -> dict[str, str]:
    """Raw prompt text for a log record — empty unless LOG_PROMPTS is set.

    Call sites log ``prompt_chars`` unconditionally and splat this in, so the
    default deployment records the shape of a question and never its content.
    """
    if prompt is None or not log_prompts_enabled():
        return {}
    return {key: prompt}


# ---------------------------------------------------------------------------
# Formatter
# ---------------------------------------------------------------------------
def _json_safe(value: Any) -> Any:
    """Replace non-finite floats so the rendered line stays parseable JSON.

    ``json.dumps`` writes ``NaN``/``Infinity`` for them, which are Python
    literals rather than JSON, and a strict parser rejects the whole line. They
    are reachable in practice: a deployment can set ``LLM_PRICE_TABLE`` to
    non-finite prices, which flow into the ``cost_usd`` field of the telemetry
    log record. Rendering them as their string form keeps the value visible
    instead of silently dropping it.
    """
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


class JsonFormatter(logging.Formatter):
    """Render a LogRecord as a single line of JSON.

    Exceptions become a nested ``exception`` object carrying the full
    traceback, so a production 500 is diagnosable from the log alone instead of
    from a stringified error message.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": getattr(record, "request_id", NO_REQUEST_ID),
        }

        for key, value in record.__dict__.items():
            if key not in _STANDARD_RECORD_FIELDS and not key.startswith("_"):
                payload[key] = value

        if record.exc_info:
            exc_type, exc_value, _ = record.exc_info
            payload["exception"] = {
                "type": exc_type.__name__ if exc_type else None,
                "message": str(exc_value) if exc_value else None,
                "traceback": self.formatException(record.exc_info),
            }
        if record.stack_info:
            payload["stack"] = self.formatStack(record.stack_info)

        # default=str keeps one unserializable extra from killing the line, and
        # allow_nan=False refuses to emit the NaN/Infinity tokens that would
        # make it invalid JSON — _json_safe has already replaced those.
        try:
            return json.dumps(_json_safe(payload), default=str, ensure_ascii=False, allow_nan=False)
        except Exception:  # noqa: BLE001 - nothing a call site attaches may kill logging
            # A formatter that raises drops the record entirely, so this catches
            # everything: json.dumps raises TypeError/ValueError itself, but
            # default=str re-raises whatever a hostile __str__ does. One line
            # that is valid JSON but thin beats no line at all, and beats a
            # traceback on stderr.
            return json.dumps(
                {
                    "timestamp": payload["timestamp"],
                    "level": payload["level"],
                    "logger": payload["logger"],
                    "message": payload["message"],
                    "request_id": payload["request_id"],
                    "log_serialization_error": True,
                }
            )


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
_configured = False


def build_handler() -> logging.Handler:
    handler = logging.StreamHandler(sys.stdout)
    if _env_flag("LOG_JSON", True):
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [%(request_id)s] %(name)s: %(message)s"))
    handler.addFilter(RequestIdFilter())
    return handler


def configure_logging(force: bool = False) -> None:
    """Install the JSON handler as the process-wide logging configuration.

    Idempotent, and safe to call before or after uvicorn configures its own
    loggers: uvicorn's loggers are stripped of their handlers and left to
    propagate to the root, so their output is formatted here too and no line
    escapes as free-form text.
    """
    global _configured
    if _configured and not force:
        return

    level = getattr(logging, os.getenv("LOG_LEVEL", "INFO").strip().upper(), logging.INFO)

    root = logging.getLogger()
    for existing in list(root.handlers):
        root.removeHandler(existing)
    root.addHandler(build_handler())
    root.setLevel(level)

    for name in ("uvicorn", "uvicorn.error", "uvicorn.access", "gunicorn.error"):
        server_logger = logging.getLogger(name)
        server_logger.handlers.clear()
        server_logger.propagate = True

    # uvicorn's access log duplicates the `request completed` record emitted by
    # the request-context middleware, minus the request id. Drop it unless an
    # operator asks for it back.
    logging.getLogger("uvicorn.access").disabled = not _env_flag("LOG_ACCESS", False)

    _configured = True


# ---------------------------------------------------------------------------
# Request-scoped logging context
# ---------------------------------------------------------------------------
class RequestContextMiddleware:
    """Pure-ASGI middleware that gives every request an id and a log line.

    Deliberately not a Starlette ``BaseHTTPMiddleware`` subclass: that wrapper
    buffers responses in ways that interfere with the SSE stream served by
    ``/chat/stream``. A raw ASGI callable passes the message stream straight
    through, so streaming behaviour is unchanged and the completion record is
    emitted once the last chunk has actually been sent.

    Per request it: honours an inbound ``X-Request-ID`` (or mints one), binds
    it for the duration so every log record carries it, echoes it on the
    response header, and logs one ``request completed`` record with the status
    code and duration.
    """

    def __init__(self, app: Any) -> None:
        self.app = app
        self._logger = logging.getLogger("request")

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        request_id = sanitize_request_id(_header(scope, b"x-request-id")) or new_request_id()
        token = set_request_id(request_id)
        started = time.perf_counter()
        status_code = 500
        header_pair = (REQUEST_ID_HEADER.lower().encode("latin-1"), request_id.encode("latin-1"))

        async def send_with_request_id(message: Any) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
                message["headers"] = [*message.get("headers", []), header_pair]
            await send(message)

        context = {
            "http_method": scope.get("method"),
            "path": scope.get("path"),
        }
        try:
            await self.app(scope, receive, send_with_request_id)
        except Exception:
            # The stack trace belongs in the log; the client still gets
            # whatever the app's own handlers decided to return.
            self._logger.exception(
                "request failed",
                extra={**context, "status_code": status_code, "duration_ms": _elapsed_ms(started)},
            )
            raise
        else:
            self._logger.info(
                "request completed",
                extra={**context, "status_code": status_code, "duration_ms": _elapsed_ms(started)},
            )
        finally:
            reset_request_id(token)


def _header(scope: Any, name: bytes) -> str | None:
    for key, value in scope.get("headers") or []:
        if key.lower() == name:
            return value.decode("latin-1")
    return None


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000.0, 2)
