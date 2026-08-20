"""LLM observability: per-request token, cost, and latency telemetry.

This module is deliberately dependency-free (standard library only) so it adds
no footprint to the small Render service. It captures what every model call
costs and how long it took, records lightweight per-stage spans, and keeps an
in-memory aggregate that ``GET /metrics`` can surface.

Redaction is a design invariant, not an afterthought: every public function
here takes counts, durations, model names, and trace IDs. Prompt and answer
text never enter this module, so it is impossible for content to leak into a
metric label or the spans logged from here. Related work:

* #11 owns structured logging + request/trace IDs, and now supplies them: a
  trace minted inside a request reuses that request's id (see
  ``new_trace_id``), so ``X-Trace-Id``, ``X-Request-ID`` and every log line
  carry one value instead of two.
* #13 owns token counting for prompt-budget purposes; we only read the actual
  ``usage_metadata`` the Gemini SDK already returns, so we do not duplicate it.
* #9 (auth/rate limiting) and #13 can consume ``estimate_cost`` /
  ``MetricsRegistry`` without this module wiring any enforcement itself.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

from logging_config import get_request_id

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cost table
# ---------------------------------------------------------------------------
# Prices are in US dollars per 1,000 tokens, keyed by model name. These are
# estimates and are meant to be overridden per deployment: set the
# LLM_PRICE_TABLE env var to a JSON object of the same shape to replace or
# extend the defaults, e.g.
#   LLM_PRICE_TABLE='{"gemini-2.5-flash": {"input": 0.000075,
#                                          "output": 0.0003}}'
# The model this service currently calls. Shared so main.py and the price
# table cannot drift apart: a rename in only one place would silently unprice
# the model (priced=False / $0) instead of erroring.
GEMINI_MODEL = "gemini-2.5-flash"

_DEFAULT_PRICE_TABLE: dict[str, dict[str, float]] = {
    # Gemini 2.5 Flash: the model this service currently calls.
    GEMINI_MODEL: {"input": 0.000075, "output": 0.0003},
    # The superseded preview alias, kept priced so telemetry replayed from
    # before the rename does not report priced=False / $0.
    "gemini-2.5-flash-preview-05-20": {"input": 0.000075, "output": 0.0003},
    # A likely alternate so a model swap does not silently unprice.
    "gemini-2.5-pro": {"input": 0.00125, "output": 0.01},
}


def _load_price_table() -> dict[str, dict[str, float]]:
    table = dict(_DEFAULT_PRICE_TABLE)
    raw = os.getenv("LLM_PRICE_TABLE")
    if raw:
        try:
            override = json.loads(raw)
            if isinstance(override, dict):
                for model, prices in override.items():
                    if isinstance(prices, dict) and "input" in prices and "output" in prices:
                        table[model] = {
                            "input": float(prices["input"]),
                            "output": float(prices["output"]),
                        }
        except (ValueError, TypeError) as exc:
            logger.warning("Ignoring invalid LLM_PRICE_TABLE: %s", exc)
    return table


PRICE_TABLE: dict[str, dict[str, float]] = _load_price_table()


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> CostEstimate:
    """Estimate the USD cost of a single model call.

    Returns a CostEstimate whose ``priced`` flag is False when the model is not
    in the table, so an unknown model surfaces as cost 0.0 with priced=False
    rather than a silently wrong number.
    """
    prices = PRICE_TABLE.get(model)
    if prices is None:
        return CostEstimate(usd=0.0, priced=False)
    usd = (input_tokens / 1000.0) * prices["input"] + (output_tokens / 1000.0) * prices["output"]
    return CostEstimate(usd=round(usd, 8), priced=True)


@dataclass(frozen=True)
class CostEstimate:
    usd: float
    priced: bool


# ---------------------------------------------------------------------------
# Per-call usage capture
# ---------------------------------------------------------------------------
@dataclass
class LlmCall:
    """One model call's telemetry. Contains no prompt or answer text."""

    model: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost_usd: float
    priced: bool
    latency_ms: float
    trace_id: str
    stage: str = "generation"

    def as_labels(self) -> dict[str, Any]:
        """A content-free dict safe to log or expose."""
        return {
            "trace_id": self.trace_id,
            "stage": self.stage,
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "cost_usd": self.cost_usd,
            "priced": self.priced,
            "latency_ms": round(self.latency_ms, 2),
        }


def _usage_field(usage: Any, name: str) -> int:
    value = getattr(usage, name, None)
    if value is None and isinstance(usage, dict):
        value = usage.get(name)
    try:
        return int(value) if value is not None else 0
    except (TypeError, ValueError):
        return 0


def capture_usage(
    response: Any,
    model: str,
    latency_ms: float,
    trace_id: str,
    stage: str = "generation",
) -> LlmCall:
    """Build an LlmCall from a Gemini response's ``usage_metadata``.

    Defensive by design: a response without usage_metadata (or with missing
    fields) yields zeroed counts rather than raising, so telemetry never breaks
    the request it is measuring.
    """
    usage = getattr(response, "usage_metadata", None)
    input_tokens = _usage_field(usage, "prompt_token_count")
    output_tokens = _usage_field(usage, "candidates_token_count")
    total_tokens = _usage_field(usage, "total_token_count") or (input_tokens + output_tokens)
    cost = estimate_cost(model, input_tokens, output_tokens)
    return LlmCall(
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        cost_usd=cost.usd,
        priced=cost.priced,
        latency_ms=latency_ms,
        trace_id=trace_id,
        stage=stage,
    )


# ---------------------------------------------------------------------------
# Spans / stage timing
# ---------------------------------------------------------------------------
@dataclass
class Span:
    name: str
    duration_ms: float


@dataclass
class Trace:
    """Accumulates the stage spans of one request, keyed by a trace id."""

    trace_id: str = field(default_factory=lambda: new_trace_id())
    spans: list[Span] = field(default_factory=list)
    calls: list[LlmCall] = field(default_factory=list)

    def request_totals(self) -> dict[str, Any]:
        """Sum this request's model calls, for optional response metadata."""
        return {
            "trace_id": self.trace_id,
            "input_tokens": sum(c.input_tokens for c in self.calls),
            "output_tokens": sum(c.output_tokens for c in self.calls),
            "total_tokens": sum(c.total_tokens for c in self.calls),
            "cost_usd": round(sum(c.cost_usd for c in self.calls), 8),
            "model_calls": len(self.calls),
        }

    def add_span(self, name: str, duration_ms: float) -> None:
        """Record a completed stage. Used by ``span`` and for manual timing of
        blocks that are awkward to wrap in a context manager."""
        rounded = round(duration_ms, 2)
        self.spans.append(Span(name=name, duration_ms=rounded))
        logger.info(
            "stage completed",
            extra={"trace_id": self.trace_id, "stage": name, "duration_ms": rounded},
        )

    @contextmanager
    def span(self, name: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self.add_span(name, (time.perf_counter() - start) * 1000.0)

    def as_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "spans": [{"stage": s.name, "duration_ms": s.duration_ms} for s in self.spans],
        }


def new_trace_id() -> str:
    """Per-request trace id, unified with the request id from #11.

    Inside a request this is the same value that every log record carries and
    that goes back on the ``X-Request-ID`` header, so a log search and a
    response header lead to the same trace. Outside a request — a background
    task, a direct unit test — it falls back to a fresh uuid4.
    """
    return get_request_id() or uuid.uuid4().hex


# ---------------------------------------------------------------------------
# In-memory aggregate registry
# ---------------------------------------------------------------------------
def _percentile(samples: list[float], pct: float) -> float:
    if not samples:
        return 0.0
    ordered = sorted(samples)
    if len(ordered) == 1:
        return round(ordered[0], 2)
    rank = (pct / 100.0) * (len(ordered) - 1)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    frac = rank - low
    return round(ordered[low] + (ordered[high] - ordered[low]) * frac, 2)


class MetricsRegistry:
    """Thread-safe, in-memory aggregate of LLM telemetry.

    Latency samples are capped so memory stays bounded on a long-lived process;
    percentiles are computed over the most recent ``max_samples`` requests.
    """

    def __init__(self, max_samples: int = 1024) -> None:
        self._lock = threading.Lock()
        self._max_samples = max_samples
        self.request_count = 0
        self.error_count = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.total_tokens = 0
        self.cost_usd = 0.0
        self._handler_latencies: list[float] = []
        self._model_latencies: list[float] = []
        self._by_model: dict[str, dict[str, float]] = {}

    def record_call(self, call: LlmCall) -> None:
        with self._lock:
            self.input_tokens += call.input_tokens
            self.output_tokens += call.output_tokens
            self.total_tokens += call.total_tokens
            self.cost_usd = round(self.cost_usd + call.cost_usd, 8)
            self._model_latencies.append(call.latency_ms)
            self._trim(self._model_latencies)
            bucket = self._by_model.setdefault(
                call.model,
                {"calls": 0, "total_tokens": 0, "cost_usd": 0.0},
            )
            bucket["calls"] += 1
            bucket["total_tokens"] += call.total_tokens
            bucket["cost_usd"] = round(bucket["cost_usd"] + call.cost_usd, 8)

    def record_request(self, handler_latency_ms: float, error: bool = False) -> None:
        with self._lock:
            self.request_count += 1
            if error:
                self.error_count += 1
            self._handler_latencies.append(handler_latency_ms)
            self._trim(self._handler_latencies)

    def _trim(self, samples: list[float]) -> None:
        if len(samples) > self._max_samples:
            del samples[: len(samples) - self._max_samples]

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "requests": {
                    "total": self.request_count,
                    "errors": self.error_count,
                    "error_rate": round(self.error_count / self.request_count, 4) if self.request_count else 0.0,
                },
                "tokens": {
                    "input": self.input_tokens,
                    "output": self.output_tokens,
                    "total": self.total_tokens,
                },
                "cost_usd": round(self.cost_usd, 6),
                "latency_ms": {
                    "handler_p50": _percentile(self._handler_latencies, 50),
                    "handler_p95": _percentile(self._handler_latencies, 95),
                    "model_p50": _percentile(self._model_latencies, 50),
                    "model_p95": _percentile(self._model_latencies, 95),
                },
                "by_model": {
                    model: {
                        "calls": b["calls"],
                        "total_tokens": b["total_tokens"],
                        "cost_usd": round(b["cost_usd"], 6),
                    }
                    for model, b in self._by_model.items()
                },
            }

    def reset(self) -> None:
        # Reset fields in place under the existing lock. Re-running __init__
        # here would replace self._lock with a new object while a thread may
        # still be blocked on the old one, letting two critical sections run at
        # once against the same instance.
        with self._lock:
            self.request_count = 0
            self.error_count = 0
            self.input_tokens = 0
            self.output_tokens = 0
            self.total_tokens = 0
            self.cost_usd = 0.0
            self._handler_latencies.clear()
            self._model_latencies.clear()
            self._by_model.clear()


# Process-wide registry the app records into and /metrics reads from.
registry = MetricsRegistry()

# Current request's Trace, so a model call made deep in the request (e.g. the
# safety classifier) can be correlated without threading the id through every
# call site. #11 should eventually own this.
current_trace: ContextVar[Trace | None] = ContextVar("current_trace", default=None)


def record_model_call(
    response: Any,
    model: str,
    latency_ms: float,
    stage: str = "generation",
    trace: Trace | None = None,
) -> LlmCall:
    """Capture one model call, record it globally and on the request's trace.

    ``trace`` falls back to the current-request contextvar. Passing it
    explicitly (e.g. from a closure) is more robust across thread boundaries.
    """
    if trace is None:
        trace = current_trace.get()
    trace_id = trace.trace_id if trace is not None else "-"
    call = capture_usage(response, model, latency_ms, trace_id, stage)
    if trace is not None:
        trace.calls.append(call)
    registry.record_call(call)
    logger.info("model call completed", extra=call.as_labels())
    return call
