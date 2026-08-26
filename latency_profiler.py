"""Latency profiling infrastructure — component timing, request tracing, budget enforcement.

Issue #211: lightweight, offline-safe latency profiling with bottleneck
identification, budget enforcement, baseline comparison, and optimization
recommendations.

Everything is in-memory with no external dependencies beyond numpy (already
in requirements.txt) for percentile computation.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Iterator

import numpy as np
from fastapi import APIRouter
from pydantic import BaseModel, Field

from config import get_settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/profiling", tags=["profiling"])


# ---------------------------------------------------------------------------
# Component latency registry
# ---------------------------------------------------------------------------

@dataclass
class _ComponentSamples:
    """Thread-safe accumulator for one component's latency samples."""

    samples: list[float] = field(default_factory=list)
    _max: int = 4096

    def record(self, duration_ms: float) -> None:
        self.samples.append(duration_ms)
        if len(self.samples) > self._max:
            self.samples = self.samples[-self._max:]

    def snapshot(self) -> list[float]:
        return list(self.samples)


class _LatencyRegistry:
    """Process-wide registry of component latency samples."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._components: dict[str, _ComponentSamples] = defaultdict(_ComponentSamples)

    def record(self, component: str, duration_ms: float) -> None:
        with self._lock:
            self._components[component].record(duration_ms)

    def get_samples(self, component: str | None = None) -> dict[str, list[float]]:
        with self._lock:
            if component:
                comp = self._components.get(component)
                if comp is None or not comp.samples:
                    return {}
                return {component: comp.snapshot()}
            return {name: s.snapshot() for name, s in self._components.items() if s.samples}

    def components(self) -> list[str]:
        with self._lock:
            return list(self._components.keys())

    def reset(self) -> None:
        with self._lock:
            self._components.clear()


_registry = _LatencyRegistry()


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def _summary_stats(samples: list[float]) -> dict[str, float]:
    """Compute p50/p95/p99, min/max/avg, throughput from a sample list."""
    if not samples:
        return {
            "count": 0,
            "min_ms": 0.0,
            "max_ms": 0.0,
            "avg_ms": 0.0,
            "p50_ms": 0.0,
            "p95_ms": 0.0,
            "p99_ms": 0.0,
            "throughput_rps": 0.0,
        }
    arr = np.array(samples, dtype=np.float64)
    return {
        "count": len(samples),
        "min_ms": round(float(np.min(arr)), 2),
        "max_ms": round(float(np.max(arr)), 2),
        "avg_ms": round(float(np.mean(arr)), 2),
        "p50_ms": round(float(np.percentile(arr, 50)), 2),
        "p95_ms": round(float(np.percentile(arr, 95)), 2),
        "p99_ms": round(float(np.percentile(arr, 99)), 2),
        "throughput_rps": round(len(samples) / max(float(np.sum(arr) / 1000.0), 0.001), 2),
    }


def get_profiler_stats(
    component: str | None = None,
    time_window_s: float | None = None,
) -> dict[str, Any]:
    """Return aggregated latency stats, optionally filtered by component/time."""
    raw = _registry.get_samples(component)
    result: dict[str, Any] = {}
    for name, samples in raw.items():
        result[name] = _summary_stats(samples)
    return result


# ---------------------------------------------------------------------------
# Context-manager timing
# ---------------------------------------------------------------------------

@contextmanager
def trace_component(name: str) -> Iterator[None]:
    """Context manager that records elapsed time for *name* into the global registry."""
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        _registry.record(name, elapsed_ms)


# ---------------------------------------------------------------------------
# Request-level tracing
# ---------------------------------------------------------------------------

@dataclass
class _Span:
    component: str
    duration_ms: float


@dataclass
class _TraceRecord:
    trace_id: str
    spans: list[_Span] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)
    ended_at: float | None = None

    @property
    def total_ms(self) -> float:
        return sum(s.duration_ms for s in self.spans)

    def bottleneck(self) -> _Span | None:
        """Return the span with the largest duration_ms."""
        return max(self.spans, key=lambda s: s.duration_ms) if self.spans else None


class _TraceStore:
    def __init__(self) -> None:
        self._lock = Lock()
        self._traces: dict[str, _TraceRecord] = {}

    def start(self) -> str:
        trace_id = uuid.uuid4().hex
        with self._lock:
            self._traces[trace_id] = _TraceRecord(trace_id=trace_id)
        return trace_id

    def add_span(self, trace_id: str, component: str, duration_ms: float) -> None:
        with self._lock:
            trace = self._traces.get(trace_id)
            if trace is None:
                return
            trace.spans.append(_Span(component=component, duration_ms=duration_ms))

    def end(self, trace_id: str) -> _TraceRecord | None:
        with self._lock:
            trace = self._traces.get(trace_id)
            if trace is None:
                return None
            trace.ended_at = time.time()
            return trace

    def get(self, trace_id: str) -> _TraceRecord | None:
        with self._lock:
            return self._traces.get(trace_id)

    def clear(self) -> None:
        with self._lock:
            self._traces.clear()


_trace_store = _TraceStore()


def start_trace() -> str:
    return _trace_store.start()


def add_span(trace_id: str, component: str, duration_ms: float) -> None:
    _trace_store.add_span(trace_id, component, duration_ms)
    _registry.record(component, duration_ms)


def end_trace(trace_id: str) -> dict[str, Any] | None:
    trace = _trace_store.end(trace_id)
    if trace is None:
        return None
    bn = trace.bottleneck()
    return {
        "trace_id": trace.trace_id,
        "spans": [{"component": s.component, "duration_ms": round(s.duration_ms, 2)} for s in trace.spans],
        "total_ms": round(trace.total_ms, 2),
        "bottleneck": {
            "component": bn.component,
            "duration_ms": round(bn.duration_ms, 2),
        }
        if bn
        else None,
        "span_count": len(trace.spans),
    }


# ---------------------------------------------------------------------------
# Latency budget enforcement
# ---------------------------------------------------------------------------

def _load_budgets() -> dict[str, float]:
    """Build the budget dict from config settings."""
    s = get_settings()
    return {
        "database": s.latency_budget_database_ms,
        "api_call": s.latency_budget_api_call_ms,
        "retrieval": s.latency_budget_retrieval_ms,
        "inference": s.latency_budget_inference_ms,
        "_default": s.latency_budget_default_ms,
    }


def check_budget(component: str, elapsed_ms: float) -> dict[str, Any]:
    """Check whether *elapsed_ms* exceeds the budget for *component*.

    Returns a dict with ``within_budget``, ``budget_ms``, and ``overage_ms``
    when violated.
    """
    budgets = _load_budgets()
    budget = budgets.get(component, budgets["_default"])
    within = elapsed_ms <= budget
    return {
        "component": component,
        "within_budget": within,
        "budget_ms": budget,
        "elapsed_ms": round(elapsed_ms, 2),
        "overage_ms": round(max(elapsed_ms - budget, 0.0), 2),
    }


# ---------------------------------------------------------------------------
# Baseline comparison
# ---------------------------------------------------------------------------

def compare_to_baseline(
    current_stats: dict[str, Any],
    baseline_stats: dict[str, Any],
    regression_threshold: float = 0.20,
) -> dict[str, Any]:
    """Compare current component stats against a stored baseline.

    A component is flagged as a *regression* when its p95 latency increased
    by more than ``regression_threshold`` (default 20 %).  Improvements are
    reported symmetrically.
    """
    regressions: list[dict[str, Any]] = []
    improvements: list[dict[str, Any]] = []

    all_components = set(current_stats) | set(baseline_stats)
    for comp in sorted(all_components):
        cur = current_stats.get(comp, {})
        bl = baseline_stats.get(comp, {})
        cur_p95 = cur.get("p95_ms", 0.0)
        bl_p95 = bl.get("p95_ms", 0.0)
        if bl_p95 <= 0:
            continue
        change_pct = (cur_p95 - bl_p95) / bl_p95
        entry = {
            "component": comp,
            "baseline_p95_ms": bl_p95,
            "current_p95_ms": cur_p95,
            "change_pct": round(change_pct * 100, 1),
        }
        if change_pct > regression_threshold:
            regressions.append(entry)
        elif change_pct < -regression_threshold:
            improvements.append(entry)

    return {
        "regressions": regressions,
        "improvements": improvements,
        "components_compared": len(all_components),
    }


# ---------------------------------------------------------------------------
# Optimization recommendations (rules-based)
# ---------------------------------------------------------------------------

_RULES: list[tuple[str, str, float | None]] = [
    ("inference", "Consider caching frequent inference results or using a smaller model for simple queries.", 1000.0),
    ("retrieval", "Add indexing or pre-filtering to reduce retrieval latency.", 200.0),
    ("database", "Review query plans and add appropriate indexes.", 100.0),
    ("api_call", "Consider connection pooling, request batching, or async parallelization.", 500.0),
]


def get_recommendations(stats: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Generate rule-based optimization recommendations from current stats."""
    if stats is None:
        stats = get_profiler_stats()
    recs: list[dict[str, Any]] = []
    for comp, advice, threshold in _RULES:
        comp_stats = stats.get(comp)
        if comp_stats is None:
            continue
        p95 = comp_stats.get("p95_ms", 0.0)
        if p95 > threshold:
            recs.append({
                "component": comp,
                "current_p95_ms": p95,
                "threshold_ms": threshold,
                "recommendation": advice,
            })
    # General: any component with high p99 relative to p95 suggests tail latency spikes.
    for comp, comp_stats in stats.items():
        p95 = comp_stats.get("p95_ms", 0.0)
        p99 = comp_stats.get("p99_ms", 0.0)
        if p95 > 0 and p99 > p95 * 1.5:
            recs.append({
                "component": comp,
                "current_p95_ms": p95,
                "current_p99_ms": p99,
                "recommendation": (
                    f"Tail latency spike detected for '{comp}' (p99={p99:.0f}ms is >1.5x p95={p95:.0f}ms). "
                    "Investigate outlier requests or add timeout/retry policies."
                ),
            })
    return recs


# ---------------------------------------------------------------------------
# Baseline store (in-memory)
# ---------------------------------------------------------------------------

_baseline_store: dict[str, Any] = {}


def store_baseline(name: str, stats: dict[str, Any]) -> None:
    _baseline_store[name] = stats


def get_baseline(name: str) -> dict[str, Any] | None:
    return _baseline_store.get(name)


def list_baselines() -> list[str]:
    return list(_baseline_store.keys())


def reset_profiler() -> None:
    """Clear all profiling data (useful between test runs)."""
    _registry.reset()
    _trace_store.clear()
    _baseline_store.clear()


# ---------------------------------------------------------------------------
# API models
# ---------------------------------------------------------------------------


class TraceStartResponse(BaseModel):
    trace_id: str


class TraceEndResponse(BaseModel):
    trace_id: str
    spans: list[dict[str, Any]]
    total_ms: float
    bottleneck: dict[str, Any] | None = None
    span_count: int


class BaselineRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=128)
    stats: dict[str, Any]


class CompareRequest(BaseModel):
    baseline_name: str | None = None
    baseline_stats: dict[str, Any] | None = None


class ComponentTiming(BaseModel):
    component: str
    duration_ms: float


class TraceSpanRequest(BaseModel):
    trace_id: str
    timings: list[ComponentTiming]


class BudgetCheckRequest(BaseModel):
    component: str
    elapsed_ms: float


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/trace", response_model=TraceStartResponse, status_code=200)
async def trace_start() -> TraceStartResponse:
    """Start a new request trace and return its unique trace_id."""
    return TraceStartResponse(trace_id=start_trace())


@router.post("/trace/span", status_code=200)
async def trace_add_span(body: TraceSpanRequest) -> dict[str, str]:
    """Add one or more timing spans to an active trace."""
    for t in body.timings:
        add_span(body.trace_id, t.component, t.duration_ms)
    return {"status": "ok"}


@router.post("/trace/end", response_model=TraceEndResponse)
async def trace_end(body: dict[str, str]) -> TraceEndResponse:
    """End a trace and return the full trace with bottleneck identified."""
    trace_id = body.get("trace_id", "")
    result = end_trace(trace_id)
    if result is None:
        return TraceEndResponse(trace_id=trace_id, spans=[], total_ms=0.0, bottleneck=None, span_count=0)
    return TraceEndResponse(**result)


@router.get("/stats")
async def profiling_stats(
    component: str | None = None,
    time_window: float | None = None,
) -> dict[str, Any]:
    """Return aggregated latency statistics per component."""
    return get_profiler_stats(component=component, time_window_s=time_window)


@router.post("/budget/check")
async def budget_check(body: BudgetCheckRequest) -> dict[str, Any]:
    """Check whether an elapsed time exceeds the configured budget for a component."""
    return check_budget(body.component, body.elapsed_ms)


@router.post("/baseline", status_code=200)
async def store_baseline_endpoint(body: BaselineRequest) -> dict[str, str]:
    """Store a named baseline snapshot for future regression comparison."""
    store_baseline(body.name, body.stats)
    return {"status": "ok", "name": body.name}


@router.get("/baseline")
async def list_baselines_endpoint() -> dict[str, Any]:
    """List stored baseline names."""
    return {"baselines": list_baselines()}


@router.post("/baseline/compare")
async def compare_baseline(body: CompareRequest) -> dict[str, Any]:
    """Compare current stats against a stored or supplied baseline."""
    baseline = None
    if body.baseline_stats:
        baseline = body.baseline_stats
    elif body.baseline_name:
        baseline = get_baseline(body.baseline_name)
    if baseline is None:
        return {"error": "Baseline not found. Provide baseline_stats or a valid baseline_name."}
    current = get_profiler_stats()
    return compare_to_baseline(current, baseline)


@router.get("/recommendations")
async def recommendations() -> dict[str, Any]:
    """Return rule-based optimization recommendations based on current stats."""
    return {"recommendations": get_recommendations()}


@router.post("/reset", status_code=200)
async def profiling_reset() -> dict[str, str]:
    """Clear all profiling data (traces, samples, baselines)."""
    reset_profiler()
    return {"status": "reset"}
