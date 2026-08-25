"""Prometheus metrics and observability for Deen Bridge AI Service (#116).

This module instruments the FastAPI application with standard HTTP metrics via
prometheus-fastapi-instrumentator and exports custom domain metrics tracking model calls,
latencies, token counts, cache performance, confidence scores, and the scholar review queue.

Safety Invariant:
Metric labels only ever contain content-free identifiers (model names, stages, cache types,
direction). Prompt, response, and user texts are strictly prohibited from metric labels.
"""

from __future__ import annotations

import ipaddress
import logging
import os
import secrets
from typing import TYPE_CHECKING

from fastapi import HTTPException, Request, status
from prometheus_client import (
    REGISTRY,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from prometheus_fastapi_instrumentator import Instrumentator

if TYPE_CHECKING:
    from fastapi import FastAPI

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Custom Prometheus Metrics
# ---------------------------------------------------------------------------

# 1. Model Calls Counter: by model and stage
MODEL_CALLS_TOTAL = Counter(
    "dnb_ai_model_calls_total",
    "Total number of model calls made by the service.",
    ["model", "stage"],
)

# 2. Model Latency Histogram (in seconds)
MODEL_LATENCY_SECONDS = Histogram(
    "dnb_ai_model_latency_seconds",
    "Latency of individual model calls in seconds.",
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
)

# 3. Tokens Total Counter: by direction (input / output)
TOKENS_TOTAL = Counter(
    "dnb_ai_tokens_total",
    "Total tokens processed by the service partitioned by direction (input/output).",
    ["direction"],
)

# 4. Cache Hits and Misses Counters
CACHE_HITS_TOTAL = Counter(
    "dnb_ai_cache_hits_total",
    "Total number of cache hits.",
    ["cache_type"],
)

CACHE_MISSES_TOTAL = Counter(
    "dnb_ai_cache_misses_total",
    "Total number of cache misses.",
    ["cache_type"],
)

# 5. Confidence Score Histogram (0.0 to 1.0)
CONFIDENCE_SCORE = Histogram(
    "dnb_ai_confidence_score",
    "Distribution of confidence scores computed for model responses.",
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)

# 6. Scholar Review Queue Depth Gauge
SCHOLAR_QUEUE_DEPTH = Gauge(
    "dnb_ai_scholar_queue_depth",
    "Current number of pending items in the scholar review queue.",
)


# ---------------------------------------------------------------------------
# Metric Recording Helpers
# ---------------------------------------------------------------------------


def record_model_call_metrics(
    model: str,
    stage: str,
    latency_ms: float,
    input_tokens: int,
    output_tokens: int,
) -> None:
    """Record model call telemetry into Prometheus counters and histograms.

    Parameters:
        model: Model identifier (e.g. 'gemini-2.5-flash').
        stage: Pipeline stage name (e.g. 'generation', 'fiqh_classification').
        latency_ms: Model response latency in milliseconds.
        input_tokens: Number of prompt/input tokens.
        output_tokens: Number of candidate/output tokens.
    """
    safe_model = model or "unknown"
    safe_stage = stage or "generation"
    MODEL_CALLS_TOTAL.labels(model=safe_model, stage=safe_stage).inc()

    # Convert milliseconds to seconds for Prometheus convention
    if latency_ms >= 0:
        MODEL_LATENCY_SECONDS.observe(latency_ms / 1000.0)

    if input_tokens > 0:
        TOKENS_TOTAL.labels(direction="input").inc(input_tokens)
    if output_tokens > 0:
        TOKENS_TOTAL.labels(direction="output").inc(output_tokens)


def record_cache_hit(cache_type: str = "semantic") -> None:
    """Record a cache hit for the given cache type."""
    CACHE_HITS_TOTAL.labels(cache_type=cache_type).inc()


def record_cache_miss(cache_type: str = "semantic") -> None:
    """Record a cache miss for the given cache type."""
    CACHE_MISSES_TOTAL.labels(cache_type=cache_type).inc()


def record_confidence_score(score: float) -> None:
    """Record a response confidence score in the histogram."""
    clamped_score = max(0.0, min(1.0, score))
    CONFIDENCE_SCORE.observe(clamped_score)


def set_scholar_queue_depth(depth: int | float) -> None:
    """Set the current pending depth of the scholar review queue."""
    SCHOLAR_QUEUE_DEPTH.set(max(0.0, float(depth)))


async def refresh_scholar_queue_depth() -> None:
    """Read current pending items from review_store and update the gauge."""
    try:
        from review_store import get_review_store

        store = get_review_store()
        stats = await store.stats()
        pending = stats.get("pending", 0)
        set_scholar_queue_depth(pending)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Failed to refresh scholar queue depth gauge: %s", exc)


# ---------------------------------------------------------------------------
# Endpoint Protection & Access Verification
# ---------------------------------------------------------------------------


def _parse_ip_networks(raw_allowlist: str) -> list[ipaddress.IPv4Network | ipaddress.IPv6Network]:
    """Parse a comma-separated list of IP addresses or CIDR networks."""
    networks: list[ipaddress.IPv4Network | ipaddress.IPv6Network] = []
    for item in raw_allowlist.split(","):
        cleaned = item.strip()
        if not cleaned:
            continue
        try:
            # ip_network with strict=False allows host IPs like '192.168.1.1' as well as CIDRs
            net = ipaddress.ip_network(cleaned, strict=False)
            networks.append(net)
        except ValueError as exc:
            logger.warning("Invalid IP or CIDR in METRICS_IP_ALLOWLIST (%r): %s", cleaned, exc)
    return networks


def _get_client_ip(request: Request) -> str | None:
    """Extract client IP from request, checking forwarded headers defensively."""
    # Check X-Forwarded-For first if present (taking the leftmost / original client IP)
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        parts = [p.strip() for p in forwarded_for.split(",") if p.strip()]
        if parts:
            return parts[0]
    if request.client and request.client.host:
        return request.client.host
    return None


def verify_metrics_access(request: Request) -> None:
    """Validate request authorization and IP allowlist for the /metrics endpoint.

    Configuration:
      - METRICS_TOKEN: If set, client must provide Bearer token in Authorization header
        or X-Metrics-Token header.
      - METRICS_IP_ALLOWLIST: If set, client IP must fall within one of the allowed networks.

    Raises:
      - HTTPException(401) if token is required and missing/invalid.
      - HTTPException(403) if client IP is not in allowlist.
    """
    token_configured = os.getenv("METRICS_TOKEN", "").strip()
    allowlist_configured = os.getenv("METRICS_IP_ALLOWLIST", "").strip()

    # 1. IP Allowlist Verification
    if allowlist_configured:
        client_ip_str = _get_client_ip(request)
        if not client_ip_str:
            logger.warning("Blocked metrics request: unable to determine client IP")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access forbidden: unable to determine client IP address.",
            )

        try:
            client_ip = ipaddress.ip_address(client_ip_str)
        except ValueError:
            logger.warning("Blocked metrics request: invalid client IP address %r", client_ip_str)
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access forbidden: invalid client IP address '{client_ip_str}'.",
            ) from None

        allowed_networks = _parse_ip_networks(allowlist_configured)
        is_allowed = any(client_ip in net for net in allowed_networks)
        if not is_allowed:
            logger.warning("Blocked metrics request from unauthorized IP %s", client_ip_str)
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access forbidden: client IP '{client_ip_str}' is not in the allowlist.",
            )

    # 2. Token Authentication Verification
    if token_configured:
        auth_header = request.headers.get("Authorization", "").strip()
        custom_header = request.headers.get("X-Metrics-Token", "").strip()

        provided_token = ""
        if auth_header.startswith("Bearer "):
            provided_token = auth_header[7:].strip()
        elif custom_header:
            provided_token = custom_header

        if not provided_token or not secrets.compare_digest(
            provided_token.encode("utf-8"),
            token_configured.encode("utf-8"),
        ):
            logger.warning("Unauthorized access attempt to /metrics (invalid or missing token)")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing metrics authentication token.",
                headers={"WWW-Authenticate": "Bearer realm='metrics'"},
            )


# ---------------------------------------------------------------------------
# FastAPI Instrumentator Setup
# ---------------------------------------------------------------------------


def setup_metrics(app: FastAPI) -> Instrumentator:
    """Instrument the FastAPI application with standard HTTP metrics and expose /metrics.

    HTTP metrics instrumented:
      - Request count by method, handler, status code
      - Latency histogram by handler
      - Content length

    Returns the initialized Instrumentator.
    """
    instrumentator = Instrumentator(
        should_group_status_codes=False,
        should_ignore_untemplated=True,
        should_respect_env_var=True,
        env_var_name="ENABLE_METRICS",
        excluded_handlers=["/metrics", "/metrics/json"],
    )

    # Instrument the app to capture all incoming HTTP traffic
    instrumentator.instrument(app)

    return instrumentator


def generate_prometheus_metrics() -> bytes:
    """Generate the latest Prometheus exposition metrics from the registry."""
    return generate_latest(REGISTRY)
