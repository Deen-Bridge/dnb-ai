#!/usr/bin/env python3
"""Failure-injection script for health and readiness verification (AI-094 / Issue #870).

Runs simulated failure scenarios against the application, polling /health (liveness)
and /ready (readiness) with timestamps, demonstrating:
1. Baseline: All systems operational -> /health 200 (ok), /ready 200 (ready).
2. Primary LLM Outage (Failover to secondary) -> /health 200 (ok), /ready 200 (degraded).
3. All LLM Providers Failing -> /health 200 (ok), /ready 503 (not_ready).
4. Provider Recovery -> /health 200 (ok), /ready 200 (ready).
5. Redis Outage (Single-Instance mode) -> /health 200 (ok), /ready 200 (degraded).
6. Redis Outage (Multi-Instance mode) -> /health 200 (ok), /ready 503 (not_ready).
7. Redis Recovery -> /health 200 (ok), /ready 200 (ready).
8. Embedding Provider Outage (Local Fallback) -> /health 200 (ok), /ready 200 (degraded).
"""

from __future__ import annotations

import os
import sys
from datetime import UTC, datetime
from pathlib import Path

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient

import main
from readiness import readiness_registry


def format_table_row(
    timestamp: str, phase: str, health_code: int, health_status: str, ready_code: int, ready_status: str, reasons: str
) -> str:
    return f"| {timestamp} | {phase:<38} | {health_code} ({health_status}) | {ready_code} ({ready_status}) | {reasons:<40} |"


def main_trace() -> int:
    client = TestClient(main.app)

    print("\n==========================================================================================")
    print("PRODUCTION AUDIT AI-094: HEALTH & READINESS FAILURE-INJECTION TRACE")
    print("==========================================================================================\n")

    header = "| Timestamp (UTC)          | Scenario / Injected Condition          | /health (Liveness) | /ready (Readiness) | Active Reasons / Degradations            |"
    divider = "|--------------------------|----------------------------------------|--------------------|--------------------|------------------------------------------|"
    print(header)
    print(divider)

    readiness_registry.reset()

    # Step 1: Baseline
    ts = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    h_resp = client.get("/health")
    r_resp = client.get("/ready")
    print(
        format_table_row(
            ts,
            "1. Normal baseline operation",
            h_resp.status_code,
            h_resp.json()["status"],
            r_resp.status_code,
            r_resp.json()["status"],
            ",".join(r_resp.json().get("reasons", [])),
        )
    )

    # Step 2: Primary LLM Outage (Failover active)
    readiness_registry.set_mock_provider_health("primary_llm", False)
    readiness_registry.set_mock_provider_health("secondary_llm", True)
    ts = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    h_resp = client.get("/health")
    r_resp = client.get("/ready")
    print(
        format_table_row(
            ts,
            "2. Primary LLM down (failover active)",
            h_resp.status_code,
            h_resp.json()["status"],
            r_resp.status_code,
            r_resp.json()["status"],
            ",".join(r_resp.json().get("reasons", [])),
        )
    )

    # Step 3: All LLM Providers Failing
    readiness_registry.set_mock_provider_health("primary_llm", False)
    readiness_registry.set_mock_provider_health("secondary_llm", False)
    ts = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    h_resp = client.get("/health")
    r_resp = client.get("/ready")
    print(
        format_table_row(
            ts,
            "3. All LLM providers down",
            h_resp.status_code,
            h_resp.json()["status"],
            r_resp.status_code,
            r_resp.json()["status"],
            ",".join(r_resp.json().get("reasons", [])),
        )
    )

    # Step 4: Primary LLM Recovers
    readiness_registry.set_mock_provider_health("primary_llm", True)
    readiness_registry.set_mock_provider_health("secondary_llm", True)
    ts = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    h_resp = client.get("/health")
    r_resp = client.get("/ready")
    print(
        format_table_row(
            ts,
            "4. Primary LLM recovered",
            h_resp.status_code,
            h_resp.json()["status"],
            r_resp.status_code,
            r_resp.json()["status"],
            ",".join(r_resp.json().get("reasons", [])),
        )
    )

    # Step 5: Redis Outage in Single-Instance Mode
    os.environ["REQUIRE_REDIS"] = "false"
    readiness_registry.set_mock_provider_health("redis", False)
    ts = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    h_resp = client.get("/health")
    r_resp = client.get("/ready")
    print(
        format_table_row(
            ts,
            "5. Redis down (single-instance)",
            h_resp.status_code,
            h_resp.json()["status"],
            r_resp.status_code,
            r_resp.json()["status"],
            ",".join(r_resp.json().get("reasons", [])),
        )
    )

    # Step 6: Redis Outage in Multi-Instance Mode
    os.environ["REQUIRE_REDIS"] = "true"
    readiness_registry.set_mock_provider_health("redis", False)
    ts = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    h_resp = client.get("/health")
    r_resp = client.get("/ready")
    print(
        format_table_row(
            ts,
            "6. Redis down (multi-instance required)",
            h_resp.status_code,
            h_resp.json()["status"],
            r_resp.status_code,
            r_resp.json()["status"],
            ",".join(r_resp.json().get("reasons", [])),
        )
    )

    # Step 7: Redis Recovers
    os.environ["REQUIRE_REDIS"] = "false"
    readiness_registry.set_mock_provider_health("redis", True)
    ts = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    h_resp = client.get("/health")
    r_resp = client.get("/ready")
    print(
        format_table_row(
            ts,
            "7. Redis recovered",
            h_resp.status_code,
            h_resp.json()["status"],
            r_resp.status_code,
            r_resp.json()["status"],
            ",".join(r_resp.json().get("reasons", [])),
        )
    )

    # Step 8: Embedding Provider Outage (Local Fallback)
    readiness_registry.record_embedding_fallback()
    ts = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    h_resp = client.get("/health")
    r_resp = client.get("/ready")
    print(
        format_table_row(
            ts,
            "8. Embedding down (local fallback)",
            h_resp.status_code,
            h_resp.json()["status"],
            r_resp.status_code,
            r_resp.json()["status"],
            ",".join(r_resp.json().get("reasons", [])),
        )
    )

    # Step 9: Full Recovery
    readiness_registry.reset()
    ts = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    h_resp = client.get("/health")
    r_resp = client.get("/ready")
    print(
        format_table_row(
            ts,
            "9. All systems fully recovered",
            h_resp.status_code,
            h_resp.json()["status"],
            r_resp.status_code,
            r_resp.json()["status"],
            ",".join(r_resp.json().get("reasons", [])),
        )
    )

    print("\n==========================================================================================")
    print("FAILURE-INJECTION TRACE COMPLETED SUCCESSFULLY")
    print("==========================================================================================\n")
    return 0


if __name__ == "__main__":
    sys.exit(main_trace())
