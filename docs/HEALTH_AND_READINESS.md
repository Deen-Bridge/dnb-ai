# PROD-AUDIT AI-094: Production Health, Readiness, and Dependency Degradation

**Issue**: [#870](https://github.com/Deen-Bridge/dnb-ai/issues/870)  
**Compliance Standard**: ISO/IEC 42001 (AI Management System), NIST AI RMF (Reliability, Transparency & Incident Management).

---

## 1. Executive Summary & Audit Context

Prior to issue #870, Render configured `healthCheckPath: /health` as both the container deploy-promotion check and instance restart signal. At the same time, `/health` performed dependency key checks. If an external service or optional cache blipped, an instance could fail health checks and enter a restart loop that a container restart could not resolve. Conversely, a deployment could pass basic checks while critical model capabilities or state stores were unavailable.

Under issue #870, the service enforces a strict separation of concerns across orchestration signals:
1. **Liveness (`GET /health`)**: Constant-time verification that the process and asyncio event loop are alive and responsive. Performs **zero** external network calls (no Redis, no provider APIs).
2. **Readiness (`GET /ready`)**: Comprehensive evaluation of whether the instance can serve production traffic based on explicit dependency criticality.
3. **Degraded Operation**: When an optional dependency fails or a fallback is active (e.g. deterministic local embeddings or in-memory queue fallback), the service remains operational (`degraded` status, HTTP 200) to prevent cascade failures. Requests traversing fallbacks emit an additive `X-Degraded` response header.
4. **Startup Configuration Gate**: Required configuration is validated at boot. In production (`ENVIRONMENT=production`), missing configuration fails fast with a non-zero exit code to prevent bad deploys from being promoted. In development and test modes, the server starts and reports `not_ready` (HTTP 503) on `/ready`.

---

## 2. Dependency Criticality Specification

Every external and platform dependency is classified in `readiness.py` as either **REQUIRED** or **OPTIONAL / DEGRADABLE**:

| Dependency | Classification | Deployment Condition | Failure Behavior & Fallback Mechanism | Readiness Impact | HTTP Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Configuration** (`GEMINI_API_KEY`, `SERVICE_API_KEY`) | **REQUIRED** | All modes (`SERVICE_API_KEY` required in production when `AUTH_DISABLED=false`) | Startup fail-fast with non-zero exit in production; reports `config_missing` outside production. | `not_ready` | **503** |
| **Primary LLM** (Google Gemini) | **REQUIRED** | All modes | Monitored passively via rolling window of real request outcomes without spending paid generation tokens. If primary fails and secondary provider is available, routes to failover. | `degraded` (if secondary OK), `not_ready` (if all fail) | **200** (degraded)<br>**503** (all down) |
| **Redis** | **REQUIRED** | Multi-instance mode (`REQUIRE_REDIS=true` or `DEPLOYMENT_MODE=multi-instance`) | Multi-instance rate limiting, distributed session sharing, and persistent queue durability mandate Redis. | `not_ready` (`redis_required_unreachable`) | **503** |
| **Redis** | **OPTIONAL** | Single-instance mode (`REQUIRE_REDIS=false`, default) | Gracefully falls back to bounded in-memory queue (`REVIEW_QUEUE_LOCAL_MAX_SIZE`), SQLite for feedback, and memory session store. | `degraded` (`redis_degraded_in_memory`) | **200** |
| **Embedding Provider** (`text-embedding-004`) | **OPTIONAL** | All modes | Deterministic local SHA-256 fallback (`_local_embedding`) in `semantic_cache.py` keeps the cache functional offline and during provider degradation. | `degraded` (`embedding_degraded_local`) | **200** |
| **Secondary LLM Providers** (Anthropic, OpenAI) | **OPTIONAL** | All modes | Model routing and agent fallback chains execute alternative models when configured. | `degraded` (`secondary_provider_degraded`) | **200** |

---

## 3. Endpoint Contract & Schemas

### 3.1 Liveness Probe: `GET /health`

- **Purpose**: Process vitality for container orchestrators (Render `healthCheckPath`).
- **Timing**: Guaranteed constant-time ($O(1)$), strictly non-blocking.
- **Network Calls**: **Zero**. External providers, Redis, and network databases are never contacted.
- **HTTP Status**: **200 OK** whenever the process and event loop can respond.
- **Payload Schema**:
  ```json
  {
    "status": "ok",
    "version": "0.1.0",
    "checks": {
      "process": "ok",
      "gemini_api_key_configured": "ok",
      "gemini_api_reachable": "skipped"
    }
  }
  ```

### 3.2 Readiness Probe: `GET /ready`

- **Purpose**: Traffic routing decisions for ingress load balancers, canary gates, and incident tooling.
- **Timing**: Bounded by hard probe timeouts (`PROBE_TIMEOUT_SECONDS`, default 1.0s), cached with single-flight concurrency deduplication (`PROBE_CACHE_TTL_SECONDS`, default 5.0s).
- **Token Spend**: **Zero**. Never calls `generate_content` or paid generation endpoints. Provider health is derived passively from real request outcomes.
- **HTTP Status Mapping**:
  - `ready` $\to$ **200 OK**
  - `degraded` $\to$ **200 OK** (instance serves traffic; downstream degradation alerted)
  - `not_ready` $\to$ **503 Service Unavailable** (instance removed from load-balancer rotation)

#### Example: Ready State (`status: "ready"`, 200 OK)

```json
{
  "status": "ready",
  "timestamp": "2026-11-23T04:17:11Z",
  "version": "0.1.0",
  "mode": "single-instance",
  "reasons": [
    "ok"
  ],
  "dependencies": {
    "config": {
      "name": "config",
      "criticality": "required",
      "status": "ok",
      "reason": "config_valid",
      "latency_ms": null
    },
    "primary_llm": {
      "name": "primary_llm",
      "criticality": "required",
      "status": "ok",
      "reason": "provider_ok",
      "latency_ms": null
    },
    "redis": {
      "name": "redis",
      "criticality": "optional",
      "status": "ok",
      "reason": "in_memory_mode",
      "latency_ms": null
    },
    "embedding": {
      "name": "embedding",
      "criticality": "optional",
      "status": "ok",
      "reason": "embedding_ok",
      "latency_ms": null
    }
  },
  "fallbacks": {
    "active_count": 0,
    "local_embedding": false,
    "redis_in_memory": false,
    "review_queue_overflow_drops": 0,
    "total_fallbacks": 0
  }
}
```

#### Example: Degraded State (`status: "degraded"`, 200 OK)

Returned when Redis is down in single-instance mode or when the embedding provider failed and local embeddings are active:

```json
{
  "status": "degraded",
  "timestamp": "2026-11-23T04:17:12Z",
  "version": "0.1.0",
  "mode": "single-instance",
  "reasons": [
    "redis:redis_degraded_in_memory",
    "embedding:embedding_degraded_local"
  ],
  "dependencies": {
    "config": {
      "name": "config",
      "criticality": "required",
      "status": "ok",
      "reason": "config_valid",
      "latency_ms": null
    },
    "primary_llm": {
      "name": "primary_llm",
      "criticality": "required",
      "status": "ok",
      "reason": "provider_ok",
      "latency_ms": null
    },
    "redis": {
      "name": "redis",
      "criticality": "optional",
      "status": "degraded",
      "reason": "redis_degraded_in_memory",
      "latency_ms": null
    },
    "embedding": {
      "name": "embedding",
      "criticality": "optional",
      "status": "degraded",
      "reason": "embedding_degraded_local",
      "latency_ms": null
    }
  },
  "fallbacks": {
    "active_count": 2,
    "local_embedding": true,
    "redis_in_memory": true,
    "review_queue_overflow_drops": 0,
    "total_fallbacks": 1
  }
}
```

#### Example: Not Ready State (`status: "not_ready"`, 503 Service Unavailable)

Returned when all LLM providers are down, required Redis is unreachable in multi-instance mode, or required configuration is missing:

```json
{
  "status": "not_ready",
  "timestamp": "2026-11-23T04:17:11Z",
  "version": "0.1.0",
  "mode": "multi-instance",
  "reasons": [
    "primary_llm:all_providers_unavailable",
    "redis:redis_required_unreachable"
  ],
  "dependencies": {
    "config": {
      "name": "config",
      "criticality": "required",
      "status": "ok",
      "reason": "config_valid",
      "latency_ms": null
    },
    "primary_llm": {
      "name": "primary_llm",
      "criticality": "required",
      "status": "fail",
      "reason": "all_providers_unavailable",
      "latency_ms": null
    },
    "redis": {
      "name": "redis",
      "criticality": "required",
      "status": "fail",
      "reason": "redis_required_unreachable",
      "latency_ms": null
    },
    "embedding": {
      "name": "embedding",
      "criticality": "optional",
      "status": "ok",
      "reason": "embedding_ok",
      "latency_ms": null
    }
  },
  "fallbacks": {
    "active_count": 0,
    "local_embedding": false,
    "redis_in_memory": false,
    "review_queue_overflow_drops": 0,
    "total_fallbacks": 0
  }
}
```

---

## 4. Consumer Routing and Alerting Recommendations

### 4.1 Render Health Check: Keep on `/health` (Liveness)
- **Recommendation**: Leave `render.yaml: healthCheckPath: /health`.
- **Architectural Rationale**: Render uses `healthCheckPath` to decide both whether a new deploy is healthy and whether to restart unhealthy instances. Restarting a container cannot resolve a third-party upstream API outage (e.g. Gemini quota/latency) or a Redis cluster incident. If Render pointed to `/ready`, a transient external provider outage would trigger container restart thrashing, compounding load and taking down healthy fallback paths. `/health` ensures the process stays alive to serve fallback and cached responses.

### 4.2 Load Balancers, Ingress Routers & Incident Response: Use `/ready`
- **Traffic Routing**: Upstream load balancers (e.g. Cloudflare, AWS ALB, NGINX) should poll `GET /ready`. Instances returning `503` are automatically pulled from the active pool until required dependencies recover.
- **Alerting on Degraded State**:
  - Alert on HTTP 200 with response body `"status": "degraded"`.
  - Severity: **P3 / Warning**. Indicates service is serving users via fallback mechanisms (e.g. deterministic local embeddings or in-memory queues) without customer-facing downtime.
  - Alert on HTTP 503 (`"status": "not_ready"`).
  - Severity: **P1 / Critical**. Core dependencies are unavailable; service cannot answer new requests.

---

## 5. Bounded Fallbacks & Memory Safety

To prevent unbounded memory growth during sustained Redis outages:
1. **In-Memory Review Queue Bound**: `ReviewStore` enforces `REVIEW_QUEUE_LOCAL_MAX_SIZE` (default: 1,000 items).
2. **Explicit Overflow Eviction**: When the local fallback buffer reaches capacity, the oldest unreviewed item is evicted. The eviction is recorded in `overflow_drops`, logged as a warning without PII, and surfaced in `/ready` under `fallbacks.review_queue_overflow_drops`.
3. **Additive Response Header**: Every request that traverses a fallback path (e.g. local embedding or in-memory queue) emits the additive header:
   ```http
   X-Degraded: local-embedding, redis-unavailable
   ```

---

## 6. Safe Diagnostics & Zero-Leakage Guarantee

All health signals and logs strictly adhere to privacy invariants:
- **No Prompt or Completion Text**: Prompts and generated text are never passed to health probes or diagnostic endpoints.
- **No PII or User IDs**: Health responses only surface system-level component states.
- **No Secret or Credential Leakage**: `GEMINI_API_KEY`, `SERVICE_API_KEY`, and connection strings containing passwords (e.g. `redis://user:pass@host`) are stripped and never appear in response bodies, headers, or log lines.
- **Machine-Readable Reason Codes**: All failure reasons use static enumerated codes (`ReasonCode`) rather than raw exception tracebacks.

---

## 7. Configuration Reference

| Environment Variable | Default | Description |
| :--- | :--- | :--- |
| `REQUIRE_REDIS` | `false` | When `true`, mandates Redis availability; Redis failure reports `not_ready` (503) instead of `degraded` (200). Required in multi-instance deployments. |
| `PROBE_CACHE_TTL_SECONDS` | `5.0` | Cache time-to-live in seconds for readiness probe results to prevent dependency stampedes. |
| `PROBE_TIMEOUT_SECONDS` | `1.0` | Hard timeout in seconds for executing any single dependency probe. |
| `REVIEW_QUEUE_LOCAL_MAX_SIZE`| `1000` | Maximum item capacity of the in-memory fallback review queue when Redis is unreachable. |
