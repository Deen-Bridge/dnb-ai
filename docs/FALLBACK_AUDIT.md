# PROD-AUDIT AI-108: Auditable External-Provider Fallback Decisions

**Issue**: [#884](https://github.com/Deen-Bridge/dnb-ai/issues/884)  
**Audit Objective**: Ensure degraded AI answers are transparent, bounded, and reviewable.  
**Compliance Standard**: ISO/IEC 42001 (AI Management System), NIST AI RMF (Transparency & Trustworthiness).

---

## 1. Executive Summary & Audit Context

The service uses multi-tier resilience mechanisms including transport retries, circuit breakers, alternate agents, deterministic fallbacks, and local knowledge fallbacks. Prior to issue #884, release evidence did not require a consistent user-visible and operator-visible fallback reason.

Under issue #884, the fallback architecture was redesigned to enforce:
1. **Auditable Fallback Taxonomy**: Stable machine-readable reason codes emitted across all fallback paths.
2. **Transparent User Messaging**: Truthful, risk-graded user-facing messages ensuring users know when a response is degraded or deterministic.
3. **Strict Control Invariants**: Fallbacks cannot bypass safety moderation, citation requirements, authentication, or cost limits.
4. **Privacy-Safe Telemetry**: Telemetry aggregates fallback rates by provider and feature without logging prompts, secrets, or PII.
5. **Threshold Disabling & Rollback**: Automatic disabling thresholds, status review triggers, and instant operator rollback capabilities.

---

## 2. Fallback Reason Taxonomy

Every fallback path emits a stable, canonical `FallbackReason` enum code, associated with a risk level and truthful user messaging:

| Reason Code (`FallbackReason`) | Trigger Condition | Risk Level | User-Facing Explanation | Operator Response / Runbook |
| :--- | :--- | :--- | :--- | :--- |
| `provider_timeout` | External LLM provider call exceeded latency deadline (`TimeoutError`, `DeadlineExceeded`). | **MEDIUM** | *"The primary provider timed out. An alternative response was generated."* | Check provider API status page and egress network latency. Increase timeout if model latency has shifted. |
| `provider_unavailable` | External provider returned HTTP 503, connection dropped, or service is unreachable. | **HIGH** | *"The primary service is currently unavailable. A fallback system handled this request."* | Verify provider outage; check DNS/firewall; fail over secondary regional endpoint if persistent. |
| `provider_rate_limit` | Provider returned HTTP 429 (`ResourceExhausted` / quota exceeded). | **MEDIUM** | *"Service capacity limit reached. An alternative model was used."* | Review account quota; check token consumption spikes in telemetry; rebalance load across provider keys. |
| `circuit_open` | Per-agent circuit breaker tripped to `OPEN` state after exceeding failure threshold. | **HIGH** | *"The service is experiencing connectivity issues with the primary provider. A degraded fallback response was generated."* | Inspect upstream health; verify if recovery timeout has elapsed; verify circuit transitions to half-open. |
| `circuit_half_open_failed` | Trial/canary probe call failed while circuit breaker was in `HALF_OPEN` state. | **HIGH** | *"Provider recovery check failed. Degraded fallback response provided."* | Upstream provider has not recovered; keep circuit open; investigate lingering provider degradation. |
| `safety_block` | Request or response flagged by input moderation gate or output safety validator. | **HIGH** | *"This response could not be generated due to safety policy constraints."* | **TERMINAL REFUSAL**: Fallback execution is blocked to prevent bypassing moderation policy. Review safety logs. |
| `deterministic_fallback` | Algorithmic or rule-based calculation fallback used due to model degradation. | **MEDIUM** | *"A rule-based deterministic response was provided due to temporary service degradation."* | Verify rule accuracy and deterministic engine outputs; review model degradation root cause. |
| `local_data_fallback` | Static reference lookup (e.g. offline Hadith/Quran database) used instead of model generation. | **LOW** | *"This response was answered using local reference records."* | Check local database indexing and ensure reference data matches latest vetted canon. |
| `stale_data` | Response served from cache past ideal freshness window due to downstream unreachability. | **LOW** | *"This response is served from cached reference data and may not reflect recent updates."* | Invalidate stale entries once upstream services recover; check semantic cache TTL configuration. |
| `handler_failure` | Unhandled application or agent internal logic exception. | **HIGH** | *"An unexpected processing error occurred. A fallback response was generated."* | Inspect application stack trace; check for regression in agent orchestration logic. |

---

## 3. Privacy-Safe Telemetry Specification & Samples

Telemetry adheres to the **zero-content invariant**: prompts, answers, user identifiers, and authentication keys never enter telemetry structures.

### 3.1 Structured Log Event Sample

When a fallback is triggered, `telemetry.py` logs an operational event with content-free labels:

```json
{
  "timestamp": "2026-11-21T10:02:45.102Z",
  "level": "INFO",
  "logger": "telemetry",
  "message": "fallback={'trace_id': '4b82d3a1f9e843c98893d7c57ef25b21', 'provider': 'google', 'feature': 'tafsir', 'reason': 'provider_timeout', 'component': 'primary_gemini', 'version': '2.1.0', 'duration_ms': 42.15, 'risk_level': 'medium'}"
}
```

### 3.2 Metrics Snapshot (`GET /metrics`)

The in-memory telemetry registry exposes aggregated counters distinguishing fallback rates across providers and features:

```json
{
  "requests": {
    "total": 1250,
    "errors": 25,
    "error_rate": 0.02
  },
  "tokens": {
    "input": 182400,
    "output": 64500,
    "total": 246900
  },
  "cost_usd": 0.03303,
  "latency_ms": {
    "handler_p50": 85.0,
    "handler_p95": 310.0,
    "model_p50": 72.0,
    "model_p95": 280.0
  },
  "by_model": {
    "gemini-2.5-flash": {
      "calls": 1180,
      "total_tokens": 234000,
      "cost_usd": 0.0315
    }
  },
  "fallbacks": {
    "total": 45,
    "rate": 0.036,
    "by_provider": {
      "google": 38,
      "anthropic": 7
    },
    "by_feature": {
      "chat": 22,
      "hadith": 14,
      "tafsir": 9
    },
    "by_reason": {
      "provider_timeout": 28,
      "circuit_open": 10,
      "circuit_half_open_failed": 3,
      "safety_block": 4
    },
    "by_provider_feature": {
      "google": {
        "chat": 20,
        "hadith": 11,
        "tafsir": 7
      },
      "anthropic": {
        "chat": 2,
        "hadith": 3,
        "tafsir": 2
      }
    },
    "latency_ms": {
      "p50": 34.5,
      "p95": 88.0
    }
  }
}
```

---

## 4. Dashboards & Alerting Configuration

### 4.1 Prometheus Alert Rules (`alerts.yaml`)

```yaml
groups:
  - name: ai_provider_fallback_alerts
    rules:
      - alert: HighFallbackRateByProvider
        expr: (sum by (provider) (rate(ai_fallback_total[5m])) / sum by (provider) (rate(ai_requests_total[5m]))) > 0.15
        for: 3m
        labels:
          severity: warning
          tier: ai-gateway
        annotations:
          summary: "High fallback rate for AI provider {{ $labels.provider }}"
          description: "Provider {{ $labels.provider }} fallback rate is {{ $value | humanizePercentage }} over 5m (threshold: 15%)."

      - alert: HighFallbackRateByFeature
        expr: (sum by (feature) (rate(ai_fallback_total[5m])) / sum by (feature) (rate(ai_requests_total[5m]))) > 0.20
        for: 3m
        labels:
          severity: warning
          tier: ai-features
        annotations:
          summary: "High fallback rate for AI feature {{ $labels.feature }}"
          description: "Feature {{ $labels.feature }} has {{ $value | humanizePercentage }} fallback answers over 5m."

      - alert: CircuitBreakerOpenElevated
        expr: sum by (component) (increase(ai_circuit_breaker_open_total[5m])) > 3
        for: 1m
        labels:
          severity: critical
          tier: resilience
        annotations:
          summary: "Circuit breaker persistently OPEN for {{ $labels.component }}"
          description: "Component {{ $labels.component }} entered OPEN circuit state {{ $value }} times in 5 minutes."

      - alert: SafetyBlockSpike
        expr: sum by (feature) (rate(ai_fallback_total{reason="safety_block"}[5m])) > 5
        for: 2m
        labels:
          severity: warning
          tier: safety
        annotations:
          summary: "Elevated safety refusals in {{ $labels.feature }}"
          description: "Content moderation safety blocks exceeded 5/sec over 5 minutes."
```

### 4.2 Grafana Dashboard Specification

- **Panel 1: Fallback Rate by Provider**:
  `sum by (provider) (rate(ai_fallback_total[5m])) / sum by (provider) (rate(ai_requests_total[5m]))`
- **Panel 2: Fallback Rate by Feature**:
  `sum by (feature) (rate(ai_fallback_total[5m])) / sum by (feature) (rate(ai_requests_total[5m]))`
- **Panel 3: Fallback Reason Breakdown (Bar Gauge / Pie)**:
  `sum by (reason) (increase(ai_fallback_total[1h]))`
- **Panel 4: Fallback Duration Latency (p50 / p95)**:
  `histogram_quantile(0.95, sum by (le) (rate(ai_fallback_duration_ms_bucket[5m])))`

---

## 5. Non-Bypassable Control Invariants

To guarantee that fallbacks cannot serve as an adversarial or accidental escape hatch, `agent_fallback.py` enforces four invariants:

```
[Incoming Request]
        │
        ▼
 ┌───────────────┐
 │ 1. Auth Guard │ ── (invalid auth) ──► Raises ControlBypassError (No fallback)
 └───────┬───────┘
         │
         ▼
 ┌───────────────┐
 │ 2. Cost Guard │ ── (quota exceeded) ─► Raises ControlBypassError (No fallback)
 └───────┬───────┘
         │
         ▼
 ┌───────────────┐
 │ 3. Safety In  │ ── (policy refusal) ─► Terminal SAFETY_BLOCK (Cannot route to fallback)
 └───────┬───────┘
         │
         ▼
 [Primary Handler Execution]
   ├── Success ──► Return Standard Response
   └── Failure / Timeout / Circuit Breaker
         │
         ▼
 ┌────────────────────────────────────────────────────────┐
 │ Fallback Execution (Alternative / Deterministic Path)  │
 └───────────────────────┬────────────────────────────────┘
                         │
                         ▼
        ┌───────────────────────────────────┐
        │ 4. Output Controls Post-Execution │
        └─────────────────┬─────────────────┘
                          │
             ┌────────────┴────────────┐
             ▼                         ▼
   [Output Safety Check]      [Citation Verification]
     └── (violates safety)      └── (unverified / missing)
           │                          │
           ▼                          ▼
Raises SafetyBypassError    Raises ControlBypassError
(Unsafe output blocked)     (Uncited output rejected)
```

1. **Safety Moderation Invariant**:
   - If an input triggers a safety block, `_execute_fallback_chain()` halts immediately with `FallbackReason.SAFETY_BLOCK` and `risk_level=HIGH`. It is forbidden from routing to alternative agents or returning unmoderated text.
   - If a fallback agent generates text that violates the `output_safety_validator`, it raises `SafetyBypassError`, blocking output delivery.
2. **Citation Verification Invariant**:
   - For Islamic knowledge features (Fiqh, Hadith, Tafsir), responses generated by fallbacks must meet citation rules (`citation_validator`). Fallbacks lacking required Quran/Hadith citations are rejected with `ControlBypassError`.
3. **Authentication Invariant**:
   - Fallbacks cannot execute for unauthenticated contexts if `enforce_auth=True`. Missing credentials raise `ControlBypassError`.
4. **Cost & Quota Invariant**:
   - Fallback processing enforces budget limits (`max_cost_usd`). Overflowing requests raise `ControlBypassError`.

---

## 6. Thresholds, Disabling, and Review/Rollback Runbook

### 6.1 Disabling Thresholds & Review Triggers

Each agent fallback route is monitored against a `FallbackThresholdConfig`:
- `max_fallback_rate` (default: 0.50 / 50% fallback rate).
- `min_requests_for_threshold` (default: 5 requests before threshold evaluation).
- `auto_disable_on_threshold` (default: False; if True, trips circuit into fail-fast mode upon threshold breach).

When `fallback_rate > max_fallback_rate`:
- If `auto_disable_on_threshold=False`: The agent status transitions to `FallbackReviewStatus.REVIEW_REQUIRED`, signaling alerts for operator review.
- If `auto_disable_on_threshold=True`: The agent fallback is automatically disabled (`FallbackReviewStatus.DISABLED`), stopping degraded answers from silently masking upstream failure.

### 6.2 Operator Commands for Rollback & Re-enable

Operators can query, disable, or re-enable fallback paths at runtime:

```python
# Check operational status of an agent's fallback path
status = registry.get_fallback_status("primary_gemini")
# Returns:
# {
#   "agent_name": "primary_gemini",
#   "total_calls": 120,
#   "fallback_count": 72,
#   "fallback_rate": 0.60,
#   "threshold": 0.50,
#   "status": "review_required",
#   "fallback_enabled": True,
#   "disabled_reason": ""
# }

# Rollback / disable a broken fallback route (emergency kill-switch)
registry.rollback_fallback("primary_gemini", reason="Quality regression: hallucinated rulings in backup agent")

# Once disabled, subsequent calls fail fast rather than returning degraded output:
# AgentExecutionResult(success=False, error="Fallback disabled for 'primary_gemini': ...", fallback_used=False)

# Re-enable fallback route after deploying patch
registry.enable_fallback("primary_gemini")
```

---

## 7. Failure-Injection Verification Results

Automated failure-injection test suites (`tests/test_agent_fallback_audit.py`, `tests/test_agent_fallback.py`, `tests/test_agent_fallback_failover.py`) executed all failure scenarios:

| Failure Injection Scenario | Simulated Fault | Observed Fallback Reason | Risk Level | User Message Emitted | Control Verification | Result |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Provider Timeout** | Injected `TimeoutError(5000ms)` into primary Gemini handler | `provider_timeout` | `medium` | *"The primary provider timed out. An alternative response was generated."* | Telemetry recorded `provider=google`, `feature=tafsir`; no prompt text leaked. | **PASS** |
| **Circuit Breaker Open** | Injected connection failure exceeding failure threshold | `circuit_open` | `high` | *"The service is experiencing connectivity issues with the primary provider. A degraded fallback response was generated."* | Circuit entered `OPEN` state; primary not invoked on subsequent call. | **PASS** |
| **Circuit Half-Open Failure** | Canary probe call failed in `HALF_OPEN` state | `circuit_half_open_failed` | `high` | *"Provider recovery check failed. Degraded fallback response provided."* | Circuit returned to `OPEN`; emitted probe failure telemetry. | **PASS** |
| **Safety Block Refusal** | Injected hate-speech policy refusal in primary | `safety_block` | `high` | *"This response could not be generated due to safety policy constraints."* | **Terminal refusal**: Both alternative agent and deterministic fallback were blocked from executing. | **PASS** |
| **Unsafe Fallback Output** | Alternative agent returned toxic text | `safety_block` | `high` | (Exception raised) | `output_safety_validator` caught output; raised `SafetyBypassError`; text dropped. | **PASS** |
| **Deterministic Fallback** | Injected generic model crash into Zakat agent | `deterministic_fallback` | `medium` | *"A rule-based deterministic response was provided due to temporary service degradation."* | Algorithmic rate math executed; emitted deterministic metadata. | **PASS** |
| **Citation Control Check** | Fallback output missing Bukhari/Muslim reference | N/A | N/A | (Exception raised) | `citation_validator` intercepted uncited output; raised `ControlBypassError`. | **PASS** |
| **Auth Control Check** | Request executed with expired token | N/A | N/A | (Exception raised) | `auth_validator` intercepted request; raised `ControlBypassError`. | **PASS** |
| **Cost Control Check** | Request with cost ($0.12) exceeding limit ($0.05) | N/A | N/A | (Exception raised) | `control_policy` intercepted request; raised `ControlBypassError`. | **PASS** |
| **Threshold Disabling** | 3 of 3 requests failed with rate (1.0) > threshold (0.4) | `provider_unavailable` | `high` | *"Fallback disabled for 'fragile'..."* | Registry auto-disabled fallback; 4th request failed fast without fallback. | **PASS** |
| **Operator Manual Rollback** | Operator invoked `rollback_fallback()` | N/A | N/A | *"Fallback disabled for 'search_agent'..."* | Status became `disabled`; fallback stopped; re-enabled cleanly with `enable_fallback()`. | **PASS** |
| **Privacy Leak Proof** | Injected secret prompt and fake API key into error body | `provider_timeout` | `medium` | Content-free message | Zero occurrences of secret prompt or API key in `FallbackMetadata.to_dict()` or `telemetry.registry.snapshot()`. | **PASS** |

---

## 8. Summary of Files Changed

- `agent_fallback.py`:
  - Added `FallbackReason`, `FallbackRiskLevel`, `FallbackReviewStatus` enums.
  - Added `FallbackMetadata` dataclass with `to_dict()`.
  - Added `FallbackControlPolicy` and `FallbackThresholdConfig`.
  - Added `SafetyBlockError`, `SafetyBypassError`, `ControlBypassError`, and `FallbackDisabledError`.
  - Added `DEFAULT_USER_MESSAGES` and `DEFAULT_RISK_LEVELS`.
  - Extended `AgentRegistry` with provider/feature/version registration, control validators, review status, disabling thresholds, and operator rollback.
  - Enhanced `_execute_fallback_chain` to enforce non-bypassable safety, citation, auth, and cost controls and record privacy-safe telemetry.
- `telemetry.py`:
  - Added `FallbackEvent` dataclass with content-free `as_labels()`.
  - Added fallback aggregation to `MetricsRegistry` (`fallback_count`, `_fallbacks_by_provider`, `_fallbacks_by_feature`, `_fallbacks_by_reason`, `_fallbacks_by_provider_feature`, latencies).
  - Added `record_fallback_event()` helper function.
- `docs/architecture.md`:
  - Updated section 3.2 to document auditable fallback decisions, reason codes, truthful messaging, control invariants, telemetry by provider/feature, and rollback runbooks.
- `docs/FALLBACK_AUDIT.md`:
  - Full release evidence document including taxonomy, telemetry samples, alerting configuration, and failure-injection results.
- `tests/test_agent_fallback_audit.py`:
  - 13 comprehensive compliance tests covering all acceptance criteria.
- `tests/test_telemetry.py`:
  - Added tests verifying fallback telemetry aggregation and privacy redaction invariants.
