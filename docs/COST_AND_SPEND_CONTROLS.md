# Cost and Spend Controls

**Issue**: [#882](https://github.com/Deen-Bridge/dnb-ai/issues/882)  
**Audit Objective**: Prevent provider usage from creating an uncontrolled financial liability.

---

## 1. Overview

DeenBridge AI makes calls to external LLM providers (primarily Google Gemini) on every
AI-generated response.  Without production-wide spend controls, a traffic spike, abuse
event, retry storm, or provider price change can exceed the intended budget even when
individual users remain within their hourly token quotas.

This document defines the cost accounting contract, budget policy, alert configuration,
and emergency cutoff procedure that satisfy audit objective AI-106.

> [!IMPORTANT]
> The zero-content invariant applies to all cost telemetry: prompt text, answer text,
> user identifiers, and API keys **never** appear in cost records, logs, or metric labels.

---

## 2. Architecture

```
Provider call (primary / retry / fallback)
         │
         ▼
  build_call_record()          ← model, input_tokens, output_tokens,
  [provider_cost.py]              price_version, estimated_cost_usd
         │
         ▼
  BudgetCutoff.check_and_record()
         │
         ├── is_critical_path?  → allow, still record spend
         │   (/health, /auth, /metrics, …)
         │
         ├── emergency_stop active? → DegradedResponse (safe, content-free)
         │
         ├── per_call_limit exceeded? → BudgetExceededError
         │
         ├── tenant_daily cap exceeded? → BudgetExceededError
         │
         ├── daily alert threshold? → BudgetAlertThresholdReached (logs, fires alert)
         │
         ├── daily limit reached? → BudgetExceededError → DegradedResponse
         │                          activates emergency_stop for fast-fail
         ├── monthly alert threshold? → BudgetAlertThresholdReached
         │
         └── monthly limit reached? → BudgetExceededError → DegradedResponse
```

**Key invariant inherited from #884**: retries and fallbacks call
`check_and_record()` for every attempt — the budget cannot be circumvented by
routing through an alternate provider or model.

---

## 3. Versioned Model Prices

Prices are defined in `telemetry.PRICE_TABLE` (USD per 1,000 tokens) and extended
by `provider_cost.py`.  The active price schedule is identified by a **price version**
string recorded in every `CallRecord`.

| Model | Input ($/1k tokens) | Output ($/1k tokens) | Price Version |
|---|---|---|---|
| `gemini-2.5-flash` | $0.000075 | $0.0003 | `2026-05` |
| `gemini-2.5-flash-preview-05-20` | $0.000075 | $0.0003 | `2026-05` |
| `gemini-2.5-pro` | $0.00125 | $0.01 | `2026-05` |

Prices are overridden at deploy time via environment variables:

```bash
# Update prices when the provider changes their schedule:
LLM_PRICE_TABLE='{"gemini-2.5-flash": {"input": 0.000090, "output": 0.0004}}'
LLM_PRICE_VERSION="2026-08"
```

> [!WARNING]
> Increment `LLM_PRICE_VERSION` whenever `LLM_PRICE_TABLE` changes.  An unchanged
> version with updated prices creates a gap in the audit trail.

---

## 4. Budget Policy

### 4.1 Global Production Budgets

| Window | Default Limit | Alert Threshold | Environment Variable |
|---|---|---|---|
| Daily | $50.00 | 80 % ($40.00) | `BUDGET_DAILY_USD` |
| Monthly (30 days) | $1,000.00 | 80 % ($800.00) | `BUDGET_MONTHLY_USD` |
| Alert fraction | 80 % | — | `BUDGET_ALERT_FRACTION` |
| Per-call hard cap | None | — | `BUDGET_PER_CALL_USD` |

Change defaults at deploy time by setting the corresponding environment variables.
The module-level `configure_default_guard(BudgetConfig(...))` function replaces the
singleton before the application starts serving requests.

### 4.2 Per-Tenant Daily Caps

Tenants with elevated usage risk can be assigned an individual daily cap:

```python
from provider_cost import default_guard

default_guard.set_tenant_cap("tenant-enterprise-42", daily_limit_usd=5.0)
```

A tenant that hits its cap receives a `BudgetExceededError` while the global budget
remains unaffected for other tenants.

### 4.3 Per-Call Limit

When `BUDGET_PER_CALL_USD` is set, any single call that estimates a cost above that
threshold is rejected before the tokens are submitted to the provider, preventing a
single runaway prompt from draining the budget.

---

## 5. Per-Call Cost Recording

Every provider interaction — primary call, retry, or fallback — must produce a
`CallRecord` and pass it through `BudgetCutoff.check_and_record()`.

```python
from provider_cost import build_call_record, default_cutoff

# After receiving the provider response:
record = build_call_record(
    model="gemini-2.5-flash",
    input_tokens=response.usage_metadata.prompt_token_count,
    output_tokens=response.usage_metadata.candidates_token_count,
    trace_id=current_trace_id,
    tenant_id=request.tenant_id,       # optional
    is_retry=attempt > 0,
    is_fallback=False,
)
degraded = default_cutoff.check_and_record(record, path=request.url.path)
if degraded is not None:
    return degraded.to_dict()          # safe, content-free response to user
```

The `CallRecord` captures:

| Field | Description |
|---|---|
| `model` | Model name as returned by the provider |
| `input_tokens` | Prompt token count from `usage_metadata` |
| `output_tokens` | Completion token count |
| `estimated_cost_usd` | Computed from `telemetry.PRICE_TABLE` |
| `priced` | `False` when model is not in the price table |
| `price_version` | Active price schedule version |
| `trace_id` | Request trace ID for correlation |
| `tenant_id` | Optional tenant identifier |
| `is_retry` | Whether this is a retry attempt |
| `is_fallback` | Whether this is a fallback call |

---

## 6. Alert Rules

### 6.1 Log-Based Alerts

The spend guard logs structured lines at `WARNING` level when thresholds are crossed.
Configure your log aggregator (Cloud Logging, Datadog, Loki) to alert on these patterns:

| Log pattern | Meaning | Recommended action |
|---|---|---|
| `cost_alert window=daily fraction=0.80` | 80 % of daily limit used | Page on-call; investigate traffic spike |
| `cost_alert window=monthly fraction=0.80` | 80 % of monthly limit used | Review usage trend; consider raising limit |
| `cost_alert tenant=<id> window=tenant_daily` | Tenant approaching their cap | Notify tenant; review their usage |
| `budget_exceeded window=daily` | Daily limit hit; cutoff activated | Immediate page; investigate root cause |
| `budget_exceeded window=monthly` | Monthly limit hit | Escalate to engineering lead |
| `EMERGENCY_CUTOFF_ACTIVATED` | Emergency stop active | All non-critical generation suspended |

### 6.2 Prometheus Alert Rules

```yaml
groups:
  - name: provider_spend_alerts
    rules:
      - alert: DailyBudgetAlert
        expr: dnb_daily_spend_usd / dnb_daily_budget_usd > 0.80
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "Daily AI spend at {{ $value | humanizePercentage }} of limit"
          description: "Daily spend has crossed the 80% alert threshold."

      - alert: MonthlyBudgetAlert
        expr: dnb_monthly_spend_usd / dnb_monthly_budget_usd > 0.80
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Monthly AI spend at {{ $value | humanizePercentage }} of limit"

      - alert: BudgetCutoffActive
        expr: dnb_emergency_cutoff_active == 1
        for: 0m
        labels:
          severity: critical
        annotations:
          summary: "AI provider budget cutoff is ACTIVE — generation suspended"
          description: "Non-critical AI generation has been suspended. Health and auth endpoints remain up."

      - alert: TenantCapExceeded
        expr: dnb_tenant_spend_usd{tenant!=""} / dnb_tenant_cap_usd{tenant!=""} > 1.0
        for: 1m
        labels:
          severity: warning
        annotations:
          summary: "Tenant {{ $labels.tenant }} has exceeded daily spend cap"
```

### 6.3 Grafana Dashboard Panels

| Panel | Query |
|---|---|
| Daily spend vs limit | `dnb_daily_spend_usd` + threshold line at `dnb_daily_budget_usd` |
| Monthly spend trend | `increase(dnb_provider_cost_total[30d])` |
| Spend by model | `sum by (model) (increase(dnb_provider_cost_total[1d]))` |
| Spend by tenant | `sum by (tenant) (increase(dnb_provider_cost_total[1d]))` |
| Retry/fallback cost share | `sum(dnb_provider_cost_total{is_retry="true"} or dnb_provider_cost_total{is_fallback="true"})` |
| Emergency cutoff state | `dnb_emergency_cutoff_active` (stat panel; red when 1) |

---

## 7. Emergency Cutoff Procedure

### 7.1 What the cutoff does

When `BudgetCutoff` activates the emergency stop (automatically on budget exhaustion
or manually by an operator):

- All non-critical generation endpoints return a **safe, content-free `DegradedResponse`**.
- `/health`, `/healthz`, `/readiness`, `/auth`, `/metrics`, and `/login` are **never blocked**.
- The degraded response includes `retry_after_seconds: 300` so clients know to wait.

### 7.2 Manual activation

```python
from provider_cost import default_cutoff

# Activate immediately (e.g., after detecting an abuse event)
default_cutoff.activate_emergency_stop(reason="abuse_spike_detected")

# Resume after the incident is resolved and budget window resets
default_cutoff.deactivate_emergency_stop()
```

### 7.3 Cutoff Drill

Run this drill before every major release to verify the cutoff works end-to-end:

```bash
# Step 1: Set a minimal budget so the cutoff triggers quickly
export BUDGET_DAILY_USD=0.00001
export BUDGET_MONTHLY_USD=0.0001

# Step 2: Start the service
uvicorn main:app --host 0.0.0.0 --port 8000

# Step 3: Send two requests to exhaust the budget
curl -s -X POST http://localhost:8000/chat -d '{"message":"test"}' -H "Content-Type: application/json"
curl -s -X POST http://localhost:8000/chat -d '{"message":"test2"}' -H "Content-Type: application/json"

# Step 4: Verify the third request returns the degraded response
curl -s -X POST http://localhost:8000/chat -d '{"message":"test3"}' -H "Content-Type: application/json" \
  | python -c "import sys,json; r=json.load(sys.stdin); assert r['status']=='budget_exhausted', r"

# Step 5: Verify health is still up
curl -sf http://localhost:8000/health | python -c "import sys,json; r=json.load(sys.stdin); assert r.get('status')=='ok', r"

# Step 6: Deactivate the emergency stop (simulate budget window reset or operator action)
python -c "from provider_cost import default_cutoff; default_cutoff.deactivate_emergency_stop()"

# Step 7: Verify generation resumes
curl -s -X POST http://localhost:8000/chat -d '{"message":"resume"}' -H "Content-Type: application/json"
```

Expected outcomes:

| Step | Expected result |
|---|---|
| Step 3 | Second `/chat` request succeeds |
| Step 4 | Third `/chat` returns `{"status":"budget_exhausted","user_message":"..."}` |
| Step 5 | `/health` returns `{"status":"ok"}` even during cutoff |
| Step 7 | Generation resumes after `deactivate_emergency_stop()` |

> [!CAUTION]
> Never run the cutoff drill in production against live traffic. Use a staging instance
> with a dedicated test API key.

---

## 8. Release Evidence

Attach the following artefacts to every production release that changes budget
thresholds or cost accounting logic:

| Artefact | Source |
|---|---|
| Budget policy | This document (§4) |
| Dashboard / alert configuration | §6 Prometheus rules + Grafana panels |
| Cutoff drill results | Output of the drill in §7.3 |
| Redacted cost report | `default_cutoff.snapshot()` with no prompt/secret content |
| Test results | `pytest tests/test_provider_cost.py -v` — all pass |

---

## 9. Runbook Owner

**Primary**: Platform team on-call  
**Escalation**: Engineering lead  
**Review cadence**: Monthly — reassess budget limits after each provider price change
and after any traffic anomaly.
