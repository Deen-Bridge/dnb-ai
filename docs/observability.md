# Observability & Monitoring with Prometheus and Grafana

Deen Bridge AI Service exposes a Prometheus-compatible `/metrics` endpoint for comprehensive real-time monitoring, metrics collection, and alerting.

---

## 1. Overview

The service utilizes `prometheus-fastapi-instrumentator` for automatic HTTP request instrumentation and exports custom domain metrics tracking LLM interactions, token usage, caching efficiency, answer confidence scores, and scholar review queue depth.

### Endpoint

- **Path**: `GET /metrics`
- **Format**: Prometheus text format (`text/plain; version=0.0.4; charset=utf-8`)
- **JSON Fallback**: `GET /metrics/json` or `GET /metrics` with `Accept: application/json`

---

## 2. Exported Metrics

### Custom Domain Metrics

| Metric Name | Type | Labels | Description |
|---|---|---|---|
| `dnb_ai_model_calls_total` | Counter | `model`, `stage` | Total number of Gemini model calls partitioned by model identifier and pipeline stage (e.g., `generation`, `fiqh_classification`). |
| `dnb_ai_model_latency_seconds` | Histogram | *None* | Latency of individual model invocations in seconds. Buckets: `[0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0]`. |
| `dnb_ai_tokens_total` | Counter | `direction` | Cumulative token consumption categorized by direction (`input` for prompt tokens, `output` for completion tokens). |
| `dnb_ai_cache_hits_total` | Counter | `cache_type` | Total number of cache hits (`semantic` or `exact`). |
| `dnb_ai_cache_misses_total` | Counter | `cache_type` | Total number of cache misses (`semantic` or `exact`). |
| `dnb_ai_confidence_score` | Histogram | *None* | Distribution of computed confidence scores (0.0 to 1.0) across generated answers. Buckets: `[0.1, 0.2, ..., 1.0]`. |
| `dnb_ai_scholar_queue_depth` | Gauge | *None* | Current number of low-confidence answers awaiting scholar vetting in the review queue. |

### Standard HTTP Metrics

Provided by `prometheus-fastapi-instrumentator`:
- `http_requests_total`: Request counts partitioned by HTTP method, handler route, and status code.
- `http_request_duration_seconds`: Request latency histogram.
- `http_request_size_bytes` & `http_response_size_bytes`: Request and response payload sizes.

---

## 3. Configuration & Access Protection

The metrics endpoint can be protected in production environments via bearer tokens or IP allowlisting.

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `METRICS_TOKEN` | *(unset)* | When set, requests to `/metrics` require `Authorization: Bearer <token>` or `X-Metrics-Token: <token>`. Returns `401 Unauthorized` if missing/invalid. |
| `METRICS_IP_ALLOWLIST` | *(unset)* | Comma-separated list of allowed IPv4/IPv6 addresses or CIDRs (e.g. `127.0.0.1, 10.0.0.0/8, 192.168.1.50`). Returns `403 Forbidden` if client IP is not matched. |
| `ENABLE_METRICS` | `true` | Set to `false` to disable HTTP metrics collection. |

---

## 4. Prometheus Setup

Add the scrape configuration to your `prometheus.yml`:

### Standard Scrape (Open / Private VPC)

```yaml
scrape_configs:
  - job_name: "dnb-ai"
    scrape_interval: 15s
    scrape_timeout: 10s
    metrics_path: "/metrics"
    static_configs:
      - targets: ["dnb-ai:8000"]
```

### Authenticated Scrape (with `METRICS_TOKEN`)

```yaml
scrape_configs:
  - job_name: "dnb-ai"
    scrape_interval: 15s
    metrics_path: "/metrics"
    bearer_token: "your-metrics-token-here"
    static_configs:
      - targets: ["dnb-ai.internal:8000"]
```

---

## 5. Grafana Dashboard & PromQL Queries

### Key PromQL Expressions

#### 1. Model Request Rate (Calls / sec by Model)
```promql
sum by (model, stage) (rate(dnb_ai_model_calls_total[5m]))
```

#### 2. Model Latency P50 & P95
```promql
# P95 Model Latency in Seconds
histogram_quantile(0.95, sum(rate(dnb_ai_model_latency_seconds_bucket[5m])) by (le))

# P50 (Median) Model Latency in Seconds
histogram_quantile(0.50, sum(rate(dnb_ai_model_latency_seconds_bucket[5m])) by (le))
```

#### 3. Token Consumption Rate (Tokens / min)
```promql
sum by (direction) (rate(dnb_ai_tokens_total[5m]) * 60)
```

#### 4. Semantic Cache Hit Ratio (%)
```promql
sum(rate(dnb_ai_cache_hits_total{cache_type="semantic"}[5m]))
/
(
  sum(rate(dnb_ai_cache_hits_total{cache_type="semantic"}[5m]))
  + sum(rate(dnb_ai_cache_misses_total{cache_type="semantic"}[5m]))
) * 100
```

#### 5. Scholar Queue Backlog (Current Pending Count)
```promql
dnb_ai_scholar_queue_depth
```

#### 6. Average Confidence Score Distribution
```promql
histogram_quantile(0.50, sum(rate(dnb_ai_confidence_score_bucket[5m])) by (le))
```

---

## 6. Example Alerting Rules (`alerts.yml`)

```yaml
groups:
  - name: dnb_ai_alerts
    rules:
      - alert: ScholarQueueBacklogHigh
        expr: dnb_ai_scholar_queue_depth > 50
        for: 15m
        labels:
          severity: warning
        annotations:
          summary: "Scholar review queue backlog is high ({{ $value }} pending)"
          description: "Low-confidence questions are accumulating in the review queue without timely verification."

      - alert: ModelHighLatency
        expr: histogram_quantile(0.95, sum(rate(dnb_ai_model_latency_seconds_bucket[5m])) by (le)) > 10
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Model call P95 latency exceeds 10s"
          description: "Upstream LLM response times are degraded (p95: {{ $value }}s)."

      - alert: HighHttp5xxErrorRate
        expr: sum(rate(http_requests_total{status=~"5.."}[5m])) / sum(rate(http_requests_total[5m])) > 0.05
        for: 3m
        labels:
          severity: critical
        annotations:
          summary: "High HTTP 5xx error rate (>5%)"
          description: "Service is experiencing elevated server errors."
```
