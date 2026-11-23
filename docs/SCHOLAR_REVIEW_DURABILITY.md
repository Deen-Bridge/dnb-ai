# Scholar-Review Durability, Degradation Policy & Operational Runbook

> **Audit Identifier**: `PROD-AUDIT AI-096` / Issue #872  
> **Status**: Production Standard  
> **Audience**: Platform Engineers, SREs, Incident Commanders, Islamic Scholars  

---

## 1. Executive Summary & Audit Finding

### The Risk
When Islamic answers fall below the confidence abstention threshold (`CONFIDENCE_LOW_THRESHOLD`), the system hedges or abstains from answering and routes the question to qualified scholars via the scholar-review queue. 

Historically, when Redis became unreachable, review items were buffered exclusively in volatile in-process Python memory (`self._local: dict`). If the process crashed, container restarted, or a cloud replica was replaced during a Redis outage, all queued religious questions were permanently and silently lost. Concurrently, the user's chat turn continued while the highest-risk unresolved questions vanished without operator awareness.

### The Solution
This document defines the fail-safe durability architecture, bounded degradation policies, automatic recovery reconciliation, operational alerting, and incident runbook for the scholar-review queue.

---

## 2. Durability Architecture

The review queue employs a two-tier durability strategy:

```
[Chat / Abstention Engine]
            │
            ▼
┌─────────────────────────┐
│       ReviewStore       │
└───────────┬─────────────┘
            │
   Healthy? │ Yes
   ─────────┴─────────► [Redis Cluster] (Primary Durable Store)
            │           - review:item:{id} (JSON record)
            │           - review:pending   (Sorted Set by created_at)
            │           - review:reviewed  (Sorted Set by reviewed_at)
         No │ (Outage)
            ▼
┌─────────────────────────┐
│       ReviewSpool       │ (Local Durable Spool)
│   (data/review/spool)   │ - item_{uuid}.json (Atomic write + fsync)
└───────────┬─────────────┘
            │
    Recovery│ (Automatic / POST /review/reconcile)
            ▼
     [Redis Cluster] (Reconciled with zero duplicates)
```

### Crash-Safe Disk Spooling
During a Redis outage, `ReviewStore` delegates persistence to `ReviewSpool`:
1. **Atomic Disk Write**: Review items are written to a unique temporary file (`item_{id}.tmp.{nonce}`) in `REVIEW_SPOOL_DIR` (default: `data/review/spool`).
2. **Synchronous Flush**: File buffers are flushed to physical storage using `os.fsync(f.fileno())` before renaming.
3. **Atomic Replace**: The temporary file is atomically renamed to `item_{id}.json` using `os.replace()`.
4. **Boot Recovery**: On process startup, `ReviewStore` inspects `REVIEW_SPOOL_DIR` and recovers all spooled items into local memory, marking the store `degraded: true` until Redis reconciles them.

### Zero-Duplicate Recovery Reconciliation
When Redis recovers (detected on `stats()`, background health checks, or explicit administrative call):
1. `reconcile_spool()` reads all `item_*.json` files from disk.
2. For each item, it checks if `review:item:{id}` already exists in Redis.
3. If not present, it commits the item and sorted set index entry transactionally via a Redis pipeline.
4. If already present (e.g. from a concurrent replica or previous partial sync), it updates verdict transitions if local carries a completed review.
5. Upon confirmation from Redis, the local spool file `item_{id}.json` is safely unlinked.
6. Once the spool is empty, the store automatically transitions back to `degraded: false` and `durable: true`.

---

## 3. Bounded Capacity & Degraded Policies

To prevent unbounded disk consumption during extended downstream outages, the spool capacity is strictly bounded by `REVIEW_SPOOL_MAX_SIZE` (default: `10000` items).

When spool capacity is breached, the store enforces `REVIEW_DEGRADED_POLICY`:

| Policy | Behavior on Capacity Breach | User Impact | Operator Impact |
|:---|:---|:---|:---|
| **`spool_or_reject`** *(Default)* | Raises `ReviewSpoolFullError`. Item is rejected from the queue. | The chat turn completes normally with an abstention message, but `assessment.queued` is set to `False`. The user is **NOT** falsely promised that a scholar will review the question (`SCHOLAR_QUEUED_NOTE` omitted). | `total_rejected` counter increments; emits `REVIEW_QUEUE_ALERT alert=queue_rejection`. Triggers P1 alert. |
| **`spool_or_drop`** | Evicts the oldest pending spooled item from disk and memory to admit the new item. | New question is accepted into the spool; user receives scholar queue confirmation. Oldest pending item is dropped. | `total_dropped` counter increments; emits `REVIEW_QUEUE_ALERT alert=queue_drop`. Triggers P1 alert. |

---

## 4. Alert Configuration & Monitoring

### 4.1 Structured Log Alert Signatures
Configure your log aggregator (Datadog, Loki, CloudWatch, GCP Cloud Logging) to alert on these structured log signatures:

| Log Signature | Severity | Condition | Action Required |
|:---|:---|:---|:---|
| `REVIEW_QUEUE_ALERT alert=degraded_mode status=active` | **WARNING (P2)** | Redis unreachable; serving from durable local spool. | Check Redis cluster connectivity and network firewalls. |
| `REVIEW_QUEUE_ALERT alert=degraded_mode status=resolved` | **INFO** | All spooled items reconciled to Redis; store is durable. | Verification only. |
| `REVIEW_QUEUE_ALERT alert=depth_exceeded` | **WARNING (P2)** | Spool item count exceeds `REVIEW_ALERT_DEPTH_THRESHOLD` (500). | Investigate persistent Redis outage and scholar throughput. |
| `REVIEW_QUEUE_ALERT alert=age_exceeded` | **WARNING (P2)** | Oldest spooled item age exceeds `REVIEW_ALERT_AGE_THRESHOLD_SECONDS` (3600s). | Check for stalled queue or un-reconciled spool instances. |
| `REVIEW_QUEUE_ALERT alert=queue_rejection` | **CRITICAL (P1)** | Spool full; question rejected under `spool_or_reject`. | Immediately expand spool capacity or restore Redis. |
| `REVIEW_QUEUE_ALERT alert=queue_drop` | **CRITICAL (P1)** | Spool full; oldest item dropped under `spool_or_drop`. | Data loss event: restore Redis and restore spool from backup. |

### 4.2 Prometheus Alert Rules (`alerts/scholar_review_alerts.yaml`)

```yaml
groups:
  - name: scholar_review_durability_alerts
    rules:
      - alert: ScholarReviewQueueDegraded
        expr: deenbridge_review_degraded == 1
        for: 3m
        labels:
          severity: warning
          tier: storage
        annotations:
          summary: "Scholar review queue running in degraded mode"
          description: "Redis is unreachable for review storage. Questions are spooled locally on disk."

      - alert: ScholarReviewSpoolHighDepth
        expr: deenbridge_review_spool_size > 500
        for: 5m
        labels:
          severity: warning
          tier: storage
        annotations:
          summary: "Scholar review spool depth elevated (> 500)"
          description: "Spool directory contains {{ $value }} un-reconciled items."

      - alert: ScholarReviewSpoolHighAge
        expr: deenbridge_review_spool_oldest_age_seconds > 3600
        for: 5m
        labels:
          severity: warning
          tier: storage
        annotations:
          summary: "Scholar review spool item age exceeds 1 hour"
          description: "Oldest pending review item has been waiting in local spool for {{ $value }}s."

      - alert: ScholarReviewItemsRejectedOrDropped
        expr: increase(deenbridge_review_rejections_total[5m]) > 0 or increase(deenbridge_review_drops_total[5m]) > 0
        for: 0m
        labels:
          severity: critical
          tier: storage
        annotations:
          summary: "Scholar review queue capacity breach: items rejected or dropped"
          description: "Review queue spool capacity exceeded. Backpressure or eviction active."
```

---

## 5. Incident Runbook

### Alert: `ScholarReviewQueueDegraded` (SEV-2)

1. **Verify Queue Status**:
   ```bash
   curl -s -H "X-Review-Token: $SCHOLAR_REVIEW_TOKEN" https://api.deenbridge.org/review/stats | jq .
   ```
   Inspect:
   - `"durable"`: Should report `false`.
   - `"degraded"`: Should report `true`.
   - `"spool.size"`: Shows items safely preserved on local disk.

2. **Check Redis Connectivity**:
   - Verify Redis service health in Cloud console or container cluster:
     ```bash
     redis-cli -u $REDIS_URL ping
     ```
   - Check application logs for connection error reasons:
     ```bash
     kubectl logs -l app=dnb-ai --tail=100 | grep "Review queue Redis"
     ```

3. **Check Local Spool Directory**:
   Verify spooled items are being written safely:
   ```bash
   ls -la /app/data/review/spool/
   ```

4. **Trigger Manual Reconciliation**:
   Once Redis connectivity is restored:
   ```bash
   curl -X POST -s -H "X-Review-Token: $SCHOLAR_REVIEW_TOKEN" https://api.deenbridge.org/review/reconcile | jq .
   ```
   Expected response:
   ```json
   {
     "status": "ok",
     "reconciled": 15,
     "remaining_spool_size": 0,
     "durable": true,
     "degraded": false
   }
   ```

### Alert: `ScholarReviewItemsRejectedOrDropped` (SEV-1)

1. **Check Spool Utilization**:
   ```bash
   curl -s -H "X-Review-Token: $SCHOLAR_REVIEW_TOKEN" https://api.deenbridge.org/review/stats | jq .spool
   ```
2. **Temporary Mitigation**:
   If Redis recovery is delayed, temporarily increase `REVIEW_SPOOL_MAX_SIZE` (e.g. from 10,000 to 50,000) via environment variable update to restore acceptance capacity.
3. **Escalate**: Page On-Call Storage Engineer and notify the Lead Scholar.

---

## 6. Failure-Injection Verification Results

Generated by `python scripts/review_durability_drill.py`:

```
===================================================================================================================
PRODUCTION AUDIT AI-096: SCHOLAR REVIEW DURABILITY FAILURE-INJECTION DRILL TRACE
===================================================================================================================

| Timestamp (UTC)      | Scenario / Failure-Injection Stage   | Durable   | Degraded   | Spool Size   | Verification Outcome   |
|----------------------|--------------------------------------|-----------|------------|--------------|------------------------|
| 2026-11-23T04:44:31Z | 1. Baseline healthy Redis operation  | True      | False      | 0            | Durable write to Redis; spool empty; 0 data loss |
| 2026-11-23T04:44:31Z | 2. Redis outage: item spooled to disk | False     | True       | 1            | Durable disk write to spool.json; degraded flag set; 0 data loss |
| 2026-11-23T04:44:31Z | 3. Process crash & restart during outage | False     | True       | 1            | Loaded 1 spooled item from disk on boot; 0 data loss |
| 2026-11-23T04:44:31Z | 4. Scholar verdict recorded while offline | False     | True       | 1            | Verdict saved to disk spool; status=approved; 0 data loss |
| 2026-11-23T04:44:31Z | 5. Redis recovers: automatic reconciliation | True      | False      | 0            | Reconciled 1 item to Redis (reviewed); spool cleared; durable restored |
| 2026-11-23T04:44:31Z | 6. Multi-replica concurrent reconciliation | True      | False      | 0            | Both replicas reconciled; exactly 2 unique items in Redis; zero duplicates |
| 2026-11-23T04:44:31Z | 7. Bounded capacity limit reached (spool_or_reject) | False     | True       | 2            | Capacity enforced (max=2); rejected=True; total_rejected=1 |

===================================================================================================================
FAILURE-INJECTION DRILL COMPLETED WITH ZERO DATA LOSS
===================================================================================================================
```
