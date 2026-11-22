# Book Moderation Outage Runbook and State Model

## 1. Moderation State Model

Book moderation decisions are authoritative records that must remain consistent across application restarts, pod redeployments, and multi-replica clusters.

### Key Separation
- **Authoritative Moderation Store**: Key prefix `moderation:book:{book_id}`. Stores serialized `BookModeration` records containing status (`active` or `taken-down`), previous status, reason, note, timestamps, and author identifier. This state is strictly durable.
- **Catalog Cache**: Key patterns `books:catalog:*`, `books:category:*`, `books:detail:*`. Ephemeral performance caches owned by the REST API. A flush or drop of the catalog cache never impacts authoritative moderation state.

## 2. Durability and Outage Policy

Production environments require strict durability (`MODERATION_STRICT_DURABILITY=true` or `ENVIRONMENT=production`). Moderation writes cannot silently exist only in process memory.

### Degradation Architecture
```
                         +-----------------------------+
                         |    Moderation Write Request |
                         +--------------+--------------+
                                        |
                                        v
                         +--------------+--------------+
                         | ResilientBookModerationStore |
                         +--------------+--------------+
                                        |
                 +----------------------+----------------------+
                 | Primary (Redis)                             | Fallback (Durable File / WAL)
                 v                                             v
        [Redis Cluster]                             [data/moderation_store.json]
                 |                                             |
       (Outage detected) -------------------------------> (Persist state)
                                                               |
                                                    (Queue pending sync)
                                                               |
                                                    (Emit operator alert)
```

1. **Redis Primary Operation**:
   - In normal operation, writes and reads hit the shared Redis store.
   - Successful writes keep the local durable store synchronized.
2. **Redis Outage Behavior**:
   - If Redis connection or write fails during `take_down` or `restore`:
     - If durable fallback store is available: state is atomically written to disk (`data/moderation_store.json`), flagged for sync, and an operator alert is emitted.
     - If no durable fallback store is configured and strict durability is enabled: the write fails closed with `ModerationStoreOutageError` (mapped to HTTP 503 Service Unavailable), preventing silent loss.
   - Reads degrade gracefully to the durable fallback store if Redis is unavailable.
3. **Multi-Replica Consistency**:
   - Replicas with access to persistent storage maintain consistent decisions even during network partitions.

## 3. Cache Invalidation and Reconciliation

When a moderation state change occurs, catalog caches must be invalidated so changes take effect immediately:
1. `BookCacheInvalidator.invalidate(book_id)` attempts to purge all matching catalog cache keys in Redis.
2. If Redis is unavailable or invalidation fails:
   - The affected `book_id` is registered in `_pending_invalidations`.
   - An alert is emitted: `MODERATION_ALERT: Catalog cache invalidation failed during outage. Queued for reconciliation. affected_count=N book_ids=[...]`.
3. After Redis recovers:
   - Operators or background workers trigger `/books/moderation/reconcile` (or `service.reconcile_outage()`).
   - All pending records from the fallback store are synchronized to Redis primary storage.
   - Catalog caches are purged for all affected books.
   - Pending queues are cleared.

## 4. Operator Alerts and Privacy Guarantee

Operator alerts are emitted via structured logging without leaking sensitive book content, author names, or notes:
- **Format**:
  ```
  MODERATION_ALERT: Redis store outage during save. affected_count=1 book_ids=['bk-404']. Error: ConnectionRefusedError
  MODERATION_ALERT: Catalog cache invalidation failed during outage. Queued for reconciliation. affected_count=2 book_ids=['bk-101', 'bk-102']
  MODERATION_RECONCILIATION: Reconciled 2 catalog cache keys for 2 books.
  ```
- **Rule**: Never include book text, notes, or moderation reasons in log alerts or telemetry.

## 5. Operator Incident Runbook

When an alert fires indicating Redis degradation:
1. **Verify Redis Health**:
   - Check Redis cluster status and network latency.
   - Inspect container logs: `docker logs deenbridge-redis`.
2. **Verify Pod File System**:
   - Ensure `/var/data/moderation_store.json` has available disk space and write permissions.
3. **Execute Post-Recovery Reconciliation**:
   - Once Redis is healthy, invoke the reconciliation endpoint:
     ```bash
     curl -X POST https://api.deenbridge.internal/books/moderation/reconcile \
       -H "X-Admin-Token: $ADMIN_TOKEN"
     ```
   - Verify the returned JSON confirms `status: reconciled` and report counts.
