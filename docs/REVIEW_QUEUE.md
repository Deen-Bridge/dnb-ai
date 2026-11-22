# Scholar review queue operations

**Owner:** DeenBridge AI service maintainers

## State contract

Review items have four states: pending, approved, corrected, and rejected.
Only a pending item can transition, and a verdict is final. Approved items
retain the original answer; corrected items retain both the original and
scholar-corrected answer; rejected answers are never cached as usable answers.
The item record and its pending/reviewed indexes are updated inside one Redis
Lua script. The in-memory backend applies the transition without yielding
between its pending check and mutation.

The Redis item record is authoritative. The pending index is for queue listing;
if a pending record is missing from that index, a verdict can still finalize
it and add it to the reviewed index. There is no automatic retry or
last-write-wins behavior.

## Data and access

Queue records contain the user question, generated answer, confidence signals,
chat ID, and reviewer identity and note. They have no TTL. Protect Redis and
REVIEW_EXPORT_PATH with deployment access controls and disk permissions.
Reviewer endpoints require SCHOLAR_REVIEW_TOKEN via X-Review-Token. The JSONL
export contains reviewed content and must be treated as sensitive.

## Failure and recovery

Set REDIS_URL for durable queue storage. Without Redis, the service uses an
in-process queue that does not survive restart. Queue writes may buffer locally
during an outage and expose degraded durability in /review/stats. A verdict
transition Redis cannot confirm returns HTTP 503; the script may have committed
before a connection failure, so inspect the queue and stats before retrying.
Never overwrite a committed verdict to resolve an ambiguous response.

## Rollback

The Lua script adds no key or schema migration. Roll back application code
normally; keep existing item and index keys. Review records and exports are
not automatically cleared.