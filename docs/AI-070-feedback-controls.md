# Feedback abuse controls (AI-070)

## Contract

- A submission must reference a (chat_id, message_id) answer snapshot recorded by this process. The client cannot create feedback records by supplying prompt or answer text. Unknown or expired snapshots return HTTP 404 and are not stored.
- The first rating for an answer wins. Retries for the same pair return the original feedback ID and do not change its rating, categories, or comment. This is an idempotent, at-most-once write contract.
- Feedback is keyed to opaque chat and answer IDs; chat deletion clears snapshots, feedback rows, and backend indexes. Chat IDs are bearer-like identifiers, not an account authentication mechanism.
- Existing per-IP sliding-window limits remain controlled by FEEDBACK_RATE_LIMIT_MAX and FEEDBACK_RATE_LIMIT_WINDOW. Forwarded IP headers are used only when TRUST_PROXY_HEADERS is enabled. This limiter is per process and should be supplemented by edge rate limiting for multi-replica deployments.

## Data and operations

Feedback retains the prompt, displayed answer, rating, categories, and optional comment for quality review. Do not include secrets or private user data in comments. SQLite uses FEEDBACK_DB_PATH and its existing bounded retention; Redis entries retain the configured feedback TTL. Admin exports remain protected by ADMIN_TOKEN.

Rejected submissions return 404; throttled submissions return 429; storage errors return 500. The application logs chat ID, answer ID, and rating on successful writes, and logs storage failures without logging prompt, answer, or comment bodies. Monitor 404/429/500 rates at the API edge and feedback storage health.

## Rollout and rollback

No schema migration is required. Deploy normally. Existing feedback rows remain readable; repeated submissions will stop changing old rows after deployment. If clients depend on submitting feedback after a process restart by resending prompt and answer, update them to retain the answer session or stop that fallback before rollout. Rollback is the normal application image rollback; it restores the previous endpoint behavior while leaving stored rows intact. Chat deletion performed by the new version removes associated feedback, so that data cannot be restored by application rollback.

## Verification evidence

The focused regression suite is pytest -q tests/test_feedback.py. It covers trusted snapshot enforcement, duplicate retry behavior, existing validation, rate limiting, and storage backends. Run the repository PR malware guard on all changed files before merge. No secrets or user data are needed to reproduce these checks.
