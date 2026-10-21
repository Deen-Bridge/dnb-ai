# Privacy, Diagnostic Logs, and Trace Retention Policy (#586)

## Overview

Deen Bridge AI interacts with the Stellar network to read public blockchain states (such as token balances and transaction confirmations) in a strictly read-only manner. Although public ledger data is publicly accessible on-chain, associating public account addresses or transaction hashes directly with private user conversations and diagnostic traces can reveal sensitive user financial activity.

This document defines the privacy policy, retention schedules, and access boundaries applied across diagnostic logs and traces.

---

## Retention Windows

| Artifact Type | Retention Period | Purge Trigger | Description |
|---|---|---|---|
| **Prompts in Routine Logs** | 0 days (`LOG_PROMPTS=false`) | Real-time redaction | Prompts are never written to standard application logs. |
| **Diagnostic Traces & Spans** | 7 days | Automated nightly expiration job | Operational traces recording model latency, status codes, and error categories. |
| **User Uploads & Media** | 24 hours | Periodic purge worker | Temporary files used for voice notes, OCR, or analysis. |
| **Conversation Memory Store** | User-managed / 30-day inactivity | User deletion request or expiration | Retained only when authenticated session persistence is enabled. |

---

## Public Chain Identifier Redaction & Keyed Correlation

To protect user privacy while maintaining operational debuggability:

1. **Response Path Isolation**: Full public account IDs and transaction hashes are returned to the authenticated client in API response bodies only when required by the client workflow.
2. **Diagnostic Trace Redaction**:
   - Secret keys (`S...`) and seed phrases are forbidden and always completely masked as `[REDACTED_SECRET_KEY]`.
   - Public account keys (`G...`) and 64-character transaction hashes in diagnostic logs are replaced with one-way, keyed pseudonymous correlation tokens (format: `corr_<16-hex-chars>`).
   - Memos, user-supplied notes, and prompt text are unconditionally replaced with `[REDACTED]`.
3. **Keyed Salt**: Correlation tokens are minted using HMAC-SHA256 with an internally secured key. This prevents correlation across external systems while allowing operators to group spans belonging to the same transaction context during an active troubleshooting incident.

---

## Access Boundary

Diagnostic traces and detailed error logs are subject to strict access controls:
- **Authorized Roles**: Access is restricted strictly to authorized Site Reliability Engineering (SRE) and platform security engineers.
- **Network Boundaries**: Trace sinks and log aggregators are accessible only via private VPC peering or IP-allowlisted ingress behind multi-factor authentication.
- **Zero Raw Identifiers**: Even internal operators querying trace sinks cannot inspect raw account keys or memo strings because redaction is applied at the application boundary prior to emission.
