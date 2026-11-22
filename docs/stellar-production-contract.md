# Stellar account freshness production contract

## Scope and ownership

`stellar.py` owns the read-only account lookup and request boundary. The pure
checks in `stellar_contract_state.py` own network, public-key evidence, asset
issuer, and Horizon latest-ledger freshness decisions. Only the configured
`testnet` or `public` network and that network's canonical USDC issuer are
accepted. Invalid request evidence is rejected before constructing or calling
the Horizon client. A read-only Horizon root request obtains freshness evidence;
the account-balance request is made only after that evidence passes. Provider
data without usable current-ledger evidence fails closed.

This service does not create, sign, or submit transactions and never requests
private keys. Authentication and the existing safety pipeline are unchanged.

## Failure behavior

| Condition | Behavior |
|---|---|
| Unsupported/mismatched network, invalid public key, or noncanonical USDC | Reject before provider access |
| Missing/malformed/zero ledger evidence | Return sanitized service-unavailable error |
| Ledger close time in the future or older than the freshness limit | Return sanitized stale-data conflict (`STELLAR_LEDGER_STALE`) |
| Horizon account not found | Preserve the existing not-found response |
| Provider transport failure | Do not present a balance as current; return service unavailable |

`STELLAR_MAX_LEDGER_AGE_SECONDS` sets the maximum accepted age (default `30`,
must be positive). Set this based on the deployment's expected Horizon cadence;
do not raise it to mask provider outages.

## Observability and retention

Logs contain only the fixed freshness failure category. They must not include
wallet addresses, balances, provider payloads, tokens, or request bodies.
No individual check records are persisted. Telemetry retains only bounded-
dimension outcome counters in process memory, reset on restart; operational log
retention remains governed by the hosting platform's existing policy. CI
exercises synthetic public-key-shaped fixtures and emits only test names and a
sanitized aggregate result; no live provider calls or credentials are required.

## Deployment and rollback

Deploy with the existing `STELLAR_NETWORK` setting and no schema migration.
Monitor sanitized 503 rates and Horizon availability after rollout. Rollback is
the normal application-image rollback; no data cleanup is needed. Do not roll
back by disabling API authentication, key validation, asset checks, or read-only
controls. If provider freshness is consistently rejected, investigate Horizon
health and clock synchronization before considering a narrowly justified
freshness-window adjustment.
