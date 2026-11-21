# Quran Provider Contract and Freshness Monitor

## AI-101 — Make Quran source availability and freshness observable

**Version:** 1.0.0  
**Status:** Active  

---

## 1. Provider Contract

### Primary Provider

| Field | Value |
|---|---|
| Provider | Quran.com API |
| Version | v4 |
| Base URL | `https://api.quran.com/api/v4` |
| Auth | None (public) |
| Rate limit | 100 req/min (unauthenticated) |
| SLA | Community-maintained; no formal uptime guarantee |

### Endpoints Used

| Endpoint | Purpose |
|---|---|
| `GET /verses/by_key/{surah}:{ayah}` | Fetch single ayah with Uthmani text |
| `GET /translations` | List available translation resources |
| `GET /chapters` | Surah metadata (validation reference) |

### Data Contract

Every ayah response from the provider **must** include:

- `text_uthmani` — the Uthmani script Arabic text
- `translations[]` — at least one translation entry with `resource_id`, `resource_name`, `language_name`, and `text`
- Implicit metadata: API version (`v4`), fetch timestamp injected by this service, language code

### Reference Validation

Before any network call, the service validates the requested reference against `data/quran/surah_index.json`:

1. Surah number must be in range [1, 114].
2. Ayah number must be in range [1, surah.verses_count].
3. Invalid references **must** return an error without making any HTTP call.

---

## 2. Caching and Freshness

| Setting | Value |
|---|---|
| Cache backend | In-process LRU (default) or Redis (production) |
| TTL per ayah | 24 hours |
| Stale threshold | > 24 hours since `fetched_at` |
| Cache key format | `quran:ayah:{surah}:{ayah}:{translation_id}` |

A cache entry is considered stale when `time.time() - fetched_at > CACHE_TTL_SECONDS`. Stale entries trigger a background refresh. If the refresh fails, the stale entry is served with a `stale: true` flag in the metadata.

---

## 3. Fallback and Degradation

| Condition | Behavior |
|---|---|
| Provider unreachable (ConnectError) | Return cached entry (stale OK) with `source_available: false` |
| Provider timeout | Same as above; log warning |
| HTTP 429 (rate limit) | Backoff + retry once; return cache on second failure |
| HTTP 5xx | Return cache; mark `source_available: false` |
| Cache miss + provider down | Return `{"error": "unavailable", "source_available": false}` |
| Cache hit + provider down | Serve stale with `stale: true` metadata |

**The service must never raise an unhandled exception to the caller** due to a provider failure.

---

## 4. Health and Readiness Endpoints

### `check_quran_provider_health() -> bool`

Performs a lightweight probe of the Quran.com API (fetches 1:1) and returns:

- `True` — provider reachable and returned a valid ayah
- `False` — any network error or non-2xx response

This function is called:
- By the `/health` readiness endpoint
- In scheduled monitoring (every 15 minutes in production)
- In CI smoke tests

### `is_cache_stale(fetched_at_timestamp: float) -> bool`

Returns `True` if `time.time() - fetched_at_timestamp > CACHE_TTL_SECONDS`.

---

## 5. Drift Detection

### `detect_source_drift(surah, ayah, expected_text) -> bool`

Fetches the live ayah and compares its `text_uthmani` against a stored fixture. Returns `True` on mismatch or provider unavailability.

**Drift triggers an alert when:**
- The Arabic text differs from the pinned fixture by more than 1 character
- The provider is unreachable during the scheduled check
- A translation `resource_id` is no longer served

### Scheduled Drift Verification

A weekly CI/cron job (`scripts/verify_quran_fixtures.py`) runs `detect_source_drift` against 10 sampled ayat from different surahs (covering different revelation periods and text complexity). Any drift fails the check and sends an alert.

---

## 6. Incident Procedure

### On Provider Outage

1. Automated alert fires within 15 minutes (via health check failure).
2. Service continues degraded: stale cache served with `source_available: false`.
3. On-call engineer verifies provider status at https://status.quran.com.
4. If outage > 6 hours, evaluate bundling a minimal offline corpus for the top-50 most-requested ayat.

### On Source Disagreement (Drift)

1. Drift alert fires from weekly verification job.
2. A qualified Quranic scholar reviews the differing text against the Uthmani mushaf.
3. If provider text is wrong: raise issue with Quran.com API team; use cached fixture.
4. If fixture is wrong: update fixture after scholar sign-off; create a PR with the corrected expected text.

---

## 7. Release Evidence Checklist

Before any release that changes Quran retrieval logic:

- [ ] `tests/test_quran_provider.py` passes (all 20 tests)
- [ ] `check_quran_provider_health()` returns `True` in staging
- [ ] Drift verification against 10 sampled ayat returns no drift
- [ ] Cache TTL and fallback behavior verified in integration test
- [ ] This document reviewed and version incremented if contract changed
