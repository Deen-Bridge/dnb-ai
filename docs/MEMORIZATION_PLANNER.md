# Memorization planner: production contract

Owner: memorization planner (`memorization_planner.py`, `memorization_i18n.py`, `memorization_store.py`).
Audit item: #843 (PROD-AUDIT AI-067).

The planner produces Quran memorization schedules (Sabaq / Sabqi / Manzil),
adaptive pace adjustments, milestone celebrations and Muraja'ah (spaced
repetition) sessions. It is deterministic: no model or network call is made.

## 1. Routes

| Route | Auth | State |
| --- | --- | --- |
| `GET /memorization/surahs` | none | read-only reference data |
| `POST /memorization/plan` | none | stateless; stores nothing |
| `POST /memorization/adjust` | none | stateless; `plan_id` is echoed, not looked up |
| `POST /memorization/milestone-check` | none | stateless; measures against a 560-ayah goal (original behaviour) |
| `POST /memorization/revision/session`, `GET .../reminder`, `GET .../stats` | none | stateless; always empty (original behaviour) |
| `POST /memorization/users/{user_id}/plans` | `X-API-Key` | creates and stores a plan |
| `GET /memorization/users/{user_id}/plans` | `X-API-Key` | lists the user's plans, newest first |
| `POST /memorization/users/{user_id}/plans/{plan_id}/adjust` | `X-API-Key` | adjusts a stored plan, stores reported progress |
| `POST /memorization/users/{user_id}/plans/{plan_id}/milestone-check` | `X-API-Key` | measures against the stored plan's size |
| `POST /memorization/users/{user_id}/revision/record` | `X-API-Key` | records one reviewed ayah (SM-2 update) |
| `POST /memorization/users/{user_id}/revision/session` | `X-API-Key` | session from the user's progress |
| `GET /memorization/users/{user_id}/revision/reminder` | `X-API-Key` | reminder from the user's progress |
| `GET /memorization/users/{user_id}/revision/stats` | `X-API-Key` | stats for the user's progress |
| `DELETE /memorization/users/{user_id}` | `X-API-Key` | deletes all of the user's planner data |

The stateless routes are unchanged apart from additive response fields, so
existing clients keep working. New clients should use the user-bound routes.

## 2. Identity and isolation

* This service authenticates its caller (the DeenBridge backend) with the
  shared `X-API-Key`; it has no end-user sessions. The backend authenticates
  the end user and passes an opaque `user_id`, exactly as it does for `/chat`
  and `/memory/{user_id}`. The user-bound routes are mounted behind
  `verify_api_key` so only the backend can read or change a user's data.
* `user_id` must match `^[A-Za-z0-9._:@-]{1,128}$`.
* Each user's data is one record under one storage key
  (`memorization:user:{user_id}`). A `plan_id` is only looked up inside that
  record, so another user's plan returns the same `404 memorization_plan_not_found`
  as a plan that never existed.
* `plan_id` is `plan_` plus 24 random hex characters. It no longer embeds the
  `user_id` (it previously did, which leaked the identifier).
* Responses never include profile data; they return only the requesting
  user's planner data.

## 3. Locale

* `locale` is an optional BCP-47 tag on plan, adjust and milestone requests
  and on the user reminder route (max 64 characters, otherwise 422).
* `ar` and `en` are supported. `_` is accepted as a separator and the language
  subtag is case-insensitive, so `ar-EG`, `AR_sa` and `ar` resolve to `ar`.
* Missing, malformed or unsupported tags fall back to `en`. Every localized
  response carries `locale = {requested, resolved, direction, fallback, fallback_reason}`;
  `direction` is `rtl` for Arabic.
* Localized: tips, summaries, milestone titles and messages, adjustment
  actions, steps and encouragement, celebration text and badges, the
  Muraja'ah reminder and the `limitations` list.
* Never localized: surah numbers, ayah references, `surah_name`
  (transliteration), hadith citations (`citation`, `authentic_citation`),
  `retention_health` and `status`. Arabic surah names are returned in the
  separate `surah_arabic_name` field and used in Arabic text.
* The Arabic catalog contains no Quran or hadith wording. Where an English
  string quotes a hadith, the Arabic string refers to the collection and
  number instead. English strings are unchanged from before localization.
* Digits are Western Arabic (0-9) in both locales; clients may reformat.

## 4. Grounding, provenance and uncertainty

* Surah facts (Arabic name, ayah count, revelation place) are checked against
  the versioned table `data/quran/surah_index.json` (built and validated by
  `scripts/build_surah_index.py`, see `data/quran/PROVENANCE.md`). A test
  fails if the embedded planner table and the index disagree.
* Plan, adjustment and milestone responses include `provenance`:
  `planner_rules_version`, `message_catalog_version`, the surah index path and
  its SHA-256, and the canonical citations quoted in that response.
* They also include `limitations`: the plan is scheduling guidance from fixed
  rules, not a religious ruling; dates and paces are estimates; difficulty
  tiers and page counts are editorial estimates.
* If the surah index cannot be read, `provenance.surah_reference.sha256` is
  null and an error is logged at import; plan generation still works because
  the embedded table is used for scheduling.

## 5. Data handling and retention

Stored per user (and nothing else):

* up to 10 plans: id, timestamps, rules version, locale, summary text, target
  surah numbers, totals, pace, days per week, start and estimated completion
  dates, reported completed ayahs. Creating an 11th drops the oldest;
* Muraja'ah progress: one entry per reviewed ayah (at most 6,236), holding SM-2
  state, counts and review dates.

Backends:

| Backend | When | Retention |
| --- | --- | --- |
| Redis | `REDIS_URL` set | `MEMORIZATION_TTL_DAYS` (default 365) after the user's last write |
| In-memory | `REDIS_URL` unset | until restart; at most `MEMORIZATION_MAX_IN_MEMORY_USERS` (default 10,000) users, least recently used evicted |

The Render blueprint does not set `REDIS_URL`, so production currently uses the
in-memory backend and planner data is lost on restart or redeploy. Set
`REDIS_URL` for durable plans.

Reset: `DELETE /memorization/users/{user_id}` removes the user's record
immediately and returns how many plans and progress entries were deleted. It is
idempotent and works even when the stored record is unreadable. It does not
touch chat history or memory profiles. Redis persistence snapshots (RDB/AOF)
and backups follow the operator's Redis backup policy; this service keeps no
other copy.

## 6. Failure modes

| Condition | Response |
| --- | --- |
| Missing or wrong `X-API-Key` on a user route | `401` |
| Malformed `user_id`, locale over 64 chars, ayah beyond the surah, more than 60 sessions | `422` |
| Goal with nothing left to memorize, invalid range, bad `start_date` | `400` with hint |
| `plan_id` not in this user's record | `404` `memorization_plan_not_found` |
| Stored record exists but cannot be decoded or validated | `500` `memorization_record_unreadable`; reset still works |
| Redis unreachable or failing | `503` `memorization_storage_unavailable`; nothing is changed |

Error bodies never contain the `user_id` or stored data.

## 7. Observability

* Plan creation and reset are logged at INFO with `user_ref` (first 12 hex
  characters of the SHA-256 of the `user_id`), never the raw identifier.
* Store failures are logged at WARNING with `user_ref` and the exception.
* Note: the user-bound routes carry `user_id` in the URL path, so HTTP access
  logs contain it, as they already do for `/memory/{user_id}` and
  `/user/{user_id}/chats`.
* Useful alerts: rate of `503 memorization_storage_unavailable` and any
  `500 memorization_record_unreadable` on `/memorization/users/*`.

## 8. Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `SERVICE_API_KEY` | none | required for user-bound routes (already required in production) |
| `REDIS_URL` | unset | enables durable Redis storage |
| `MEMORIZATION_TTL_DAYS` | `365` | Redis TTL for a user's record |
| `MEMORIZATION_MAX_IN_MEMORY_USERS` | `10000` | in-memory backend cap |

## 9. Rollout and rollback

Rollout is additive: stateless routes keep their paths and existing fields,
and English text is identical. New request limits (max 64-character locale,
128-character `user_id` with the pattern above, at most 60 recent sessions,
at most 6,236 memorized ayahs) can turn previously accepted extreme inputs into
`422`; watch the `422` rate on `/memorization/*` after deploy.

Rollback: redeploy the previous image. Stateless routes behave as before. The
user-bound routes disappear, so their clients get `404` until the next roll
forward. Data in Redis is left untouched under `memorization:user:*` and is
read again on roll forward (records carry `schema_version: 1`, and unknown
fields are ignored). To discard it instead, delete those keys or let the TTL
expire them. With the in-memory backend there is nothing to migrate.
