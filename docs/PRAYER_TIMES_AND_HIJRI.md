# Prayer times and Hijri calendar (AI-060)

`worship.py` ports the `dev` branch prayer-time and Hijri endpoints (#52) onto
`main` and makes them timezone-safe. All four endpoints are stateless, offline
and need no secrets.

| Endpoint | Purpose |
| --- | --- |
| `GET /prayer-times?lat&lon&date&tz[&method][&asr]` | Fajr, sunrise, Dhuhr, Asr, Maghrib, Isha for a **local civil date** in an IANA zone |
| `GET /hijri?date=YYYY-MM-DD` | Gregorian civil date to Umm al-Qura Hijri date |
| `GET /hijri/today?tz=` | Today's Hijri date resolved in the **caller's** zone, not the server's |
| `GET /gregorian?hijri=YYYY-MM-DD` | Hijri date to Gregorian; accepts Western, Arabic-Indic (٠-٩) and Persian (۰-۹) digits |

## Production behaviour

- **Local-date anchoring.** Times are computed around the solar noon nearest
  to local noon of `date` in `tz`. The dev port used the UTC calendar day, so
  zones whose offset is far from their longitude came back on the wrong
  local day. For example, `Pacific/Kiritimati` (UTC+14, longitude −157°)
  returned Dhuhr on the next local day.
- **Aware output only.** Every time is an ISO-8601 datetime with the zone's
  offset for that instant, so DST is applied per prayer. On a DST change day
  the wall clock moves by an hour while the solar instant moves by under a
  minute. `utc_offset_at_noon` reports the offset in force at local noon.
- **The zone only changes how a time is displayed.** The same place and date
  give the same UTC instants whatever zone you ask for (tested with
  Kashgar on `Asia/Shanghai` and on `Asia/Urumqi`).
- **Times after midnight are not wrapped back.** If a time really falls on
  the next local date (for example Isha in Kashgar on Beijing time in June,
  or the high-latitude fallback in summer), it is returned on that date and
  named in `notes`.
- **Rounding.** Times are rounded to the nearest second. The dev port cut the
  seconds off instead.

### Failure modes

| Condition | Response |
| --- | --- |
| Unknown, empty, path-like, raw-offset (`+03:00`) or longer than 64 characters `tz` | `400 Invalid timezone` |
| `lat`/`lon` out of range, impossible `date`, unknown `method`, missing `tz` | `422` (FastAPI validation) |
| Twilight angle never reached (high latitude, summer) | Fajr/Isha from the one-seventh-of-the-night rule, `notes` says so |
| Sun never rises or never sets (polar day or night) | `sunrise`/`maghrib`/`isha`/`fajr` are `null`, `notes` explains |
| Gregorian date outside 1924-08-01..2077-11-16 | `400 Date out of supported range` |
| Hijri date outside 1343..1500 AH, impossible day (e.g. 30 in a 29-day month), bad format | `400 Invalid Hijri date` |

## Provenance and uncertainty

- `provenance` on `/prayer-times` gives the algorithm (NOAA/Meeus
  low-precision solar position, as formulated by PrayTimes.org), the
  `algorithm_version`, the Fajr/Isha angles or Isha interval, the Asr shadow
  factor, and the timezone database version (`tzdata X`).
- `uncertainty_minutes` is 2. The low-precision solar model and differences
  in local practice (elevation, adjustments to round to whole minutes) mean
  published mosque timetables can differ by about that much. The Makkah test
  checks against the published Umm al-Qura times within this tolerance.
- Hijri responses name the calendar (Umm al-Qura), the `hijridate` version,
  the supported range, and an `uncertainty` note. Local moon-sighting
  authorities may differ by one day, and the Islamic day begins at Maghrib,
  not at civil midnight.

## Configuration

There are no environment variables. Results are reproducible because both
data sources are pinned in `requirements.txt`:

- `hijridate==2.5.0` provides the Umm al-Qura tables (range read from the package).
- `tzdata==2026.6` provides the IANA zone rules, so results don't depend on
  the host OS. Bump it deliberately when governments change DST rules.

## Data handling and user isolation

Nothing is stored. Coordinates and dates are used only to compute the
response and are not logged. The single INFO log line records only the
method, Asr school, zone name and whether the fallback or polar paths were
taken. There is no user-scoped state to isolate or reset.

## Observability

Watch the `worship` logger for `prayer_times computed ... fallback=True` and
`polar=True` rates. A rise in `400 Invalid timezone` usually means a client
is sending abbreviations or raw offsets instead of IANA names.

## Verification

```bash
pytest -q tests/test_worship.py
```

The tests run fully offline. `/hijri/today` gets its clock through the
`worship.utc_now` dependency, which tests override with
`app.dependency_overrides`.

## Rollback

The change adds a module, one `include_router` line in `main.py`, two
pinned requirements and a CI step. To roll back, revert the commit. No
data migration or cache flush is needed because the endpoints keep no
state. Rolling back removes the four routes; clients get `404`.
