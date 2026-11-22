"""Offline tests for timezone-safe prayer times and Hijri conversion (AI-060).

No secrets and no network: solar positions are computed locally, Hijri dates
come from the pinned ``hijridate`` Umm al-Qura tables, and zones come from the
pinned ``tzdata`` package. The clock used by ``/hijri/today`` is injected.
"""

import os
from datetime import UTC, date, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest
from httpx import ASGITransport, AsyncClient

os.environ.setdefault("GEMINI_API_KEY", "test-key")

import main  # noqa: E402
import worship  # noqa: E402
from worship import (  # noqa: E402
    AsrMethod,
    CalculationMethod,
    calculate_prayer_times,
    parse_hijri,
    resolve_timezone,
)

PRAYERS = ("fajr", "sunrise", "dhuhr", "asr", "maghrib", "isha")


@pytest.fixture()
async def client():
    transport = ASGITransport(app=main.app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
    main.app.dependency_overrides.pop(worship.utc_now, None)


def _minutes(a: str, b: str) -> float:
    return (datetime.fromisoformat(b) - datetime.fromisoformat(a)).total_seconds() / 60.0


# ---------------------------------------------------------------------------
# Timezone safety (the core of AI-060)
# ---------------------------------------------------------------------------

# (label, lat, lon, tz, date) chosen because the zone offset is far from the
# longitude, uses a non-hour offset, or the date is a DST transition day.
TRICKY_ZONES = [
    ("kiritimati_utc+14", 1.87, -157.36, "Pacific/Kiritimati", date(2026, 3, 1)),
    ("pago_pago_utc-11", -14.27, -170.70, "Pacific/Pago_Pago", date(2026, 3, 1)),
    ("kashgar_on_beijing_time", 39.47, 75.99, "Asia/Shanghai", date(2026, 12, 1)),
    ("chatham_utc+12:45", -43.95, -176.56, "Pacific/Chatham", date(2026, 1, 10)),
    ("kathmandu_utc+5:45", 27.71, 85.32, "Asia/Kathmandu", date(2026, 5, 1)),
    ("tehran_utc+3:30", 35.69, 51.39, "Asia/Tehran", date(2026, 9, 1)),
    ("new_york_dst_start", 40.71, -74.0, "America/New_York", date(2026, 3, 8)),
    ("new_york_dst_end", 40.71, -74.0, "America/New_York", date(2026, 11, 1)),
    ("london_dst_end", 51.5072, -0.1276, "Europe/London", date(2026, 10, 25)),
    ("auckland_dst_end", -36.85, 174.76, "Pacific/Auckland", date(2026, 4, 5)),
    ("sydney_dst_start", -33.87, 151.21, "Australia/Sydney", date(2026, 10, 4)),
    ("utc_equator", 0.0, 0.0, "UTC", date(2026, 3, 20)),
]


@pytest.mark.parametrize(("label", "lat", "lon", "tz", "day"), TRICKY_ZONES, ids=[z[0] for z in TRICKY_ZONES])
def test_all_times_fall_on_requested_local_date(label, lat, lon, tz, day):
    zone = ZoneInfo(tz)
    result = calculate_prayer_times(lat, lon, day, CalculationMethod.MWL, AsrMethod.STANDARD, zone)
    assert not result.polar
    for name in PRAYERS:
        moment = result.times[name]
        assert moment is not None, name
        assert moment.tzinfo is not None, f"{name} must be timezone-aware"
        assert moment.date() == day, f"{label}: {name}={moment.isoformat()} is not on {day}"
        assert moment.microsecond == 0
    ordered = [result.times[n] for n in PRAYERS]
    assert ordered == sorted(ordered), f"{label}: prayers out of order"


def test_kiritimati_regression_dev_port_returned_next_local_day():
    # The dev port anchored on the UTC calendar day, so at UTC+14 with a
    # longitude of -157 degrees Dhuhr came back on 2 March local time.
    result = calculate_prayer_times(
        1.87, -157.36, date(2026, 3, 1), CalculationMethod.MWL, AsrMethod.STANDARD, ZoneInfo("Pacific/Kiritimati")
    )
    dhuhr = result.times["dhuhr"]
    assert dhuhr is not None
    assert dhuhr.isoformat().startswith("2026-03-01T12:")


async def test_isha_after_local_midnight_is_flagged_not_wrapped(client):
    # Kashgar on Beijing time in June: solar time runs ~3h behind the clock, so
    # Isha genuinely falls after local midnight. It must be reported on the next
    # date and flagged, never silently wrapped back onto the requested date.
    res = await client.get(
        "/prayer-times",
        params={"lat": 39.47, "lon": 75.99, "date": "2026-06-01", "method": "MWL", "tz": "Asia/Shanghai"},
    )
    data = res.json()
    assert data["isha"].startswith("2026-06-02T00:")
    assert data["dhuhr"].startswith("2026-06-01T")
    assert "adjacent local date: isha" in data["notes"]


def test_timezone_only_changes_presentation_not_the_instant():
    # Kashgar observed on Beijing time vs Urumqi time: same place, same civil
    # date, so the UTC instants must be identical.
    beijing = calculate_prayer_times(
        39.47, 75.99, date(2026, 6, 1), CalculationMethod.MWL, AsrMethod.STANDARD, ZoneInfo("Asia/Shanghai")
    )
    urumqi = calculate_prayer_times(
        39.47, 75.99, date(2026, 6, 1), CalculationMethod.MWL, AsrMethod.STANDARD, ZoneInfo("Asia/Urumqi")
    )
    for name in PRAYERS:
        assert beijing.times[name] == urumqi.times[name], name
        assert beijing.times[name].utcoffset() != urumqi.times[name].utcoffset()  # type: ignore[union-attr]


def test_dst_transition_shifts_wall_clock_not_solar_instant():
    zone = ZoneInfo("America/New_York")
    before = calculate_prayer_times(40.71, -74.0, date(2026, 3, 7), CalculationMethod.ISNA, AsrMethod.STANDARD, zone)
    after = calculate_prayer_times(40.71, -74.0, date(2026, 3, 8), CalculationMethod.ISNA, AsrMethod.STANDARD, zone)
    b, a = before.times["dhuhr"], after.times["dhuhr"]
    assert b is not None and a is not None
    assert b.utcoffset() == timedelta(hours=-5)
    assert a.utcoffset() == timedelta(hours=-4)
    assert a.hour - b.hour == 1  # wall clock jumps an hour
    # ...but solar noon moves by well under a minute from one day to the next.
    # (compare in UTC: same-tzinfo subtraction in Python is wall-clock arithmetic)
    assert abs((a.astimezone(UTC) - b.astimezone(UTC)) - timedelta(days=1)) < timedelta(minutes=1)


# ---------------------------------------------------------------------------
# Calculation correctness
# ---------------------------------------------------------------------------


async def test_makkah_matches_published_umm_al_qura_times(client):
    # Umm al-Qura calendar, Makkah, 15 June 2023 (Asia/Riyadh, UTC+3).
    published = {"fajr": "04:11", "sunrise": "05:39", "dhuhr": "12:22", "maghrib": "19:05", "isha": "20:35"}
    res = await client.get(
        "/prayer-times",
        params={"lat": 21.4225, "lon": 39.8262, "date": "2023-06-15", "method": "MAKKAH", "tz": "Asia/Riyadh"},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["tz"] == "Asia/Riyadh"
    assert data["utc_offset_at_noon"] == "+03:00"
    for name, hhmm in published.items():
        expected = datetime.fromisoformat(f"2023-06-15T{hhmm}:00+03:00")
        actual = datetime.fromisoformat(data[name])
        assert abs((actual - expected).total_seconds()) <= data["uncertainty_minutes"] * 60, name
    assert _minutes(data["maghrib"], data["isha"]) == pytest.approx(90.0, abs=0.02)


async def test_london_isna_winter(client):
    res = await client.get(
        "/prayer-times",
        params={"lat": 51.5072, "lon": -0.1276, "date": "2023-01-15", "method": "ISNA", "tz": "Europe/London"},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["fajr"].endswith("+00:00")
    assert data["notes"] is None


def test_hanafi_asr_is_later_than_standard():
    zone = ZoneInfo("Asia/Karachi")
    std = calculate_prayer_times(24.86, 67.0, date(2026, 5, 1), CalculationMethod.KARACHI, AsrMethod.STANDARD, zone)
    han = calculate_prayer_times(24.86, 67.0, date(2026, 5, 1), CalculationMethod.KARACHI, AsrMethod.HANAFI, zone)
    assert std.times["asr"] is not None and han.times["asr"] is not None
    assert han.times["asr"] > std.times["asr"]


async def test_high_latitude_summer_uses_fallback(client):
    res = await client.get(
        "/prayer-times",
        params={"lat": 64.1466, "lon": -21.9426, "date": "2023-06-21", "method": "ISNA", "tz": "Atlantic/Reykjavik"},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["fajr"] is not None and data["isha"] is not None
    assert "High-latitude fallback" in data["notes"]
    # Isha after local midnight is reported honestly rather than wrapped onto the wrong day.
    assert "adjacent local date" in data["notes"]
    assert data["isha"].startswith("2023-06-22")


@pytest.mark.parametrize("day", ["2026-06-21", "2026-12-21"], ids=["midnight_sun", "polar_night"])
async def test_polar_day_and_night_return_nulls_with_explanation(client, day):
    res = await client.get(
        "/prayer-times",
        params={"lat": 78.22, "lon": 15.65, "date": day, "method": "MWL", "tz": "Arctic/Longyearbyen"},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["sunrise"] is None and data["maghrib"] is None
    assert data["dhuhr"].startswith(day)
    assert "does not rise or set" in data["notes"]


async def test_response_carries_provenance_and_uncertainty(client):
    res = await client.get(
        "/prayer-times",
        params={"lat": 30.04, "lon": 31.24, "date": "2026-02-01", "method": "EGYPT", "tz": "Africa/Cairo"},
    )
    data = res.json()
    prov = data["provenance"]
    assert prov["fajr_angle"] == 19.5 and prov["isha_angle"] == 17.5
    assert prov["algorithm_version"] == worship.PRAYER_TIMES_ALGORITHM_VERSION
    assert prov["timezone_database"].startswith(("tzdata ", "system zoneinfo"))
    assert data["uncertainty_minutes"] == worship.PRAYER_TIME_UNCERTAINTY_MINUTES


async def test_endpoints_are_stateless_and_deterministic(client):
    params = {"lat": 3.14, "lon": 101.69, "date": "2026-07-01", "method": "MWL", "tz": "Asia/Kuala_Lumpur"}
    first = (await client.get("/prayer-times", params=params)).json()
    await client.get("/prayer-times", params={**params, "tz": "UTC"})
    second = (await client.get("/prayer-times", params=params)).json()
    assert first == second


# ---------------------------------------------------------------------------
# Failure states
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tz", ["Fake/Timezone", "", "   ", "../../etc/passwd", "/etc/localtime", "+03:00", "x" * 80])
async def test_invalid_timezone_is_rejected(client, tz):
    res = await client.get("/prayer-times", params={"lat": 0, "lon": 0, "date": "2023-01-01", "tz": tz})
    assert res.status_code == 400
    assert "Invalid timezone" in res.json()["detail"]


def test_resolve_timezone_strips_whitespace():
    assert resolve_timezone(" Asia/Riyadh ").key == "Asia/Riyadh"


@pytest.mark.parametrize(
    "params",
    [
        {"lat": 100, "lon": 0, "date": "2023-01-01", "tz": "UTC"},
        {"lat": 0, "lon": 181, "date": "2023-01-01", "tz": "UTC"},
        {"lat": 0, "lon": 0, "date": "2023-02-30", "tz": "UTC"},
        {"lat": 0, "lon": 0, "date": "2023-01-01", "tz": "UTC", "method": "NOPE"},
        {"lat": 0, "lon": 0, "date": "2023-01-01"},
    ],
    ids=["lat", "lon", "date", "method", "missing_tz"],
)
async def test_invalid_query_parameters_are_422(client, params):
    res = await client.get("/prayer-times", params=params)
    assert res.status_code == 422


# ---------------------------------------------------------------------------
# Hijri / Gregorian conversion
# ---------------------------------------------------------------------------


async def test_hijri_conversion(client):
    res = await client.get("/hijri", params={"date": "2023-03-23"})
    assert res.status_code == 200
    data = res.json()
    assert (data["year"], data["month"], data["day"]) == (1444, 9, 1)
    assert data["month_en"] == "Ramadan"
    assert data["month_ar"] == "رمضان"
    assert data["provenance"]["calendar"] == "Umm al-Qura"
    assert data["provenance"]["library"].startswith("hijridate ")
    assert "Maghrib" in data["uncertainty"]


@pytest.mark.parametrize("day", ["1000-01-01", "1924-07-31", "2077-11-17"])
async def test_hijri_out_of_range(client, day):
    res = await client.get("/hijri", params={"date": day})
    assert res.status_code == 400
    assert "Date out of supported range" in res.json()["detail"]


async def test_hijri_range_boundaries_are_inclusive(client):
    assert (await client.get("/hijri", params={"date": "1924-08-01"})).status_code == 200
    assert (await client.get("/hijri", params={"date": "2077-11-16"})).status_code == 200


@pytest.mark.parametrize(
    "hijri",
    ["1444-09-01", "١٤٤٤-٠٩-٠١", "۱۴۴۴-۰۹-۰۱", " 1444-9-1 "],
    ids=["western", "arabic_indic", "persian", "unpadded"],
)
async def test_gregorian_conversion_accepts_arabic_and_english_digits(client, hijri):
    res = await client.get("/gregorian", params={"hijri": hijri})
    assert res.status_code == 200
    data = res.json()
    assert (data["year"], data["month"], data["day"]) == (2023, 3, 23)
    assert data["hijri_date"] == "1444-09-01"


@pytest.mark.parametrize(
    "hijri",
    ["1444-09-30", "1444-13-01", "1342-12-29", "1501-01-01", "1444/09/01", "abc", "", "1444-09-01-02", "1444-٠٩"],
)
async def test_gregorian_conversion_rejects_bad_input(client, hijri):
    res = await client.get("/gregorian", params={"hijri": hijri})
    assert res.status_code == 400
    assert res.json()["detail"].startswith("Invalid Hijri date")


def test_parse_hijri_rejects_non_ascii_lookalike_digits():
    with pytest.raises(ValueError):
        parse_hijri("１４４４-０９-０１")  # fullwidth digits are not a supported input form


def test_round_trip_is_stable_across_supported_range():
    day = date(1925, 1, 1)
    while day < date(2077, 1, 1):
        h = worship.gregorian_to_hijri(day)
        g = h.to_gregorian()
        assert date(g.year, g.month, g.day) == day
        day += timedelta(days=97)


# ---------------------------------------------------------------------------
# /hijri/today resolves "today" in the caller's zone
# ---------------------------------------------------------------------------


async def test_hijri_today_depends_on_caller_timezone(client):
    # 22:30 UTC on 19 March 2026: already 20 March in Riyadh, still 19 March in Los Angeles.
    main.app.dependency_overrides[worship.utc_now] = lambda: datetime(2026, 3, 19, 22, 30, tzinfo=UTC)
    riyadh = (await client.get("/hijri/today", params={"tz": "Asia/Riyadh"})).json()
    la = (await client.get("/hijri/today", params={"tz": "America/Los_Angeles"})).json()
    assert riyadh["gregorian_date"] == "2026-03-20"
    assert la["gregorian_date"] == "2026-03-19"
    assert (riyadh["year"], riyadh["month"], riyadh["day"]) != (la["year"], la["month"], la["day"])
    assert riyadh["local_datetime"] == "2026-03-20T01:30:00+03:00"
    assert riyadh["provenance"]["timezone_database"]


async def test_hijri_today_treats_naive_clock_as_utc(client):
    main.app.dependency_overrides[worship.utc_now] = lambda: datetime(2026, 3, 19, 22, 30)
    data = (await client.get("/hijri/today", params={"tz": "Asia/Tokyo"})).json()
    assert data["local_datetime"] == "2026-03-20T07:30:00+09:00"


async def test_hijri_today_invalid_timezone(client):
    res = await client.get("/hijri/today", params={"tz": "Mars/Olympus_Mons"})
    assert res.status_code == 400
    assert "Invalid timezone" in res.json()["detail"]


async def test_hijri_today_outside_table_range(client):
    main.app.dependency_overrides[worship.utc_now] = lambda: datetime(2090, 1, 1, tzinfo=UTC)
    res = await client.get("/hijri/today", params={"tz": "UTC"})
    assert res.status_code == 400
    assert "Date out of supported range" in res.json()["detail"]


async def test_routes_are_registered_on_main_app(client):
    schema = (await client.get("/openapi.json")).json()
    for path in ("/prayer-times", "/hijri", "/hijri/today", "/gregorian"):
        assert path in schema["paths"], path
