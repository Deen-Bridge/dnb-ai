"""Timezone-safe prayer times and Hijri calendar conversion (AI-060).

Ported from the ``dev`` branch (#52) and reconciled for production:

* Every prayer time is an aware ISO-8601 datetime in the caller's IANA zone.
  Times are anchored on the solar noon nearest to *local* noon of the
  requested civil date, so zones whose offset is far from their longitude
  (``Pacific/Kiritimati`` at UTC+14, ``Asia/Shanghai``, ...) and DST
  transition days never return times for the wrong local day.
* Times are rounded to the nearest second (the dev port truncated them) and a
  computed hour angle of ``0.0`` is no longer mistaken for "not calculable".
* Hijri conversion uses the Umm al-Qura tables shipped with the pinned
  ``hijridate`` package. ``/hijri/today`` resolves "today" in the caller's
  zone instead of the server's, because the same instant can be a different
  civil (and therefore Hijri) date in Makkah and in Los Angeles.
* Responses carry provenance (calculation angles, data package versions) and
  an explicit uncertainty note.

The endpoints are stateless: nothing about the caller is stored, so there is
no user-scoped data to isolate or reset. No secrets or network access are
needed; see ``docs/PRAYER_TIMES_AND_HIJRI.md``.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from enum import Enum
from importlib import metadata
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from fastapi import APIRouter, Depends, HTTPException, Query
from hijridate import Gregorian, Hijri
from hijridate.ummalqura import GREGORIAN_RANGE, HIJRI_RANGE
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(tags=["worship"])

PRAYER_TIMES_ALGORITHM_VERSION = "2026.11-ai060"
PRAYER_TIME_UNCERTAINTY_MINUTES = 2
HIJRI_UNCERTAINTY_NOTE = (
    "Umm al-Qura is a pre-computed civil calendar. Local moon-sighting authorities may start a month one day "
    "earlier or later, and the Islamic day begins at Maghrib, not at civil midnight."
)

# Umm al-Qura range covered by the pinned hijridate tables (read from the package, not hard-coded).
HIJRI_MIN, HIJRI_MAX = HIJRI_RANGE
GREGORIAN_MIN, GREGORIAN_MAX = (date(*GREGORIAN_RANGE[0]), date(*GREGORIAN_RANGE[1]))

_ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
_HIJRI_DATE_RE = re.compile(r"^\s*(\d{1,4})-(\d{1,2})-(\d{1,2})\s*$", re.ASCII)


class CalculationMethod(str, Enum):
    MWL = "MWL"  # Muslim World League (Fajr 18, Isha 17)
    ISNA = "ISNA"  # Islamic Society of North America (Fajr 15, Isha 15)
    EGYPT = "EGYPT"  # Egyptian General Authority (Fajr 19.5, Isha 17.5)
    MAKKAH = "MAKKAH"  # Umm al-Qura University, Makkah (Fajr 18.5, Isha 90 min)
    KARACHI = "KARACHI"  # University of Islamic Sciences, Karachi (Fajr 18, Isha 18)


class AsrMethod(str, Enum):
    STANDARD = "STANDARD"  # Shafi'i, Maliki, Hanbali (shadow factor 1)
    HANAFI = "HANAFI"  # shadow factor 2


@dataclass(frozen=True)
class MethodParams:
    fajr_angle: float
    isha_angle: float | None
    isha_interval_minutes: float | None


METHOD_PARAMS: dict[CalculationMethod, MethodParams] = {
    CalculationMethod.MWL: MethodParams(18.0, 17.0, None),
    CalculationMethod.ISNA: MethodParams(15.0, 15.0, None),
    CalculationMethod.EGYPT: MethodParams(19.5, 17.5, None),
    CalculationMethod.MAKKAH: MethodParams(18.5, None, 90.0),
    CalculationMethod.KARACHI: MethodParams(18.0, 18.0, None),
}

# Sun's apparent radius plus standard atmospheric refraction, in degrees.
SUNRISE_ANGLE = 0.833
PRAYER_NAMES = ("fajr", "sunrise", "dhuhr", "asr", "maghrib", "isha")


# --- Math utilities ---------------------------------------------------------


def _dsin(d: float) -> float:
    return math.sin(math.radians(d))


def _dcos(d: float) -> float:
    return math.cos(math.radians(d))


def _dtan(d: float) -> float:
    return math.tan(math.radians(d))


def _fix(a: float, b: float) -> float:
    return a - b * math.floor(a / b)


def julian_day(moment: datetime) -> float:
    """Julian Day of an aware datetime (Meeus, Astronomical Algorithms ch. 7)."""
    utc = moment.astimezone(UTC)
    y, m = utc.year, utc.month
    d = utc.day + (utc.hour + (utc.minute + utc.second / 60.0) / 60.0) / 24.0
    if m <= 2:
        y -= 1
        m += 12
    a = math.floor(y / 100)
    b = 2 - a + math.floor(a / 4)
    return math.floor(365.25 * (y + 4716)) + math.floor(30.6001 * (m + 1)) + d + b - 1524.5


def sun_position(jd: float) -> tuple[float, float]:
    """Return (declination in degrees, equation of time in hours)."""
    d = jd - 2451545.0
    g = _fix(357.529 + 0.98560028 * d, 360.0)
    q = _fix(280.459 + 0.98564736 * d, 360.0)
    ecl_lon = _fix(q + 1.915 * _dsin(g) + 0.020 * _dsin(2 * g), 360.0)
    obliquity = 23.439 - 0.00000036 * d
    declination = math.degrees(math.asin(_dsin(obliquity) * _dsin(ecl_lon)))
    ra = _fix(math.degrees(math.atan2(_dcos(obliquity) * _dsin(ecl_lon), _dcos(ecl_lon))) / 15.0, 24.0)
    eq_t = q / 15.0 - ra
    # Keep the equation of time in its natural (-0.5h, 0.5h) band across the 0/360 wrap.
    eq_t = (eq_t + 12.0) % 24.0 - 12.0
    return declination, eq_t


def _hour_angle(angle: float, lat: float, declination: float) -> float | None:
    """Hours between solar noon and the sun reaching ``angle`` degrees below the horizon."""
    denom = _dcos(declination) * _dcos(lat)
    if denom == 0:
        return None
    val = (-_dsin(angle) - _dsin(declination) * _dsin(lat)) / denom
    if val < -1.0 or val > 1.0:
        return None  # the sun never reaches this angle on this day (high latitude)
    return math.degrees(math.acos(val)) / 15.0


def _asr_hour_angle(lat: float, declination: float, shadow_factor: float) -> float | None:
    denom = _dcos(declination) * _dcos(lat)
    if denom == 0:
        return None
    altitude = math.degrees(math.atan(1.0 / (shadow_factor + _dtan(abs(lat - declination)))))
    val = (_dsin(altitude) - _dsin(declination) * _dsin(lat)) / denom
    if val < -1.0 or val > 1.0:
        return None
    return math.degrees(math.acos(val)) / 15.0


# --- Timezone handling ------------------------------------------------------


def resolve_timezone(tz: str) -> ZoneInfo:
    """Return the IANA zone named ``tz`` or raise ``ValueError``.

    Raw offsets such as ``+03:00`` are rejected because they cannot follow
    DST; path-like input is rejected before it reaches the zoneinfo loader.
    """
    name = (tz or "").strip()
    if not name or len(name) > 64 or ".." in name or name.startswith("/") or "\\" in name:
        raise ValueError(f"Invalid timezone: {tz!r}. Use an IANA name such as 'Asia/Riyadh'.")
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError, OSError):
        raise ValueError(f"Invalid timezone: {tz!r}. Use an IANA name such as 'Asia/Riyadh'.") from None


def _local_noon(day: date, zone: ZoneInfo) -> datetime:
    return datetime.combine(day, time(12, 0), tzinfo=zone)


def solar_noon_near(day: date, lon: float, zone: ZoneInfo) -> tuple[datetime, float]:
    """Solar noon (UTC) closest to local civil noon of ``day`` in ``zone``, and the declination there.

    Anchoring on local noon (not on the UTC calendar day) is what keeps every
    returned time on the requested local date, regardless of how far the
    zone's offset is from the longitude or whether DST is in force.
    """
    target = _local_noon(day, zone).astimezone(UTC)
    utc_midnight = datetime.combine(target.date(), time(0), tzinfo=UTC)
    noon = utc_midnight
    declination = 0.0
    for _ in range(2):  # one refinement pass is plenty (sub-second change)
        declination, eq_t = sun_position(julian_day(noon))
        hours = 12.0 - eq_t - lon / 15.0
        noon = utc_midnight + timedelta(hours=hours)
        # Pick the solar noon within 12h of local noon (handles offsets far from longitude).
        while noon - target > timedelta(hours=12):
            noon -= timedelta(days=1)
            utc_midnight -= timedelta(days=1)
        while target - noon > timedelta(hours=12):
            noon += timedelta(days=1)
            utc_midnight += timedelta(days=1)
    return noon, declination


def _round_second(moment: datetime) -> datetime:
    if moment.microsecond >= 500_000:
        moment += timedelta(seconds=1)
    return moment.replace(microsecond=0)


@dataclass(frozen=True)
class PrayerTimesResult:
    times: dict[str, datetime | None]
    fallback_applied: bool
    polar: bool


def calculate_prayer_times(
    lat: float,
    lon: float,
    day: date,
    method: CalculationMethod,
    asr_method: AsrMethod,
    zone: ZoneInfo,
) -> PrayerTimesResult:
    """Compute prayer times for the local civil ``day`` at (lat, lon) in ``zone``."""
    params = METHOD_PARAMS[method]
    noon, decl = solar_noon_near(day, lon, zone)
    shadow = 1.0 if asr_method == AsrMethod.STANDARD else 2.0

    offsets: dict[str, float | None] = {
        "fajr": _neg(_hour_angle(params.fajr_angle, lat, decl)),
        "sunrise": _neg(_hour_angle(SUNRISE_ANGLE, lat, decl)),
        "dhuhr": 0.0,
        "asr": _asr_hour_angle(lat, decl, shadow),
        "maghrib": _hour_angle(SUNRISE_ANGLE, lat, decl),
        "isha": _hour_angle(params.isha_angle, lat, decl) if params.isha_angle is not None else None,
    }
    if params.isha_interval_minutes is not None and offsets["maghrib"] is not None:
        offsets["isha"] = offsets["maghrib"] + params.isha_interval_minutes / 60.0

    fallback = False
    polar = offsets["sunrise"] is None or offsets["maghrib"] is None
    if not polar:
        sunrise, maghrib = offsets["sunrise"], offsets["maghrib"]
        assert sunrise is not None and maghrib is not None
        night = 24.0 - (maghrib - sunrise)
        # One-seventh-of-the-night rule when twilight angles are never reached.
        if offsets["fajr"] is None:
            offsets["fajr"] = sunrise - night / 7.0
            fallback = True
        if offsets["isha"] is None:
            offsets["isha"] = maghrib + night / 7.0
            fallback = True

    times: dict[str, datetime | None] = {}
    for name in PRAYER_NAMES:
        off = offsets[name]
        times[name] = None if off is None else _round_second(noon + timedelta(hours=off)).astimezone(zone)
    return PrayerTimesResult(times=times, fallback_applied=fallback, polar=polar)


def _neg(value: float | None) -> float | None:
    return None if value is None else -value


# --- Provenance ----------------------------------------------------------------


def _package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "unavailable"


def _tzdata_source() -> str:
    version = _package_version("tzdata")
    return f"tzdata {version}" if version != "unavailable" else "system zoneinfo"


class PrayerProvenance(BaseModel):
    algorithm: str
    algorithm_version: str
    fajr_angle: float
    isha_angle: float | None
    isha_interval_minutes: float | None
    asr_shadow_factor: float
    timezone_database: str


class PrayerTimesResponse(BaseModel):
    fajr: str | None
    sunrise: str | None
    dhuhr: str
    asr: str | None
    maghrib: str | None
    isha: str | None
    method: CalculationMethod
    asr_method: AsrMethod
    lat: float
    lon: float
    date: date
    tz: str
    utc_offset_at_noon: str
    uncertainty_minutes: int
    provenance: PrayerProvenance
    notes: str | None = None


class HijriProvenance(BaseModel):
    calendar: str
    library: str
    supported_gregorian_range: str
    timezone_database: str | None = None


class HijriResponse(BaseModel):
    year: int
    month: int
    month_en: str
    month_ar: str
    day: int
    gregorian_date: date
    uncertainty: str
    provenance: HijriProvenance


class HijriTodayResponse(HijriResponse):
    tz: str
    local_datetime: str


class GregorianResponse(BaseModel):
    year: int
    month: int
    day: int
    hijri_date: str
    uncertainty: str
    provenance: HijriProvenance


def _hijri_provenance(zone_used: bool = False) -> HijriProvenance:
    return HijriProvenance(
        calendar="Umm al-Qura",
        library=f"hijridate {_package_version('hijridate')}",
        supported_gregorian_range=f"{GREGORIAN_MIN.isoformat()}..{GREGORIAN_MAX.isoformat()}",
        timezone_database=_tzdata_source() if zone_used else None,
    )


def _format_offset(moment: datetime) -> str:
    offset = moment.utcoffset() or timedelta(0)
    sign = "-" if offset < timedelta(0) else "+"
    total = abs(int(offset.total_seconds()))
    return f"{sign}{total // 3600:02d}:{(total % 3600) // 60:02d}"


# --- Hijri helpers -------------------------------------------------------------


def gregorian_to_hijri(day: date) -> Hijri:
    if not GREGORIAN_MIN <= day <= GREGORIAN_MAX:
        raise OverflowError(
            f"Date out of supported range: {day.isoformat()} is outside "
            f"{GREGORIAN_MIN.isoformat()}..{GREGORIAN_MAX.isoformat()} (Umm al-Qura tables)."
        )
    try:
        return Gregorian(day.year, day.month, day.day).to_hijri()
    except OverflowError as exc:
        raise OverflowError(f"Date out of supported range: {exc}") from exc


def parse_hijri(text: str) -> tuple[int, int, int]:
    """Parse ``YYYY-MM-DD`` with Western, Arabic-Indic or Persian digits."""
    normalized = (text or "").translate(_ARABIC_DIGITS)
    match = _HIJRI_DATE_RE.match(normalized)
    if not match:
        raise ValueError("Invalid format. Use YYYY-MM-DD (Western or Arabic-Indic digits).")
    year, month, day = (int(part) for part in match.groups())
    return year, month, day


def _hijri_response_fields(h: Hijri, day: date) -> dict[str, object]:
    return {
        "year": h.year,
        "month": h.month,
        "month_en": h.month_name(),
        "month_ar": h.month_name(language="ar"),
        "day": h.day,
        "gregorian_date": day,
        "uncertainty": HIJRI_UNCERTAINTY_NOTE,
    }


def utc_now() -> datetime:
    """Clock dependency; tests override it with ``app.dependency_overrides``."""
    return datetime.now(UTC)


# --- Endpoints -----------------------------------------------------------------


@router.get("/prayer-times", response_model=PrayerTimesResponse)
async def get_prayer_times(
    lat: float = Query(..., ge=-90, le=90, description="Latitude in decimal degrees"),
    lon: float = Query(..., ge=-180, le=180, description="Longitude in decimal degrees"),
    date: date = Query(..., description="Local civil date (YYYY-MM-DD) in `tz`"),
    method: CalculationMethod = Query(CalculationMethod.MAKKAH, description="Calculation method"),
    asr: AsrMethod = Query(AsrMethod.STANDARD, description="Asr juristic method"),
    tz: str = Query(..., description="IANA timezone name, e.g. Asia/Riyadh"),
) -> PrayerTimesResponse:
    """Prayer times for a location on a local civil date, as aware datetimes in ``tz``."""
    try:
        zone = resolve_timezone(tz)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None

    try:
        result = calculate_prayer_times(lat, lon, date, method, asr, zone)
    except (OverflowError, ValueError):
        raise HTTPException(status_code=400, detail=f"Date out of supported range: {date.isoformat()}") from None

    notes: list[str] = []
    if result.polar:
        notes.append("Sun does not rise or set on this date at this latitude; affected times are null.")
    if result.fallback_applied:
        notes.append("High-latitude fallback applied (one-seventh of the night) for Fajr and/or Isha.")
    off_day = [n for n, t in result.times.items() if t is not None and t.date() != date]
    if off_day:
        notes.append(f"These times fall on an adjacent local date: {', '.join(off_day)}.")

    logger.info(
        "prayer_times computed method=%s asr=%s tz=%s fallback=%s polar=%s",
        method.value,
        asr.value,
        zone.key,
        result.fallback_applied,
        result.polar,
    )

    params = METHOD_PARAMS[method]
    dhuhr = result.times["dhuhr"]
    assert dhuhr is not None  # solar noon always exists

    def iso(name: str) -> str | None:
        value = result.times[name]
        return value.isoformat() if value is not None else None

    return PrayerTimesResponse(
        fajr=iso("fajr"),
        sunrise=iso("sunrise"),
        dhuhr=dhuhr.isoformat(),
        asr=iso("asr"),
        maghrib=iso("maghrib"),
        isha=iso("isha"),
        method=method,
        asr_method=asr,
        lat=lat,
        lon=lon,
        date=date,
        tz=zone.key,
        utc_offset_at_noon=_format_offset(_local_noon(date, zone)),
        uncertainty_minutes=PRAYER_TIME_UNCERTAINTY_MINUTES,
        provenance=PrayerProvenance(
            algorithm="NOAA/Meeus low-precision solar position (PrayTimes.org formulation)",
            algorithm_version=PRAYER_TIMES_ALGORITHM_VERSION,
            fajr_angle=params.fajr_angle,
            isha_angle=params.isha_angle,
            isha_interval_minutes=params.isha_interval_minutes,
            asr_shadow_factor=1.0 if asr == AsrMethod.STANDARD else 2.0,
            timezone_database=_tzdata_source(),
        ),
        notes=" ".join(notes) or None,
    )


@router.get("/hijri", response_model=HijriResponse)
async def get_hijri(date: date = Query(..., description="Gregorian date (YYYY-MM-DD)")) -> HijriResponse:
    """Convert a Gregorian civil date to its Umm al-Qura Hijri date."""
    try:
        h = gregorian_to_hijri(date)
    except OverflowError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    return HijriResponse(**_hijri_response_fields(h, date), provenance=_hijri_provenance())  # type: ignore[arg-type]


@router.get("/hijri/today", response_model=HijriTodayResponse)
async def get_hijri_today(
    tz: str = Query(..., description="IANA timezone name; 'today' is resolved in this zone"),
    instant: datetime = Depends(utc_now),
) -> HijriTodayResponse:
    """Today's Hijri date for the caller's timezone (not the server's)."""
    try:
        zone = resolve_timezone(tz)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    if instant.tzinfo is None:  # defensive: never interpret a naive clock as local time
        instant = instant.replace(tzinfo=UTC)
    local = instant.astimezone(zone)
    try:
        h = gregorian_to_hijri(local.date())
    except OverflowError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    return HijriTodayResponse(
        **_hijri_response_fields(h, local.date()),  # type: ignore[arg-type]
        provenance=_hijri_provenance(zone_used=True),
        tz=zone.key,
        local_datetime=_round_second(local).isoformat(),
    )


@router.get("/gregorian", response_model=GregorianResponse)
async def get_gregorian(
    hijri: str = Query(..., description="Hijri date YYYY-MM-DD (Western or Arabic-Indic digits)"),
) -> GregorianResponse:
    """Convert an Umm al-Qura Hijri date to its Gregorian civil date."""
    try:
        y, m, d = parse_hijri(hijri)
        if not HIJRI_MIN <= (y, m, d) <= HIJRI_MAX:
            raise OverflowError(f"Hijri date out of supported range {HIJRI_MIN[0]}..{HIJRI_MAX[0]} AH")
        g = Hijri(y, m, d).to_gregorian()
    except (ValueError, OverflowError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid Hijri date: {exc}") from None
    return GregorianResponse(
        year=g.year,
        month=g.month,
        day=g.day,
        hijri_date=f"{y:04d}-{m:02d}-{d:02d}",
        uncertainty=HIJRI_UNCERTAINTY_NOTE,
        provenance=_hijri_provenance(),
    )
