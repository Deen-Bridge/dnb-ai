"""Islamic utilities for the Deen Bridge AI service.

Provides deterministic endpoints for prayer times and Hijri/Gregorian
calendar conversions.

Prayer times are calculated using standard solar-position formulas
(declination + equation of time) derived from NOAA / Astronomical Algorithms (Meeus),
matching the logic of PrayTimes.org.
"""

import math
from datetime import date, datetime, timedelta
from enum import Enum
from zoneinfo import ZoneInfo, available_timezones
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from hijridate import Hijri, Gregorian

router = APIRouter(tags=["worship"])


class CalculationMethod(str, Enum):
    MWL = "MWL"  # Muslim World League (Fajr 18, Isha 17)
    ISNA = "ISNA"  # Islamic Society of North America (Fajr 15, Isha 15)
    EGYPT = "EGYPT"  # Egyptian General Authority (Fajr 19.5, Isha 17.5)
    MAKKAH = "MAKKAH"  # Umm al-Qura University, Makkah (Fajr 18.5, Isha 90 min)
    KARACHI = "KARACHI"  # University of Islamic Sciences, Karachi (Fajr 18, Isha 18)


class AsrMethod(str, Enum):
    STANDARD = "STANDARD"  # Shafi'i, Maliki, Hanbali
    HANAFI = "HANAFI"


# --- Math Utilities ---
def dsin(d):
    return math.sin(math.radians(d))


def dcos(d):
    return math.cos(math.radians(d))


def dtan(d):
    return math.tan(math.radians(d))


def darcsin(x):
    return math.degrees(math.asin(x))


def darccos(x):
    return math.degrees(math.acos(x))


def darctan2(y, x):
    return math.degrees(math.atan2(y, x))


def darccot(x):
    return math.degrees(math.atan(1.0 / x))


def fix_angle(a):
    a = a - 360.0 * math.floor(a / 360.0)
    return a if a >= 0 else a + 360.0


def fix_hour(a):
    a = a - 24.0 * math.floor(a / 24.0)
    return a if a >= 0 else a + 24.0


# --- Solar Position Calculations ---
def julian_date(y, m, d):
    if m <= 2:
        y -= 1
        m += 12
    A = math.floor(y / 100)
    B = 2 - A + math.floor(A / 4)
    return math.floor(365.25 * (y + 4716)) + math.floor(30.6001 * (m + 1)) + d + B - 1524.5


def sun_position(jd):
    # Number of days from J2000.0
    D = jd - 2451545.0
    # Mean anomaly of the sun
    g = fix_angle(357.529 + 0.98560028 * D)
    # Mean longitude of the sun
    q = fix_angle(280.459 + 0.98564736 * D)
    # Geocentric apparent ecliptic longitude
    L = fix_angle(q + 1.915 * dsin(g) + 0.020 * dsin(2 * g))

    # Mean obliquity of the ecliptic
    e = 23.439 - 0.00000036 * D

    # Sun's declination
    declination = darcsin(dsin(e) * dsin(L))

    # Right ascension
    RA = darctan2(dcos(e) * dsin(L), dcos(L)) / 15.0
    RA = fix_hour(RA)

    # Equation of time
    eq_t = q / 15.0 - RA
    return declination, eq_t


def compute_time(angle, lat, declination):
    """Compute the hour angle for a given solar angle."""
    try:
        val = (-dsin(angle) - dsin(declination) * dsin(lat)) / (dcos(declination) * dcos(lat))
        if val < -1.0 or val > 1.0:
            return None  # Angle does not occur (e.g. high latitudes)
        return darccos(val) / 15.0
    except ValueError:
        return None


def compute_asr(lat, declination, asr_factor):
    """Compute hour angle for Asr."""
    try:
        val = dsin(darccot(asr_factor + dtan(abs(lat - declination))))
        val = (val - dsin(declination) * dsin(lat)) / (dcos(declination) * dcos(lat))
        if val < -1.0 or val > 1.0:
            return None
        return darccos(val) / 15.0
    except ValueError:
        return None


def calculate_prayer_times(lat: float, lon: float, date_obj: date, method: CalculationMethod,
                           asr_method: AsrMethod, tz: ZoneInfo):
    jd = julian_date(date_obj.year, date_obj.month, date_obj.day)

    offset_h = lon / 15.0
    jd_noon = jd + 0.5 - offset_h / 24.0

    declination, eq_t = sun_position(jd_noon)

    # Dhuhr
    dhuhr_time = 12.0 - eq_t - offset_h

    # Fajr and Isha Angles
    fajr_angle = 18.0
    isha_angle = 18.0  # default
    isha_interval = None

    if method == CalculationMethod.MWL:
        fajr_angle, isha_angle = 18.0, 17.0
    elif method == CalculationMethod.ISNA:
        fajr_angle, isha_angle = 15.0, 15.0
    elif method == CalculationMethod.EGYPT:
        fajr_angle, isha_angle = 19.5, 17.5
    elif method == CalculationMethod.MAKKAH:
        fajr_angle = 18.5
        isha_interval = 90.0  # minutes after Maghrib
    elif method == CalculationMethod.KARACHI:
        fajr_angle, isha_angle = 18.0, 18.0

    asr_factor = 1.0 if asr_method == AsrMethod.STANDARD else 2.0

    # Compute base times
    t_fajr = compute_time(fajr_angle, lat, declination)
    t_sunrise = compute_time(0.833, lat, declination)  # 0.833 accounts for refraction and sun radius
    t_sunset = compute_time(0.833, lat, declination)
    t_asr = compute_asr(lat, declination, asr_factor)
    t_isha = compute_time(isha_angle, lat, declination) if isha_interval is None else None

    times_hours = {
        "fajr": dhuhr_time - t_fajr if t_fajr else None,
        "sunrise": dhuhr_time - t_sunrise if t_sunrise else None,
        "dhuhr": dhuhr_time,
        "asr": dhuhr_time + t_asr if t_asr else None,
        "maghrib": dhuhr_time + t_sunset if t_sunset else None,
        "isha": dhuhr_time + t_isha if t_isha else None,
    }

    # Handle Umm al-Qura Isha
    if isha_interval is not None and times_hours["maghrib"] is not None:
        times_hours["isha"] = times_hours["maghrib"] + (isha_interval / 60.0)

    # High-latitude fallback (Angle-based method: e.g. 1/7th of night)
    fallback_applied = False
    if (times_hours["fajr"] is None or times_hours["isha"] is None or
            times_hours["sunrise"] is None or times_hours["maghrib"] is None):
        if times_hours["sunrise"] is not None and times_hours["maghrib"] is not None:
            night_duration = 24.0 - (times_hours["maghrib"] - times_hours["sunrise"])
            # Fallback: Fajr is 1/7th of night before sunrise, Isha is 1/7th of night after sunset
            if times_hours["fajr"] is None:
                times_hours["fajr"] = times_hours["sunrise"] - (night_duration / 7.0)
                fallback_applied = True
            if times_hours["isha"] is None:
                times_hours["isha"] = times_hours["maghrib"] + (night_duration / 7.0)
                fallback_applied = True

    # Convert float hours to UTC datetime
    res = {}
    for k, v in times_hours.items():
        if v is None:
            res[k] = None
        else:
            v_utc = fix_hour(v)
            hours = int(v_utc)
            minutes = int((v_utc - hours) * 60)
            seconds = int((((v_utc - hours) * 60) - minutes) * 60)

            # This is UTC time.
            dt_utc = datetime(
                date_obj.year, date_obj.month, date_obj.day,
                hours, minutes, seconds, tzinfo=ZoneInfo("UTC")
            )
            # If the calculated UTC time wrapped around the day, adjust date
            if v < 0:
                dt_utc -= timedelta(days=1)
            elif v >= 24:
                dt_utc += timedelta(days=1)

            res[k] = dt_utc.astimezone(tz).isoformat()

    return res, fallback_applied


class PrayerTimesResponse(BaseModel):
    fajr: Optional[str]
    sunrise: Optional[str]
    dhuhr: str
    asr: Optional[str]
    maghrib: Optional[str]
    isha: Optional[str]
    method: CalculationMethod
    asr_method: AsrMethod
    lat: float
    lon: float
    date: date
    tz: str
    notes: Optional[str] = None


class HijriResponse(BaseModel):
    year: int
    month: int
    month_en: str
    month_ar: str
    day: int
    gregorian_date: date


class GregorianResponse(BaseModel):
    year: int
    month: int
    day: int
    hijri_date: str


@router.get("/prayer-times", response_model=PrayerTimesResponse)
async def get_prayer_times(
    lat: float = Query(..., ge=-90, le=90, description="Latitude"),
    lon: float = Query(..., ge=-180, le=180, description="Longitude"),
    date: date = Query(..., description="Date for calculation"),
    method: CalculationMethod = Query(CalculationMethod.MAKKAH, description="Calculation method"),
    asr: AsrMethod = Query(AsrMethod.STANDARD, description="Asr juristic method"),
    tz: str = Query(..., description="IANA timezone name")
):
    """Get Islamic prayer times for a given location, date, and calculation method."""
    try:
        zone = ZoneInfo(tz)
    except Exception:
        raise HTTPException(status_code=400, detail=f"Invalid timezone: {tz}")

    times, fallback_applied = calculate_prayer_times(lat, lon, date, method, asr, zone)

    notes = None
    if fallback_applied:
        notes = "High-latitude fallback applied or times not calculable."

    return PrayerTimesResponse(
        fajr=times.get("fajr"),
        sunrise=times.get("sunrise"),
        dhuhr=times["dhuhr"],
        asr=times.get("asr"),
        maghrib=times.get("maghrib"),
        isha=times.get("isha"),
        method=method,
        asr_method=asr,
        lat=lat,
        lon=lon,
        date=date,
        tz=tz,
        notes=notes
    )


@router.get("/hijri", response_model=HijriResponse)
async def get_hijri(date: date = Query(..., description="Gregorian date (YYYY-MM-DD)")):
    """Convert a Gregorian date to Hijri."""
    try:
        g = Gregorian(date.year, date.month, date.day)
        h = g.to_hijri()
        return HijriResponse(
            year=h.year,
            month=h.month,
            month_en=h.month_name(),
            month_ar=h.month_name(language='ar'),
            day=h.day,
            gregorian_date=date
        )
    except OverflowError as e:
        raise HTTPException(status_code=400, detail=f"Date out of supported range: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/gregorian", response_model=GregorianResponse)
async def get_gregorian(hijri: str = Query(..., description="Hijri date (YYYY-MM-DD)")):
    """Convert a Hijri date to Gregorian."""
    try:
        parts = hijri.split('-')
        if len(parts) != 3:
            raise ValueError("Invalid format. Use YYYY-MM-DD")
        y, m, d = int(parts[0]), int(parts[1]), int(parts[2])
        h = Hijri(y, m, d)
        g = h.to_gregorian()
        return GregorianResponse(
            year=g.year,
            month=g.month,
            day=g.day,
            hijri_date=hijri
        )
    except (ValueError, OverflowError) as e:
        raise HTTPException(status_code=400, detail=f"Invalid Hijri date: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
