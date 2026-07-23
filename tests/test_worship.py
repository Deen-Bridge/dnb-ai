import pytest
from fastapi.testclient import TestClient
from datetime import date
from zoneinfo import ZoneInfo
from main import app

client = TestClient(app)

def test_prayer_times_makkah():
    # Makkah coords approx 21.4225, 39.8262
    # Reference: general Makkah times in summer
    res = client.get("/prayer-times", params={
        "lat": 21.4225,
        "lon": 39.8262,
        "date": "2023-06-15",
        "method": "MAKKAH",
        "tz": "Asia/Riyadh"
    })
    assert res.status_code == 200
    data = res.json()
    assert data["method"] == "MAKKAH"
    
    # Makkah Isha is strictly 90 mins after Maghrib for non-Ramadan
    maghrib_str = data["maghrib"]
    isha_str = data["isha"]
    
    maghrib_time = maghrib_str.split("T")[1][:5] # "HH:MM"
    isha_time = isha_str.split("T")[1][:5]
    
    mh, mm = map(int, maghrib_time.split(':'))
    ih, im = map(int, isha_time.split(':'))
    
    maghrib_mins = mh * 60 + mm
    isha_mins = ih * 60 + im
    if isha_mins < maghrib_mins:
        isha_mins += 24 * 60
        
    assert isha_mins - maghrib_mins == 90

def test_prayer_times_london_isna():
    # London approx 51.5072, -0.1276
    res = client.get("/prayer-times", params={
        "lat": 51.5072,
        "lon": -0.1276,
        "date": "2023-01-15",
        "method": "ISNA",
        "tz": "Europe/London"
    })
    assert res.status_code == 200
    data = res.json()
    assert data["tz"] == "Europe/London"
    assert data["fajr"] is not None

def test_prayer_times_high_latitude_summer():
    # Reykjavik in summer (64.1466, -21.9426)
    # Sun does not set properly or angles are not reached.
    res = client.get("/prayer-times", params={
        "lat": 64.1466,
        "lon": -21.9426,
        "date": "2023-06-21",
        "method": "ISNA",
        "tz": "Atlantic/Reykjavik"
    })
    assert res.status_code == 200
    data = res.json()
    # High-latitude fallback should populate all times
    assert data["fajr"] is not None
    assert data["isha"] is not None
    assert "High-latitude fallback" in data["notes"]

def test_hijri_conversion():
    # 1 Ramadan 1444 was approximately March 23, 2023
    res = client.get("/hijri", params={"date": "2023-03-23"})
    assert res.status_code == 200
    data = res.json()
    assert data["year"] == 1444
    assert data["month"] == 9
    assert data["day"] == 1

def test_hijri_out_of_range():
    # Extremely old date not supported by umm al-qura
    res = client.get("/hijri", params={"date": "1000-01-01"})
    assert res.status_code == 400
    assert "Date out of supported range" in res.json()["detail"]

def test_gregorian_conversion():
    res = client.get("/gregorian", params={"hijri": "1444-09-01"})
    assert res.status_code == 200
    data = res.json()
    assert data["year"] == 2023
    assert data["month"] == 3
    assert data["day"] == 23

def test_invalid_lat_lon():
    res = client.get("/prayer-times", params={
        "lat": 100, # Invalid
        "lon": 0,
        "date": "2023-01-01",
        "method": "ISNA",
        "tz": "UTC"
    })
    assert res.status_code == 422 # Pydantic validation error

def test_invalid_tz():
    res = client.get("/prayer-times", params={
        "lat": 0,
        "lon": 0,
        "date": "2023-01-01",
        "method": "ISNA",
        "tz": "Fake/Timezone"
    })
    assert res.status_code == 400
    assert "Invalid timezone" in res.json()["detail"]
