"""Tests for the narrator biography (rijal) lookup system (issue #197)."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from narrator_biography import (
    NarratorDatabase,
    router,
)


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


# ---------------------------------------------------------------------------
# Database init & data integrity
# ---------------------------------------------------------------------------


def test_database_loads_and_count() -> None:
    db = NarratorDatabase()
    assert db.count >= 20


def test_all_records_have_id_and_name() -> None:
    db = NarratorDatabase()
    db._ensure_loaded()
    for rec in db._records:
        assert rec["id"]
        assert rec["name"]


def test_all_records_have_reliability() -> None:
    db = NarratorDatabase()
    db._ensure_loaded()
    for rec in db._records:
        assert rec.get("reliability_assessment"), f"Missing reliability for {rec['id']}"


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


def test_search_by_name() -> None:
    db = NarratorDatabase()
    results = db.search("bukhari")
    assert len(results) >= 1
    assert results[0].id == "bukhari"
    assert results[0].name == "Muhammad ibn Isma'il al-Bukhari"


def test_search_by_kunyah() -> None:
    db = NarratorDatabase()
    results = db.search("Abu Abdullah")
    ids = [r.id for r in results]
    assert "bukhari" in ids


def test_search_by_nisba() -> None:
    db = NarratorDatabase()
    results = db.search("Naysaburi")
    ids = [r.id for r in results]
    assert "muslim" in ids


def test_search_fuzzy_match() -> None:
    db = NarratorDatabase()
    results = db.search("Tirmiz")
    ids = [r.id for r in results]
    assert "tirmidhi" in ids


def test_search_no_match() -> None:
    db = NarratorDatabase()
    results = db.search("zzznonexistent99999")
    assert results == []


# ---------------------------------------------------------------------------
# Profile
# ---------------------------------------------------------------------------


def test_profile_fields_populated() -> None:
    db = NarratorDatabase()
    profile = db.get_narrator("bukhari")
    assert profile.name == "Muhammad ibn Isma'il al-Bukhari"
    assert profile.kunyah == "Abu Abdullah"
    assert profile.nisba == "al-Bukhari"
    assert profile.birth_year == 810
    assert profile.death_year == 870
    assert profile.region == "Bukhara, Central Asia"
    assert profile.biography_summary


def test_profile_reliability_present() -> None:
    db = NarratorDatabase()
    profile = db.get_narrator("bukhari")
    assert len(profile.reliability_assessment) >= 1
    scholars = {a.scholar for a in profile.reliability_assessment}
    assert "al-Dhahabi" in scholars


def test_profile_404() -> None:
    db = NarratorDatabase()
    import pytest
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc_info:
        db.get_narrator("does-not-exist")
    assert exc_info.value.status_code == 404


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def test_compare_two_narrators() -> None:
    db = NarratorDatabase()
    result = db.compare_narrators(["bukhari", "muslim"])
    assert len(result.narrators) == 2
    assert result.narrators[0].id == "bukhari"
    assert result.narrators[1].id == "muslim"
    # Both studied under Ahmad Hanbal
    assert "ahmad-hanbal" in result.shared_teachers


# ---------------------------------------------------------------------------
# Isnad resolution
# ---------------------------------------------------------------------------


def test_isnad_lookup_resolves_names() -> None:
    db = NarratorDatabase()
    profiles = db.lookup_isnad(["bukhari", "muslim"])
    assert len(profiles) == 2
    ids = {p.id for p in profiles}
    assert "bukhari" in ids
    assert "muslim" in ids


def test_isnad_lookup_partial_match() -> None:
    db = NarratorDatabase()
    profiles = db.lookup_isnad(["Abu Hanifa", "zzznonexistent"])
    assert len(profiles) == 1
    assert profiles[0].id == "abu-hanifa"


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------


def test_network_returns_center_and_teachers() -> None:
    db = NarratorDatabase()
    graph = db.get_network("bukhari", depth=1)
    assert graph.center == "bukhari"
    relations = {n.id: n.relation for n in graph.nodes}
    assert relations["bukhari"] == "self"
    assert "ahmad-hanbal" in relations
    assert "malik-ibn-anas" in relations


def test_network_depth_2() -> None:
    db = NarratorDatabase()
    graph = db.get_network("bukhari", depth=2)
    ids = {n.id for n in graph.nodes}
    # Bukhari's teachers' teachers should appear at depth 2
    assert len(ids) > 4


def test_network_404() -> None:
    db = NarratorDatabase()
    import pytest
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc_info:
        db.get_network("zzznonexistent")
    assert exc_info.value.status_code == 404


# ---------------------------------------------------------------------------
# HTTP routes
# ---------------------------------------------------------------------------


def test_get_search_route() -> None:
    resp = _client().get("/narrators/search?q=bukhari")
    assert resp.status_code == 200
    body = resp.json()
    assert len(body) >= 1
    assert body[0]["id"] == "bukhari"
    assert "relevance_score" in body[0]


def test_get_profile_route() -> None:
    resp = _client().get("/narrators/muslim")
    assert resp.status_code == 200
    body = resp.json()
    assert body["id"] == "muslim"
    assert body["reliability_assessment"]


def test_get_profile_not_found() -> None:
    resp = _client().get("/narrators/zzznonexistent")
    assert resp.status_code == 404


def test_post_compare_route() -> None:
    resp = _client().post("/narrators/compare", json={"ids": ["bukhari", "muslim"]})
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["narrators"]) == 2
    assert "shared_teachers" in body


def test_post_compare_needs_at_least_two() -> None:
    resp = _client().post("/narrators/compare", json={"ids": ["bukhari"]})
    assert resp.status_code == 422


def test_post_isnad_lookup_route() -> None:
    resp = _client().post("/narrators/isnad-lookup", json={"names": ["Abu Dawud", "Tirmidhi"]})
    assert resp.status_code == 200
    body = resp.json()
    assert len(body) == 2


def test_get_network_route() -> None:
    resp = _client().get("/narrators/ahmad-hanbal/network?depth=1")
    assert resp.status_code == 200
    body = resp.json()
    assert body["center"] == "ahmad-hanbal"
    assert len(body["nodes"]) >= 2


# ---------------------------------------------------------------------------
# API schema validation
# ---------------------------------------------------------------------------


def test_search_schema() -> None:
    resp = _client().get("/narrators/search?q=malik")
    assert resp.status_code == 200
    for item in resp.json():
        assert "id" in item
        assert "name" in item
        assert "relevance_score" in item


def test_profile_schema() -> None:
    resp = _client().get("/narrators/abu-hanifa")
    assert resp.status_code == 200
    body = resp.json()
    expected_fields = {
        "id",
        "name",
        "kunyah",
        "birth_year",
        "death_year",
        "reliability_assessment",
        "teachers",
        "students",
        "biography_summary",
    }
    assert expected_fields.issubset(set(body.keys()))
