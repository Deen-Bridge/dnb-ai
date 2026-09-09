"""Unit and integration tests for the Hadith search module (#120).

No live API keys or external network required.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from hadith_search import (
    HadithSearchEngine,
    HadithSearchResponse,
    HadithSearchResult,
    get_chapter_title,
    get_search_engine,
    normalize_search_text,
    router,
    search_hadith,
    strip_arabic_diacritics,
    tokenize_query,
)
from main import app

client = TestClient(app)


# ---------------------------------------------------------------------------
# Text Normalization & Tokenization Unit Tests
# ---------------------------------------------------------------------------


def test_strip_arabic_diacritics() -> None:
    text_with_harakat = "إِنَّمَا الأَعْمَالُ بِالنِّيَّاتِ"
    stripped = strip_arabic_diacritics(text_with_harakat)
    assert "َ" not in stripped
    assert "ِ" not in stripped
    assert "ُ" not in stripped
    assert "ّ" not in stripped
    assert "ا" in stripped


def test_normalize_search_text() -> None:
    text = "Sahih al-Bukhari: Hadith #1 (Revelation)!"
    normalized = normalize_search_text(text)
    assert normalized == "sahih al bukhari hadith 1 revelation"


def test_tokenize_query_synonym_and_stopwords() -> None:
    tokens = tokenize_query("the messenger and the prophet regarding intentions")
    assert "the" not in tokens
    assert "and" not in tokens
    assert "prophet" in tokens
    assert "intention" in tokens


# ---------------------------------------------------------------------------
# Chapter Title Resolution Tests
# ---------------------------------------------------------------------------


def test_get_chapter_title_bukhari() -> None:
    assert get_chapter_title("bukhari", 1) == "Revelation"
    assert get_chapter_title("bukhari", 2) == "Belief"
    assert get_chapter_title("bukhari", 24) == "Zakat (Obligatory Charity)"
    assert get_chapter_title("bukhari", 31) == "Fasting (Sawm)"
    assert get_chapter_title("bukhari", None) == "General"


def test_get_chapter_title_muslim_and_others() -> None:
    assert "Faith" in get_chapter_title("muslim", 1)
    assert "Purification" in get_chapter_title("abudawud", 1)
    assert "Purification" in get_chapter_title("tirmidhi", 1)
    assert "Purification" in get_chapter_title("nasai", 1)


# ---------------------------------------------------------------------------
# Core Engine & Search Tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def search_engine() -> HadithSearchEngine:
    return get_search_engine()


def test_search_by_topic_intention(search_engine: HadithSearchEngine) -> None:
    response = search_hadith("intention", engine=search_engine)
    assert isinstance(response, HadithSearchResponse)
    assert response.total > 0
    assert len(response.results) > 0

    first = response.results[0]
    assert isinstance(first, HadithSearchResult)
    assert first.collection == "Sahih al-Bukhari"
    assert first.number == 1
    assert first.grading == "sahih"
    assert first.chapter == "Revelation"
    assert first.narrator == "Umar ibn al-Khattab"
    assert "intentions" in first.text_english.lower()
    assert first.text_arabic is not None
    assert "النيات" in strip_arabic_diacritics(first.text_arabic)


def test_search_by_english_keyword_phrase(search_engine: HadithSearchEngine) -> None:
    response = search_hadith("good word is charity", engine=search_engine)
    assert response.total > 0
    found = any("good" in r.text_english.lower() and "charity" in r.text_english.lower() for r in response.results)
    assert found


def test_search_by_arabic_text(search_engine: HadithSearchEngine) -> None:
    response = search_hadith("طلب العلم فريضة", engine=search_engine)
    assert response.total > 0
    found = any("knowledge" in r.text_english.lower() or "علم" in (r.text_arabic or "") for r in response.results)
    assert found


def test_search_by_narrator(search_engine: HadithSearchEngine) -> None:
    response = search_hadith("Abu Hurayrah", engine=search_engine)
    assert response.total > 0
    # Abu Hurayrah is a primary narrator
    found_narrator = any(r.narrator and "Abu Hurayrah" in r.narrator for r in response.results)
    assert found_narrator


def test_search_by_chapter(search_engine: HadithSearchEngine) -> None:
    response = search_hadith("Revelation", engine=search_engine)
    assert response.total > 0
    assert any(r.chapter and "Revelation" in r.chapter for r in response.results)


# ---------------------------------------------------------------------------
# Filtering Tests (Collection and Grading)
# ---------------------------------------------------------------------------


def test_filter_by_collection_bukhari(search_engine: HadithSearchEngine) -> None:
    response = search_hadith("knowledge", collection="bukhari", engine=search_engine)
    assert response.total > 0
    for r in response.results:
        assert r.collection == "Sahih al-Bukhari"


def test_filter_by_collection_muslim(search_engine: HadithSearchEngine) -> None:
    response = search_hadith("faith", collection="muslim", engine=search_engine)
    assert response.total > 0
    for r in response.results:
        assert r.collection == "Sahih al-Muslim" or r.collection == "Sahih Muslim"


def test_filter_by_collection_alias(search_engine: HadithSearchEngine) -> None:
    response = search_hadith("purification", collection="Sunan Abu Dawud", engine=search_engine)
    assert response.total > 0
    for r in response.results:
        assert r.collection == "Sunan Abu Dawud"


def test_filter_by_grading_sahih(search_engine: HadithSearchEngine) -> None:
    response = search_hadith("prayer", grading="sahih", engine=search_engine)
    assert response.total > 0
    for r in response.results:
        assert r.grading == "sahih"


def test_filter_by_grading_hasan(search_engine: HadithSearchEngine) -> None:
    response = search_hadith("knowledge", grading="hasan", engine=search_engine)
    assert response.total > 0
    for r in response.results:
        assert r.grading == "hasan"


def test_filter_by_collection_and_grading_combined(search_engine: HadithSearchEngine) -> None:
    response = search_hadith("knowledge", collection="ibnmajah", grading="hasan", engine=search_engine)
    assert response.total > 0
    for r in response.results:
        assert r.collection == "Sunan Ibn Majah"
        assert r.grading == "hasan"


# ---------------------------------------------------------------------------
# Pagination Tests
# ---------------------------------------------------------------------------


def test_pagination_limit_and_offset(search_engine: HadithSearchEngine) -> None:
    page_1 = search_hadith("prayer", limit=5, offset=0, engine=search_engine)
    page_2 = search_hadith("prayer", limit=5, offset=5, engine=search_engine)

    assert page_1.limit == 5
    assert page_1.offset == 0
    assert len(page_1.results) == 5
    assert page_2.limit == 5
    assert page_2.offset == 5
    assert len(page_2.results) > 0

    # Ensure no duplicate results between page 1 and page 2
    page_1_ids = {(r.collection, r.number) for r in page_1.results}
    page_2_ids = {(r.collection, r.number) for r in page_2.results}
    assert len(page_1_ids.intersection(page_2_ids)) == 0


def test_pagination_offset_beyond_total(search_engine: HadithSearchEngine) -> None:
    response = search_hadith("intention", limit=10, offset=100000, engine=search_engine)
    assert response.total > 0
    assert response.results == []
    assert response.offset == 100000


# ---------------------------------------------------------------------------
# Edge Cases & Special Characters Tests
# ---------------------------------------------------------------------------


def test_search_special_characters(search_engine: HadithSearchEngine) -> None:
    # Queries with punctuation / operators should not raise sqlite3 syntax errors
    special_queries = [
        "what is: (intention)?",
        "faith & prayer",
        "charity*",
        "da'if OR 'sahih'",
        '"actions are by intentions"',
        "test / query - not crash",
    ]
    for sq in special_queries:
        response = search_hadith(sq, engine=search_engine)
        assert isinstance(response, HadithSearchResponse)


def test_search_unknown_collection_returns_empty(search_engine: HadithSearchEngine) -> None:
    response = search_hadith("prayer", collection="nonexistent_collection", engine=search_engine)
    assert response.total == 0
    assert response.results == []


def test_search_unknown_keyword_returns_empty(search_engine: HadithSearchEngine) -> None:
    response = search_hadith("zyxwvutsrqponmlkjihgfedcba", engine=search_engine)
    assert response.total == 0
    assert response.results == []


# ---------------------------------------------------------------------------
# FastAPI Route & Endpoint Integration Tests
# ---------------------------------------------------------------------------


def test_api_search_endpoint_success() -> None:
    resp = client.get("/hadith/search?q=intention")
    assert resp.status_code == 200
    data = resp.json()

    assert "results" in data
    assert "total" in data
    assert "offset" in data
    assert "limit" in data
    assert data["total"] >= 1
    assert data["offset"] == 0
    assert data["limit"] == 10

    first = data["results"][0]
    assert first["collection"] == "Sahih al-Bukhari"
    assert first["number"] == 1
    assert first["grading"] == "sahih"
    assert first["chapter"] == "Revelation"
    assert first["narrator"] == "Umar ibn al-Khattab"
    assert "intention" in first["text_english"].lower()


def test_api_search_endpoint_with_filters() -> None:
    resp = client.get("/hadith/search?q=knowledge&collection=bukhari&grading=sahih&limit=3&offset=0")
    assert resp.status_code == 200
    data = resp.json()
    assert data["limit"] == 3
    assert len(data["results"]) <= 3
    for r in data["results"]:
        assert r["collection"] == "Sahih al-Bukhari"
        assert r["grading"] == "sahih"


def test_api_search_endpoint_validation_missing_q() -> None:
    resp = client.get("/hadith/search")
    assert resp.status_code == 422  # Query parameter 'q' is required


def test_api_search_endpoint_validation_empty_q() -> None:
    resp = client.get("/hadith/search?q=   ")
    assert resp.status_code == 400
    assert "Search query 'q' must not be empty" in resp.json().get("detail", "")


def test_api_search_endpoint_validation_invalid_limit() -> None:
    # limit max is 50
    resp = client.get("/hadith/search?q=prayer&limit=100")
    assert resp.status_code == 422

    # limit min is 1
    resp = client.get("/hadith/search?q=prayer&limit=0")
    assert resp.status_code == 422


def test_api_search_endpoint_validation_invalid_offset() -> None:
    resp = client.get("/hadith/search?q=prayer&offset=-1")
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# OpenAPI Documentation Tests
# ---------------------------------------------------------------------------


def test_openapi_schema_contains_hadith_search() -> None:
    resp = client.get("/openapi.json")
    assert resp.status_code == 200
    schema = resp.json()
    paths = schema.get("paths", {})

    assert "/hadith/search" in paths
    search_op = paths["/hadith/search"].get("get")
    assert search_op is not None
    assert search_op.get("summary") == "Search hadith by topic and keyword"

    param_names = [p.get("name") for p in search_op.get("parameters", [])]
    assert "q" in param_names
    assert "collection" in param_names
    assert "grading" in param_names
    assert "limit" in param_names
    assert "offset" in param_names


def test_router_prefix_and_routes() -> None:
    assert router.prefix == "/hadith"
    paths = {route.path for route in router.routes}  # type: ignore[attr-defined]
    assert "/hadith/search" in paths
