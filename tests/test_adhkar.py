"""Tests for the /adhkar/recommend endpoint and the AdhkarCorpus module.

Coverage areas
--------------
* Happy-path category and free-text query matching.
* Empty / no-match state.
* Provenance fields present and well-formed on every response.
* Freshness: ``loaded_at`` is a UTC ISO-8601 timestamp.
* Arabic text round-trips through the JSON response unchanged.
* Edge cases: empty strings, whitespace-only inputs, mixed-case category.
* AdhkarCorpus unit tests: missing file, malformed file, provenance() contract.
"""

import json
import re
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from adhkar import AdhkarCorpus
from main import app

client = TestClient(app)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ISO8601_UTC_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


def _assert_provenance(provenance: dict) -> None:
    """Assert that a provenance dict satisfies the production contract."""
    assert isinstance(provenance["data_version"], str) and provenance["data_version"]
    assert isinstance(provenance["schema_version"], int) and provenance["schema_version"] >= 1
    assert isinstance(provenance["source"], str) and provenance["source"]
    assert isinstance(provenance["data_last_updated"], str) and provenance["data_last_updated"]
    assert _ISO8601_UTC_RE.match(provenance["loaded_at"]), (
        f"loaded_at must be UTC ISO-8601 (YYYY-MM-DDTHH:MM:SSZ), got {provenance['loaded_at']!r}"
    )
    assert isinstance(provenance["entry_count"], int) and provenance["entry_count"] > 0


# ---------------------------------------------------------------------------
# Endpoint: happy-path
# ---------------------------------------------------------------------------


def test_recommends_by_category():
    response = client.post("/adhkar/recommend", json={"category": "morning"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["matches"]
    assert payload["matches"][0]["category"] == "morning"
    assert payload["matches"][0]["source"] == "Hisnul Muslim"


def test_recommends_by_free_text_query():
    response = client.post("/adhkar/recommend", json={"query": "anxiety and grief"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["matches"]
    assert payload["matches"][0]["category"] == "distress"


def test_category_matching_is_case_insensitive():
    response_lower = client.post("/adhkar/recommend", json={"category": "morning"})
    response_upper = client.post("/adhkar/recommend", json={"category": "Morning"})
    response_mixed = client.post("/adhkar/recommend", json={"category": "MORNING"})

    for resp in (response_lower, response_upper, response_mixed):
        assert resp.status_code == 200
        assert resp.json()["matches"], "Case-insensitive category match should return results"


def test_no_filters_returns_all_entries():
    response = client.post("/adhkar/recommend", json={})

    assert response.status_code == 200
    payload = response.json()
    # Corpus has 4 entries; at minimum we should get them all when no filter applied
    assert len(payload["matches"]) >= 4


# ---------------------------------------------------------------------------
# Endpoint: empty / no-match states
# ---------------------------------------------------------------------------


def test_returns_no_match_message_when_nothing_matches():
    response = client.post("/adhkar/recommend", json={"category": "travel", "query": "nothing here"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["matches"] == []
    assert "No authenticated supplication" in payload["message"]


def test_empty_string_query_returns_all_entries():
    response = client.post("/adhkar/recommend", json={"query": ""})

    assert response.status_code == 200
    assert len(response.json()["matches"]) >= 4


def test_whitespace_only_query_treated_as_no_query():
    response = client.post("/adhkar/recommend", json={"query": "   "})

    assert response.status_code == 200
    # Whitespace-only should not filter — same as omitting the query
    assert len(response.json()["matches"]) >= 4


def test_unknown_category_returns_empty_matches():
    response = client.post("/adhkar/recommend", json={"category": "nonexistent_xyz"})

    assert response.status_code == 200
    assert response.json()["matches"] == []


# ---------------------------------------------------------------------------
# Endpoint: provenance contract
# ---------------------------------------------------------------------------


def test_provenance_present_on_match_response():
    response = client.post("/adhkar/recommend", json={"category": "morning"})

    assert response.status_code == 200
    payload = response.json()
    assert "provenance" in payload, "Response must include 'provenance' field"
    _assert_provenance(payload["provenance"])


def test_provenance_present_on_no_match_response():
    response = client.post("/adhkar/recommend", json={"category": "travel", "query": "nothing here"})

    assert response.status_code == 200
    payload = response.json()
    assert "provenance" in payload, "Provenance must be present even when no entries match"
    _assert_provenance(payload["provenance"])


def test_provenance_data_version_is_semver():
    response = client.post("/adhkar/recommend", json={})

    assert response.status_code == 200
    version = response.json()["provenance"]["data_version"]
    parts = version.split(".")
    assert len(parts) == 3 and all(p.isdigit() for p in parts), (
        f"data_version must be MAJOR.MINOR.PATCH, got {version!r}"
    )


def test_provenance_entry_count_matches_corpus():
    response_all = client.post("/adhkar/recommend", json={})
    response_filtered = client.post("/adhkar/recommend", json={"category": "morning"})

    all_payload = response_all.json()
    filtered_payload = response_filtered.json()

    # entry_count always reflects the total corpus size, not the filtered result set
    assert all_payload["provenance"]["entry_count"] == filtered_payload["provenance"]["entry_count"]
    assert all_payload["provenance"]["entry_count"] == len(all_payload["matches"])


# ---------------------------------------------------------------------------
# Endpoint: Arabic text round-trip
# ---------------------------------------------------------------------------


def test_arabic_text_preserved_in_response():
    """Arabic Unicode text must survive JSON serialisation unchanged."""
    response = client.post("/adhkar/recommend", json={"category": "morning"})

    assert response.status_code == 200
    entry = response.json()["matches"][0]
    assert entry["arabic"] == "أَصْبَحْنَا وَأَصْبَحَ الْمُلْكُ لِلَّهِ"


def test_arabic_query_search():
    """Searching by Arabic text should match the correct entry."""
    response = client.post("/adhkar/recommend", json={"query": "أَمْسَيْنَا"})

    assert response.status_code == 200
    matches = response.json()["matches"]
    assert any(m["id"] == "evening_1" for m in matches), (
        "Arabic query 'أَمْسَيْنَا' should match the evening adhkar entry"
    )


# ---------------------------------------------------------------------------
# Endpoint: validation errors
# ---------------------------------------------------------------------------


def test_invalid_body_returns_422():
    # Sending a non-object body should trigger Pydantic validation failure
    response = client.post("/adhkar/recommend", content="not-json", headers={"Content-Type": "application/json"})

    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Unit tests: AdhkarCorpus
# ---------------------------------------------------------------------------


def _make_corpus(entries: list[dict], *, version: str = "2.0.0", schema_version: int = 1) -> AdhkarCorpus:
    """Write a temporary data file and return an AdhkarCorpus instance for it."""
    payload = {
        "version": version,
        "schema_version": schema_version,
        "source_description": "Test source",
        "last_updated": "2024-06-01",
        "entries": entries,
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False)
        tmp_path = Path(fh.name)
    return AdhkarCorpus(data_file=tmp_path)


def test_corpus_loads_entries_from_file():
    entries = [{"id": "t1", "category": "morning", "arabic": "بِسْمِ اللهِ", "source": "Test"}]
    c = _make_corpus(entries)
    assert len(c.entries) == 1
    assert c.entries[0]["id"] == "t1"


def test_corpus_provenance_fields_populated():
    c = _make_corpus([], version="3.1.0", schema_version=2)
    prov = c.provenance()
    assert prov["data_version"] == "3.1.0"
    assert prov["schema_version"] == 2
    assert prov["source"] == "Test source"
    assert prov["data_last_updated"] == "2024-06-01"
    assert _ISO8601_UTC_RE.match(prov["loaded_at"]), f"loaded_at malformed: {prov['loaded_at']!r}"
    assert prov["entry_count"] == 0


def test_corpus_provenance_entry_count_reflects_entries():
    entries = [{"id": f"e{i}", "category": "morning"} for i in range(5)]
    c = _make_corpus(entries)
    assert c.provenance()["entry_count"] == 5


def test_corpus_missing_file_gives_empty_entries():
    c = AdhkarCorpus(data_file=Path("/nonexistent/adhkar_xyz.json"))
    assert c.entries == []
    prov = c.provenance()
    assert prov["data_version"] == "unknown"
    assert prov["entry_count"] == 0
    # loaded_at must still be a valid timestamp even when the file is missing
    assert _ISO8601_UTC_RE.match(prov["loaded_at"])


def test_corpus_search_no_filters_returns_all():
    entries = [{"id": f"e{i}", "category": "morning", "arabic": "", "transliteration": "", "translation": "", "source": ""} for i in range(3)]
    c = _make_corpus(entries)
    assert len(c.search()) == 3


def test_corpus_search_by_category():
    entries = [
        {"id": "m1", "category": "morning", "arabic": "", "transliteration": "", "translation": "", "source": ""},
        {"id": "e1", "category": "evening", "arabic": "", "transliteration": "", "translation": "", "source": ""},
    ]
    c = _make_corpus(entries)
    result = c.search(category="morning")
    assert len(result) == 1
    assert result[0]["id"] == "m1"


def test_corpus_search_by_arabic_text():
    arabic = "بِسْمِ اللهِ الرَّحْمَنِ الرَّحِيمِ"
    entries = [
        {"id": "b1", "category": "general", "arabic": arabic, "transliteration": "", "translation": "", "source": ""},
        {"id": "b2", "category": "general", "arabic": "other", "transliteration": "", "translation": "", "source": ""},
    ]
    c = _make_corpus(entries)
    result = c.search(query="بِسْمِ")
    assert len(result) == 1
    assert result[0]["id"] == "b1"


def test_corpus_search_empty_query_returns_all():
    entries = [{"id": f"e{i}", "category": "morning", "arabic": "", "transliteration": "", "translation": "", "source": ""} for i in range(3)]
    c = _make_corpus(entries)
    assert len(c.search(query="")) == 3


def test_corpus_search_whitespace_query_returns_all():
    entries = [{"id": f"e{i}", "category": "morning", "arabic": "", "transliteration": "", "translation": "", "source": ""} for i in range(2)]
    c = _make_corpus(entries)
    assert len(c.search(query="   ")) == 2


def test_corpus_search_no_match_returns_empty_list():
    c = _make_corpus([{"id": "m1", "category": "morning", "arabic": "", "transliteration": "", "translation": "", "source": ""}])
    assert c.search(category="evening") == []


def test_corpus_category_match_case_insensitive():
    entries = [{"id": "m1", "category": "Morning", "arabic": "", "transliteration": "", "translation": "", "source": ""}]
    c = _make_corpus(entries)
    assert len(c.search(category="morning")) == 1
    assert len(c.search(category="MORNING")) == 1
