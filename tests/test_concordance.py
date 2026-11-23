"""Offline tests for the Quranic Concordance API (#125).

No secrets and no network: the taxonomy and the curated theme-verse mapping
(``data/theme_verses.json``) are bundled, and the app is exercised through
httpx's ASGI transport like the other endpoint tests.
"""

import os

import pytest
from httpx import ASGITransport, AsyncClient

os.environ.setdefault("GEMINI_API_KEY", "test-key")

import main  # noqa: E402


@pytest.fixture()
async def client():
    transport = ASGITransport(app=main.app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


# ---------------------------------------------------------------------------
# Hierarchy & navigation
# ---------------------------------------------------------------------------


class TestHierarchy:
    async def test_topic_hierarchy_returns_main_themes(self, client):
        resp = await client.get("/concordance/topics")
        assert resp.status_code == 200
        data = resp.json()
        roots = data["roots"]
        assert len(roots) >= 10  # the ten main themes
        ids = {node["id"] for node in roots}
        assert "tawhid" in ids and "worship" in ids and "ethics" in ids

    async def test_hierarchy_has_children_and_verse_counts(self, client):
        resp = await client.get("/concordance/topics")
        roots = resp.json()["roots"]
        tawhid = next(node for node in roots if node["id"] == "tawhid")
        assert tawhid["level"] == 0
        assert len(tawhid["children"]) > 0
        assert all(child["parent_id"] == "tawhid" or child["id"].startswith("tawhid") for child in tawhid["children"])
        # The bundled dataset must actually map verses, or the feature is hollow.
        assert tawhid["verse_count"] > 0

    async def test_unknown_topic_404(self, client):
        resp = await client.get("/concordance/topics/not-a-theme")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


class TestSearch:
    async def test_search_english_keyword(self, client):
        resp = await client.get("/concordance/search", params={"q": "monotheism"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] >= 1
        assert any(t["id"] == "tawhid" for t in data["themes"])
        tawhid = next(t for t in data["themes"] if t["id"] == "tawhid")
        assert tawhid["verse_count"] > 0
        assert len(tawhid["verses"]) > 0
        assert "reference" in tawhid["verses"][0]
        assert "relevance_score" in tawhid["verses"][0]

    async def test_search_arabic(self, client):
        resp = await client.get("/concordance/search", params={"q": "التوحيد"})
        assert resp.status_code == 200
        data = resp.json()
        assert any(t["id"] == "tawhid" for t in data["themes"])

    async def test_search_without_verses(self, client):
        resp = await client.get("/concordance/search", params={"q": "prayer", "include_verses": "false"})
        assert resp.status_code == 200
        for theme in resp.json()["themes"]:
            assert "verses" not in theme


# ---------------------------------------------------------------------------
# Browse topic
# ---------------------------------------------------------------------------


class TestBrowseTopic:
    async def test_browse_returns_verses_and_children(self, client):
        resp = await client.get("/concordance/topics/worship-salah")
        assert resp.status_code == 200
        data = resp.json()
        assert data["theme"]["id"] == "worship-salah"
        assert data["total_verses"] > 0
        assert len(data["verses"]) > 0
        assert data["verses"][0]["reference"].startswith("2:")

    async def test_browse_filter_by_min_relevance(self, client):
        resp = await client.get("/concordance/topics/tawhid", params={"min_relevance": "0.9"})
        assert resp.status_code == 200
        for verse in resp.json()["verses"]:
            assert verse["relevance_score"] >= 0.9

    async def test_browse_filter_by_context_type(self, client):
        resp = await client.get("/concordance/topics/history", params={"context_type": "primary"})
        assert resp.status_code == 200
        for verse in resp.json()["verses"]:
            assert verse["context_type"] == "primary"


# ---------------------------------------------------------------------------
# Multi-topic AND/OR queries
# ---------------------------------------------------------------------------


class TestMultiTopicQuery:
    async def test_or_union(self, client):
        resp = await client.post(
            "/concordance/query",
            json={"topics": ["worship-salah", "ethics-patience"], "operator": "OR"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["operator"] == "OR"
        assert data["total"] > 0
        refs = {v["reference"] for v in data["verses"]}
        # 2:45 is about both prayer and patience; 2:43 is prayer only.
        assert "2:45" in refs

    async def test_and_intersection(self, client):
        resp = await client.post(
            "/concordance/query",
            json={"topics": ["worship-salah", "ethics-patience"], "operator": "AND"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["operator"] == "AND"
        refs = {v["reference"] for v in data["verses"]}
        # Only verses mapped to BOTH topics survive the intersection.
        assert "2:45" in refs
        assert "2:43" not in refs

    async def test_invalid_operator(self, client):
        resp = await client.post(
            "/concordance/query",
            json={"topics": ["tawhid"], "operator": "XOR"},
        )
        assert resp.status_code == 422

    async def test_unknown_topic_404(self, client):
        resp = await client.post("/concordance/query", json={"topics": ["nope"], "operator": "OR"})
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Frequency statistics
# ---------------------------------------------------------------------------


class TestFrequency:
    async def test_frequency_by_surah(self, client):
        resp = await client.get("/concordance/topics/tawhid/frequency")
        assert resp.status_code == 200
        data = resp.json()
        assert data["theme_id"] == "tawhid"
        assert data["total_verses"] > 0
        assert data["surahs_covered"] > 0
        first = data["by_surah"][0]
        assert first["surah"] >= 1
        assert first["verses"] >= 1
        assert first["references"]  # non-empty reference list

    async def test_frequency_unknown_theme(self, client):
        resp = await client.get("/concordance/topics/nope/frequency")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Related topics
# ---------------------------------------------------------------------------


class TestRelatedTopics:
    async def test_related_suggestions(self, client):
        resp = await client.get("/concordance/topics/worship-salah/related")
        assert resp.status_code == 200
        data = resp.json()
        assert data["theme_id"] == "worship-salah"
        # 2:45 maps to both salah and patience, so patience must be suggested.
        ids = [t["id"] for t in data["related"]]
        assert "ethics-patience" in ids
        patience = next(t for t in data["related"] if t["id"] == "ethics-patience")
        assert patience["co_occurrence"] >= 1
        assert "2:45" in patience["shared_verses"]


# ---------------------------------------------------------------------------
# Reverse lookup
# ---------------------------------------------------------------------------


class TestVerseThemes:
    async def test_verse_lookup(self, client):
        resp = await client.get("/concordance/verse/2/255")
        assert resp.status_code == 200
        data = resp.json()
        assert data["reference"] == "2:255"
        ids = [t["theme"]["id"] for t in data["themes"]]
        assert "tawhid" in ids

    async def test_verse_without_mapping(self, client):
        resp = await client.get("/concordance/verse/1/1")
        assert resp.status_code == 200
        assert resp.json()["themes"] == []

    async def test_invalid_surah(self, client):
        resp = await client.get("/concordance/verse/999/1")
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Surah-scoped relevance filtering
# ---------------------------------------------------------------------------


class TestBrowseSurahFilter:
    async def test_browse_filter_by_surah(self, client):
        resp = await client.get("/concordance/topics/worship-salah", params={"surah": 2})
        assert resp.status_code == 200
        verses = resp.json()["verses"]
        assert verses
        assert all(verse["surah"] == 2 for verse in verses)

    async def test_browse_filter_by_surah_with_no_matches(self, client):
        resp = await client.get("/concordance/topics/worship-salah", params={"surah": 114})
        assert resp.status_code == 200
        data = resp.json()
        assert data["verses"] == []
        assert data["total_verses"] > 0  # the theme total is unchanged

    async def test_browse_filter_by_surah_and_context(self, client):
        resp = await client.get(
            "/concordance/topics/history",
            params={"surah": 2, "context_type": "primary"},
        )
        assert resp.status_code == 200
        for verse in resp.json()["verses"]:
            assert verse["surah"] == 2
            assert verse["context_type"] == "primary"

    @pytest.mark.parametrize("surah", [0, 115, -1])
    async def test_browse_rejects_invalid_surah_filter(self, client, surah):
        resp = await client.get("/concordance/topics/worship-salah", params={"surah": surah})
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Versioned provenance and explainable matches
# ---------------------------------------------------------------------------


def _assert_provenance(payload: dict) -> None:
    provenance = payload["provenance"]
    assert provenance["dataset_version"] and provenance["dataset_version"] != "unversioned"
    assert provenance["source"]
    assert provenance["mapping_count"] > 0
    assert provenance["uncertainty_note"]


class TestProvenanceAndExplainability:
    async def test_hierarchy_exposes_provenance(self, client):
        resp = await client.get("/concordance/topics")
        assert resp.status_code == 200
        _assert_provenance(resp.json())

    async def test_search_exposes_provenance_and_explanations(self, client):
        resp = await client.get("/concordance/search", params={"q": "prayer"})
        assert resp.status_code == 200
        data = resp.json()
        _assert_provenance(data)
        verse = next(v for theme in data["themes"] for v in theme.get("verses", []))
        assert verse["theme_id"]
        assert verse["relevance_label"] in {"high", "moderate", "low"}
        assert "relevance" in verse["explanation"]

    async def test_browse_exposes_provenance_and_explanations(self, client):
        resp = await client.get("/concordance/topics/worship-salah")
        assert resp.status_code == 200
        data = resp.json()
        _assert_provenance(data)
        verse = data["verses"][0]
        assert verse["theme_id"] == "worship-salah"
        assert verse["relevance_label"] in {"high", "moderate", "low"}
        assert "worship-salah" in verse["explanation"]
        assert str(verse["relevance_score"]) in verse["explanation"] or "relevance" in verse["explanation"]

    async def test_multi_topic_exposes_provenance_and_explanations(self, client):
        resp = await client.post(
            "/concordance/query",
            json={"topics": ["worship-salah", "ethics-patience"], "operator": "OR", "limit": 5},
        )
        assert resp.status_code == 200
        data = resp.json()
        _assert_provenance(data)
        assert data["verses"]
        assert all(verse["explanation"] and verse["theme_id"] for verse in data["verses"])

    async def test_frequency_related_and_verse_expose_provenance(self, client):
        for path in (
            "/concordance/topics/worship-salah/frequency",
            "/concordance/topics/worship-salah/related",
            "/concordance/verse/2/255",
        ):
            resp = await client.get(path)
            assert resp.status_code == 200, path
            _assert_provenance(resp.json())
