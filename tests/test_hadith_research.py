"""Tests for the Hadith Research Agent — retrieval, isnad analysis, rijal
cross-referencing, grading justification, and variant alignment.

Everything runs offline against the bundled datasets (data/hadith/*.json,
data/hadith/variants.json, data/rijal/narrators.json); no network, no API key.
"""

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

import hadith_research as hr
from hadith import Strength
from hadith_research import (
    analyze_isnad,
    build_variant_match,
    find_narrator,
    find_variants_by_text,
    format_citation,
    justify_grading,
    load_collections,
    load_variants,
    lookup,
    normalize_collection,
    resolve_reference,
    router,
)


def run(coro):
    import asyncio

    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------


class TestSources:
    def test_ten_major_collections_are_bundled(self):
        collections = load_collections()
        assert set(collections) >= {
            "bukhari",
            "muslim",
            "abudawud",
            "tirmidhi",
            "nasai",
            "ibnmajah",
            "malik",
            "nawawi",
            "qudsi",
            "dehlawi",
        }
        assert len(collections) == 10

    def test_every_collection_has_a_name_and_records(self):
        for key, info in load_collections().items():
            assert info["name"], key
            assert info["hadith_count"] > 0, key


# ---------------------------------------------------------------------------
# Reference parsing and retrieval
# ---------------------------------------------------------------------------


class TestResolveReference:
    def test_plain_citation(self):
        assert resolve_reference("Sahih al-Bukhari 1") == ("bukhari", 1, None)

    def test_book_qualified_citation(self):
        assert resolve_reference("Sunan Abu Dawud, Book 13, Hadith 27") == ("abudawud", 27, 13)

    def test_unparseable_returns_none(self):
        assert resolve_reference("some random sentence") is None

    def test_collection_alias_normalization(self):
        assert normalize_collection("Jami' at-Tirmidhi") == "tirmidhi"
        assert normalize_collection("Sahih Muslim") == "muslim"


class TestLookup:
    def test_bukhari_one_is_consensus_sahih(self):
        result = lookup("bukhari", 1)
        assert result is not None
        assert result.grade == Strength.SAHIH.value
        assert result.collection_name == "Sahih al-Bukhari"
        assert result.citation == "Sahih al-Bukhari, no. 1 (Book 1, Hadith 1)"
        assert result.grading.justification
        assert "Scholarly consensus" in result.grading.justification

    def test_weak_hadith_is_graded_daif(self):
        result = lookup("abudawud", 3)
        assert result is not None
        assert result.grade == Strength.DAIF.value
        assert len(result.grading.graders) == 4
        assert all(opinion.grader for opinion in result.grading.graders)

    def test_unknown_number_returns_none(self):
        assert lookup("abudawud", 999999) is None

    def test_unknown_collection_returns_none(self):
        assert lookup("nonexistent", 1) is None

    def test_book_qualified_lookup_resolves(self):
        # Abu Dawud 2201 (intentions) is Book 13, Hadith 27 in the dataset.
        result = lookup("abudawud", 2201)
        assert result is not None
        assert result.hadith_number == 2201
        result_by_book = lookup("abudawud", 27, book=13)
        assert result_by_book is not None
        assert result_by_book.hadith_number == 2201

    def test_disputed_graders_are_surfaced(self):
        result = lookup("abudawud", 1)
        assert result is not None
        assert result.disputed is True
        assert result.grade == Strength.HASAN.value  # weakest-wins
        assert "disagree" in result.grading.justification

    def test_agreeing_graders_are_not_disputed(self):
        result = lookup("abudawud", 3)
        assert result is not None
        assert result.disputed is False

    def test_nawawi_forty_are_consensus_sahih(self):
        result = lookup("nawawi", 1)
        assert result is not None
        assert result.grade == Strength.SAHIH.value
        assert "al-Nawawi" in result.grading.justification

    def test_qudsi_is_consensus_sahih(self):
        result = lookup("qudsi", 1)
        assert result is not None
        assert result.grade == Strength.SAHIH.value

    def test_lookup_reports_variant_membership(self):
        result = lookup("bukhari", 1)
        assert result is not None
        assert any(variant.variant_id == "intentions" for variant in result.variants)
        intentions = next(variant for variant in result.variants if variant.variant_id == "intentions")
        assert len(intentions.references) == 6
        assert all(reference.verified for reference in intentions.references)
        assert all(reference.citation for reference in intentions.references)


# ---------------------------------------------------------------------------
# Grading justification and citations
# ---------------------------------------------------------------------------


class TestGradingAndCitations:
    def test_justify_grading_preserves_each_grader(self):
        record = hr.hadith.get_default_source().get("abudawud", 3)
        assert record is not None
        disputed, explanation = justify_grading(record)
        assert disputed is False
        assert "Al-Albani" in explanation
        assert "Daif" in explanation

    def test_justify_grading_flags_non_marfu_chain(self):
        record = hr.hadith.get_default_source().get("malik", 6)
        assert record is not None
        assert record.chain_type.value == "MAUQUF"
        disputed, explanation = justify_grading(record)
        assert "MAUQUF" in explanation or "mauquf" in explanation.lower() or "Companion" in explanation

    def test_format_citation_academic(self):
        assert format_citation("bukhari", 1) == "Sahih al-Bukhari, no. 1"
        assert (
            format_citation("abudawud", 27, book=13, book_number=27) == "Sunan Abu Dawud, no. 27 (Book 13, Hadith 27)"
        )


# ---------------------------------------------------------------------------
# Rijal knowledge base and isnad analysis
# ---------------------------------------------------------------------------


class TestRijal:
    def test_known_narrator_resolves(self):
        narrator = find_narrator("Abu Hurayrah")
        assert narrator is not None
        assert narrator["reliability"] == "thiqa"
        assert narrator["generation"] == "Sahabah"

    def test_alias_resolves(self):
        assert find_narrator("abu huraira") is not None
        assert find_narrator("Ibn 'Abbas") is not None

    def test_unknown_narrator_returns_none(self):
        assert find_narrator("Someone Not In The Book") is None

    def test_weak_narrator_is_flagged(self):
        narrator = find_narrator("Abdullah ibn Lahi'a")
        assert narrator is not None
        assert narrator["reliability"] == "daif"


class TestIsnadAnalysis:
    def test_golden_chain_is_continuous(self):
        analysis = analyze_isnad(["Malik ibn Anas", "Nafi'", "Abdullah ibn Umar"])
        assert analysis.continuous is True
        assert analysis.flagged is False
        assert all(entry.profile.matched for entry in analysis.narrators)

    def test_reversed_generation_order_is_flagged(self):
        analysis = analyze_isnad(["Abdullah ibn Umar", "Nafi'", "Malik ibn Anas"])
        assert analysis.continuous is False
        assert analysis.flagged is True
        assert any("implausible" in note for note in analysis.notes)

    def test_generation_jump_is_flagged_as_possible_gap(self):
        analysis = analyze_isnad(["Malik ibn Anas", "Abdullah ibn Umar"])
        assert analysis.continuous is False
        assert any("missing intermediate" in note for note in analysis.notes)

    def test_unknown_narrator_prevents_verdict(self):
        analysis = analyze_isnad(["Malik ibn Anas", "A Completely Unknown Man"])
        assert any(not entry.profile.matched for entry in analysis.narrators)
        assert any("unverified" in entry.warnings[0] for entry in analysis.narrators if not entry.profile.matched)

    def test_weak_narrator_warns(self):
        analysis = analyze_isnad(["Ibn Lahi'a", "Abdullah ibn Umar"])
        entry = analysis.narrators[0]
        assert entry.profile.reliability == "daif"
        assert any("do not rely" in warning for warning in entry.warnings)
        assert analysis.flagged is True

    def test_empty_chain_rejected_by_endpoint(self):
        app = FastAPI()
        app.include_router(router)

        async def post():
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                return await client.post("/hadith-research/chain", json={"narrators": []})

        assert run(post()).status_code == 422


# ---------------------------------------------------------------------------
# Variant alignment
# ---------------------------------------------------------------------------


class TestVariants:
    def test_variant_map_is_curated_and_resolvable(self):
        variants = load_variants()
        assert "intentions" in variants
        for variant in variants.values():
            assert variant["references"], variant["id"]
            for ref in variant["references"]:
                assert ref["collection"] in load_collections(), (variant["id"], ref["collection"])

    def test_find_by_text_matches_known_variant(self):
        matches = find_variants_by_text("Actions are but by intentions, and every person will have what they intended")
        assert matches and matches[0].variant_id == "intentions"
        assert matches[0].score is not None and matches[0].score > 0.5

    def test_find_by_text_returns_nothing_for_unrelated_text(self):
        assert find_variants_by_text("The price of rice in the market rose today") == []

    def test_build_variant_match_grades_every_reference(self):
        match = build_variant_match("intentions")
        assert match is not None
        grades = {ref.collection: ref.grade for ref in match.references}
        assert grades["bukhari"] == Strength.SAHIH.value
        assert grades["ibnmajah"] == Strength.SAHIH.value

    def test_qudsi_variant_resolves(self):
        match = build_variant_match("mercy-prevails")
        assert match is not None
        qudsi = next(ref for ref in match.references if ref.collection == "qudsi")
        assert qudsi.verified is True
        assert qudsi.grade == Strength.SAHIH.value

    def test_unknown_variant_id_returns_none(self):
        assert build_variant_match("not-a-variant") is None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


class TestEndpoints:
    @pytest.fixture()
    def app(self):
        app = FastAPI()
        app.include_router(router)
        return app

    def test_sources_endpoint(self, app):
        async def get_sources():
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                return await client.get("/hadith-research/sources")

        response = run(get_sources())
        assert response.status_code == 200
        keys = {item["key"] for item in response.json()}
        assert len(keys) == 10
        assert "bukhari" in keys and "qudsi" in keys

    def test_lookup_endpoint_by_reference(self, app):
        async def post():
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                return await client.post("/hadith-research/lookup", json={"reference": "Sahih al-Bukhari 1"})

        response = run(post())
        assert response.status_code == 200
        payload = response.json()
        assert payload["grade"] == "SAHIH"
        assert payload["citation"] == "Sahih al-Bukhari, no. 1 (Book 1, Hadith 1)"
        assert any(v["variant_id"] == "intentions" for v in payload["variants"])

    def test_lookup_endpoint_by_collection_and_number(self, app):
        async def post():
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                return await client.post("/hadith-research/lookup", json={"collection": "abudawud", "number": 3})

        response = run(post())
        assert response.status_code == 200
        assert response.json()["grade"] == "DAIF"

    def test_lookup_endpoint_unknown_returns_404(self, app):
        async def post():
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                return await client.post("/hadith-research/lookup", json={"collection": "bukhari", "number": 999999})

        response = run(post())
        assert response.status_code == 404

    def test_lookup_endpoint_invalid_request_returns_400(self, app):
        async def post():
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                return await client.post("/hadith-research/lookup", json={"reference": "not a citation"})

        response = run(post())
        assert response.status_code == 400

    def test_chain_endpoint(self, app):
        async def post():
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                return await client.post(
                    "/hadith-research/chain",
                    json={"narrators": ["Malik ibn Anas", "Nafi'", "Abdullah ibn Umar"]},
                )

        response = run(post())
        assert response.status_code == 200
        assert response.json()["continuous"] is True

    def test_variants_endpoint_by_reference(self, app):
        async def post():
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                return await client.post("/hadith-research/variants", json={"collection": "bukhari", "number": 1})

        response = run(post())
        assert response.status_code == 200
        payload = response.json()
        assert payload["matches"][0]["variant_id"] == "intentions"
        assert payload["disclaimer"]

    def test_variants_endpoint_by_text(self, app):
        async def post():
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                return await client.post(
                    "/hadith-research/variants",
                    json={"text": "none of you believes until he loves for his brother"},
                )

        response = run(post())
        assert response.status_code == 200
        assert any(match["variant_id"] == "love-for-brother" for match in response.json()["matches"])

    def test_variants_endpoint_rejects_empty_request(self, app):
        async def post():
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                return await client.post("/hadith-research/variants", json={})

        response = run(post())
        assert response.status_code == 400
