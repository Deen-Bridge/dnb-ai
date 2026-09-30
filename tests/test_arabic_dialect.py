"""Offline tests for the Arabic Dialect Support subsystem (#136).

No secrets and no network: dialect detection and MSA normalization are
deterministic marker-based engines, and the app is exercised through httpx's
ASGI transport.
"""

import os

import pytest
from httpx import ASGITransport, AsyncClient

os.environ.setdefault("GEMINI_API_KEY", "test-key")

import main  # noqa: E402
from arabic_dialect import (  # noqa: E402
    ArabicDialect,
    analyze_arabic_dialect,
    dialect_classifier,
    extract_terms_from_text,
    terminology_db,
)


@pytest.fixture()
async def client():
    transport = ASGITransport(app=main.app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


class TestClassification:
    def test_msa_text_classified_as_msa(self):
        profile = dialect_classifier.classify_dialect("الحمد لله رب العالمين")
        assert profile.primary_dialect is ArabicDialect.MSA
        assert profile.is_msa is True
        assert profile.confidence >= 0.9

    def test_egyptian_detected(self):
        profile = dialect_classifier.classify_dialect("عايز أسأل عن حكم الصلاة")
        assert profile.primary_dialect is ArabicDialect.EGYPTIAN
        assert profile.is_msa is False
        assert profile.confidence >= 0.6

    def test_gulf_detected(self):
        profile = dialect_classifier.classify_dialect("شنو حكم الزكاة في رمضان؟")
        assert profile.primary_dialect is ArabicDialect.GULF

    def test_levantine_detected(self):
        profile = dialect_classifier.classify_dialect("شو حكم الصيام؟ بدي أعرف")
        assert profile.primary_dialect is ArabicDialect.LEVANTINE

    def test_confidence_scales_with_markers(self):
        light = dialect_classifier.classify_dialect("عايز أعرف")
        heavy = dialect_classifier.classify_dialect("عايز أعرف إزيك دلوقتي امبارح كده")
        assert heavy.confidence > light.confidence


# ---------------------------------------------------------------------------
# Normalization to MSA
# ---------------------------------------------------------------------------


class TestNormalization:
    def test_egyptian_to_msa(self):
        normalized, replaced = dialect_classifier.normalize_to_msa("عايز أسأل عن الحكم")
        assert replaced.get("عايز") == "أريد"
        assert "أريد" in normalized

    def test_gulf_to_msa(self):
        normalized, replaced = dialect_classifier.normalize_to_msa("وين الزكاة؟")
        assert replaced.get("وين") == "أين"
        assert "أين" in normalized

    def test_levantine_to_msa(self):
        normalized, replaced = dialect_classifier.normalize_to_msa("بدي أعرف الحكم")
        assert replaced.get("بدي") == "أريد"
        assert "أريد" in normalized

    def test_msa_text_unchanged(self):
        normalized, replaced = dialect_classifier.normalize_to_msa("الحمد لله رب العالمين")
        assert replaced == {}
        assert normalized == "الحمد لله رب العالمين"


# ---------------------------------------------------------------------------
# Terminology lexicon
# ---------------------------------------------------------------------------


class TestTerminology:
    def test_lexicon_has_terms_for_all_regional_dialects(self):
        for dialect in (ArabicDialect.EGYPTIAN, ArabicDialect.GULF, ArabicDialect.LEVANTINE):
            terms = [t for t in terminology_db.terms if t.dialect is dialect]
            assert terms, f"no terms for {dialect}"

    def test_lookup_by_term(self):
        term = terminology_db.lookup("عايز")
        assert term is not None
        assert term.dialect is ArabicDialect.EGYPTIAN
        assert term.msa_equivalent == "أريد"

    def test_extract_terms_from_text(self):
        found = extract_terms_from_text("عايز أعرف عن الزكاة")
        assert any(t.id == "eg-3ayez" for t in found)

    def test_search_terms_by_dialect(self):
        results = terminology_db.search_terms(dialect=ArabicDialect.GULF)
        assert results
        assert all(t.dialect is ArabicDialect.GULF for t in results)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


class TestEndpoints:
    async def test_analyze_endpoint(self, client):
        resp = await client.post(
            "/arabic-dialect/analyze",
            json={"text": "عايز أسأل عن حكم الصلاة"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["dialect"]["primary_dialect"] == "egyptian"
        assert data["normalized_msa"]
        assert data["detected_terms"]
        assert data["segments"]

    async def test_analyze_msa(self, client):
        resp = await client.post(
            "/arabic-dialect/analyze",
            json={"text": "الحمد لله رب العالمين"},
        )
        assert resp.status_code == 200
        assert resp.json()["dialect"]["primary_dialect"] == "msa"
        assert resp.json()["dialect"]["is_msa"] is True

    async def test_normalize_endpoint(self, client):
        resp = await client.post(
            "/arabic-dialect/normalize",
            json={"text": "بدي أعرف الحكم"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["replaced_terms"].get("بدي") == "أريد"
        assert "أريد" in data["normalized_msa"]

    async def test_list_dialects(self, client):
        resp = await client.get("/arabic-dialect/dialects")
        assert resp.status_code == 200
        data = resp.json()
        supported = set(data["supported_dialects"])
        assert {"egyptian", "gulf", "levantine", "msa"} <= supported
        assert "egyptian" in data["dialect_profiles"]

    async def test_search_terms_endpoint(self, client):
        resp = await client.get("/arabic-dialect/terms", params={"query": "عايز", "dialect": "egyptian"})
        assert resp.status_code == 200
        terms = resp.json()
        assert terms
        assert all(t["dialect"] == "egyptian" for t in terms)

    async def test_get_term_by_id(self, client):
        resp = await client.get("/arabic-dialect/terms/eg-3ayez")
        assert resp.status_code == 200
        assert resp.json()["msa_equivalent"] == "أريد"

    async def test_get_unknown_term_404(self, client):
        resp = await client.get("/arabic-dialect/terms/nope")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Facade helper
# ---------------------------------------------------------------------------


class TestFacade:
    def test_analyze_arabic_dialect_helper(self):
        result = analyze_arabic_dialect("شنو حكم الزكاة؟")
        assert result.dialect.primary_dialect is ArabicDialect.GULF
        assert result.normalized_msa
