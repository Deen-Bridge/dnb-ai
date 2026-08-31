"""Tests for the Fabricated Scholarly Attribution Prevention System (#173).

All offline — no model calls, no GEMINI_API_KEY. Tests hit the real
``validate_scholarly_attribution`` entry point and the FastAPI ``/scholarly-attribution``
endpoints, not helpers in isolation.
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from scholarly_attribution import (
    AttributionVerdict,
    SCHOLARS_DB,
    get_scholars_by_school,
    get_scholars_list,
    validate_single_attribution,
    validate_scholarly_attribution,
)
from scholarly_attribution_api import router as scholarly_attribution_router

# Build a minimal app from the router so tests hit real endpoints
# without importing main.py (which has unrelated dependency issues on this branch).
test_app = FastAPI()
test_app.include_router(scholarly_attribution_router)

client = TestClient(test_app, raise_server_exceptions=False)

# -----------------------------------------------------------------------
# Scholar database
# -----------------------------------------------------------------------


class TestScholarDatabase:
    def test_database_not_empty(self):
        assert len(SCHOLARS_DB) > 10

    def test_get_all_scholars(self):
        scholars = get_scholars_list()
        assert isinstance(scholars, list)
        assert len(scholars) > 10
        assert "id" in scholars[0]

    def test_filter_by_school(self):
        hanafi = get_scholars_by_school("Hanafi")
        assert all(s["school"] == "Hanafi" for s in hanafi)

    def test_unknown_school_returns_empty(self):
        items = get_scholars_by_school("NonexistentSchoolXYZ")
        assert items == []

    def test_all_scholars_have_required_fields(self):
        for scholar in SCHOLARS_DB.values():
            assert scholar.id
            assert scholar.name
            assert scholar.full_name
            assert scholar.era.birth_ce is not None
            assert scholar.era.death_ce is not None
            assert scholar.school
            assert len(scholar.known_positions) > 0


# -----------------------------------------------------------------------
# Core validation — free-text (the real entry point)
# -----------------------------------------------------------------------


class TestAttributionValidationFreeText:
    """Test the main ``validate_scholarly_attribution`` function with real text."""

    def test_clean_text_no_issues(self):
        text = "The five pillars of Islam are Shahada, Salat, Zakat, Sawm, and Hajj."
        result = validate_scholarly_attribution(text)
        assert result.has_fabrication is False
        assert result.should_block is False
        assert result.overall_verdict == AttributionVerdict.VERIFIED

    def test_known_opinion_verified(self):
        text = "Imam Abu Hanifa said that wiping over leather socks is permissible."
        result = validate_scholarly_attribution(text)
        assert result.overall_verdict == AttributionVerdict.VERIFIED
        assert result.has_fabrication is False
        assert any(e.get("action") == "verified" for e in result.audit_trail)

    def test_fabricated_opinion_detected(self):
        text = "Imam Abu Hanifa said that the Quran is created and not eternal."
        result = validate_scholarly_attribution(text)
        assert result.has_fabrication is True
        assert result.overall_verdict in (
            AttributionVerdict.SUSPICIOUS,
            AttributionVerdict.FABRICATED,
        )
        assert result.stats["fabricated_opinions"] >= 1

    def test_scholar_not_in_database_flagged(self):
        text = "According to Sheikh Dr. SomeUnknownPerson, all music is absolutely haram."
        result = validate_scholarly_attribution(text)
        # The scholar is not in the DB, should be flagged as unverifiable
        assert result.stats["unverifiable_scholars"] >= 1

    def test_audit_trail_populated(self):
        text = "Imam Malik said that the practice of people of madinah is authoritative."
        result = validate_scholarly_attribution(text)
        assert len(result.audit_trail) > 0
        assert any(entry.get("action") in ("verified", "flagged") for entry in result.audit_trail)

    def test_stats_computed(self):
        text = "Imam Abu Hanifa said that wiping over leather socks is permissible."
        result = validate_scholarly_attribution(text)
        assert "total_issues" in result.stats
        assert "fabricated_opinions" in result.stats
        assert "anachronisms" in result.stats
        assert "false_consensus" in result.stats


# -----------------------------------------------------------------------
# Anachronism detection
# -----------------------------------------------------------------------


class TestAnachronismDetection:
    def test_modern_concept_flagged_for_classical_scholar(self):
        text = "Imam Abu Hanifa stated that the internet is a useful tool for spreading knowledge."
        result = validate_scholarly_attribution(text)
        assert result.has_fabrication is True
        assert result.stats["anachronisms"] >= 1

    def test_modern_science_flagged(self):
        text = "Imam al-Ghazali argued that evolution is compatible with Islamic beliefs."
        result = validate_scholarly_attribution(text)
        assert result.has_fabrication is True
        assert result.stats["anachronisms"] >= 1


# -----------------------------------------------------------------------
# Consensus validation
# -----------------------------------------------------------------------


class TestConsensusValidation:
    def test_false_consensus_on_debated_topic(self):
        text = "All scholars agree unanimously that music is haram."
        result = validate_scholarly_attribution(text)
        assert result.stats["false_consensus"] >= 1

    def test_no_false_consensus_on_non_debated_topic(self):
        text = "All scholars agree that the Shahada is an obligation."
        result = validate_scholarly_attribution(text)
        assert result.stats["false_consensus"] == 0


# -----------------------------------------------------------------------
# Single-attraction validation
# -----------------------------------------------------------------------


class TestSingleAttributionValidation:
    def test_verified_opinion(self):
        result = validate_single_attribution(
            "Imam Abu Hanifa",
            "wiping over leather socks is permissible",
        )
        assert result["verdict"] == "verified"
        assert result["matched"] is True

    def test_unknown_scholar(self):
        result = validate_single_attribution(
            "Unknown Scholar XYZ",
            "some opinion",
        )
        assert result["verdict"] == "unverifiable"
        assert result["matched"] is False

    def test_fabricated_opinion(self):
        result = validate_single_attribution(
            "Imam Abu Hanifa",
            "the Quran is created and not eternal",
        )
        assert result["verdict"] == "suspicious"

    def test_suspicious_returns_similar_positions(self):
        result = validate_single_attribution(
            "Imam Malik",
            "wiping over leather socks is permissible",
        )
        # This should be verified since Malik permits it too
        assert result["verdict"] in ("verified", "plausible")


# -----------------------------------------------------------------------
# Endpoints — the real paths
# -----------------------------------------------------------------------


class TestEndpoints:
    def test_validate_endpoint_returns_200(self):
        response = client.post(
            "/scholarly-attribution/validate",
            json={"text": "The five pillars of Islam are foundational."},
        )
        assert response.status_code == 200
        data = response.json()
        assert "issues" in data
        assert "has_fabrication" in data
        assert "audit_trail" in data

    def test_validate_endpoint_blocks_fabrication(self):
        response = client.post(
            "/scholarly-attribution/validate",
            json={
                "text": "Imam Abu Hanifa said that the internet is essential for dawah."
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["has_fabrication"] is True

    def test_validate_single_endpoint(self):
        response = client.post(
            "/scholarly-attribution/validate-single",
            json={
                "scholar_name": "Imam Abu Hanifa",
                "opinion": "wiping over leather socks is permissible",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["verdict"] == "verified"
        assert data["matched"] is True

    def test_scholars_list_endpoint(self):
        response = client.get("/scholarly-attribution/scholars")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] > 10
        assert len(data["scholars"]) > 10

    def test_scholars_list_filter_by_school(self):
        response = client.get("/scholarly-attribution/scholars?school=Hanafi")
        assert response.status_code == 200
        data = response.json()
        assert all(s["school"] == "Hanafi" for s in data["scholars"])

    def test_scholar_by_id_endpoint(self):
        response = client.get("/scholarly-attribution/scholars/imam_abu_hanifa")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "imam_abu_hanifa"
        assert data["school"] == "Hanafi"

    def test_scholar_by_id_not_found(self):
        response = client.get("/scholarly-attribution/scholars/nonexistent_xyz")
        assert response.status_code == 404

    def test_validate_empty_text_rejected(self):
        response = client.post(
            "/scholarly-attribution/validate",
            json={"text": ""},
        )
        assert response.status_code == 422


# -----------------------------------------------------------------------
# Edge cases
# -----------------------------------------------------------------------


class TestEdgeCases:
    def test_text_with_no_scholar_names(self):
        text = "Islam is a religion of peace and submission to Allah."
        result = validate_scholarly_attribution(text)
        assert result.has_fabrication is False
        assert result.should_block is False

    def test_multiple_scholars(self):
        text = (
            "Imam Abu Hanifa said that wiping over leather socks is permissible. "
            "Imam al-Shafi'i also said that qunut in fajr prayer is sunnah."
        )
        result = validate_scholarly_attribution(text)
        # Both attributions should be verified
        assert result.has_fabrication is False

    def test_absolutist_language_flagged(self):
        text = (
            "Imam Abu Hanifa absolutely categorically stated that "
            "wiping over leather socks is permissible. "
            "All scholars unanimously agree on this matter."
        )
        result = validate_scholarly_attribution(text)
        assert len(result.issues) > 0
        assert any(
            i.issue_type.value == "flattened_nuance"
            for i in result.issues
        )
