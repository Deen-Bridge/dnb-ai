"""Tests for advanced citation verification (#132).

Covers format validation, completeness checking, cross-reference validation,
volume/page/edition verification, and citation drift detection.
"""

from citation_verification import CitationGraph, verify_citations
from citations import (
    CitationExtraction,
    HadithCitation,
    QuranCitation,
    ScholarlyReference,
    parse_citations,
)


def extraction_with(*citations) -> CitationExtraction:
    return CitationExtraction(citations=list(citations), attempted=len(citations))


class TestQuranVerification:
    def test_valid_quran_is_compliant_and_cross_referenced(self):
        citation = QuranCitation(surah=2, ayah_start=153, surah_name="Al-Baqarah")
        result = verify_citations(extraction_with(citation))
        assert result.compliant
        assert result.format_compliance_rate == 1.0
        assert result.cross_reference_rate == 1.0
        assert result.findings[0].cross_referenced is True
        assert result.findings[0].issues == []


class TestHadithVerification:
    def test_hadith_with_number_is_complete_and_cross_referenced(self):
        citation = HadithCitation(collection="Sahih al-Bukhari", number="1")
        result = verify_citations(extraction_with(citation))
        assert result.compliant
        assert result.findings[0].complete is True
        assert result.findings[0].cross_referenced is True

    def test_hadith_without_number_is_incomplete(self):
        citation = HadithCitation(collection="Sahih al-Bukhari")
        result = verify_citations(extraction_with(citation))
        assert result.compliant is False
        assert result.findings[0].complete is False
        assert any("missing a hadith number" in issue for issue in result.findings[0].issues)


class TestScholarlyVerification:
    def test_complete_scholarly_reference_is_compliant(self):
        citation = ScholarlyReference(
            work="Ihya Ulum al-Din",
            author="Al-Ghazali",
            detail="Book of Sincerity",
        )
        result = verify_citations(extraction_with(citation))
        assert result.compliant
        assert result.findings[0].complete is True
        assert result.findings[0].format_compliant is True

    def test_missing_author_is_incomplete(self):
        citation = ScholarlyReference(work="Some Work")
        result = verify_citations(extraction_with(citation))
        assert result.compliant is False
        assert result.findings[0].complete is False
        assert any("missing 'author'" in issue for issue in result.findings[0].issues)

    def test_malformed_volume_is_flagged(self):
        citation = ScholarlyReference(
            work="Ihya Ulum al-Din",
            author="Al-Ghazali",
            volume="vol. one",  # not numeric
        )
        result = verify_citations(extraction_with(citation))
        assert result.findings[0].volume_verified is False
        assert result.findings[0].format_compliant is False
        assert any("malformed volume" in issue for issue in result.findings[0].issues)

    def test_malformed_page_range_is_flagged(self):
        citation = ScholarlyReference(
            work="Ihya Ulum al-Din",
            author="Al-Ghazali",
            pages="pages 1-2",  # not a bare numeric range
        )
        result = verify_citations(extraction_with(citation))
        assert result.findings[0].page_verified is False
        assert result.findings[0].format_compliant is False

    def test_valid_volume_page_edition_pass(self):
        citation = ScholarlyReference(
            work="Ihya Ulum al-Din",
            author="Al-Ghazali",
            volume="1",
            pages="1-400",
            edition="1",
        )
        result = verify_citations(extraction_with(citation))
        assert result.findings[0].volume_verified is True
        assert result.findings[0].page_verified is True
        assert result.findings[0].edition_verified is True
        assert result.findings[0].format_compliant is True

    def test_known_work_is_cross_referenced(self):
        citation = ScholarlyReference(
            work="Ihya Ulum al-Din",
            author="Al-Ghazali",
        )
        result = verify_citations(extraction_with(citation))
        assert result.findings[0].cross_referenced is True

    def test_unknown_work_is_not_cross_referenced_but_not_failed(self):
        citation = ScholarlyReference(
            work="An Obscure Work",
            author="Someone",
        )
        result = verify_citations(extraction_with(citation))
        assert result.findings[0].cross_referenced is False
        assert result.findings[0].format_compliant is True

    def test_drift_detected_on_volume_mismatch(self):
        citation = ScholarlyReference(
            work="Ihya Ulum al-Din",
            author="Al-Ghazali",
            volume="9",  # known editions only have volumes 1 and 2
        )
        result = verify_citations(extraction_with(citation))
        assert result.findings[0].drift_detected is True
        assert result.drift_count == 1
        assert any("does not match known edition" in issue for issue in result.findings[0].issues)


class TestAggregates:
    def test_empty_extraction_returns_default_verification(self):
        result = verify_citations(CitationExtraction())
        assert result.total_citations == 0
        assert result.compliant is True
        assert result.format_compliance_rate == 1.0

    def test_aggregate_rates_are_computed(self):
        good = QuranCitation(surah=2, ayah_start=153, surah_name="Al-Baqarah")
        bad = ScholarlyReference(work="Only a Title")  # missing author
        result = verify_citations(extraction_with(good, bad))
        assert result.total_citations == 2
        assert result.format_compliance_rate == 1.0  # both format-compliant
        assert result.completeness_rate == 0.5  # one of two complete
        assert result.compliant is False


class TestCitationGraph:
    def test_graph_tracks_citing_works(self):
        graph = CitationGraph()
        graph.add_edge("answer", "Ihya Ulum al-Din")
        graph.add_edge("answer", "Sahih al-Bukhari")
        assert graph.citing("Ihya Ulum al-Din") == {"answer"}
        assert graph.citing("Sahih al-Bukhari") == {"answer"}
        assert len(graph) == 2


class TestParseIntegration:
    def test_parse_populates_verification(self):
        extraction = parse_citations('{"citations": [{"type": "quran", "surah": 2, "ayah_start": 153}]}')
        assert extraction.verification
        assert extraction.verification["total_citations"] == 1
        assert extraction.verification["compliant"] is True

    def test_parse_verification_survives_bad_input(self):
        extraction = parse_citations("not json")
        assert extraction.citations == []
        # Verification runs on an empty extraction and returns defaults.
        assert extraction.verification["total_citations"] == 0

    def test_parse_scholarly_with_bibliographic_fields(self):
        extraction = parse_citations(
            '{"citations": [{"type": "scholarly", "work": "Ihya Ulum al-Din", '
            '"author": "Al-Ghazali", "volume": "1", "pages": "1-400", '
            '"edition": "1", "publisher": "Dar al-Ma\'arif"}]}'
        )
        assert len(extraction.citations) == 1
        citation = extraction.citations[0]
        assert isinstance(citation, ScholarlyReference)
        assert citation.volume == "1"
        assert citation.pages == "1-400"
        assert citation.edition == "1"
        assert citation.publisher == "Dar al-Ma'arif"
