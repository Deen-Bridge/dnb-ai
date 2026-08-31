"""Tests for query expansion module."""

import pytest

from query_expansion import (
    ExpandedQuery,
    expand_query,
    expand_query_for_search,
    find_matching_term,
    get_arabic_equivalent,
    get_english_equivalent,
    get_related_concepts,
    normalize_transliteration,
)


class TestNormalizeTransliteration:
    """Tests for transliteration normalization."""

    def test_lowercase_conversion(self):
        assert normalize_transliteration("SALAH") == "salah"

    def test_multiple_a_normalization(self):
        assert normalize_transliteration("salaat") == "salat"
        assert normalize_transliteration("salaah") == "salah"

    def test_ee_to_i_conversion(self):
        assert normalize_transliteration("tajweed") == "tajwid"

    def test_oo_to_u_conversion(self):
        assert normalize_transliteration("wudoo") == "wudu"

    def test_apostrophe_removal(self):
        assert normalize_transliteration("qur'an") == "quran"

    def test_hyphen_removal(self):
        assert normalize_transliteration("al-quran") == "alquran"

    def test_whitespace_stripping(self):
        assert normalize_transliteration("  salah  ") == "salah"


class TestFindMatchingTerm:
    """Tests for term matching in knowledge graph."""

    def test_direct_match(self):
        assert find_matching_term("salah") == "salah"
        assert find_matching_term("quran") == "quran"

    def test_transliteration_match(self):
        assert find_matching_term("salaat") == "salah"
        assert find_matching_term("wudhu") == "wudu"

    def test_english_match(self):
        assert find_matching_term("prayer") == "prayer"
        assert find_matching_term("fasting") == "fasting"

    def test_case_insensitive(self):
        assert find_matching_term("SALAH") == "salah"
        assert find_matching_term("Quran") == "quran"

    def test_no_match(self):
        assert find_matching_term("unknown_term") is None
        assert find_matching_term("xyz123") is None


class TestExpandQuery:
    """Tests for query expansion functionality."""

    def test_basic_expansion(self):
        result = expand_query("salah")
        assert result.original == "salah"
        assert len(result.arabic_terms) > 0
        assert "صلاة" in result.arabic_terms

    def test_english_expansion(self):
        result = expand_query("wudu")
        assert "ablution" in result.english_terms

    def test_transliteration_expansion(self):
        result = expand_query("quran")
        assert any("qur" in t.lower() for t in result.expanded_terms)

    def test_related_concepts_included(self):
        result = expand_query("salah", include_related=True)
        assert "wudu" in result.related_concepts or "qibla" in result.related_concepts

    def test_related_concepts_excluded(self):
        result = expand_query("salah", include_related=False)
        assert len(result.related_concepts) == 0

    def test_multi_word_query(self):
        result = expand_query("how to perform wudu")
        assert len(result.arabic_terms) > 0
        assert "وضوء" in result.arabic_terms or "الوضوء" in result.arabic_terms

    def test_all_terms_includes_original(self):
        result = expand_query("salah")
        all_terms = result.all_terms()
        assert "salah" in all_terms

    def test_all_terms_deduplication(self):
        result = expand_query("prayer salah")
        all_terms = result.all_terms()
        # Should not have duplicates
        assert len(all_terms) == len(set(all_terms))

    def test_max_expansions_limit(self):
        result = expand_query("quran", max_expansions=3)
        assert len(result.expanded_terms) <= 3
        assert len(result.arabic_terms) <= 3


class TestExpandQueryForSearch:
    """Tests for the search-optimized expansion function."""

    def test_returns_list(self):
        result = expand_query_for_search("salah")
        assert isinstance(result, list)

    def test_includes_original(self):
        result = expand_query_for_search("wudu")
        assert "wudu" in result

    def test_default_no_related(self):
        result = expand_query_for_search("salah")
        # Without related concepts, should be more focused
        assert len(result) < 20

    def test_with_related(self):
        result = expand_query_for_search("salah", include_related=True)
        # With related concepts, should have more terms
        assert len(result) > 1


class TestGetArabicEquivalent:
    """Tests for Arabic equivalent lookup."""

    def test_known_term(self):
        result = get_arabic_equivalent("salah")
        assert "صلاة" in result

    def test_transliteration_input(self):
        result = get_arabic_equivalent("wudhu")
        assert "وضوء" in result or "الوضوء" in result

    def test_unknown_term(self):
        result = get_arabic_equivalent("unknown_xyz")
        assert result == []


class TestGetEnglishEquivalent:
    """Tests for English equivalent lookup."""

    def test_known_term(self):
        result = get_english_equivalent("wudu")
        assert "ablution" in result

    def test_arabic_input(self):
        # Note: Arabic matching requires exact match
        result = get_english_equivalent("salah")
        assert "prayer" in result or len(result) > 0

    def test_unknown_term(self):
        result = get_english_equivalent("unknown_xyz")
        assert result == []


class TestGetRelatedConcepts:
    """Tests for related concepts lookup."""

    def test_known_term(self):
        result = get_related_concepts("salah")
        assert len(result) > 0
        assert any(
            term in result for term in ["wudu", "qibla", "rakah", "sujud", "ruku"]
        )

    def test_unknown_term(self):
        result = get_related_concepts("unknown_xyz")
        assert result == []


class TestExpandedQueryDataclass:
    """Tests for ExpandedQuery dataclass."""

    def test_default_values(self):
        eq = ExpandedQuery(original="test")
        assert eq.original == "test"
        assert eq.expanded_terms == []
        assert eq.arabic_terms == []
        assert eq.english_terms == []
        assert eq.related_concepts == []

    def test_all_terms_aggregation(self):
        eq = ExpandedQuery(
            original="test",
            expanded_terms=["a", "b"],
            arabic_terms=["ا"],
            english_terms=["english"],
            related_concepts=["related"],
        )
        all_terms = eq.all_terms()
        assert "test" in all_terms
        assert "a" in all_terms
        assert "ا" in all_terms
        assert "english" in all_terms
        assert "related" in all_terms


class TestIslamicTermsCoverage:
    """Tests for comprehensive Islamic terminology coverage."""

    @pytest.mark.parametrize(
        "term",
        [
            "salah",
            "wudu",
            "sawm",
            "hajj",
            "zakat",
            "quran",
            "hadith",
            "fiqh",
            "tawhid",
            "dua",
            "dhikr",
        ],
    )
    def test_core_pillars_and_concepts(self, term):
        result = expand_query(term)
        assert len(result.arabic_terms) > 0, f"{term} should have Arabic equivalents"

    @pytest.mark.parametrize(
        "term",
        ["hanafi", "maliki", "shafii", "hanbali"],
    )
    def test_madhhab_coverage(self, term):
        result = expand_query(term)
        assert "madhhab" in result.related_concepts or len(result.arabic_terms) > 0

    def test_ramadan_expansion(self):
        result = expand_query("ramadan")
        assert "صيام" in result.related_concepts or "sawm" in result.related_concepts or "رمضان" in result.arabic_terms
