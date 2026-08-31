"""Tests for the enriched asbab al-nuzul knowledge base (#160)."""

from history import (
    ASBAB_AL_NUZUL,
    build_historical_context,
    filter_asbab,
    get_asbab_for_verse,
    list_all_asbab_periods,
)


class TestAsbabDatabase:
    """Verify the enriched database structure."""

    def test_minimum_entry_count(self):
        assert len(ASBAB_AL_NUZUL) >= 20

    def test_all_entries_have_required_fields(self):
        for ref, entry in ASBAB_AL_NUZUL.items():
            assert "summary" in entry, f"{ref} missing summary"
            assert "attribution" in entry, f"{ref} missing attribution"
            assert "period" in entry, f"{ref} missing period"
            assert "narrations" in entry, f"{ref} missing narrations"
            assert "suggested_interpretation" in entry, f"{ref} missing suggested_interpretation"

    def test_all_periods_are_valid(self):
        valid = {"Makki", "Madani", "unknown"}
        for ref, entry in ASBAB_AL_NUZUL.items():
            assert entry["period"] in valid, f"{ref} has invalid period: {entry['period']}"

    def test_narrations_are_lists_of_dicts(self):
        for ref, entry in ASBAB_AL_NUZUL.items():
            assert isinstance(entry["narrations"], list), f"{ref} narrations not a list"
            for n in entry["narrations"]:
                assert isinstance(n, dict), f"{ref} narration not a dict"
                assert "scholar" in n, f"{ref} narration missing scholar"
                assert "text" in n, f"{ref} narration missing text"

    def test_alcohol_prohibition_sequence(self):
        """The three alcohol verses form a known tadarruj sequence."""
        refs = ["2:219", "4:43", "5:90"]
        for r in refs:
            assert r in ASBAB_AL_NUZUL
            assert ASBAB_AL_NUZUL[r]["period"] in {"Makki", "Madani"}


class TestGetAsbabForVerse:
    """Verse-specific retrieval."""

    def test_known_verse(self):
        entry = get_asbab_for_verse("2:256")
        assert entry is not None
        assert entry.reference == "2:256"
        assert "compulsion" in entry.summary.lower() or "compulsion" in entry.summary

    def test_unknown_verse(self):
        assert get_asbab_for_verse("999:999") is None

    def test_strips_whitespace(self):
        entry = get_asbab_for_verse("  5:90  ")
        assert entry is not None
        assert entry.reference == "5:90"

    def test_narrations_populated(self):
        entry = get_asbab_for_verse("2:256")
        assert entry is not None
        assert len(entry.narrations) >= 1
        assert entry.narrations[0].scholar

    def test_period_populated(self):
        entry = get_asbab_for_verse("17:1")
        assert entry is not None
        assert entry.period == "Makki"


class TestFilterAsbab:
    """Filtering helpers."""

    def test_filter_makki(self):
        makki = filter_asbab(period="Makki")
        assert len(makki) >= 5
        for e in makki:
            assert e.period == "Makki"

    def test_filter_madani(self):
        madani = filter_asbab(period="Madani")
        assert len(madani) >= 5
        for e in madani:
            assert e.period == "Madani"

    def test_filter_has_narrations(self):
        with_narrations = filter_asbab(has_narrations=True)
        assert len(with_narrations) >= 5
        for e in with_narrations:
            assert len(e.narrations) >= 1

    def test_filter_no_results(self):
        result = filter_asbab(period="Pre-Islamic")
        assert result == []


class TestListAllAsbabPeriods:
    """Period summary endpoint."""

    def test_returns_periods(self):
        periods = list_all_asbab_periods()
        assert "Makki" in periods
        assert "Madani" in periods
        assert len(periods["Makki"]) >= 5
        assert len(periods["Madani"]) >= 5

    def test_references_are_sorted(self):
        periods = list_all_asbab_periods()
        for refs in periods.values():
            assert refs == sorted(refs)


class TestBuildHistoricalContext:
    """Relevance detection picks up new entries."""

    def test_picks_up_verse_reference(self):
        ctx = build_historical_context("What about 2:256?")
        assert ctx.has_context is True
        refs = [e.reference for e in ctx.asbab]
        assert "2:256" in refs

    def test_picks_up_multiple_verses(self):
        ctx = build_historical_context("Compare 2:219 and 5:90")
        refs = [e.reference for e in ctx.asbab]
        assert "2:219" in refs
        assert "5:90" in refs

    def test_no_match_for_irrelevant_text(self):
        ctx = build_historical_context("What is the weather like today?")
        assert ctx.has_context is False
        assert len(ctx.asbab) == 0

    def test_context_block_contains_summary(self):
        ctx = build_historical_context("Tell me about 9:5")
        assert "sword" in ctx.context_block.lower() or "treat" in ctx.context_block.lower()
