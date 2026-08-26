"""Tests for thematic Quran retrieval system (#144)."""

import pytest

from thematic_quran import (
    MAIN_THEMES,
    RevelationPeriod,
    ThematicRetriever,
    ThematicTaxonomy,
    Theme,
    ThemeVerseStore,
    VerseThemeMapping,
)


class TestTheme:
    """Tests for Theme dataclass."""

    def test_theme_creation(self):
        theme = Theme(
            id="test",
            name="Test Theme",
            name_arabic="اختبار",
            description="A test theme",
        )
        assert theme.id == "test"
        assert theme.name == "Test Theme"
        assert theme.level == 0
        assert theme.keywords == []

    def test_theme_to_dict(self):
        theme = Theme(
            id="test",
            name="Test Theme",
            name_arabic="اختبار",
            description="A test theme",
            parent_id="parent",
            level=1,
            keywords=["test", "example"],
        )
        d = theme.to_dict()
        assert d["id"] == "test"
        assert d["parent_id"] == "parent"
        assert d["level"] == 1
        assert d["keywords"] == ["test", "example"]


class TestThematicTaxonomy:
    """Tests for ThematicTaxonomy."""

    def test_default_taxonomy_loaded(self):
        taxonomy = ThematicTaxonomy()
        main_themes = taxonomy.get_main_themes()
        assert len(main_themes) == len(MAIN_THEMES)

    def test_get_theme(self):
        taxonomy = ThematicTaxonomy()
        theme = taxonomy.get_theme("tawhid")
        assert theme is not None
        assert theme.name == "Monotheism (Tawhid)"
        assert theme.name_arabic == "التوحيد"

    def test_get_nonexistent_theme(self):
        taxonomy = ThematicTaxonomy()
        theme = taxonomy.get_theme("nonexistent")
        assert theme is None

    def test_get_children(self):
        taxonomy = ThematicTaxonomy()
        children = taxonomy.get_children("tawhid")
        assert len(children) > 0
        for child in children:
            assert child.parent_id == "tawhid"

    def test_get_ancestors(self):
        taxonomy = ThematicTaxonomy()
        ancestors = taxonomy.get_ancestors("tawhid-rububiyyah")
        assert len(ancestors) == 1
        assert ancestors[0].id == "tawhid"

    def test_search_themes_by_name(self):
        taxonomy = ThematicTaxonomy()
        results = taxonomy.search_themes("monotheism")
        assert len(results) >= 1
        assert any(t.id == "tawhid" for t in results)

    def test_search_themes_by_keyword(self):
        taxonomy = ThematicTaxonomy()
        results = taxonomy.search_themes("prayer")
        assert len(results) >= 1
        assert any("prayer" in t.keywords for t in results)

    def test_search_themes_arabic(self):
        taxonomy = ThematicTaxonomy()
        results = taxonomy.search_themes("التوحيد")
        assert len(results) >= 1
        assert any(t.id == "tawhid" for t in results)

    def test_add_custom_theme(self):
        taxonomy = ThematicTaxonomy()
        custom = Theme(
            id="custom-theme",
            name="Custom Theme",
            name_arabic="مخصص",
            description="A custom theme",
            parent_id="ethics",
            level=2,
        )
        taxonomy.add_theme(custom)
        retrieved = taxonomy.get_theme("custom-theme")
        assert retrieved is not None
        assert retrieved.parent_id == "ethics"


class TestThemeVerseStore:
    """Tests for ThemeVerseStore."""

    def test_add_mapping(self):
        store = ThemeVerseStore(data_file="/tmp/test_verses.json")
        mapping = VerseThemeMapping(
            surah=2,
            ayah=255,
            theme_id="tawhid",
            relevance_score=1.0,
            annotation="Ayat al-Kursi",
        )
        store.add_mapping(mapping)

        # Retrieve by verse
        themes = store.get_themes_for_verse(2, 255)
        assert len(themes) == 1
        assert themes[0].theme_id == "tawhid"

        # Retrieve by theme
        verses = store.get_verses_for_theme("tawhid")
        assert len(verses) == 1
        assert verses[0].surah == 2
        assert verses[0].ayah == 255

    def test_filter_by_relevance(self):
        store = ThemeVerseStore(data_file="/tmp/test_verses2.json")
        store.add_mapping(VerseThemeMapping(surah=1, ayah=1, theme_id="tawhid", relevance_score=1.0))
        store.add_mapping(VerseThemeMapping(surah=1, ayah=2, theme_id="tawhid", relevance_score=0.5))
        store.add_mapping(VerseThemeMapping(surah=1, ayah=3, theme_id="tawhid", relevance_score=0.3))

        # High relevance only
        high = store.get_verses_for_theme("tawhid", min_relevance=0.8)
        assert len(high) == 1

        # Medium relevance
        medium = store.get_verses_for_theme("tawhid", min_relevance=0.4)
        assert len(medium) == 2

    def test_filter_by_context_type(self):
        store = ThemeVerseStore(data_file="/tmp/test_verses3.json")
        store.add_mapping(VerseThemeMapping(surah=1, ayah=1, theme_id="tawhid", context_type="primary"))
        store.add_mapping(VerseThemeMapping(surah=1, ayah=2, theme_id="tawhid", context_type="secondary"))

        primary = store.get_verses_for_theme("tawhid", context_type="primary")
        assert len(primary) == 1
        assert primary[0].context_type == "primary"


class TestThematicRetriever:
    """Tests for ThematicRetriever."""

    @pytest.fixture
    def retriever(self):
        taxonomy = ThematicTaxonomy()
        store = ThemeVerseStore(data_file="/tmp/test_retriever.json")
        # Add some test mappings
        store.add_mapping(
            VerseThemeMapping(
                surah=2,
                ayah=255,
                theme_id="tawhid",
                relevance_score=1.0,
                annotation="Ayat al-Kursi - supreme verse about Allah's attributes",
            )
        )
        store.add_mapping(
            VerseThemeMapping(
                surah=112,
                ayah=1,
                theme_id="tawhid",
                relevance_score=1.0,
                annotation="Surah Al-Ikhlas - pure monotheism",
            )
        )
        store.add_mapping(VerseThemeMapping(surah=2, ayah=255, theme_id="tawhid-asma-sifat", relevance_score=0.9))
        return ThematicRetriever(taxonomy, store)

    def test_get_theme_hierarchy(self, retriever):
        hierarchy = retriever.get_theme_hierarchy()
        assert "main_themes" in hierarchy
        assert len(hierarchy["main_themes"]) == len(MAIN_THEMES)
        # Check that children are included
        tawhid = next(t for t in hierarchy["main_themes"] if t["id"] == "tawhid")
        assert "children" in tawhid
        assert len(tawhid["children"]) > 0

    def test_browse_theme(self, retriever):
        result = retriever.browse_theme("tawhid")
        assert "theme" in result
        assert result["theme"]["id"] == "tawhid"
        assert "verses" in result
        assert len(result["verses"]) == 2
        assert "children" in result
        assert "related" in result

    def test_browse_nonexistent_theme(self, retriever):
        result = retriever.browse_theme("nonexistent")
        assert "error" in result

    def test_search_themes(self, retriever):
        results = retriever.search_themes("worship")
        assert len(results) >= 1
        for r in results:
            assert "verse_count" in r

    def test_get_verse_themes(self, retriever):
        themes = retriever.get_verse_themes(2, 255)
        assert len(themes) >= 1
        theme_ids = [t["theme"]["id"] for t in themes]
        assert "tawhid" in theme_ids

    def test_get_theme_cooccurrence(self, retriever):
        cooccur = retriever.get_theme_cooccurrence("tawhid")
        # tawhid and tawhid-asma-sifat share verse 2:255
        assert "tawhid-asma-sifat" in cooccur

    def test_get_chronological_distribution(self, retriever):
        dist = retriever.get_chronological_distribution("tawhid")
        assert dist["theme_id"] == "tawhid"
        assert dist["total"] == 2
        # Surah 2 is Medinan, Surah 112 is Meccan
        assert dist["medinan"] >= 1
        assert dist["meccan"] >= 1

    def test_compare_themes(self, retriever):
        result = retriever.compare_themes(["tawhid", "tawhid-asma-sifat"])
        assert "themes" in result
        assert len(result["themes"]) == 2
        assert "overlap" in result

    def test_generate_theme_summary(self, retriever):
        summary = retriever.generate_theme_summary("tawhid")
        assert summary is not None
        assert summary.theme_id == "tawhid"
        assert summary.total_verses == 2
        assert len(summary.key_verses) <= 5
        assert len(summary.summary_text) > 0

    def test_generate_summary_nonexistent(self, retriever):
        summary = retriever.generate_theme_summary("nonexistent")
        assert summary is None


class TestRevelationPeriod:
    """Tests for RevelationPeriod enum."""

    def test_enum_values(self):
        assert RevelationPeriod.MECCAN.value == "meccan"
        assert RevelationPeriod.MEDINAN.value == "medinan"
        assert RevelationPeriod.UNKNOWN.value == "unknown"
