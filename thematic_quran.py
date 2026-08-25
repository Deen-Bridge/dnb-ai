"""Thematic Quran Retrieval System (#144)

A comprehensive system for organizing and accessing Quranic content by themes.
Provides a multi-tiered classification system grounded in Islamic scholarly
traditions with complete associations between themes and individual verses.

Architecture:
- ThematicTaxonomy: Multi-tiered theme classification
- ThemeVerseMapping: Verse-to-theme associations with annotations
- ThematicRetriever: API for querying and navigating themes

Features:
- Theme co-occurrence analysis
- Chronological tracking (Meccan vs Medinan)
- Comparative theme analysis
- Scholarly definitions and explanations
- Thematic summaries
"""

import json
import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


class RevelationPeriod(str, Enum):
    """Period of revelation for a surah or verse."""
    MECCAN = "meccan"
    MEDINAN = "medinan"
    UNKNOWN = "unknown"


@dataclass
class Theme:
    """A theme in the taxonomy hierarchy."""
    id: str
    name: str
    name_arabic: str
    description: str
    scholarly_definition: Optional[str] = None
    parent_id: Optional[str] = None
    level: int = 0  # 0 = main, 1 = sub, 2 = micro
    keywords: list[str] = field(default_factory=list)
    related_themes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "name_arabic": self.name_arabic,
            "description": self.description,
            "scholarly_definition": self.scholarly_definition,
            "parent_id": self.parent_id,
            "level": self.level,
            "keywords": self.keywords,
            "related_themes": self.related_themes,
        }


@dataclass
class VerseThemeMapping:
    """Association between a verse and a theme."""
    surah: int
    ayah: int
    theme_id: str
    relevance_score: float = 1.0
    annotation: Optional[str] = None
    scholarly_notes: Optional[str] = None
    context_type: str = "primary"  # primary, secondary, tangential


@dataclass
class ThematicSummary:
    """Summary of a theme across the Quran."""
    theme_id: str
    total_verses: int
    meccan_verses: int
    medinan_verses: int
    co_occurring_themes: dict[str, int]  # theme_id -> count
    summary_text: str
    key_verses: list[tuple[int, int]]  # (surah, ayah)


# ─────────────────────────────────────────────────────────────────────────────
# Core Taxonomy
# ─────────────────────────────────────────────────────────────────────────────

# Main themes based on classical Islamic scholarship (Usul al-Din & Usul al-Fiqh)
MAIN_THEMES = [
    Theme(
        id="tawhid",
        name="Monotheism (Tawhid)",
        name_arabic="التوحيد",
        description="The oneness and uniqueness of Allah",
        scholarly_definition="The Islamic concept of monotheism, affirming that Allah is One in His essence, attributes, and actions.",
        level=0,
        keywords=["Allah", "one", "Lord", "worship", "deity"],
    ),
    Theme(
        id="prophethood",
        name="Prophethood (Nubuwwah)",
        name_arabic="النبوة",
        description="The concept of prophets and messengers",
        scholarly_definition="The belief in prophets sent by Allah to guide humanity, culminating in Prophet Muhammad (PBUH).",
        level=0,
        keywords=["prophet", "messenger", "revelation", "guidance"],
    ),
    Theme(
        id="afterlife",
        name="Afterlife (Akhirah)",
        name_arabic="الآخرة",
        description="The Day of Judgment and life after death",
        scholarly_definition="Belief in the resurrection, Day of Judgment, Paradise, and Hellfire.",
        level=0,
        keywords=["day", "judgment", "paradise", "hellfire", "resurrection"],
    ),
    Theme(
        id="worship",
        name="Worship (Ibadah)",
        name_arabic="العبادة",
        description="Acts of worship and devotion to Allah",
        scholarly_definition="All acts of obedience to Allah, including ritual worship and daily conduct.",
        level=0,
        keywords=["prayer", "fasting", "charity", "pilgrimage", "remembrance"],
    ),
    Theme(
        id="ethics",
        name="Ethics & Morality (Akhlaq)",
        name_arabic="الأخلاق",
        description="Moral principles and ethical conduct",
        scholarly_definition="The Islamic ethical framework governing individual and social behavior.",
        level=0,
        keywords=["justice", "honesty", "patience", "kindness", "righteousness"],
    ),
    Theme(
        id="social",
        name="Social Relations (Muamalat)",
        name_arabic="المعاملات",
        description="Social interactions and community life",
        scholarly_definition="Rules governing interpersonal and social relationships in Islam.",
        level=0,
        keywords=["family", "community", "marriage", "inheritance", "contract"],
    ),
    Theme(
        id="history",
        name="Historical Narratives (Qisas)",
        name_arabic="القصص",
        description="Stories of past nations and prophets",
        scholarly_definition="Accounts of previous prophets and nations for guidance and lessons.",
        level=0,
        keywords=["story", "people", "nation", "prophet", "destroyed"],
    ),
    Theme(
        id="creation",
        name="Creation & Nature (Khalq)",
        name_arabic="الخلق",
        description="Signs of Allah in creation",
        scholarly_definition="Verses pointing to Allah's creative power as evidence of His existence.",
        level=0,
        keywords=["heaven", "earth", "creation", "signs", "nature"],
    ),
    Theme(
        id="guidance",
        name="Divine Guidance (Hidayah)",
        name_arabic="الهداية",
        description="Guidance from Allah to humanity",
        scholarly_definition="The Quran and Sunnah as sources of divine guidance.",
        level=0,
        keywords=["guide", "path", "straight", "light", "truth"],
    ),
    Theme(
        id="law",
        name="Islamic Law (Shariah)",
        name_arabic="الشريعة",
        description="Legal rulings and obligations",
        scholarly_definition="Divine law derived from the Quran and Sunnah.",
        level=0,
        keywords=["lawful", "forbidden", "obligatory", "ruling", "command"],
    ),
]

# Sub-themes (level 1) - examples under main themes
SUB_THEMES = [
    # Under Tawhid
    Theme(
        id="tawhid-rububiyyah",
        name="Lordship (Rububiyyah)",
        name_arabic="الربوبية",
        description="Allah's sovereignty and control over creation",
        parent_id="tawhid",
        level=1,
        keywords=["Lord", "sovereign", "control", "sustainer"],
    ),
    Theme(
        id="tawhid-uluhiyyah",
        name="Worship (Uluhiyyah)",
        name_arabic="الألوهية",
        description="Allah alone deserves worship",
        parent_id="tawhid",
        level=1,
        keywords=["worship", "deity", "serve", "devotion"],
    ),
    Theme(
        id="tawhid-asma-sifat",
        name="Names & Attributes",
        name_arabic="الأسماء والصفات",
        description="Allah's beautiful names and perfect attributes",
        parent_id="tawhid",
        level=1,
        keywords=["name", "attribute", "merciful", "knowing", "powerful"],
    ),
    # Under Worship
    Theme(
        id="worship-salah",
        name="Prayer (Salah)",
        name_arabic="الصلاة",
        description="The five daily prayers",
        parent_id="worship",
        level=1,
        keywords=["prayer", "prostration", "bow", "worship"],
    ),
    Theme(
        id="worship-zakah",
        name="Charity (Zakah)",
        name_arabic="الزكاة",
        description="Obligatory charity",
        parent_id="worship",
        level=1,
        keywords=["charity", "alms", "spend", "poor"],
    ),
    Theme(
        id="worship-sawm",
        name="Fasting (Sawm)",
        name_arabic="الصوم",
        description="Fasting in Ramadan",
        parent_id="worship",
        level=1,
        keywords=["fast", "Ramadan", "abstain"],
    ),
    Theme(
        id="worship-hajj",
        name="Pilgrimage (Hajj)",
        name_arabic="الحج",
        description="Pilgrimage to Mecca",
        parent_id="worship",
        level=1,
        keywords=["pilgrimage", "Mecca", "Kaaba", "Hajj"],
    ),
    # Under Ethics
    Theme(
        id="ethics-justice",
        name="Justice (Adl)",
        name_arabic="العدل",
        description="Fairness and justice in all matters",
        parent_id="ethics",
        level=1,
        keywords=["justice", "fair", "equity", "balance"],
    ),
    Theme(
        id="ethics-patience",
        name="Patience (Sabr)",
        name_arabic="الصبر",
        description="Patience and perseverance",
        parent_id="ethics",
        level=1,
        keywords=["patience", "endure", "persevere", "steadfast"],
    ),
    Theme(
        id="ethics-gratitude",
        name="Gratitude (Shukr)",
        name_arabic="الشكر",
        description="Thankfulness to Allah",
        parent_id="ethics",
        level=1,
        keywords=["grateful", "thankful", "blessings"],
    ),
    # Under Afterlife
    Theme(
        id="afterlife-paradise",
        name="Paradise (Jannah)",
        name_arabic="الجنة",
        description="The eternal abode of the righteous",
        parent_id="afterlife",
        level=1,
        keywords=["paradise", "garden", "reward", "eternal"],
    ),
    Theme(
        id="afterlife-hellfire",
        name="Hellfire (Jahannam)",
        name_arabic="جهنم",
        description="The punishment for the wicked",
        parent_id="afterlife",
        level=1,
        keywords=["hellfire", "punishment", "fire", "torment"],
    ),
    Theme(
        id="afterlife-judgment",
        name="Day of Judgment",
        name_arabic="يوم القيامة",
        description="The final day of reckoning",
        parent_id="afterlife",
        level=1,
        keywords=["day", "judgment", "reckoning", "account"],
    ),
]

# Surah revelation periods (partial list - expand as needed)
SURAH_PERIODS: dict[int, RevelationPeriod] = {
    1: RevelationPeriod.MECCAN,   # Al-Fatiha
    2: RevelationPeriod.MEDINAN,  # Al-Baqarah
    3: RevelationPeriod.MEDINAN,  # Ali 'Imran
    4: RevelationPeriod.MEDINAN,  # An-Nisa
    5: RevelationPeriod.MEDINAN,  # Al-Ma'idah
    6: RevelationPeriod.MECCAN,   # Al-An'am
    7: RevelationPeriod.MECCAN,   # Al-A'raf
    # ... (would be expanded for all 114 surahs)
}


class ThematicTaxonomy:
    """Manages the hierarchical theme taxonomy."""

    def __init__(self) -> None:
        self._themes: dict[str, Theme] = {}
        self._children: dict[str, list[str]] = {}  # parent_id -> child_ids
        self._load_default_taxonomy()

    def _load_default_taxonomy(self) -> None:
        """Load the default taxonomy of themes."""
        for theme in MAIN_THEMES + SUB_THEMES:
            self.add_theme(theme)

    def add_theme(self, theme: Theme) -> None:
        """Add a theme to the taxonomy."""
        self._themes[theme.id] = theme
        if theme.parent_id:
            if theme.parent_id not in self._children:
                self._children[theme.parent_id] = []
            self._children[theme.parent_id].append(theme.id)

    def get_theme(self, theme_id: str) -> Optional[Theme]:
        """Get a theme by ID."""
        return self._themes.get(theme_id)

    def get_children(self, theme_id: str) -> list[Theme]:
        """Get all child themes of a theme."""
        child_ids = self._children.get(theme_id, [])
        return [self._themes[cid] for cid in child_ids if cid in self._themes]

    def get_ancestors(self, theme_id: str) -> list[Theme]:
        """Get all ancestor themes (parent, grandparent, etc.)."""
        ancestors = []
        theme = self._themes.get(theme_id)
        while theme and theme.parent_id:
            parent = self._themes.get(theme.parent_id)
            if parent:
                ancestors.append(parent)
                theme = parent
            else:
                break
        return ancestors

    def get_main_themes(self) -> list[Theme]:
        """Get all main (top-level) themes."""
        return [t for t in self._themes.values() if t.level == 0]

    def get_all_themes(self) -> list[Theme]:
        """Get all themes in the taxonomy."""
        return list(self._themes.values())

    def search_themes(self, query: str) -> list[Theme]:
        """Search themes by name or keywords."""
        query_lower = query.lower()
        results = []
        for theme in self._themes.values():
            if (
                query_lower in theme.name.lower()
                or query_lower in theme.name_arabic
                or any(query_lower in kw.lower() for kw in theme.keywords)
            ):
                results.append(theme)
        return results

    def get_related_themes(self, theme_id: str) -> list[Theme]:
        """Get themes related to the given theme."""
        theme = self._themes.get(theme_id)
        if not theme:
            return []
        return [
            self._themes[rid]
            for rid in theme.related_themes
            if rid in self._themes
        ]

    def to_dict(self) -> dict[str, Any]:
        """Export taxonomy as a dictionary."""
        return {
            "themes": [t.to_dict() for t in self._themes.values()],
            "hierarchy": self._children,
        }


class ThemeVerseStore:
    """Manages verse-to-theme mappings."""

    def __init__(self, data_file: Optional[str] = None) -> None:
        self._mappings: list[VerseThemeMapping] = []
        self._by_verse: dict[tuple[int, int], list[VerseThemeMapping]] = {}
        self._by_theme: dict[str, list[VerseThemeMapping]] = {}
        self._data_file = data_file or os.getenv(
            "THEME_VERSE_DATA", "./data/theme_verses.json"
        )
        self._load_data()

    def _load_data(self) -> None:
        """Load mappings from file."""
        if not os.path.exists(self._data_file):
            logger.info("No theme verse data file found; starting empty")
            return
        try:
            with open(self._data_file) as f:
                data = json.load(f)
            for item in data.get("mappings", []):
                mapping = VerseThemeMapping(
                    surah=item["surah"],
                    ayah=item["ayah"],
                    theme_id=item["theme_id"],
                    relevance_score=item.get("relevance_score", 1.0),
                    annotation=item.get("annotation"),
                    scholarly_notes=item.get("scholarly_notes"),
                    context_type=item.get("context_type", "primary"),
                )
                self.add_mapping(mapping)
            logger.info("Loaded %d theme-verse mappings", len(self._mappings))
        except Exception as e:
            logger.warning("Failed to load theme verse data: %s", e)

    def add_mapping(self, mapping: VerseThemeMapping) -> None:
        """Add a verse-theme mapping."""
        self._mappings.append(mapping)
        key = (mapping.surah, mapping.ayah)
        if key not in self._by_verse:
            self._by_verse[key] = []
        self._by_verse[key].append(mapping)
        if mapping.theme_id not in self._by_theme:
            self._by_theme[mapping.theme_id] = []
        self._by_theme[mapping.theme_id].append(mapping)

    def get_themes_for_verse(
        self, surah: int, ayah: int
    ) -> list[VerseThemeMapping]:
        """Get all theme mappings for a verse."""
        return self._by_verse.get((surah, ayah), [])

    def get_verses_for_theme(
        self,
        theme_id: str,
        min_relevance: float = 0.0,
        context_type: Optional[str] = None,
    ) -> list[VerseThemeMapping]:
        """Get all verse mappings for a theme with optional filters."""
        mappings = self._by_theme.get(theme_id, [])
        if min_relevance > 0:
            mappings = [m for m in mappings if m.relevance_score >= min_relevance]
        if context_type:
            mappings = [m for m in mappings if m.context_type == context_type]
        return sorted(mappings, key=lambda m: -m.relevance_score)

    def save_data(self) -> None:
        """Save mappings to file."""
        data = {
            "mappings": [
                {
                    "surah": m.surah,
                    "ayah": m.ayah,
                    "theme_id": m.theme_id,
                    "relevance_score": m.relevance_score,
                    "annotation": m.annotation,
                    "scholarly_notes": m.scholarly_notes,
                    "context_type": m.context_type,
                }
                for m in self._mappings
            ]
        }
        os.makedirs(os.path.dirname(self._data_file) or ".", exist_ok=True)
        with open(self._data_file, "w") as f:
            json.dump(data, f, indent=2)


class ThematicRetriever:
    """Main API for thematic Quran retrieval."""

    def __init__(
        self,
        taxonomy: Optional[ThematicTaxonomy] = None,
        verse_store: Optional[ThemeVerseStore] = None,
    ) -> None:
        self.taxonomy = taxonomy or ThematicTaxonomy()
        self.verse_store = verse_store or ThemeVerseStore()

    def get_theme_hierarchy(self) -> dict[str, Any]:
        """Get the full theme hierarchy."""
        main_themes = self.taxonomy.get_main_themes()
        return {
            "main_themes": [
                {
                    **t.to_dict(),
                    "children": [c.to_dict() for c in self.taxonomy.get_children(t.id)],
                }
                for t in main_themes
            ]
        }

    def browse_theme(
        self,
        theme_id: str,
        include_children: bool = True,
        limit: int = 50,
    ) -> dict[str, Any]:
        """Browse a theme and its verses."""
        theme = self.taxonomy.get_theme(theme_id)
        if not theme:
            return {"error": f"Theme '{theme_id}' not found"}

        verses = self.verse_store.get_verses_for_theme(theme_id)[:limit]
        children = self.taxonomy.get_children(theme_id) if include_children else []
        related = self.taxonomy.get_related_themes(theme_id)

        return {
            "theme": theme.to_dict(),
            "verses": [
                {
                    "surah": v.surah,
                    "ayah": v.ayah,
                    "relevance_score": v.relevance_score,
                    "annotation": v.annotation,
                    "context_type": v.context_type,
                }
                for v in verses
            ],
            "children": [c.to_dict() for c in children],
            "related": [r.to_dict() for r in related],
            "total_verses": len(self.verse_store.get_verses_for_theme(theme_id)),
        }

    def search_themes(self, query: str) -> list[dict[str, Any]]:
        """Search for themes by name or keyword."""
        themes = self.taxonomy.search_themes(query)
        return [
            {
                **t.to_dict(),
                "verse_count": len(self.verse_store.get_verses_for_theme(t.id)),
            }
            for t in themes
        ]

    def get_verse_themes(self, surah: int, ayah: int) -> list[dict[str, Any]]:
        """Get all themes associated with a verse."""
        mappings = self.verse_store.get_themes_for_verse(surah, ayah)
        results = []
        for m in mappings:
            theme = self.taxonomy.get_theme(m.theme_id)
            if theme:
                results.append({
                    "theme": theme.to_dict(),
                    "relevance_score": m.relevance_score,
                    "annotation": m.annotation,
                    "context_type": m.context_type,
                })
        return results

    def get_theme_cooccurrence(self, theme_id: str) -> dict[str, int]:
        """Analyze which themes frequently co-occur with the given theme."""
        verses = self.verse_store.get_verses_for_theme(theme_id)
        cooccurrence: dict[str, int] = {}

        for v in verses:
            other_themes = self.verse_store.get_themes_for_verse(v.surah, v.ayah)
            for ot in other_themes:
                if ot.theme_id != theme_id:
                    cooccurrence[ot.theme_id] = cooccurrence.get(ot.theme_id, 0) + 1

        return dict(sorted(cooccurrence.items(), key=lambda x: -x[1]))

    def get_chronological_distribution(
        self, theme_id: str
    ) -> dict[str, Any]:
        """Get distribution of theme verses by revelation period."""
        verses = self.verse_store.get_verses_for_theme(theme_id)
        meccan = 0
        medinan = 0
        unknown = 0

        for v in verses:
            period = SURAH_PERIODS.get(v.surah, RevelationPeriod.UNKNOWN)
            if period == RevelationPeriod.MECCAN:
                meccan += 1
            elif period == RevelationPeriod.MEDINAN:
                medinan += 1
            else:
                unknown += 1

        return {
            "theme_id": theme_id,
            "meccan": meccan,
            "medinan": medinan,
            "unknown": unknown,
            "total": len(verses),
        }

    def compare_themes(
        self, theme_ids: list[str]
    ) -> dict[str, Any]:
        """Compare multiple themes by their verse coverage and overlap."""
        themes_data = []
        all_verses: dict[str, set[tuple[int, int]]] = {}

        for tid in theme_ids:
            theme = self.taxonomy.get_theme(tid)
            if not theme:
                continue
            verses = self.verse_store.get_verses_for_theme(tid)
            verse_set = {(v.surah, v.ayah) for v in verses}
            all_verses[tid] = verse_set
            themes_data.append({
                "theme": theme.to_dict(),
                "verse_count": len(verses),
                "chronology": self.get_chronological_distribution(tid),
            })

        # Calculate overlap between themes
        overlap = {}
        for i, tid1 in enumerate(theme_ids):
            for tid2 in theme_ids[i + 1:]:
                if tid1 in all_verses and tid2 in all_verses:
                    shared = len(all_verses[tid1] & all_verses[tid2])
                    overlap[f"{tid1}:{tid2}"] = shared

        return {
            "themes": themes_data,
            "overlap": overlap,
        }

    def generate_theme_summary(self, theme_id: str) -> Optional[ThematicSummary]:
        """Generate a summary of a theme's presence in the Quran."""
        theme = self.taxonomy.get_theme(theme_id)
        if not theme:
            return None

        verses = self.verse_store.get_verses_for_theme(theme_id)
        chrono = self.get_chronological_distribution(theme_id)
        cooccur = self.get_theme_cooccurrence(theme_id)

        # Get top 5 key verses (highest relevance)
        key_verses = sorted(verses, key=lambda v: -v.relevance_score)[:5]

        summary_text = (
            f"The theme '{theme.name}' ({theme.name_arabic}) appears in "
            f"{len(verses)} verses across the Quran. "
            f"It is found in {chrono['meccan']} Meccan verses and "
            f"{chrono['medinan']} Medinan verses. "
        )

        if cooccur:
            top_cooccur = list(cooccur.items())[:3]
            related_names = []
            for tid, _ in top_cooccur:
                related_theme = self.taxonomy.get_theme(tid)
                if related_theme:
                    related_names.append(related_theme.name)
            if related_names:
                summary_text += (
                    f"It frequently co-occurs with themes of {', '.join(related_names)}."
                )

        return ThematicSummary(
            theme_id=theme_id,
            total_verses=len(verses),
            meccan_verses=chrono["meccan"],
            medinan_verses=chrono["medinan"],
            co_occurring_themes=cooccur,
            summary_text=summary_text,
            key_verses=[(v.surah, v.ayah) for v in key_verses],
        )


# ─────────────────────────────────────────────────────────────────────────────
# Singleton instance for the application
# ─────────────────────────────────────────────────────────────────────────────

_retriever: Optional[ThematicRetriever] = None


def get_thematic_retriever() -> ThematicRetriever:
    """Get or create the singleton thematic retriever instance."""
    global _retriever
    if _retriever is None:
        _retriever = ThematicRetriever()
    return _retriever
