"""
Query Expansion for Islamic Terms

This module provides query expansion capabilities for Islamic terminology,
handling Arabic-English equivalents, transliteration variations, and
concept-based expansion for improved search results.
"""

import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ExpandedQuery:
    """Represents an expanded query with all variations."""

    original: str
    expanded_terms: list[str] = field(default_factory=list)
    arabic_terms: list[str] = field(default_factory=list)
    english_terms: list[str] = field(default_factory=list)
    related_concepts: list[str] = field(default_factory=list)

    def all_terms(self) -> list[str]:
        """Return all expanded terms including original."""
        terms = [self.original]
        terms.extend(self.expanded_terms)
        terms.extend(self.arabic_terms)
        terms.extend(self.english_terms)
        terms.extend(self.related_concepts)
        return list(set(terms))


# Islamic terminology knowledge graph
# Maps terms to their equivalents in different forms
ISLAMIC_TERMS_GRAPH: dict[str, dict] = {
    # Prayer-related terms
    "salah": {
        "arabic": ["صلاة", "الصلاة"],
        "english": ["prayer", "prayers", "salat", "namaz"],
        "transliterations": ["salaat", "salaah", "solat", "solah"],
        "related": ["wudu", "qibla", "rakah", "sujud", "ruku"],
    },
    "prayer": {
        "arabic": ["صلاة", "الصلاة"],
        "english": ["salah", "salat", "namaz"],
        "transliterations": ["salaat", "salaah"],
        "related": ["wudu", "ablution", "qibla", "direction"],
    },
    "wudu": {
        "arabic": ["وضوء", "الوضوء"],
        "english": ["ablution", "purification", "ritual washing"],
        "transliterations": ["wudhu", "wudoo", "wazu"],
        "related": ["tayammum", "ghusl", "tahara", "salah"],
    },
    "ablution": {
        "arabic": ["وضوء", "الوضوء"],
        "english": ["wudu", "ritual washing", "purification"],
        "transliterations": ["wudhu", "wudoo"],
        "related": ["tayammum", "ghusl", "purity"],
    },
    # Fasting-related terms
    "sawm": {
        "arabic": ["صوم", "الصوم", "صيام"],
        "english": ["fasting", "fast"],
        "transliterations": ["saum", "siyam", "siyaam", "roza", "puasa"],
        "related": ["ramadan", "iftar", "suhoor", "eid"],
    },
    "fasting": {
        "arabic": ["صوم", "صيام"],
        "english": ["sawm", "saum"],
        "transliterations": ["siyam", "roza"],
        "related": ["ramadan", "iftar", "suhoor", "breaking fast"],
    },
    "ramadan": {
        "arabic": ["رمضان", "شهر رمضان"],
        "english": ["ramadhan", "month of fasting"],
        "transliterations": ["ramazan", "ramadaan", "ramzan"],
        "related": ["sawm", "fasting", "iftar", "suhoor", "taraweeh", "laylatul qadr"],
    },
    # Pilgrimage-related terms
    "hajj": {
        "arabic": ["حج", "الحج"],
        "english": ["pilgrimage", "greater pilgrimage"],
        "transliterations": ["hadj", "haj"],
        "related": ["umrah", "mecca", "kaaba", "ihram", "tawaf", "sai"],
    },
    "umrah": {
        "arabic": ["عمرة", "العمرة"],
        "english": ["lesser pilgrimage", "minor pilgrimage"],
        "transliterations": ["umra", "omrah", "omra"],
        "related": ["hajj", "mecca", "kaaba", "tawaf", "sai"],
    },
    # Charity-related terms
    "zakat": {
        "arabic": ["زكاة", "الزكاة"],
        "english": ["alms", "obligatory charity", "charity tax", "poor due"],
        "transliterations": ["zakaat", "zakah", "zekat"],
        "related": ["sadaqah", "nisab", "wealth", "purification of wealth"],
    },
    "sadaqah": {
        "arabic": ["صدقة", "الصدقة"],
        "english": ["charity", "voluntary charity", "alms"],
        "transliterations": ["sadaqa", "sadaka", "sedekah"],
        "related": ["zakat", "infaq", "giving", "donation"],
    },
    # Quran-related terms
    "quran": {
        "arabic": ["قرآن", "القرآن", "القرآن الكريم"],
        "english": ["koran", "holy book", "scripture"],
        "transliterations": ["qur'an", "quraan", "kuran", "coran"],
        "related": ["surah", "ayah", "tafsir", "tajweed", "recitation"],
    },
    "surah": {
        "arabic": ["سورة"],
        "english": ["chapter"],
        "transliterations": ["sura", "soorah", "surat"],
        "related": ["quran", "ayah", "juz"],
    },
    "ayah": {
        "arabic": ["آية", "آيات"],
        "english": ["verse", "verses", "sign"],
        "transliterations": ["ayat", "aayah", "aya"],
        "related": ["quran", "surah", "tafsir"],
    },
    "tafsir": {
        "arabic": ["تفسير", "التفسير"],
        "english": ["exegesis", "interpretation", "commentary"],
        "transliterations": ["tafseer", "tafsiir"],
        "related": ["quran", "ayah", "ulama", "scholar"],
    },
    "tajweed": {
        "arabic": ["تجويد", "التجويد"],
        "english": ["recitation rules", "proper pronunciation"],
        "transliterations": ["tajwid", "tajwied"],
        "related": ["quran", "recitation", "qiraat", "makharij"],
    },
    # Hadith-related terms
    "hadith": {
        "arabic": ["حديث", "الحديث", "أحاديث"],
        "english": ["prophetic tradition", "narration", "saying of prophet"],
        "transliterations": ["hadeeth", "hadis", "hadiths"],
        "related": ["sunnah", "isnad", "sahih", "bukhari", "muslim"],
    },
    "sunnah": {
        "arabic": ["سنة", "السنة"],
        "english": ["prophetic practice", "tradition", "way of prophet"],
        "transliterations": ["sunna", "sunnaat"],
        "related": ["hadith", "prophet", "fiqh", "sharia"],
    },
    # Jurisprudence-related terms
    "fiqh": {
        "arabic": ["فقه", "الفقه"],
        "english": ["jurisprudence", "islamic law", "understanding"],
        "transliterations": ["fikh", "feqh"],
        "related": ["sharia", "fatwa", "madhhab", "ulama", "halal", "haram"],
    },
    "halal": {
        "arabic": ["حلال", "الحلال"],
        "english": ["permissible", "lawful", "allowed"],
        "transliterations": ["halaal", "helal"],
        "related": ["haram", "fiqh", "sharia", "food"],
    },
    "haram": {
        "arabic": ["حرام", "الحرام"],
        "english": ["forbidden", "prohibited", "unlawful"],
        "transliterations": ["haraam"],
        "related": ["halal", "fiqh", "sharia", "sin"],
    },
    "fatwa": {
        "arabic": ["فتوى", "الفتوى", "فتاوى"],
        "english": ["religious ruling", "legal opinion", "edict"],
        "transliterations": ["fatwaa", "fetwa"],
        "related": ["fiqh", "mufti", "ulama", "sharia"],
    },
    # Belief-related terms
    "iman": {
        "arabic": ["إيمان", "الإيمان"],
        "english": ["faith", "belief"],
        "transliterations": ["imaan", "eeman", "eemaan"],
        "related": ["islam", "ihsan", "aqeedah", "tawhid"],
    },
    "tawhid": {
        "arabic": ["توحيد", "التوحيد"],
        "english": ["monotheism", "oneness of god", "unity of god"],
        "transliterations": ["tawheed", "tauhid", "tauheed"],
        "related": ["iman", "aqeedah", "shirk", "allah"],
    },
    # Remembrance and supplication
    "dua": {
        "arabic": ["دعاء", "الدعاء"],
        "english": ["supplication", "prayer", "invocation"],
        "transliterations": ["duaa", "du'a", "doaa"],
        "related": ["dhikr", "adhkar", "worship", "asking allah"],
    },
    "dhikr": {
        "arabic": ["ذكر", "الذكر", "أذكار"],
        "english": ["remembrance", "mention", "invocation"],
        "transliterations": ["zikr", "zikir", "thikr"],
        "related": ["dua", "adhkar", "tasbeeh", "istighfar"],
    },
    "adhkar": {
        "arabic": ["أذكار", "الأذكار"],
        "english": ["remembrances", "invocations", "daily supplications"],
        "transliterations": ["azkar", "athkar", "adkhar"],
        "related": ["dhikr", "dua", "morning adhkar", "evening adhkar"],
    },
    # Schools of thought
    "madhhab": {
        "arabic": ["مذهب", "المذهب", "مذاهب"],
        "english": ["school of thought", "school of jurisprudence", "legal school"],
        "transliterations": ["mazhab", "madhab", "mathab"],
        "related": ["fiqh", "hanafi", "maliki", "shafii", "hanbali"],
    },
    "hanafi": {
        "arabic": ["حنفي", "الحنفية"],
        "english": ["hanafi school", "abu hanifa"],
        "transliterations": ["hanafee", "hanafiyya"],
        "related": ["madhhab", "fiqh", "abu hanifa"],
    },
    "maliki": {
        "arabic": ["مالكي", "المالكية"],
        "english": ["maliki school", "imam malik"],
        "transliterations": ["malikee", "malikiyya"],
        "related": ["madhhab", "fiqh", "imam malik"],
    },
    "shafii": {
        "arabic": ["شافعي", "الشافعية"],
        "english": ["shafii school", "imam shafii"],
        "transliterations": ["shafiee", "shafei", "shafiyya"],
        "related": ["madhhab", "fiqh", "imam shafii"],
    },
    "hanbali": {
        "arabic": ["حنبلي", "الحنابلة"],
        "english": ["hanbali school", "imam ahmad"],
        "transliterations": ["hanbalee", "hanbaliyya"],
        "related": ["madhhab", "fiqh", "imam ahmad"],
    },
}

# Common transliteration patterns for normalization
TRANSLITERATION_PATTERNS = [
    (r"aa+", "a"),  # Multiple 'a's -> single 'a'
    (r"ee+", "i"),  # 'ee' -> 'i'
    (r"oo+", "u"),  # 'oo' -> 'u'
    (r"'", ""),     # Remove apostrophes
    (r"-", ""),     # Remove hyphens
]


def normalize_transliteration(text: str) -> str:
    """Normalize transliteration variations to a standard form."""
    result = text.lower().strip()
    for pattern, replacement in TRANSLITERATION_PATTERNS:
        result = re.sub(pattern, replacement, result)
    return result


def find_matching_term(query: str) -> Optional[str]:
    """Find a matching term in the knowledge graph."""
    normalized_query = normalize_transliteration(query)

    # Direct match
    if normalized_query in ISLAMIC_TERMS_GRAPH:
        return normalized_query

    # Check all entries for matches in transliterations or english terms
    for term, data in ISLAMIC_TERMS_GRAPH.items():
        # Check transliterations
        for trans in data.get("transliterations", []):
            if normalize_transliteration(trans) == normalized_query:
                return term
        # Check english equivalents
        for eng in data.get("english", []):
            if normalize_transliteration(eng) == normalized_query:
                return term
        # Check arabic (exact match only)
        if query in data.get("arabic", []):
            return term

    return None


def expand_query(query: str, include_related: bool = True, max_expansions: int = 20) -> ExpandedQuery:
    """
    Expand a query with Islamic terminology equivalents.

    Args:
        query: The original search query
        include_related: Whether to include related concepts
        max_expansions: Maximum number of expanded terms to return

    Returns:
        ExpandedQuery object with all expansions
    """
    result = ExpandedQuery(original=query)

    # Split query into words and process each
    words = query.lower().split()
    processed_words = set()

    for word in words:
        if word in processed_words:
            continue
        processed_words.add(word)

        # Find matching term in knowledge graph
        matched_term = find_matching_term(word)
        if not matched_term:
            continue

        term_data = ISLAMIC_TERMS_GRAPH[matched_term]

        # Add Arabic equivalents
        result.arabic_terms.extend(term_data.get("arabic", []))

        # Add English equivalents
        result.english_terms.extend(term_data.get("english", []))

        # Add transliteration variations
        result.expanded_terms.extend(term_data.get("transliterations", []))

        # Add related concepts if requested
        if include_related:
            result.related_concepts.extend(term_data.get("related", []))

    # Deduplicate and limit
    result.expanded_terms = list(set(result.expanded_terms))[:max_expansions]
    result.arabic_terms = list(set(result.arabic_terms))[:max_expansions]
    result.english_terms = list(set(result.english_terms))[:max_expansions]
    result.related_concepts = list(set(result.related_concepts))[:max_expansions]

    return result


def expand_query_for_search(query: str, include_related: bool = False) -> list[str]:
    """
    Expand a query and return a flat list of all search terms.

    This is a convenience function for use in search implementations.

    Args:
        query: The original search query
        include_related: Whether to include related concepts (default False for precision)

    Returns:
        List of all search terms including original and expansions
    """
    expanded = expand_query(query, include_related=include_related)
    return expanded.all_terms()


def get_arabic_equivalent(term: str) -> list[str]:
    """Get Arabic equivalents for an English/transliterated term."""
    matched = find_matching_term(term)
    if matched and matched in ISLAMIC_TERMS_GRAPH:
        return ISLAMIC_TERMS_GRAPH[matched].get("arabic", [])
    return []


def get_english_equivalent(term: str) -> list[str]:
    """Get English equivalents for an Arabic/transliterated term."""
    matched = find_matching_term(term)
    if matched and matched in ISLAMIC_TERMS_GRAPH:
        return ISLAMIC_TERMS_GRAPH[matched].get("english", [])
    return []


def get_related_concepts(term: str) -> list[str]:
    """Get related concepts for a term."""
    matched = find_matching_term(term)
    if matched and matched in ISLAMIC_TERMS_GRAPH:
        return ISLAMIC_TERMS_GRAPH[matched].get("related", [])
    return []


# Example usage and testing
if __name__ == "__main__":
    # Test cases
    test_queries = [
        "how to perform wudu",
        "salah times",
        "ramadan fasting rules",
        "zakat calculation",
        "quran tafsir",
        "hadith about prayer",
        "hanafi fiqh",
    ]

    for query in test_queries:
        print(f"\nQuery: {query}")
        expanded = expand_query(query)
        print(f"  Arabic: {expanded.arabic_terms}")
        print(f"  English: {expanded.english_terms}")
        print(f"  Transliterations: {expanded.expanded_terms}")
        print(f"  Related: {expanded.related_concepts}")
        print(f"  All terms: {len(expanded.all_terms())} total")
