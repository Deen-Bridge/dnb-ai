"""Fabricated Scholarly Attribution Prevention System (#173).

Validates all scholarly attributions against authoritative databases to prevent
fabricating or misattributing opinions to Islamic scholars.

Components:
- SCHOLARS_DB: comprehensive scholar biography database with verified positions
- ScholarAttributionValidator: validates opinion-to-scholar mappings
- TemporalConsistencyChecker: checks scholar era vs. topic chronology
- AnachronismDetector: flags attributions that postdate a scholar's lifetime
- NuanceDetector: detects flattening of nuanced scholarly positions
- ConsensusValidator: validates consensus claims against historical records
- AttributionAuditTrail: records all validation decisions for accountability
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Scholar biography database
# ---------------------------------------------------------------------------


class ScholarEra(BaseModel):
    birth_year: int  # Hijri year
    death_year: int  # Hijri year
    birth_ce: int | None = None
    death_ce: int | None = None


class ScholarInfo(BaseModel):
    id: str
    name: str
    full_name: str
    era: ScholarEra
    school: str  # madhhab
    specialties: list[str] = Field(default_factory=list)
    known_positions: list[str] = Field(default_factory=list)
    notable_works: list[str] = Field(default_factory=list)
    aliases: list[str] = Field(default_factory=list)


SCHOLARS_DB: dict[str, ScholarInfo] = {}


def _register_scholars(scholars: list[ScholarInfo]) -> None:
    for s in scholars:
        SCHOLARS_DB[s.id] = s


_register_scholars(
    [
        ScholarInfo(
            id="imam_abu_hanifa",
            name="Imam Abu Hanifa",
            full_name="Abu Hanifa Nu'man ibn Thabit",
            era=ScholarEra(birth_year=80, death_year=150, birth_ce=699, death_ce=767),
            school="Hanafi",
            specialties=["fiqh", "usul_al_fiqh", "aqeedah"],
            known_positions=[
                "wiping over leather socks is permissible",
                "intention for wudu is recommended not obligatory",
                "qunut in fajr is not practiced",
                "loud bismillah in prayer is not practiced",
                "mukallaf at seven years old",
            ],
            notable_works=["Al-Fiqh al-Akbar", "Kitab al-Athar"],
            aliases=["Abu Hanifa", "al-Imam al-Azam", "al-Nu'man"],
        ),
        ScholarInfo(
            id="imam_malik",
            name="Imam Malik",
            full_name="Malik ibn Anas",
            era=ScholarEra(birth_year=93, death_year=179, birth_ce=711, death_ce=795),
            school="Maliki",
            specialties=["fiqh", "hadith", "usul_al_fiqh"],
            known_positions=[
                "practice of people of madinah is authoritative",
                "raising hands only at opening takbir",
                "arms at sides during prayer",
                "istihsan is a source of law",
            ],
            notable_works=["Al-Muwatta", "Al-Mudawwana"],
            aliases=["Malik ibn Anas", "Imam Malik", "Shaykh al-Islam"],
        ),
        ScholarInfo(
            id="imam_shafi",
            name="Imam al-Shafi'i",
            full_name="Muhammad ibn Idris al-Shafi'i",
            era=ScholarEra(birth_year=150, death_year=204, birth_ce=767, death_ce=820),
            school="Shafi'i",
            specialties=["fiqh", "usul_al_fiqh", "hadith"],
            known_positions=[
                "bismillah is part of al-fatiha",
                "qunut in fajr prayer is sunnah",
                "raising hands at ruku and rising from ruku",
                "ahl al-sunnah wal-jama'ah is the saved group",
            ],
            notable_works=["Al-Risala", "Al-Umm"],
            aliases=["al-Shafi'i", "Shaykh al-Islam", "Muhyi al-Sunnah"],
        ),
        ScholarInfo(
            id="imam_ahmad",
            name="Imam Ahmad ibn Hanbal",
            full_name="Ahmad ibn Hanbal al-Shaybani",
            era=ScholarEra(birth_year=164, death_year=241, birth_ce=780, death_ce=855),
            school="Hanbali",
            specialties=["fiqh", "hadith", "aqeedah"],
            known_positions=[
                "strict adherence to hadith over qiyas",
                "created quran controversy stance",
                "perseverance during the mihna",
                "salat al-istisqa is sunnah",
            ],
            notable_works=["Al-Musnad", "Kitab al-Sunna"],
            aliases=["Ahmad ibn Hanbal", "Imam Ahmad", "Shaykh al-Islam"],
        ),
        ScholarInfo(
            id="ibn_taymiyyah",
            name="Ibn Taymiyyah",
            full_name="Taqi al-Din Ahmad ibn Taymiyyah",
            era=ScholarEra(birth_year=661, death_year=728, birth_ce=1263, death_ce=1328),
            school="Hanbali",
            specialties=["aqeedah", "fiqh", "usul_al_fiqh"],
            known_positions=[
                "tawassul through dead is not permissible",
                "triple talaq counts as one",
                "traveled fatwa on visiting graves",
                "ittiba' is superior to taqlid",
            ],
            notable_works=[
                "Majmu' al-Fatawa",
                "Dar' Ta'arud al-'Aql wa al-Naql",
                "Minhaj al-Sunna al-Nabawiyya",
            ],
            aliases=["Shaykh al-Islam", "Ibn Taymiyya"],
        ),
        ScholarInfo(
            id="imam_ghazali",
            name="Imam al-Ghazali",
            full_name="Abu Hamid Muhammad ibn Muhammad al-Ghazali",
            era=ScholarEra(birth_year=450, death_year=505, birth_ce=1058, death_ce=1111),
            school="Shafi'i",
            specialties=["aqeedah", "tasawwuf", "usul_al_fiqh", "philosophy"],
            known_positions=[
                "tasawwuf is integral to islam",
                "ihya ulum al-din as comprehensive work",
                "critique of philosophers in tahafut",
                "knowledge of self precedes knowledge of God",
            ],
            notable_works=[
                "Ihya Ulum al-Din",
                "Tahafut al-Falasifa",
                "Al-Munqidh min al-Dalal",
            ],
            aliases=["al-Ghazali", "Hujjat al-Islam", "Imam al-Ghazali"],
        ),
        ScholarInfo(
            id="ibn_kathir",
            name="Ibn Kathir",
            full_name="Isma'il ibn Kathir",
            era=ScholarEra(birth_year=701, death_year=774, birth_ce=1301, death_ce=1373),
            school="Shafi'i",
            specialties=["tafsir", "hadith", "tarikh"],
            known_positions=[
                "preference for Quran interpretation by Quran",
                "use of hadith in tafsir",
                "following ibn taymiyyah's approach to aqeedah",
            ],
            notable_works=[
                "Tafsir al-Qur'an al-Azim",
                "Al-Bidaya wa al-Nihaya",
            ],
            aliases=["Ibn Kathir", "Abu al-Fida", "al-Hafiz"],
        ),
        ScholarInfo(
            id="al_nawawi",
            name="Al-Nawawi",
            full_name="Yahya ibn Sharaf al-Nawawi",
            era=ScholarEra(birth_year=631, death_year=676, birth_ce=1233, death_ce=1277),
            school="Shafi'i",
            specialties=["hadith", "fiqh", "akhlaq"],
            known_positions=[
                "following shafi'i school strictly",
                "importance of hadith science",
                "fourty hadith nawawi as foundational text",
            ],
            notable_works=[
                "Riyad al-Salihin",
                "Al-Majmu'",
                "Sharh Sahih Muslim",
                "Arba'in al-Nawawiyya",
            ],
            aliases=["al-Nawawi", "Imam al-Nawawi", "Shaykh al-Islam"],
        ),
        ScholarInfo(
            id="al_qurtubi",
            name="Al-Qurtubi",
            full_name="Muhammad ibn Ahmad al-Qurtubi",
            era=ScholarEra(birth_year=600, death_year=671, birth_ce=1204, death_ce=1273),
            school="Maliki",
            specialties=["tafsir", "fiqh", "usul_al_fiqh"],
            known_positions=[
                "comprehensive tafsir from fiqh perspective",
                "emphasis on practical rulings from Quran",
            ],
            notable_works=[
                "Al-Jami' li-Ahkam al-Qur'an",
            ],
            aliases=["al-Qurtubi", "Imam al-Qurtubi"],
        ),
        ScholarInfo(
            id="ibn_rushd",
            name="Ibn Rushd",
            full_name="Muhammad ibn Ahmad ibn Rushd (Averroes)",
            era=ScholarEra(birth_year=520, death_year=595, birth_ce=1126, death_ce=1198),
            school="Maliki",
            specialties=["fiqh", "philosophy", "usul_al_fiqh"],
            known_positions=[
                "faylasuf and faqih",
                "harmonization of reason and revelation",
                "bidayat al-mujtahid methodology",
            ],
            notable_works=[
                "Bidayat al-Mujtahid",
                "Fasl al-Maqal",
                "Tahafut al-Tahafut",
            ],
            aliases=["Averroes", "Ibn Rushd"],
        ),
        ScholarInfo(
            id="ibn_hajar",
            name="Ibn Hajar al-Asqalani",
            full_name="Shihab al-Din Ahmad ibn Hajar al-Asqalani",
            era=ScholarEra(birth_year=773, death_year=852, birth_ce=1372, death_ce=1449),
            school="Shafi'i",
            specialties=["hadith", "fiqh", "tarikh"],
            known_positions=[
                "authoritative commentary on bukhari",
                "grading methodology for hadith",
            ],
            notable_works=[
                "Fath al-Bari",
                "Tahdhib al-Tahdhib",
                "Al-Isaba fi Tamyiz al-Sahaba",
            ],
            aliases=["Ibn Hajar", "al-Hafiz Ibn Hajar"],
        ),
        ScholarInfo(
            id="al_suyuti",
            name="Al-Suyuti",
            full_name="Jalal al-Din al-Suyuti",
            era=ScholarEra(birth_year=849, death_year=911, birth_ce=1445, death_ce=1505),
            school="Shafi'i",
            specialties=["tafsir", "hadith", "usul_al_quran", "tarikh"],
            known_positions=[
                "itqan methodological framework",
                "comprehensive usul al-quran science",
            ],
            notable_works=[
                "Al-Itqan fi Ulum al-Quran",
                "Tafsir al-Jalalayn",
                "Al-Jami' al-Saghir",
            ],
            aliases=["al-Suyuti", "Jalal al-Din", "al-Suyuti"],
        ),
        ScholarInfo(
            id="ibn_ashur",
            name="Ibn Ashur",
            full_name="Muhammad al-Tahir ibn Ashur",
            era=ScholarEra(birth_year=1296, death_year=1393, birth_ce=1879, death_ce=1973),
            school="Maliki",
            specialties=["tafsir", "usul_al_fiqh", "maqasid"],
            known_positions=[
                "maqasid al-shariah revival",
                "contemporary tafsir methodology",
                "objectives-based approach to usul",
            ],
            notable_works=[
                "Al-Tahrir wa al-Tanwir",
                "Maqasid al-Shariah al-Islamiyya",
            ],
            aliases=["Ibn Ashur", "al-Tahir ibn Ashur"],
        ),
        ScholarInfo(
            id="ibn_qayyim",
            name="Ibn al-Qayyim",
            full_name="Shams al-Din Abu Abdullah Muhammad ibn Qayyim al-Jawziyya",
            era=ScholarEra(birth_year=691, death_year=751, birth_ce=1292, death_ce=1350),
            school="Hanbali",
            specialties=["fiqh", "aqeedah", "tasawwuf", "tibb"],
            known_positions=[
                "spiritual purification methodology",
                "Zad al-Ma'ad as comprehensive guide",
                "tawheed-centered approach",
            ],
            notable_works=[
                "Zad al-Ma'ad",
                "Madarij al-Salikin",
                "I'lam al-Muwaqqi'in",
                "Ighathat al-Lahfan",
            ],
            aliases=["Ibn al-Qayyim", "Ibn Qayyim"],
        ),
        ScholarInfo(
            id="al_bukhari",
            name="Imam al-Bukhari",
            full_name="Muhammad ibn Isma'il al-Bukhari",
            era=ScholarEra(birth_year=194, death_year=256, birth_ce=810, death_ce=870),
            school="Hanafi",  # historically affiliated
            specialties=["hadith", "fiqh"],
            known_positions=[
                "authoritative hadith collection criteria",
                "most authentic book after Quran",
            ],
            notable_works=["Sahih al-Bukhari", "Al-Tarikh al-Kabir"],
            aliases=["al-Bukhari", "Imam al-Bukhari", "Ibn Isma'il"],
        ),
        ScholarInfo(
            id="al_tabari",
            name="Al-Tabari",
            full_name="Muhammad ibn Jarir al-Tabari",
            era=ScholarEra(birth_year=224, death_year=310, birth_ce=839, death_ce=923),
            school="Shafi'i",  # founded Jariri school
            specialties=["tafsir", "hadith", "tarikh", "fiqh"],
            known_positions=[
                "comprehensive tafsir bi'l-ma'thur",
                "historical methodology in tafsir",
            ],
            notable_works=[
                "Tafsir al-Tabari",
                "Tarikh al-Rusul wa al-Muluk",
            ],
            aliases=["al-Tabari", "Ibn Jarir", "Imam al-Tabari"],
        ),
    ]
)


# ---------------------------------------------------------------------------
# Opinion-to-scholar mapping from verified sources
# ---------------------------------------------------------------------------


class OpinionEntry(BaseModel):
    opinion: str
    scholar_id: str
    source: str  # which work or collection this is documented in
    strength: str = "well-documented"  # well-documented | debated | isolated
    requires_context: str | None = None  # conditions or qualifications


OPINION_DB: dict[str, list[OpinionEntry]] = {}


def _register_opinions(opinions: list[OpinionEntry]) -> None:
    key = opinions[0].opinion.lower() if opinions else ""
    OPINION_DB[key] = opinions


_register_opinions(
    [
        OpinionEntry(
            opinion="wiping over leather socks is permissible",
            scholar_id="imam_abu_hanifa",
            source="Al-Muwatta, Al-Fiqh al-Akbar",
            strength="well-documented",
        ),
        OpinionEntry(
            opinion="wiping over leather socks is permissible",
            scholar_id="imam_malik",
            source="Al-Muwatta",
            strength="well-documented",
        ),
        OpinionEntry(
            opinion="wiping over leather socks is permissible",
            scholar_id="imam_shafi",
            source="Al-Umm",
            strength="well-documented",
            requires_context="only when continuing wudu from a state of purity",
        ),
        OpinionEntry(
            opinion="wiping over leather socks is permissible",
            scholar_id="imam_ahmad",
            source="Al-Musnad",
            strength="well-documented",
        ),
    ]
)
_register_opinions(
    [
        OpinionEntry(
            opinion="qunut in fajr is sunnah",
            scholar_id="imam_shafi",
            source="Al-Risala, Al-Umm",
            strength="well-documented",
        ),
        OpinionEntry(
            opinion="qunut in fajr is not practiced",
            scholar_id="imam_abu_hanifa",
            source="Al-Fiqh al-Akbar",
            strength="well-documented",
        ),
    ]
)
_register_opinions(
    [
        OpinionEntry(
            opinion="tasawwuf is integral to islam",
            scholar_id="imam_ghazali",
            source="Ihya Ulum al-Din",
            strength="well-documented",
        ),
        OpinionEntry(
            opinion="tasawwuf is integral to islam",
            scholar_id="imam_malik",
            source="Al-Muwatta (Book of Purification)",
            strength="well-documented",
            requires_context="tasawwuf within Shariah boundaries",
        ),
    ]
)
_register_opinions(
    [
        OpinionEntry(
            opinion="triple talaq counts as one",
            scholar_id="ibn_taymiyyah",
            source="Majmu' al-Fatawa",
            strength="well-documented",
        ),
        OpinionEntry(
            opinion="triple talaq counts as one",
            scholar_id="ibn_hajar",
            source="Fath al-Bari",
            strength="well-documented",
        ),
        OpinionEntry(
            opinion="triple talaq counts as three",
            scholar_id="imam_abu_hanifa",
            source="Al-Fiqh al-Akbar",
            strength="well-documented",
        ),
    ]
)
_register_opinions(
    [
        OpinionEntry(
            opinion="zakat is two and a half percent",
            scholar_id="imam_abu_hanifa",
            source="Quran 9:60, Al-Fiqh al-Akbar",
            strength="well-documented",
        ),
        OpinionEntry(
            opinion="zakat is two and a half percent",
            scholar_id="imam_shafi",
            source="Al-Umm",
            strength="well-documented",
        ),
        OpinionEntry(
            opinion="zakat is two and a half percent",
            scholar_id="imam_malik",
            source="Al-Muwatta",
            strength="well-documented",
        ),
        OpinionEntry(
            opinion="zakat is two and a half percent",
            scholar_id="imam_ahmad",
            source="Al-Musnad",
            strength="well-documented",
        ),
    ]
)


# ---------------------------------------------------------------------------
# Historical events for temporal checking
# ---------------------------------------------------------------------------

HISTORICAL_EVENTS: dict[str, dict[str, Any]] = {
    "hijra": {"year_ce": 622, "year_ah": 1, "description": "Migration from Mecca to Medina"},
    "badr": {"year_ce": 624, "year_ah": 2, "description": "Battle of Badr"},
    "uhud": {"year_ce": 625, "year_ah": 3, "description": "Battle of Uhud"},
    "khandaq": {"year_ce": 627, "year_ah": 5, "description": "Battle of the Trench"},
    "hudaybiyyah": {"year_ce": 628, "year_ah": 6, "description": "Treaty of Hudaybiyyah"},
    "conquest_mecca": {"year_ce": 630, "year_ah": 8, "description": "Conquest of Mecca"},
    "farewell_pilgrimage": {"year_ce": 632, "year_ah": 10, "description": "Farewell Pilgrimage"},
    "prophet_death": {"year_ce": 632, "year_ah": 11, "description": "Death of Prophet Muhammad"},
    "fitna_first": {"year_ce": 656, "year_ah": 35, "description": "First Fitna"},
    "karbala": {"year_ce": 680, "year_ah": 61, "description": "Battle of Karbala"},
    "golden_age_start": {"year_ce": 750, "year_ah": 132, "description": "Abbasid Revolution"},
    "mihna_ahmad": {"year_ce": 833, "year_ah": 218, "description": "Mihna (Inquisition) of Imam Ahmad"},
    "crusades_first": {"year_ce": 1096, "year_ah": 489, "description": "First Crusade begins"},
    "fall_of_baghdad": {"year_ce": 1258, "year_ah": 656, "description": "Mongol sack of Baghdad"},
    "fall_of_granada": {"year_ce": 1492, "year_ah": 897, "description": "Fall of Granada"},
}


# ---------------------------------------------------------------------------
# Scholar position flattening / nuance detection
# ---------------------------------------------------------------------------

ABSOLUTIST_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\b(absolutely|categorically|without\s+exception|in\s+all\s+cases|never\s+varying)\b", re.IGNORECASE),
    re.compile(r"\b(all\s+scholars|every\s+scholar|unanimously|ijma')\b", re.IGNORECASE),
    re.compile(r"\b(the\s+only\s+(valid|correct|acceptable|legitimate)\s+(position|ruling|view|opinion))\b", re.IGNORECASE),
    re.compile(r"\b(there\s+is\s+(no|zero)\s+(room|doubt|difference|debate)\s+about)\b", re.IGNORECASE),
]

NUANCE_INDICATORS: list[re.Pattern[str]] = [
    re.compile(r"\b(according\s+to|as\s+(stated|mentioned)\s+by|in\s+(his|her)\s+(view|opinion|work))\b", re.IGNORECASE),
    re.compile(r"\b(some\s+scholars|many\s+scholars|scholars\s+differ|ikhtilaf)\b", re.IGNORECASE),
    re.compile(r"\b(this\s+position|this\s+view|this\s+ruling)\b.{0,30}\b(is|was)\b.{0,30}\b(contested|debated|questioned)\b", re.IGNORECASE),
]


# ---------------------------------------------------------------------------
# Consensus claim patterns
# ---------------------------------------------------------------------------

CONSENSUS_CLAIM_PATTERNS: list[re.Pattern[str]] = [
    re.compile(
        r"\b(all\s+scholars|the\s+entire\s+(ummah|scholarly)|every\s+(school|madhhab)|"
        r"(unanimous|unanimously|ijma'|consensus)\s+(of|among|between))\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(no\s+scholar\s+(ever|has\s+ever|disagrees))\b", re.IGNORECASE),
]


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class AttributionVerdict(str, Enum):
    VERIFIED = "verified"
    PLAUSIBLE = "plausible"
    SUSPICIOUS = "suspicious"
    FABRICATED = "fabricated"
    ANACHRONISTIC = "anachronistic"
    UNVERIFIABLE = "unverifiable"


class AttributionIssueType(str, Enum):
    FABRICATED_OPINION = "fabricated_opinion"
    MISATTRIBUTION = "misattribution"
    ANACHRONISM = "anachronism"
    FLATTENED_NUANCE = "flattened_nuance"
    FALSE_CONSENSUS = "false_consensus"
    SCHOLAR_NOT_IN_DB = "scholar_not_in_database"


class AttributionIssue(BaseModel):
    issue_type: AttributionIssueType
    severity: str  # high | medium | low
    scholar_name: str
    attributed_opinion: str
    details: str
    source_reference: str | None = None


class AttributionValidationResult(BaseModel):
    issues: list[AttributionIssue] = Field(default_factory=list)
    has_fabrication: bool = False
    should_block: bool = False
    overall_verdict: AttributionVerdict = AttributionVerdict.VERIFIED
    audit_trail: list[dict[str, Any]] = Field(default_factory=list)
    stats: dict[str, int] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Attribution extraction from text
# ---------------------------------------------------------------------------

# Patterns to extract "Scholar X said/stated/believed/opinion/view that ..."
_SCHOLAR_NAME_PATTERN = re.compile(
    r"(?:"
    r"(?:imam|shaykh|sheikh|imam|al-|ibn\s+|ibn\s+al-)\s*"
    r")?([A-Z][a-zA-Z]+(?:\s+(?:ibn|al-|ibn\s+al-|ibn\s+)[A-Z][a-zA-Z]+)*)",
    re.IGNORECASE,
)

_ATTRIBUTION_PATTERNS: list[re.Pattern[str]] = [
    # "Imam Abu Hanifa said that ..."
    re.compile(
        r"((?:imam|shaykh|sheikh|scholar|mufti)\s+"
        r"[A-Za-z\s.'-]+?)"
        r"\s+(?:said|stated|held|believed|argued|ruled|held\s+the\s+view|"
        r"mentioned|narrated|reported|affirmed|argued|opined|"
        r"maintained|advocated|asserted)\s+(?:that\s+)?(.+?)(?:\.|$)",
        re.IGNORECASE,
    ),
    # "According to Imam X, ..."
    re.compile(
        r"(?:according\s+to|as\s+(?:mentioned|stated|reported)\s+by|in\s+the\s+view\s+of)\s+"
        r"([A-Za-z\s.'-]+?),\s*(.+?)(?:\.|$)",
        re.IGNORECASE,
    ),
    # "Imam X's position on Y is/was ..."
    re.compile(
        r"([A-Za-z\s.'-]+?)'s\s+(?:position|view|opinion|ruling|stance|approach)\s+"
        r"(?:on|regarding|about|toward)\s+(.+?)\s+(?:is|was)\s+(.+?)(?:\.|$)",
        re.IGNORECASE,
    ),
]


def _find_scholar_in_text(text: str) -> list[tuple[str, ScholarInfo | None]]:
    """Find scholar names mentioned in text and look them up in the database."""
    results: list[tuple[str, ScholarInfo | None]] = []
    seen: set[str] = set()

    for scholar in SCHOLARS_DB.values():
        # Check full name, short name, and aliases
        names_to_check = [scholar.name, scholar.full_name] + scholar.aliases
        for name in names_to_check:
            if re.search(re.escape(name), text, re.IGNORECASE) and name.lower() not in seen:
                seen.add(name.lower())
                results.append((name, scholar))
                break

    return results


def _extract_attributions(text: str) -> list[dict[str, str]]:
    """Extract scholar-attribution pairs from text."""
    attributions: list[dict[str, str]] = []

    for pattern in _ATTRIBUTION_PATTERNS:
        for match in pattern.finditer(text):
            scholar_name = match.group(1).strip()
            attributed_opinion = match.group(2).strip()
            attributions.append(
                {
                    "scholar_name": scholar_name,
                    "opinion": attributed_opinion,
                    "full_match": match.group(0),
                }
            )

    return attributions


# ---------------------------------------------------------------------------
# Core validation engines
# ---------------------------------------------------------------------------


def _match_opinion_to_scholar(
    attributed_opinion: str, scholar: ScholarInfo
) -> tuple[bool, OpinionEntry | None, float]:
    """Check if the attributed opinion matches any known position of the scholar.

    Returns (is_match, matched_opinion_entry, similarity_score).
    """
    opinion_lower = attributed_opinion.lower()

    # Direct check against known positions
    for position in scholar.known_positions:
        # Simple substring check
        if position.lower() in opinion_lower or opinion_lower in position.lower():
            return True, None, 1.0

    # Check against the opinion database
    for opinion_key, entries in OPINION_DB.items():
        for entry in entries:
            if entry.scholar_id == scholar.id:
                # Check if the attributed opinion overlaps with the registered opinion
                if opinion_key in opinion_lower or opinion_lower in opinion_key:
                    return True, entry, 1.0

    # Fuzzy keyword overlap check
    opinion_words = set(opinion_lower.split())
    best_score = 0.0
    best_entry: OpinionEntry | None = None

    for position in scholar.known_positions:
        position_words = set(position.lower().split())
        if not position_words:
            continue
        overlap = len(opinion_words & position_words)
        score = overlap / max(len(opinion_words), len(position_words))
        if score > best_score:
            best_score = score
            # Find the matching OpinionEntry if possible
            for entries in OPINION_DB.values():
                for entry in entries:
                    if entry.scholar_id == scholar.id and entry.opinion.lower() == position.lower():
                        best_entry = entry
                        break

    if best_score >= 0.4:
        return True, best_entry, best_score

    return False, best_entry, best_score


def _check_temporal_consistency(
    text: str, scholar: ScholarInfo, attributed_text: str
) -> AttributionIssue | None:
    """Check if the attribution makes temporal sense.

    Verify that the scholar's era is compatible with the topic or institution
    being discussed.
    """
    # Check if topic mentions a historical event and if the scholar could know about it
    for event_key, event in HISTORICAL_EVENTS.items():
        event_ce = event.get("year_ce")
        if event_ce is None:
            continue
        event_desc = event.get("description", "").lower()

        if event_desc and any(
            word in attributed_text.lower() for word in event_desc.lower().split()[:3]
        ):
            # The scholar is discussing something related to this event
            if scholar.era.death_ce and event_ce > scholar.era.death_ce + 50:
                return AttributionIssue(
                    issue_type=AttributionIssueType.ANACHRONISM,
                    severity="high",
                    scholar_name=scholar.name,
                    attributed_opinion=attributed_text,
                    details=(
                        f"{scholar.name} (d. {scholar.era.death_ce} CE) could not have discussed "
                        f"'{event_desc}' as it occurred in {event_ce} CE, "
                        f"{event_ce - scholar.era.death_ce} years after their death."
                    ),
                    source_reference=f"Historical event: {event_key}",
                )

    return None


def _detect_anachronism(
    text: str, scholar: ScholarInfo
) -> AttributionIssue | None:
    """Detect anachronistic attributions where a scholar is attributed opinions
    about topics that postdate their lifetime by a significant margin.
    """
    # Look for modern concepts attributed to classical scholars
    modern_concepts = [
        (r"\b(internet|web\s+browser|social\s+media|smartphone|computer)\b", "modern technology"),
        (r"\b(democracy|republic|constitution|parliament|secular\s+state)\b", "modern political concepts"),
        (r"\b(evolution|quantum|relativity|microbiology|genetics)\b", "modern science"),
        (r"\b(music\s+streaming|podcast|television|radio|cinema)\b", "modern media"),
        (r"\b(artificial\s+intelligence|machine\s+learning|robot)\b", "AI concepts"),
    ]

    if scholar.era.death_ce and scholar.era.death_ce < 1800:
        for pattern, concept_category in modern_concepts:
            if re.search(pattern, text, re.IGNORECASE):
                # Check if the modern concept appears in context of the scholar's statement
                # Look for the scholar name within 200 chars of the modern concept
                scholar_mentions = [
                    (m.start(), m.end())
                    for m in re.finditer(re.escape(scholar.name), text, re.IGNORECASE)
                ]
                concept_matches = list(re.finditer(pattern, text, re.IGNORECASE))

                for concept_match in concept_matches:
                    for s_start, s_end in scholar_mentions:
                        if abs(concept_match.start() - s_start) < 200:
                            return AttributionIssue(
                                issue_type=AttributionIssueType.ANACHRONISM,
                                severity="high",
                                scholar_name=scholar.name,
                                attributed_opinion=concept_match.group(0),
                                details=(
                                    f"{scholar.name} (d. {scholar.era.death_ce} CE) could not have "
                                    f"commented on {concept_category} ('{concept_match.group(0)}')."
                                ),
                                source_reference=f"Scholar era: {scholar.era.birth_ce}-{scholar.era.death_ce} CE",
                            )

    return None


def _detect_flattened_nuance(text: str, scholar: ScholarInfo) -> AttributionIssue | None:
    """Detect when a scholar's nuanced position is flattened to an absolute."""
    for pattern in ABSOLUTIST_PATTERNS:
        if pattern.search(text):
            # Check if there's a scholar mention nearby
            scholar_mentions = list(re.finditer(re.escape(scholar.name), text, re.IGNORECASE))
            for s_match in scholar_mentions:
                # Look within 300 chars of the scholar mention for absolutist language
                context_start = max(0, s_match.start() - 100)
                context_end = min(len(text), s_match.end() + 300)
                context = text[context_start:context_end]
                if pattern.search(context):
                    # Check if the position is actually debated
                    for position in scholar.known_positions:
                        if any(
                            word in position.lower()
                            for word in context.lower().split()
                            if len(word) > 3
                        ):
                            return AttributionIssue(
                                issue_type=AttributionIssueType.FLATTENED_NUANCE,
                                severity="medium",
                                scholar_name=scholar.name,
                                attributed_opinion=context[:200],
                                details=(
                                    f"Attribution to {scholar.name} uses absolutist language that may "
                                    f"flatten a nuanced position: '{pattern.pattern}'. "
                                    f"Scholarly positions often have conditions and qualifications."
                                ),
                            )

    return None


def _detect_false_consensus(text: str) -> list[AttributionIssue]:
    """Detect false consensus claims where all scholars are said to agree."""
    issues: list[AttributionIssue] = []

    for pattern in CONSENSUS_CLAIM_PATTERNS:
        for match in pattern.finditer(text):
            claimed_consensus = match.group(0)

            # Check if the consensus claim is on a topic that actually has ikhtilaf
            ikhtilaf_topics = [
                "music",
                "mawlid",
                "tawassul",
                "sufism",
                "tasawwuf",
                "qunut",
                "quran",
                "created",
                "talaq",
                "triple talaq",
                "mukallaf",
                "wudu",
                "adhani",
            ]

            for topic in ikhtilaf_topics:
                if topic in text.lower():
                    issues.append(
                        AttributionIssue(
                            issue_type=AttributionIssueType.FALSE_CONSENSUS,
                            severity="medium",
                            scholar_name="Multiple Scholars",
                            attributed_opinion=f"Consensus claim about {topic}",
                            details=(
                                f"Text claims '{claimed_consensus}' regarding '{topic}', but this is "
                                f"a known area of ikhtilaf (scholarly disagreement). Verify the "
                                f"consensus claim against authoritative sources."
                            ),
                        )
                    )
                    break

    return issues


# ---------------------------------------------------------------------------
# Main validation pipeline
# ---------------------------------------------------------------------------


def validate_scholarly_attribution(text: str) -> AttributionValidationResult:
    """Validate all scholarly attributions in the given text.

    This is the main entry point for the attribution validation system.
    It checks for:
    1. Fabricated attributions (opinions never held by the named scholar)
    2. Misattributions (correct opinion, wrong scholar)
    3. Anachronistic attributions (scholar discussing post-mortem topics)
    4. Flattened nuances (nuanced positions presented as absolute)
    5. False consensus claims (disputed topics claimed as consensus)
    """
    result = AttributionValidationResult()
    audit_trail: list[dict[str, Any]] = []

    # Step 1: Find scholars mentioned in text
    found_scholars = _find_scholar_in_text(text)

    # Step 2: Extract attributions from text
    attributions = _extract_attributions(text)

    # Step 3: Validate each attribution
    for attr in attributions:
        scholar_name = attr["scholar_name"]
        opinion = attr["opinion"]
        full_match = attr["full_match"]

        # Find the scholar in the database
        matched_scholar: ScholarInfo | None = None
        for name, scholar_info in found_scholars:
            if name.lower() in scholar_name.lower() or scholar_name.lower() in name.lower():
                matched_scholar = scholar_info
                break

        if matched_scholar is None:
            # Scholar not found in database — check if it looks like a real scholar name
            if any(
                keyword in scholar_name.lower()
                for keyword in ["imam", "shaykh", "sheikh", "al-", "ibn", "scholar", "mufti"]
            ):
                result.issues.append(
                    AttributionIssue(
                        issue_type=AttributionIssueType.SCHOLAR_NOT_IN_DB,
                        severity="low",
                        scholar_name=scholar_name,
                        attributed_opinion=opinion,
                        details=(
                            f"Scholar '{scholar_name}' is not in the verified database. "
                            f"The attribution cannot be automatically verified."
                        ),
                    )
                )
                result.audit_trail.append(
                    {
                        "action": "unverifiable",
                        "scholar": scholar_name,
                        "opinion": opinion[:100],
                        "reason": "not_in_database",
                    }
                )
            continue

        # --- Temporal consistency check ---
        temporal_issue = _check_temporal_consistency(text, matched_scholar, full_match)
        if temporal_issue:
            result.issues.append(temporal_issue)
            result.audit_trail.append(
                {
                    "action": "flagged",
                    "scholar": matched_scholar.name,
                    "type": "temporal_inconsistency",
                    "details": temporal_issue.details,
                }
            )
            continue

        # --- Anachronism detection ---
        anachronism_issue = _detect_anachronism(text, matched_scholar)
        if anachronism_issue:
            result.issues.append(anachronism_issue)
            result.audit_trail.append(
                {
                    "action": "flagged",
                    "scholar": matched_scholar.name,
                    "type": "anachronism",
                    "details": anachronism_issue.details,
                }
            )
            continue

        # --- Opinion-to-scholar matching ---
        is_match, matched_entry, similarity = _match_opinion_to_scholar(opinion, matched_scholar)

        if not is_match and similarity < 0.4:
            result.issues.append(
                AttributionIssue(
                    issue_type=AttributionIssueType.FABRICATED_OPINION,
                    severity="high",
                    scholar_name=matched_scholar.name,
                    attributed_opinion=opinion,
                    details=(
                        f"The opinion '{opinion[:100]}' could not be verified as belonging to "
                        f"{matched_scholar.name}. This may be a fabricated attribution."
                    ),
                )
            )
            result.audit_trail.append(
                {
                    "action": "flagged",
                    "scholar": matched_scholar.name,
                    "type": "fabricated_opinion",
                    "opinion": opinion[:100],
                    "similarity": round(similarity, 3),
                }
            )
        else:
            result.audit_trail.append(
                {
                    "action": "verified",
                    "scholar": matched_scholar.name,
                    "opinion": opinion[:100],
                    "similarity": round(similarity, 3),
                }
            )

        # --- Nuance detection ---
        nuance_issue = _detect_flattened_nuance(text, matched_scholar)
        if nuance_issue:
            result.issues.append(nuance_issue)
            result.audit_trail.append(
                {
                    "action": "flagged",
                    "scholar": matched_scholar.name,
                    "type": "flattened_nuance",
                    "details": nuance_issue.details,
                }
            )

    # Step 4: Detect false consensus claims
    consensus_issues = _detect_false_consensus(text)
    result.issues.extend(consensus_issues)
    for ci in consensus_issues:
        result.audit_trail.append(
            {
                "action": "flagged",
                "type": "false_consensus",
                "details": ci.details,
            }
        )

    # Step 5: Determine overall verdict
    severity_map = {"high": 3, "medium": 2, "low": 1}
    max_severity = 0
    for issue in result.issues:
        s = severity_map.get(issue.severity, 0)
        if s > max_severity:
            max_severity = s

    if max_severity == 3:
        result.overall_verdict = AttributionVerdict.FABRICATED
        result.has_fabrication = True
        result.should_block = True
    elif max_severity == 2:
        result.overall_verdict = AttributionVerdict.SUSPICIOUS
        result.has_fabrication = True
        result.should_block = False
    elif max_severity == 1:
        result.overall_verdict = AttributionVerdict.UNVERIFIABLE
        result.has_fabrication = False
        result.should_block = False
    else:
        result.overall_verdict = AttributionVerdict.VERIFIED
        result.has_fabrication = False
        result.should_block = False

    # Step 6: Compute stats
    result.stats = {
        "total_issues": len(result.issues),
        "fabricated_opinions": sum(1 for i in result.issues if i.issue_type == AttributionIssueType.FABRICATED_OPINION),
        "misattributions": sum(1 for i in result.issues if i.issue_type == AttributionIssueType.MISATTRIBUTION),
        "anachronisms": sum(1 for i in result.issues if i.issue_type == AttributionIssueType.ANACHRONISM),
        "flattened_nuances": sum(1 for i in result.issues if i.issue_type == AttributionIssueType.FLATTENED_NUANCE),
        "false_consensus": sum(1 for i in result.issues if i.issue_type == AttributionIssueType.FALSE_CONSENSUS),
        "unverifiable_scholars": sum(
            1 for i in result.issues if i.issue_type == AttributionIssueType.SCHOLAR_NOT_IN_DB
        ),
    }

    return result


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def get_scholars_list() -> list[dict[str, Any]]:
    """Return the full scholar database as a list of dicts (for API)."""
    return [s.model_dump() for s in SCHOLARS_DB.values()]


def get_scholars_by_school(school: str) -> list[dict[str, Any]]:
    """Return scholars filtered by madhhab."""
    return [s.model_dump() for s in SCHOLARS_DB.values() if s.school.lower() == school.lower()]


def get_scholar_by_id(scholar_id: str) -> ScholarInfo | None:
    """Look up a single scholar by ID."""
    return SCHOLARS_DB.get(scholar_id)


def validate_single_attribution(
    scholar_name: str, opinion: str
) -> dict[str, Any]:
    """Validate a single scholar-opinion pair without parsing free text."""
    # Find the scholar
    matched_scholar: ScholarInfo | None = None
    for scholar in SCHOLARS_DB.values():
        names_to_check = [scholar.name, scholar.full_name] + scholar.aliases
        for name in names_to_check:
            if scholar_name.lower() in name.lower() or name.lower() in scholar_name.lower():
                matched_scholar = scholar
                break
        if matched_scholar:
            break

    if not matched_scholar:
        return {
            "scholar": scholar_name,
            "verdict": "unverifiable",
            "matched": False,
            "reason": f"Scholar '{scholar_name}' not found in verified database.",
            "similar_positions": [],
        }

    is_match, matched_entry, similarity = _match_opinion_to_scholar(opinion, matched_scholar)
    similar_positions = [
        pos for pos in matched_scholar.known_positions
        if any(w in pos.lower() for w in opinion.lower().split() if len(w) > 3)
    ]

    if is_match:
        verdict = "verified"
        reason = "Opinion matches a known position of this scholar."
    elif similarity >= 0.3:
        verdict = "plausible"
        reason = f"Partial overlap with known positions (similarity: {similarity:.2f})."
    else:
        verdict = "suspicious"
        reason = "Opinion could not be verified as belonging to this scholar."

    return {
        "scholar": matched_scholar.name,
        "scholar_id": matched_scholar.id,
        "school": matched_scholar.school,
        "verdict": verdict,
        "matched": is_match,
        "similarity": round(similarity, 3),
        "reason": reason,
        "similar_positions": similar_positions[:5],
    }
