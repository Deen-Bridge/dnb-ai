"""Personalized Quran Memorization Planner System (#252)

Generates intelligent, customized Quran memorization schedules based on user's
available time, current progress, memorization pace, and revision requirements.
Incorporates traditional Hifz pedagogy (Sabaq / Sabqi / Manzil) with modern
adaptive scheduling, difficulty-weighted surah metrics, and milestone celebrations.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field, field_validator, model_validator

from errors import APIException

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/memorization", tags=["memorization-planner"])

# ---------------------------------------------------------------------------
# Quran Structure & Metadata
# ---------------------------------------------------------------------------

TOTAL_QURAN_AYAHS = 6236
TOTAL_QURAN_PAGES = 604
TOTAL_QURAN_JUZ = 30
TOTAL_QURAN_SURAHS = 114

# Traditional Hifz time allocation proportions
SABAQ_RATIO = 0.45  # New memorization (~45% of study time)
SABQI_RATIO = 0.30  # Recent revision / Sabaq Para (~30% of study time)
MANZIL_RATIO = 0.25  # Cumulative review / Dhor (~25% of study time)

DATA_DIR = Path(__file__).parent / "data" / "quran"
SURAH_INDEX_FILE = DATA_DIR / "surah_index.json"


class GoalType(str, Enum):
    """Scope of memorization target."""

    FULL_QURAN = "full_quran"
    JUZ = "juz"
    SPECIFIC_SURAHS = "specific_surahs"
    CUSTOM_RANGE = "custom_range"


class PaceLevel(str, Enum):
    """User baseline memorization speed."""

    RELAXED = "relaxed"  # ~3-5 ayahs / day
    MODERATE = "moderate"  # ~8-12 ayahs / day (~1 page)
    INTENSIVE = "intensive"  # ~15-25 ayahs / day (~2 pages)
    CUSTOM = "custom"  # Explicit user-provided target


class DifficultyTier(str, Enum):
    """Difficulty classification of Surah memorization."""

    EASY = "easy"  # Short verses, high rhyme, e.g., Juz Amma (tier weight 1.0)
    MEDIUM = "medium"  # Moderate length narratives, e.g., Yusuf, Kahf (tier weight 1.3)
    HARD = "hard"  # Long legalistic verses, dense vocabulary, e.g., Baqarah (tier weight 1.6)


DIFFICULTY_WEIGHTS: dict[DifficultyTier, float] = {
    DifficultyTier.EASY: 1.0,
    DifficultyTier.MEDIUM: 1.3,
    DifficultyTier.HARD: 1.6,
}


@dataclass(frozen=True)
class SurahInfo:
    number: int
    name: str
    arabic_name: str
    ayah_count: int
    revelation_place: str
    difficulty: DifficultyTier
    estimated_pages: float
    juz_start: int
    juz_end: int
    notes: str = ""


# Comprehensive embedded dataset for all 114 Surahs with difficulty rating and Juz mapping
_SURAH_DATA: dict[int, SurahInfo] = {
    1: SurahInfo(
        1,
        "Al-Fatihah",
        "Ø§Ù„ÙØ§ØªØ­Ø©",
        7,
        "meccan",
        DifficultyTier.EASY,
        1.0,
        1,
        1,
        "The Opening - Essential foundational",
    ),
    2: SurahInfo(
        2,
        "Al-Baqarah",
        "Ø§Ù„Ø¨Ù‚Ø±Ø©",
        286,
        "medinan",
        DifficultyTier.HARD,
        48.0,
        1,
        3,
        "Longest surah, intricate legal rulings and long ayahs",
    ),
    3: SurahInfo(
        3,
        "Ali 'Imran",
        "Ø¢Ù„ Ø¹Ù…Ø±Ø§Ù†",
        200,
        "medinan",
        DifficultyTier.HARD,
        27.0,
        3,
        4,
        "Extensive theological and historical discourses",
    ),
    4: SurahInfo(
        4,
        "An-Nisa",
        "Ø§Ù„Ù†Ø³Ø§Ø¡",
        176,
        "medinan",
        DifficultyTier.HARD,
        29.5,
        4,
        6,
        "Detailed jurisprudence, inheritance (Fara'id) and social laws",
    ),
    5: SurahInfo(
        5,
        "Al-Ma'idah",
        "Ø§Ù„Ù…Ø§Ø¦Ø¯Ø©",
        120,
        "medinan",
        DifficultyTier.HARD,
        22.0,
        6,
        7,
        "Covenants, dietary laws and legal judgments",
    ),
    6: SurahInfo(
        6,
        "Al-An'am",
        "Ø§Ù„Ø£Ù†Ø¹Ø§Ù…",
        165,
        "meccan",
        DifficultyTier.HARD,
        23.5,
        7,
        8,
        "Deep theological arguments, revealed in a single night",
    ),
    7: SurahInfo(
        7,
        "Al-A'raf",
        "Ø§Ù„Ø£Ø¹Ø±Ø§Ù",
        206,
        "meccan",
        DifficultyTier.HARD,
        26.0,
        8,
        9,
        "Extensive prophetic stories with shared phrasing",
    ),
    8: SurahInfo(
        8,
        "Al-Anfal",
        "Ø§Ù„Ø£Ù†ÙØ§Ù„",
        75,
        "medinan",
        DifficultyTier.MEDIUM,
        10.0,
        9,
        10,
        "Badr and rules of engagement",
    ),
    9: SurahInfo(
        9,
        "At-Tawbah",
        "Ø§Ù„ØªÙˆØ¨Ø©",
        129,
        "medinan",
        DifficultyTier.HARD,
        21.0,
        10,
        11,
        "Tabuk expedition, strict hypocrite distinctions",
    ),
    10: SurahInfo(
        10,
        "Yunus",
        "ÙŠÙˆÙ†Ø³",
        109,
        "meccan",
        DifficultyTier.MEDIUM,
        13.0,
        11,
        11,
        "Prophetic warnings and divine signs",
    ),
    11: SurahInfo(
        11,
        "Hud",
        "Ù‡ÙˆØ¯",
        123,
        "meccan",
        DifficultyTier.HARD,
        14.5,
        11,
        12,
        "Stories of the prophets, high mutashabihat with A'raf",
    ),
    12: SurahInfo(
        12,
        "Yusuf",
        "ÙŠÙˆØ³Ù",
        111,
        "meccan",
        DifficultyTier.MEDIUM,
        14.5,
        12,
        13,
        "Cohesive narrative of Prophet Yusuf",
    ),
    13: SurahInfo(
        13,
        "Ar-Ra'd",
        "Ø§Ù„Ø±Ø¹Ø¯",
        43,
        "medinan",
        DifficultyTier.MEDIUM,
        6.5,
        13,
        13,
        "Cosmic signs and revelation truth",
    ),
    14: SurahInfo(
        14,
        "Ibrahim",
        "Ø¥Ø¨Ø±Ø§Ù‡ÙŠÙ…",
        52,
        "meccan",
        DifficultyTier.MEDIUM,
        7.0,
        13,
        13,
        "Prayer of Ibrahim, gratitude and parables",
    ),
    15: SurahInfo(
        15,
        "Al-Hijr",
        "Ø§Ù„Ø­Ø¬Ø±",
        99,
        "meccan",
        DifficultyTier.EASY,
        5.0,
        14,
        14,
        "Short rhythmic verses, divine preservation of Quran",
    ),
    16: SurahInfo(
        16,
        "An-Nahl",
        "Ø§Ù„Ù†Ø­Ù„",
        128,
        "meccan",
        DifficultyTier.MEDIUM,
        14.5,
        14,
        14,
        "Blessings of Allah and moral conduct",
    ),
    17: SurahInfo(
        17,
        "Al-Isra",
        "Ø§Ù„Ø¥Ø³Ø±Ø§Ø¡",
        111,
        "meccan",
        DifficultyTier.MEDIUM,
        11.5,
        15,
        15,
        "Night journey and ten commandments",
    ),
    18: SurahInfo(
        18,
        "Al-Kahf",
        "Ø§Ù„ÙƒÙ‡Ù",
        110,
        "meccan",
        DifficultyTier.MEDIUM,
        12.0,
        15,
        16,
        "Four central parables, weekly Sunnah recitation",
    ),
    19: SurahInfo(
        19,
        "Maryam",
        "Ù…Ø±ÙŠÙ…",
        98,
        "meccan",
        DifficultyTier.EASY,
        7.5,
        16,
        16,
        "Distinctive musical rhyme scheme ending in 'yya'",
    ),
    20: SurahInfo(
        20,
        "Ta-Ha",
        "Ø·Ù‡",
        135,
        "meccan",
        DifficultyTier.EASY,
        10.0,
        16,
        16,
        "Story of Musa with consistent melodic cadence",
    ),
    21: SurahInfo(
        21,
        "Al-Anbiya",
        "Ø§Ù„Ø£Ù†Ø¨ÙŠØ§Ø¡",
        112,
        "meccan",
        DifficultyTier.MEDIUM,
        10.0,
        17,
        17,
        "Chronicle of eighteen prophets",
    ),
    22: SurahInfo(
        22,
        "Al-Hajj",
        "Ø§Ù„Ø­Ø¬",
        78,
        "medinan",
        DifficultyTier.MEDIUM,
        10.0,
        17,
        17,
        "Pilgrimage rites, prostrations and divine sovereignty",
    ),
    23: SurahInfo(
        23,
        "Al-Mu'minun",
        "Ø§Ù„Ù…Ø¤Ù…Ù†ÙˆÙ†",
        118,
        "meccan",
        DifficultyTier.EASY,
        8.5,
        18,
        18,
        "Characteristics of true believers",
    ),
    24: SurahInfo(
        24,
        "An-Nur",
        "Ø§Ù„Ù†ÙˆØ±",
        64,
        "medinan",
        DifficultyTier.HARD,
        10.5,
        18,
        18,
        "Verse of Light, social modesty, legal rulings",
    ),
    25: SurahInfo(
        25,
        "Al-Furqan",
        "Ø§Ù„ÙØ±Ù‚Ø§Ù†",
        77,
        "meccan",
        DifficultyTier.EASY,
        7.5,
        18,
        19,
        "Attributes of 'Ibad ar-Rahman",
    ),
    26: SurahInfo(
        26,
        "Ash-Shu'ara",
        "Ø§Ù„Ø´Ø¹Ø±Ø§Ø¡",
        227,
        "meccan",
        DifficultyTier.MEDIUM,
        10.0,
        19,
        19,
        "Short verses, frequent refrains across prophetic stories",
    ),
    27: SurahInfo(
        27,
        "An-Naml",
        "Ø§Ù„Ù†Ù…Ù„",
        93,
        "meccan",
        DifficultyTier.MEDIUM,
        9.0,
        19,
        20,
        "Sulaiman, the ant and the Queen of Sheba",
    ),
    28: SurahInfo(
        28,
        "Al-Qasas",
        "Ø§Ù„Ù‚ØµØµ",
        88,
        "meccan",
        DifficultyTier.MEDIUM,
        11.5,
        20,
        20,
        "Detailed early life and mission of Musa",
    ),
    29: SurahInfo(
        29,
        "Al-'Ankabut",
        "Ø§Ù„Ø¹Ù†ÙƒØ¨ÙˆØª",
        69,
        "meccan",
        DifficultyTier.MEDIUM,
        8.5,
        20,
        21,
        "Trials of faith and the spider parable",
    ),
    30: SurahInfo(
        30,
        "Ar-Rum",
        "Ø§Ù„Ø±ÙˆÙ…",
        60,
        "meccan",
        DifficultyTier.MEDIUM,
        7.0,
        21,
        21,
        "Prophecy of Byzantine victory, signs in creation",
    ),
    31: SurahInfo(
        31, "Luqman", "Ù„Ù‚Ù…Ø§Ù†", 34, "meccan", DifficultyTier.EASY, 4.5, 21, 21, "Wisdom of Luqman advising his son"
    ),
    32: SurahInfo(
        32,
        "As-Sajdah",
        "Ø§Ù„Ø³Ø¬Ø¯Ø©",
        30,
        "meccan",
        DifficultyTier.EASY,
        3.5,
        21,
        21,
        "Prostration, recommended Friday Fajr recitation",
    ),
    33: SurahInfo(
        33,
        "Al-Ahzab",
        "Ø§Ù„Ø£Ø­Ø²Ø§Ø¨",
        73,
        "medinan",
        DifficultyTier.HARD,
        10.5,
        21,
        22,
        "Battle of the Trench, mothers of the believers",
    ),
    34: SurahInfo(
        34,
        "Saba",
        "Ø³Ø¨Ø£",
        54,
        "meccan",
        DifficultyTier.MEDIUM,
        6.5,
        22,
        22,
        "Kingdom of Sheba and resurrection proofs",
    ),
    35: SurahInfo(
        35,
        "Fatir",
        "ÙØ§Ø·Ø±",
        45,
        "meccan",
        DifficultyTier.MEDIUM,
        6.0,
        22,
        22,
        "The Originator and ranks of the angels",
    ),
    36: SurahInfo(
        36,
        "Ya-Sin",
        "ÙŠØ³",
        83,
        "meccan",
        DifficultyTier.EASY,
        6.0,
        22,
        23,
        "Heart of the Quran, highly memorable rhythm",
    ),
    37: SurahInfo(
        37,
        "As-Saffat",
        "Ø§Ù„ØµØ§ÙØ§Øª",
        182,
        "meccan",
        DifficultyTier.EASY,
        7.5,
        23,
        23,
        "Short verses, vivid imagery of angelic ranks",
    ),
    38: SurahInfo(
        38, "Sad", "Øµ", 88, "meccan", DifficultyTier.MEDIUM, 5.5, 23, 23, "Prophets Dawud, Sulaiman, Ayyub in trial"
    ),
    39: SurahInfo(
        39,
        "Az-Zumar",
        "Ø§Ù„Ø²Ù…Ø±",
        75,
        "meccan",
        DifficultyTier.MEDIUM,
        9.5,
        23,
        24,
        "Sincere devotion and divine forgiveness",
    ),
    40: SurahInfo(
        40,
        "Ghafir",
        "ØºØ§ÙØ±",
        85,
        "meccan",
        DifficultyTier.MEDIUM,
        10.0,
        24,
        24,
        "First of the Hawamim, believer of Pharaoh's family",
    ),
    41: SurahInfo(
        41,
        "Fussilat",
        "ÙØµÙ„Øª",
        54,
        "meccan",
        DifficultyTier.MEDIUM,
        6.5,
        24,
        25,
        "Clear verses, human testimony against itself",
    ),
    42: SurahInfo(
        42,
        "Ash-Shura",
        "Ø§Ù„Ø´ÙˆØ±Ù‰",
        53,
        "meccan",
        DifficultyTier.MEDIUM,
        6.5,
        25,
        25,
        "Mutual consultation and unified revelation",
    ),
    43: SurahInfo(
        43,
        "Az-Zukhruf",
        "Ø§Ù„Ø²Ø®Ø±Ù",
        89,
        "meccan",
        DifficultyTier.EASY,
        7.0,
        25,
        25,
        "Gold ornaments vs everlasting afterlife",
    ),
    44: SurahInfo(
        44,
        "Ad-Dukhan",
        "Ø§Ù„Ø¯Ø®Ø§Ù†",
        59,
        "meccan",
        DifficultyTier.EASY,
        3.5,
        25,
        25,
        "Night of decree and the visible smoke",
    ),
    45: SurahInfo(
        45,
        "Al-Jathiyah",
        "Ø§Ù„Ø¬Ø§Ø«ÙŠØ©",
        37,
        "meccan",
        DifficultyTier.EASY,
        4.0,
        25,
        26,
        "Kneeling nations before judgment",
    ),
    46: SurahInfo(
        46,
        "Al-Ahqaf",
        "Ø§Ù„Ø£Ø­Ù‚Ø§Ù",
        35,
        "meccan",
        DifficultyTier.MEDIUM,
        4.5,
        26,
        26,
        "The sand dunes and filial piety",
    ),
    47: SurahInfo(
        47,
        "Muhammad",
        "Ù…Ø­Ù…Ø¯",
        38,
        "medinan",
        DifficultyTier.MEDIUM,
        4.5,
        26,
        26,
        "Victory, jihad and descriptions of Jannah rivers",
    ),
    48: SurahInfo(
        48,
        "Al-Fath",
        "Ø§Ù„ÙØªØ­",
        29,
        "medinan",
        DifficultyTier.MEDIUM,
        4.5,
        26,
        26,
        "Treaty of Hudaybiyyah and manifest triumph",
    ),
    49: SurahInfo(
        49,
        "Al-Hujurat",
        "Ø§Ù„Ø­Ø¬Ø±Ø§Øª",
        18,
        "medinan",
        DifficultyTier.MEDIUM,
        2.5,
        26,
        26,
        "Social manners, brotherhood and universal human origin",
    ),
    50: SurahInfo(
        50,
        "Qaf",
        "Ù‚",
        45,
        "meccan",
        DifficultyTier.EASY,
        3.0,
        26,
        26,
        "Resurrection certainty, frequent Friday sermon surah",
    ),
    51: SurahInfo(
        51,
        "Adh-Dhariyat",
        "Ø§Ù„Ø°Ø§Ø±ÙŠØ§Øª",
        60,
        "meccan",
        DifficultyTier.EASY,
        3.0,
        26,
        27,
        "Scattering winds, Ibrahim's guests",
    ),
    52: SurahInfo(
        52,
        "At-Tur",
        "Ø§Ù„Ø·ÙˆØ±",
        49,
        "meccan",
        DifficultyTier.EASY,
        2.5,
        27,
        27,
        "The Mount and vivid delight of Jannah",
    ),
    53: SurahInfo(
        53,
        "An-Najm",
        "Ø§Ù„Ù†Ø¬Ù…",
        62,
        "meccan",
        DifficultyTier.EASY,
        2.5,
        27,
        27,
        "The Star, ascension and Sidrat al-Muntaha",
    ),
    54: SurahInfo(
        54,
        "Al-Qamar",
        "Ø§Ù„Ù‚Ù…Ø±",
        55,
        "meccan",
        DifficultyTier.EASY,
        3.0,
        27,
        27,
        "The Moon, recurring refrain 'And We have certainly made the Quran easy'",
    ),
    55: SurahInfo(
        55,
        "Ar-Rahman",
        "Ø§Ù„Ø±Ø­Ù…Ù†",
        78,
        "medinan",
        DifficultyTier.EASY,
        3.0,
        27,
        27,
        "Known as the bride of the Quran, thirty-one refrains",
    ),
    56: SurahInfo(
        56,
        "Al-Waqi'ah",
        "Ø§Ù„ÙˆØ§Ù‚Ø¹Ø©",
        96,
        "meccan",
        DifficultyTier.EASY,
        3.5,
        27,
        27,
        "The Inevitable Event, three groups of humanity",
    ),
    57: SurahInfo(
        57,
        "Al-Hadid",
        "Ø§Ù„Ø­Ø¯ÙŠØ¯",
        29,
        "medinan",
        DifficultyTier.MEDIUM,
        4.5,
        27,
        27,
        "Iron, sacrifice and internal light on the Day of Judgment",
    ),
    58: SurahInfo(
        58,
        "Al-Mujadila",
        "Ø§Ù„Ù…Ø¬Ø§Ø¯Ù„Ø©",
        22,
        "medinan",
        DifficultyTier.MEDIUM,
        3.5,
        28,
        28,
        "Every verse contains the divine name Allah",
    ),
    59: SurahInfo(
        59,
        "Al-Hashr",
        "Ø§Ù„Ø­Ø´Ø±",
        24,
        "medinan",
        DifficultyTier.MEDIUM,
        3.5,
        28,
        28,
        "Exile of Banu Nadir, concluding with the supreme divine names",
    ),
    60: SurahInfo(
        60,
        "Al-Mumtahanah",
        "Ø§Ù„Ù…Ù…ØªØ­Ù†Ø©",
        13,
        "medinan",
        DifficultyTier.MEDIUM,
        2.5,
        28,
        28,
        "Testing faith and international relations",
    ),
    61: SurahInfo(
        61, "As-Saff", "Ø§Ù„ØµÙ", 14, "medinan", DifficultyTier.EASY, 1.5, 28, 28, "Solid ranks, prophecy of Ahmad"
    ),
    62: SurahInfo(
        62,
        "Al-Jumu'ah",
        "Ø§Ù„Ø¬Ù…Ø¹Ø©",
        11,
        "medinan",
        DifficultyTier.EASY,
        1.5,
        28,
        28,
        "Friday prayer, donkey carrying books parable",
    ),
    63: SurahInfo(
        63,
        "Al-Munafiqun",
        "Ø§Ù„Ù…Ù†Ø§ÙÙ‚ÙˆÙ†",
        11,
        "medinan",
        DifficultyTier.EASY,
        1.5,
        28,
        28,
        "Hypocrite psychology and heedlessness warning",
    ),
    64: SurahInfo(
        64,
        "At-Taghabun",
        "Ø§Ù„ØªØºØ§Ø¨Ù†",
        18,
        "medinan",
        DifficultyTier.EASY,
        2.0,
        28,
        28,
        "Day of Mutual Loss and Gain",
    ),
    65: SurahInfo(
        65,
        "At-Talaq",
        "Ø§Ù„Ø·Ù„Ø§Ù‚",
        12,
        "medinan",
        DifficultyTier.MEDIUM,
        2.0,
        28,
        28,
        "Divorce rulings and promises of divine relief",
    ),
    66: SurahInfo(
        66,
        "At-Tahrim",
        "Ø§Ù„ØªØ­Ø±ÙŠÙ…",
        12,
        "medinan",
        DifficultyTier.MEDIUM,
        2.0,
        28,
        28,
        "Domestic propriety and exemplary believing women",
    ),
    67: SurahInfo(
        67,
        "Al-Mulk",
        "Ø§Ù„Ù…Ù„Ùƒ",
        30,
        "meccan",
        DifficultyTier.EASY,
        2.5,
        29,
        29,
        "Sovereignty, nightly protection from grave punishment",
    ),
    68: SurahInfo(
        68,
        "Al-Qalam",
        "Ø§Ù„Ù‚Ù„Ù…",
        52,
        "meccan",
        DifficultyTier.EASY,
        2.0,
        29,
        29,
        "The Pen, Prophet's character, parable of the blighted garden",
    ),
    69: SurahInfo(
        69,
        "Al-Haqqah",
        "Ø§Ù„Ø­Ø§Ù‚Ø©",
        52,
        "meccan",
        DifficultyTier.EASY,
        2.0,
        29,
        29,
        "The Indubitable Reality and record of deeds",
    ),
    70: SurahInfo(
        70,
        "Al-Ma'arij",
        "Ø§Ù„Ù…Ø¹Ø§Ø±Ø¬",
        44,
        "meccan",
        DifficultyTier.EASY,
        2.0,
        29,
        29,
        "Ascending stairways and human vulnerability",
    ),
    71: SurahInfo(
        71,
        "Nuh",
        "Ù†ÙˆØ­",
        28,
        "meccan",
        DifficultyTier.EASY,
        1.5,
        29,
        29,
        "Nine hundred and fifty years of patient calling",
    ),
    72: SurahInfo(
        72,
        "Al-Jinn",
        "Ø§Ù„Ø¬Ù†",
        28,
        "meccan",
        DifficultyTier.MEDIUM,
        2.0,
        29,
        29,
        "Unseen realm listening to Quranic recitation",
    ),
    73: SurahInfo(
        73,
        "Al-Muzzammil",
        "Ø§Ù„Ù…Ø²Ù…Ù„",
        20,
        "meccan",
        DifficultyTier.EASY,
        1.5,
        29,
        29,
        "Night prayer and measured recitation (Tartil)",
    ),
    74: SurahInfo(
        74,
        "Al-Muddaththir",
        "Ø§Ù„Ù…Ø¯Ø«Ø±",
        56,
        "meccan",
        DifficultyTier.EASY,
        2.0,
        29,
        29,
        "Arise and warn, short dynamic verses",
    ),
    75: SurahInfo(
        75,
        "Al-Qiyamah",
        "Ø§Ù„Ù‚ÙŠØ§Ù…Ø©",
        40,
        "meccan",
        DifficultyTier.EASY,
        1.5,
        29,
        29,
        "Self-reproaching soul and resurrection",
    ),
    76: SurahInfo(
        76,
        "Al-Insan",
        "Ø§Ù„Ø¥Ù†Ø³Ø§Ù†",
        31,
        "medinan",
        DifficultyTier.EASY,
        2.0,
        29,
        29,
        "Creation of man and delights of Paradise",
    ),
    77: SurahInfo(
        77,
        "Al-Mursalat",
        "Ø§Ù„Ù…Ø±Ø³Ù„Ø§Øª",
        50,
        "meccan",
        DifficultyTier.EASY,
        2.0,
        29,
        29,
        "Winds sent forth with recurring woe to deniers",
    ),
    78: SurahInfo(
        78,
        "An-Naba",
        "Ø§Ù„Ù†Ø¨Ø£",
        40,
        "meccan",
        DifficultyTier.EASY,
        1.5,
        30,
        30,
        "The Great Tidings, opening of Juz Amma",
    ),
    79: SurahInfo(
        79,
        "An-Nazi'at",
        "Ø§Ù„Ù†Ø§Ø²Ø¹Ø§Øª",
        46,
        "meccan",
        DifficultyTier.EASY,
        1.5,
        30,
        30,
        "Angels tearing out souls, story of Musa and Pharaoh",
    ),
    80: SurahInfo(
        80,
        "'Abasa",
        "Ø¹Ø¨Ø³",
        42,
        "meccan",
        DifficultyTier.EASY,
        1.5,
        30,
        30,
        "He frowned, priority of sincere spiritual seeker",
    ),
    81: SurahInfo(
        81,
        "At-Takwir",
        "Ø§Ù„ØªÙƒÙˆÙŠØ±",
        29,
        "meccan",
        DifficultyTier.EASY,
        1.0,
        30,
        30,
        "The Enfolding of the sun and stars",
    ),
    82: SurahInfo(
        82,
        "Al-Infitar",
        "Ø§Ù„Ø§Ù†ÙØ·Ø§Ø±",
        19,
        "meccan",
        DifficultyTier.EASY,
        1.0,
        30,
        30,
        "The Cleaving Asunder and guardian recording angels",
    ),
    83: SurahInfo(
        83,
        "Al-Mutaffifin",
        "Ø§Ù„Ù…Ø·ÙÙÙŠÙ†",
        36,
        "meccan",
        DifficultyTier.EASY,
        1.0,
        30,
        30,
        "Those who give short measure and Sijjin vs 'Illiyyin",
    ),
    84: SurahInfo(
        84,
        "Al-Inshiqaq",
        "Ø§Ù„Ø§Ù†Ø´Ù‚Ø§Ù‚",
        25,
        "meccan",
        DifficultyTier.EASY,
        1.0,
        30,
        30,
        "The Splitting Sky and receiving the record in right hand",
    ),
    85: SurahInfo(
        85,
        "Al-Buruj",
        "Ø§Ù„Ø¨Ø±ÙˆØ¬",
        22,
        "meccan",
        DifficultyTier.EASY,
        1.0,
        30,
        30,
        "Constellations and people of the ditch (Ashab al-Ukhdud)",
    ),
    86: SurahInfo(
        86,
        "At-Tariq",
        "Ø§Ù„Ø·Ø§Ø±Ù‚",
        17,
        "meccan",
        DifficultyTier.EASY,
        1.0,
        30,
        30,
        "The Night-Comer and human origin",
    ),
    87: SurahInfo(
        87,
        "Al-A'la",
        "Ø§Ù„Ø£Ø¹Ù„Ù‰",
        19,
        "meccan",
        DifficultyTier.EASY,
        1.0,
        30,
        30,
        "The Most High, regular Sunnah in Witr and Eid",
    ),
    88: SurahInfo(
        88,
        "Al-Ghashiyah",
        "Ø§Ù„ØºØ§Ø´ÙŠØ©",
        26,
        "meccan",
        DifficultyTier.EASY,
        1.0,
        30,
        30,
        "The Overwhelming Event, contrasts of hell and paradise",
    ),
    89: SurahInfo(
        89,
        "Al-Fajr",
        "Ø§Ù„ÙØ¬Ø±",
        30,
        "meccan",
        DifficultyTier.EASY,
        1.0,
        30,
        30,
        "The Dawn, ten nights and the soul at peace",
    ),
    90: SurahInfo(
        90,
        "Al-Balad",
        "Ø§Ù„Ø¨Ù„Ø¯",
        20,
        "meccan",
        DifficultyTier.EASY,
        1.0,
        30,
        30,
        "The City and ascending the steep path of charity",
    ),
    91: SurahInfo(
        91,
        "Ash-Shams",
        "Ø§Ù„Ø´Ù…Ø³",
        15,
        "meccan",
        DifficultyTier.EASY,
        1.0,
        30,
        30,
        "The Sun and eleven oaths on human purification",
    ),
    92: SurahInfo(
        92,
        "Al-Layl",
        "Ø§Ù„Ù„ÙŠÙ„",
        21,
        "meccan",
        DifficultyTier.EASY,
        1.0,
        30,
        30,
        "The Night and contrasting fates of generous vs stingy",
    ),
    93: SurahInfo(
        93,
        "Ad-Duha",
        "Ø§Ù„Ø¶Ø­Ù‰",
        11,
        "meccan",
        DifficultyTier.EASY,
        0.5,
        30,
        30,
        "The Morning Light, comfort to the Prophet",
    ),
    94: SurahInfo(
        94,
        "Ash-Sharh",
        "Ø§Ù„Ø´Ø±Ø­",
        8,
        "meccan",
        DifficultyTier.EASY,
        0.5,
        30,
        30,
        "Expansion of the chest and relief with hardship",
    ),
    95: SurahInfo(
        95,
        "At-Tin",
        "Ø§Ù„ØªÙŠÙ†",
        8,
        "meccan",
        DifficultyTier.EASY,
        0.5,
        30,
        30,
        "The Fig, Olive and creation in best mold",
    ),
    96: SurahInfo(
        96,
        "Al-'Alaq",
        "Ø§Ù„Ø¹Ù„Ù‚",
        19,
        "meccan",
        DifficultyTier.EASY,
        1.0,
        30,
        30,
        "First revelation: 'Read in the name of your Lord'",
    ),
    97: SurahInfo(
        97,
        "Al-Qadr",
        "Ø§Ù„Ù‚Ø¯Ø±",
        5,
        "meccan",
        DifficultyTier.EASY,
        0.5,
        30,
        30,
        "The Night of Power, better than a thousand months",
    ),
    98: SurahInfo(
        98,
        "Al-Bayyinah",
        "Ø§Ù„Ø¨ÙŠÙ†Ø©",
        8,
        "medinan",
        DifficultyTier.EASY,
        1.0,
        30,
        30,
        "The Clear Evidence and best of creation",
    ),
    99: SurahInfo(
        99,
        "Az-Zalzalah",
        "Ø§Ù„Ø²Ù„Ø²Ù„Ø©",
        8,
        "medinan",
        DifficultyTier.EASY,
        0.5,
        30,
        30,
        "The Earthquakes and atom's weight of deed",
    ),
    100: SurahInfo(
        100,
        "Al-'Adiyat",
        "Ø§Ù„Ø¹Ø§Ø¯ÙŠØ§Øª",
        11,
        "meccan",
        DifficultyTier.EASY,
        0.5,
        30,
        30,
        "The Charging Steeds and human ingratitude",
    ),
    101: SurahInfo(
        101,
        "Al-Qari'ah",
        "Ø§Ù„Ù‚Ø§Ø±Ø¹Ø©",
        11,
        "meccan",
        DifficultyTier.EASY,
        0.5,
        30,
        30,
        "The Striking Calamity and scales of deeds",
    ),
    102: SurahInfo(
        102,
        "At-Takathur",
        "Ø§Ù„ØªÙƒØ§Ø«Ø±",
        8,
        "meccan",
        DifficultyTier.EASY,
        0.5,
        30,
        30,
        "Rivalry in worldly increase until visiting graves",
    ),
    103: SurahInfo(
        103,
        "Al-'Asr",
        "Ø§Ù„Ø¹ØµØ±",
        3,
        "meccan",
        DifficultyTier.EASY,
        0.5,
        30,
        30,
        "Time, loss of humanity except those of faith and truth",
    ),
    104: SurahInfo(
        104,
        "Al-Humazah",
        "Ø§Ù„Ù‡Ù…Ø²Ø©",
        9,
        "meccan",
        DifficultyTier.EASY,
        0.5,
        30,
        30,
        "The Slanderer and wealth hoarding illusion",
    ),
    105: SurahInfo(
        105,
        "Al-Fil",
        "Ø§Ù„ÙÙŠÙ„",
        5,
        "meccan",
        DifficultyTier.EASY,
        0.5,
        30,
        30,
        "The Elephant army and divine intervention",
    ),
    106: SurahInfo(
        106,
        "Quraysh",
        "Ù‚Ø±ÙŠØ´",
        4,
        "meccan",
        DifficultyTier.EASY,
        0.5,
        30,
        30,
        "Security of Quraysh journeys and worship of Lord of Ka'bah",
    ),
    107: SurahInfo(
        107,
        "Al-Ma'un",
        "Ø§Ù„Ù…Ø§Ø¹ÙˆÙ†",
        7,
        "meccan",
        DifficultyTier.EASY,
        0.5,
        30,
        30,
        "Small kindnesses and true piety",
    ),
    108: SurahInfo(
        108,
        "Al-Kawthar",
        "Ø§Ù„ÙƒÙˆØ«Ø±",
        3,
        "meccan",
        DifficultyTier.EASY,
        0.5,
        30,
        30,
        "The Abundant Good bestowed on the Prophet",
    ),
    109: SurahInfo(
        109,
        "Al-Kafirun",
        "Ø§Ù„ÙƒØ§ÙØ±ÙˆÙ†",
        6,
        "meccan",
        DifficultyTier.EASY,
        0.5,
        30,
        30,
        "Disavowal of idolatry, pure monotheism",
    ),
    110: SurahInfo(
        110,
        "An-Nasr",
        "Ø§Ù„Ù†ØµØ±",
        3,
        "medinan",
        DifficultyTier.EASY,
        0.5,
        30,
        30,
        "Divine aid, conquest and glorification with forgiveness",
    ),
    111: SurahInfo(
        111,
        "Al-Masad",
        "Ø§Ù„Ù…Ø³Ø¯",
        5,
        "meccan",
        DifficultyTier.EASY,
        0.5,
        30,
        30,
        "The Palm Fiber rope, ruin of Abu Lahab",
    ),
    112: SurahInfo(
        112,
        "Al-Ikhlas",
        "Ø§Ù„Ø¥Ø®Ù„Ø§Øµ",
        4,
        "meccan",
        DifficultyTier.EASY,
        0.5,
        30,
        30,
        "Pure sincerity, equals one-third of the Quran",
    ),
    113: SurahInfo(
        113,
        "Al-Falaq",
        "Ø§Ù„ÙÙ„Ù‚",
        5,
        "meccan",
        DifficultyTier.EASY,
        0.5,
        30,
        30,
        "Daybreak, refuge from external evils and envy",
    ),
    114: SurahInfo(
        114,
        "An-Nas",
        "Ø§Ù„Ù†Ø§Ø³",
        6,
        "meccan",
        DifficultyTier.EASY,
        0.5,
        30,
        30,
        "Mankind, refuge from internal whispers of Shaytan",
    ),
}


def get_surah_info(number: int) -> SurahInfo | None:
    return _SURAH_DATA.get(number)


def get_juz_surahs(juz: int) -> list[SurahInfo]:
    """Return all surahs that intersect with the specified Juz."""
    if not (1 <= juz <= 30):
        return []
    return [s for s in _SURAH_DATA.values() if s.juz_start <= juz <= s.juz_end]


# ---------------------------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------------------------


class AyahRange(BaseModel):
    surah_number: int = Field(..., ge=1, le=114, description="Surah number (1-114)")
    surah_name: str = Field(..., description="Transliterated Surah name")
    start_ayah: int = Field(..., ge=1, description="Starting Ayah index")
    end_ayah: int = Field(..., ge=1, description="Ending Ayah index")
    ayah_count: int = Field(..., ge=1, description="Number of ayahs in this range")

    @model_validator(mode="after")
    def validate_range(self) -> AyahRange:
        if self.end_ayah < self.start_ayah:
            raise ValueError(f"end_ayah ({self.end_ayah}) cannot be less than start_ayah ({self.start_ayah})")
        info = get_surah_info(self.surah_number)
        if info and self.end_ayah > info.ayah_count:
            raise ValueError(f"Surah {self.surah_number} ({info.name}) only has {info.ayah_count} ayahs")
        return self


class UserCapacity(BaseModel):
    daily_available_minutes: int = Field(30, ge=10, le=240, description="Minutes available per day")
    days_per_week: int = Field(6, ge=1, le=7, description="Number of active memorization days per week")
    pace_level: PaceLevel = Field(PaceLevel.MODERATE, description="Memorization pace preference")
    custom_daily_ayahs: int | None = Field(
        None, ge=1, le=100, description="Explicit ayahs/day when pace_level is custom"
    )

    @field_validator("days_per_week")
    @classmethod
    def validate_days(cls, v: int) -> int:
        if not (1 <= v <= 7):
            raise ValueError("days_per_week must be between 1 and 7")
        return v


class PlanGoal(BaseModel):
    goal_type: GoalType = Field(GoalType.JUZ, description="Type of memorization goal")
    target_juz: list[int] | None = Field(default_factory=lambda: [30], description="Target Juz numbers (1-30)")
    target_surahs: list[int] | None = Field(None, description="Target Surah numbers (1-114)")
    custom_start: tuple[int, int] | None = Field(None, description="(surah, ayah) start for custom range")
    custom_end: tuple[int, int] | None = Field(None, description="(surah, ayah) end for custom range")
    already_memorized_surahs: list[int] = Field(default_factory=list, description="Surah numbers already memorized")


class DailySessionPlan(BaseModel):
    day_number: int = Field(..., description="Day index within the schedule")
    date_str: str = Field(..., description="Scheduled date (YYYY-MM-DD)")
    is_rest_day: bool = Field(False, description="Whether this day is a designated rest/catch-up day")
    is_consolidation_day: bool = Field(False, description="Whether this day is dedicated to revision and testing")
    sabaq_new: AyahRange | None = Field(None, description="New memorization portion (Sabaq)")
    sabaq_minutes: int = Field(0, description="Allocated minutes for new memorization")
    sabqi_recent: list[AyahRange] = Field(default_factory=list, description="Recent revision portion (Sabqi)")
    sabqi_minutes: int = Field(0, description="Allocated minutes for recent revision")
    manzil_cumulative: list[AyahRange] = Field(default_factory=list, description="Cumulative review portion (Manzil)")
    manzil_minutes: int = Field(0, description="Allocated minutes for cumulative review")
    total_minutes: int = Field(..., description="Total session duration in minutes")
    focal_tip: str = Field(..., description="Pedagogical tip and tajweed/mutashabihat focus")


class Milestone(BaseModel):
    id: str
    title: str
    target_description: str
    estimated_day: int
    estimated_date: str
    citation: str
    celebration_message: str


class GeneratePlanRequest(BaseModel):
    user_id: str = Field("default_user", min_length=1, description="User identifier")
    goal: PlanGoal
    capacity: UserCapacity
    start_date: str | None = Field(None, description="Start date in YYYY-MM-DD format (defaults to today)")


class GeneratePlanResponse(BaseModel):
    user_id: str
    plan_id: str
    created_at: str
    goal_summary: str
    total_target_ayahs: int
    total_target_pages: float
    difficulty_profile: dict[str, int]
    daily_capacity_ayahs: int
    estimated_study_days: int
    estimated_calendar_days: int
    estimated_completion_date: str
    daily_time_split: dict[str, int]
    weekly_schedule: list[DailySessionPlan]
    milestones: list[Milestone]


# ---------------------------------------------------------------------------
# Adaptive Plan Adjustment Schemas
# ---------------------------------------------------------------------------


class SessionPerformance(str, Enum):
    STRUGGLING = "struggling"  # Frequent stumbling, needed multiple prompts (Rating 1-2)
    MODERATE = "moderate"  # Finished with minor hesitations (Rating 3)
    STRONG = "strong"  # Smooth, accurate, confident recall (Rating 4-5)


class SessionLog(BaseModel):
    date_str: str
    completed: bool
    performance: SessionPerformance = SessionPerformance.STRONG
    actual_minutes: int | None = None
    notes: str | None = None


class AdjustPlanRequest(BaseModel):
    plan_id: str
    completed_ayahs_count: int = Field(..., ge=0, description="Total ayahs successfully memorized so far")
    recent_sessions: list[SessionLog] = Field(default_factory=list, description="Recent session history")
    updated_capacity: UserCapacity | None = Field(None, description="New user time availability or pace")
    days_missed: int = Field(0, ge=0, description="Count of missed days since last update")


class AdjustPlanResponse(BaseModel):
    plan_id: str
    status: str
    adjustment_action: str
    retention_health: str
    recommended_pace_ayahs: int
    new_estimated_completion_date: str
    days_added_or_subtracted: int
    immediate_next_steps: list[str]
    encouragement: str


# ---------------------------------------------------------------------------
# Milestone Celebration Schemas
# ---------------------------------------------------------------------------


class CheckMilestoneRequest(BaseModel):
    plan_id: str
    current_memorized_ayahs: int = Field(..., ge=0)
    current_streak_days: int = Field(0, ge=0)
    recently_finished_surah: int | None = Field(None, ge=1, le=114)


class MilestoneCelebration(BaseModel):
    achieved: bool
    milestone_title: str
    badge: str
    celebration_message: str
    authentic_citation: str
    next_objective: str


# ---------------------------------------------------------------------------
# Core Engine & Planning Logic
# ---------------------------------------------------------------------------


class MemorizationPlannerEngine:
    """Core domain calculator and scheduler for Quran memorization."""

    @staticmethod
    def resolve_target_surahs(goal: PlanGoal) -> list[SurahInfo]:
        """Determine the exact ordered list of Surahs to be memorized, excluding completed."""
        candidates: list[SurahInfo] = []
        already_set = set(goal.already_memorized_surahs)

        if goal.goal_type == GoalType.FULL_QURAN:
            candidates = [s for s in _SURAH_DATA.values() if s.number not in already_set]
        elif goal.goal_type == GoalType.JUZ:
            juz_set = set(goal.target_juz or [30])
            matched_numbers = set()
            for j in sorted(juz_set):
                for s in get_juz_surahs(j):
                    if s.number not in already_set and s.number not in matched_numbers:
                        matched_numbers.add(s.number)
                        candidates.append(s)
        elif goal.goal_type == GoalType.SPECIFIC_SURAHS:
            for num in goal.target_surahs or []:
                surah_entry = get_surah_info(num)
                if surah_entry and surah_entry.number not in already_set:
                    candidates.append(surah_entry)
        elif goal.goal_type == GoalType.CUSTOM_RANGE and goal.custom_start and goal.custom_end:
            start_s, end_s = goal.custom_start[0], goal.custom_end[0]
            for num in range(start_s, end_s + 1):
                surah_entry = get_surah_info(num)
                if surah_entry and surah_entry.number not in already_set:
                    candidates.append(surah_entry)
        else:
            # Default to Juz 30 (Amma)
            candidates = [s for s in get_juz_surahs(30) if s.number not in already_set]

        # Sort Juz 30 in pedagogical order (from An-Nas backwards to An-Naba if targeting Juz 30)
        if goal.goal_type == GoalType.JUZ and goal.target_juz == [30]:
            candidates.sort(key=lambda x: x.number, reverse=True)

        return candidates

    @staticmethod
    def calculate_base_daily_ayahs(capacity: UserCapacity) -> int:
        """Calculate nominal daily ayah quota based on pace level and available time."""
        if capacity.pace_level == PaceLevel.CUSTOM and capacity.custom_daily_ayahs:
            return capacity.custom_daily_ayahs

        time_factor = capacity.daily_available_minutes / 30.0

        if capacity.pace_level == PaceLevel.RELAXED:
            base = 4.0
        elif capacity.pace_level == PaceLevel.INTENSIVE:
            base = 18.0
        else:  # MODERATE
            base = 10.0

        return max(2, int(round(base * time_factor)))

    @classmethod
    def generate_plan(cls, req: GeneratePlanRequest) -> GeneratePlanResponse:
        surahs = cls.resolve_target_surahs(req.goal)
        if not surahs:
            raise APIException(
                status_code=400,
                detail="No pending surahs found for the specified goal or all chosen surahs are already memorized.",
                hint="Verify target_surahs, target_juz, or already_memorized_surahs parameters.",
            )

        total_ayahs = sum(s.ayah_count for s in surahs)
        total_pages = round(sum(s.estimated_pages for s in surahs), 1)

        diff_profile = {
            "easy": sum(1 for s in surahs if s.difficulty == DifficultyTier.EASY),
            "medium": sum(1 for s in surahs if s.difficulty == DifficultyTier.MEDIUM),
            "hard": sum(1 for s in surahs if s.difficulty == DifficultyTier.HARD),
        }

        # Weighted difficulty adjustment
        weighted_ayah_sum = sum(s.ayah_count * DIFFICULTY_WEIGHTS[s.difficulty] for s in surahs)
        avg_difficulty_multiplier = weighted_ayah_sum / max(1, total_ayahs)

        base_daily_ayahs = cls.calculate_base_daily_ayahs(req.capacity)
        # Difficulty adjusts daily throughput: hard surahs yield slightly fewer ayahs per day to preserve retention
        effective_daily_ayahs = max(2, int(round(base_daily_ayahs / avg_difficulty_multiplier)))

        # Timeline calculations
        effective_active_days_needed = max(1, int(round(total_ayahs / effective_daily_ayahs)))
        # Include consolidation / buffer day
        weekly_active = req.capacity.days_per_week
        total_calendar_days = int(round(effective_active_days_needed * (7.0 / weekly_active)))

        start_dt = date.today()
        if req.start_date:
            try:
                start_dt = datetime.strptime(req.start_date, "%Y-%m-%d").date()
            except ValueError:
                raise APIException(
                    status_code=400,
                    detail="Invalid start_date format. Must be YYYY-MM-DD.",
                    hint="Use format like 2026-10-21.",
                ) from None

        est_completion = start_dt + timedelta(days=total_calendar_days)

        # Time split per session
        sabaq_min = int(round(req.capacity.daily_available_minutes * SABAQ_RATIO))
        sabqi_min = int(round(req.capacity.daily_available_minutes * SABQI_RATIO))
        manzil_min = max(5, req.capacity.daily_available_minutes - sabaq_min - sabqi_min)
        time_split = {
            "sabaq_new_minutes": sabaq_min,
            "sabqi_recent_revision_minutes": sabqi_min,
            "manzil_cumulative_minutes": manzil_min,
        }

        # Generate sample weekly schedule (first 7 calendar days)
        schedule: list[DailySessionPlan] = []
        surah_idx = 0
        current_surah_ayah = 1
        recent_history: list[AyahRange] = []

        for day_i in range(1, 8):
            day_date = start_dt + timedelta(days=day_i - 1)
            is_active_day = day_i <= weekly_active

            if not is_active_day:
                # Rest Day
                schedule.append(
                    DailySessionPlan(
                        day_number=day_i,
                        date_str=day_date.isoformat(),
                        is_rest_day=True,
                        is_consolidation_day=False,
                        sabaq_new=None,
                        sabaq_minutes=0,
                        sabqi_recent=[],
                        sabqi_minutes=0,
                        manzil_cumulative=[],
                        manzil_minutes=0,
                        total_minutes=0,
                        focal_tip="Designated rest and spiritual rejuvenation day. Listen to your memorized portion by a master reciter (e.g., Al-Husary).",
                    )
                )
                continue

            if day_i == weekly_active:
                # Weekly Consolidation & Testing Day
                schedule.append(
                    DailySessionPlan(
                        day_number=day_i,
                        date_str=day_date.isoformat(),
                        is_rest_day=False,
                        is_consolidation_day=True,
                        sabaq_new=None,
                        sabaq_minutes=0,
                        sabqi_recent=recent_history[-3:] if recent_history else [],
                        sabqi_minutes=req.capacity.daily_available_minutes // 2,
                        manzil_cumulative=recent_history[:-3] if len(recent_history) > 3 else recent_history,
                        manzil_minutes=req.capacity.daily_available_minutes // 2,
                        total_minutes=req.capacity.daily_available_minutes,
                        focal_tip="Weekly Consolidation: Recite this week's new verses in voluntary Sunnah prayers to solidify retention.",
                    )
                )
                continue

            # Active memorization day
            if surah_idx < len(surahs):
                current_surah = surahs[surah_idx]
                target_end = min(current_surah.ayah_count, current_surah_ayah + effective_daily_ayahs - 1)
                count = target_end - current_surah_ayah + 1

                new_range = AyahRange(
                    surah_number=current_surah.number,
                    surah_name=current_surah.name,
                    start_ayah=current_surah_ayah,
                    end_ayah=target_end,
                    ayah_count=count,
                )
                recent_history.append(new_range)

                # Prep next day's pointer
                if target_end >= current_surah.ayah_count:
                    surah_idx += 1
                    current_surah_ayah = 1
                else:
                    current_surah_ayah = target_end + 1
            else:
                new_range = None

            # Sabqi: previous 2-3 sessions
            sabqi_portions = recent_history[-3:-1] if len(recent_history) >= 2 else []
            # Manzil: older sessions
            manzil_portions = recent_history[:-3] if len(recent_history) > 3 else []

            tip = (
                f"Sabaq tip: Repeat each ayah at least 15-20 times before linking to the next. "
                f"Notice meaning and word endings in Surah {new_range.surah_name}."
                if new_range
                else "Review and consolidate previously memorized portions."
            )

            schedule.append(
                DailySessionPlan(
                    day_number=day_i,
                    date_str=day_date.isoformat(),
                    is_rest_day=False,
                    is_consolidation_day=False,
                    sabaq_new=new_range,
                    sabaq_minutes=sabaq_min if new_range else 0,
                    sabqi_recent=sabqi_portions,
                    sabqi_minutes=sabqi_min,
                    manzil_cumulative=manzil_portions,
                    manzil_minutes=manzil_min,
                    total_minutes=req.capacity.daily_available_minutes,
                    focal_tip=tip,
                )
            )

        # Milestones setup
        milestones = cls.generate_milestones(surahs, total_ayahs, effective_daily_ayahs, weekly_active, start_dt)

        plan_id = f"plan_{req.user_id}_{int(datetime.now().timestamp())}"
        summary = (
            f"Personalized schedule for {len(surahs)} surah(s), {total_ayahs} ayahs ({total_pages} pages) "
            f"at {effective_daily_ayahs} ayahs/day ({req.capacity.daily_available_minutes} mins, {weekly_active} days/wk)."
        )

        return GeneratePlanResponse(
            user_id=req.user_id,
            plan_id=plan_id,
            created_at=datetime.now().isoformat(),
            goal_summary=summary,
            total_target_ayahs=total_ayahs,
            total_target_pages=total_pages,
            difficulty_profile=diff_profile,
            daily_capacity_ayahs=effective_daily_ayahs,
            estimated_study_days=effective_active_days_needed,
            estimated_calendar_days=total_calendar_days,
            estimated_completion_date=est_completion.isoformat(),
            daily_time_split=time_split,
            weekly_schedule=schedule,
            milestones=milestones,
        )

    @staticmethod
    def generate_milestones(
        surahs: list[SurahInfo],
        total_ayahs: int,
        daily_ayahs: int,
        days_per_week: int,
        start_date: date,
    ) -> list[Milestone]:
        """Generate structured pedagogical milestones with authentic prophetic citations."""
        milestones: list[Milestone] = []
        fraction_targets = [
            ("quarter", 0.25, "First Quarter Milestone (25%)", "Completed 25% of your target memorization goal."),
            (
                "half",
                0.50,
                "Halfway Milestone (50%)",
                "Alhamdulillah! You have completed half of your memorization journey.",
            ),
            ("three_quarters", 0.75, "Three-Quarter Milestone (75%)", "75% completed! The finish line is in sight."),
            (
                "completion",
                1.00,
                "Goal Completion (100%) - Khatm Milestone",
                "Mubarak! You have completed your full memorization goal!",
            ),
        ]

        for m_id, frac, title, desc in fraction_targets:
            ayah_mark = int(round(total_ayahs * frac))
            active_days = max(1, int(round(ayah_mark / max(1, daily_ayahs))))
            calendar_days = int(round(active_days * (7.0 / days_per_week)))
            target_date = start_date + timedelta(days=calendar_days)

            if frac == 0.25:
                citation = "Sahih al-Bukhari 5027: 'The best among you are those who learn the Quran and teach it.'"
                msg = "A strong start is half the battle. Maintain consistent daily review."
            elif frac == 0.50:
                citation = "Sahih Muslim 798: 'The one who recites the Quran skillfully is with the noble obedient angel scribes.'"
                msg = "Your commitment has borne fruit. Keep your Sabqi revision rock solid."
            elif frac == 0.75:
                citation = "Jami` at-Tirmidhi 2910: 'Whoever reads a letter from Allah's Book receives a good deed multiplied tenfold.'"
                msg = "Remain humble, steadfast, and review the beginning portions regularly."
            else:
                citation = "Sunan Abi Dawud 1464: 'Recite and ascend, and recite slowly as you recited in the world, for your rank is at the last verse you recite.'"
                msg = "May Allah make the Holy Quran the light of your heart and a guide throughout your life."

            milestones.append(
                Milestone(
                    id=f"milestone_{m_id}",
                    title=title,
                    target_description=f"{desc} ({ayah_mark}/{total_ayahs} ayahs)",
                    estimated_day=active_days,
                    estimated_date=target_date.isoformat(),
                    citation=citation,
                    celebration_message=msg,
                )
            )

        return milestones

    @classmethod
    def adjust_plan(cls, req: AdjustPlanRequest) -> AdjustPlanResponse:
        """Dynamically adjust pace, revision weighting, and timeline based on actual performance logs."""
        struggle_count = sum(
            1 for s in req.recent_sessions if s.performance == SessionPerformance.STRUGGLING or not s.completed
        )
        strong_count = sum(1 for s in req.recent_sessions if s.performance == SessionPerformance.STRONG and s.completed)
        total_recent = len(req.recent_sessions) or 1

        struggle_ratio = struggle_count / total_recent
        strong_ratio = strong_count / total_recent

        # Base nominal pace
        current_base = 10
        if req.updated_capacity:
            current_base = cls.calculate_base_daily_ayahs(req.updated_capacity)

        days_diff = 0
        immediate_steps: list[str] = []

        if struggle_ratio >= 0.4 or req.days_missed >= 3:
            # User is struggling or has fallen behind -> slow down new pace, increase review
            retention_health = "Needs Consolidation"
            action = "Slow down new memorization pace by 30% and dedicate 2 days to Sabqi revision."
            recommended_pace = max(2, int(round(current_base * 0.70)))
            days_diff = 7 + req.days_missed
            immediate_steps.extend(
                [
                    "Pause new Sabaq for the next 48 hours.",
                    "Review the last 20 ayahs memorized with a Hafiz or recorded audio repetition.",
                    "Recite your recent portions in optional (Nawafil) prayers.",
                ]
            )
            encouragement = (
                "Do not be disheartened. In the words of the Prophet ï·º: 'The one who recites the Quran "
                "with difficulty, stammering over its verses, will receive a double reward.' (Sahih Muslim 798)."
            )
        elif strong_ratio >= 0.8 and req.days_missed == 0:
            # User is excelling effortlessly -> safe optional pace acceleration
            retention_health = "Excellent"
            action = "Pace optimal. Opportunity to safely accelerate new memorization by 15%."
            recommended_pace = int(round(current_base * 1.15))
            days_diff = -3
            immediate_steps.extend(
                [
                    "Maintain your disciplined daily schedule.",
                    "Ensure Manzil cumulative review keeps rotating without neglect.",
                ]
            )
            encouragement = (
                "Masha'Allah, your consistency is commendable. 'Continue to review the Quran, "
                "for it slips away faster than untethered camels.' (Sahih al-Bukhari 5033)."
            )
        else:
            # Normal balanced progression
            retention_health = "Steady"
            action = "Maintain current balanced pace and revision cycle."
            recommended_pace = current_base
            days_diff = req.days_missed
            immediate_steps.extend(
                [
                    "Complete today's designated Sabaq portion.",
                    "Spend at least 15 minutes on recent Sabqi review before sleeping.",
                ]
            )
            encouragement = "Consistency in small deeds is most beloved to Allah. Keep going steadfastly."

        new_est_date = (date.today() + timedelta(days=max(1, 30 + days_diff))).isoformat()

        return AdjustPlanResponse(
            plan_id=req.plan_id,
            status="adjusted",
            adjustment_action=action,
            retention_health=retention_health,
            recommended_pace_ayahs=recommended_pace,
            new_estimated_completion_date=new_est_date,
            days_added_or_subtracted=days_diff,
            immediate_next_steps=immediate_steps,
            encouragement=encouragement,
        )

    @classmethod
    def check_milestone(cls, req: CheckMilestoneRequest, total_goal_ayahs: int = 560) -> MilestoneCelebration:
        """Check if current progress triggers a milestone celebration."""
        progress_ratio = req.current_memorized_ayahs / max(1, total_goal_ayahs)

        if req.recently_finished_surah:
            s_info = get_surah_info(req.recently_finished_surah)
            surah_name = s_info.name if s_info else f"Surah {req.recently_finished_surah}"
            return MilestoneCelebration(
                achieved=True,
                milestone_title=f"Completed {surah_name}!",
                badge="ðŸ† Surah Finisher",
                celebration_message=f"Congratulations on fully memorizing {surah_name}! What an incredible spiritual achievement.",
                authentic_citation="Sahih al-Bukhari 5027: 'The best among you are those who learn the Quran and teach it.'",
                next_objective=f"Dedicate the next session to reciting {surah_name} in its entirety from memory before beginning the next chapter.",
            )

        if req.current_streak_days >= 30:
            return MilestoneCelebration(
                achieved=True,
                milestone_title="30-Day Unbroken Streak!",
                badge="ðŸ”¥ Istiqamah (Steadfastness) Star",
                celebration_message="A full month of daily commitment to the Book of Allah! Your discipline is exemplary.",
                authentic_citation="Sahih al-Bukhari 6464: 'The deeds most loved by Allah are those done regularly, even if small.'",
                next_objective="Continue to protect your daily revision time as sacred.",
            )

        if progress_ratio >= 1.0:
            return MilestoneCelebration(
                achieved=True,
                milestone_title="Goal Completed! Khatm al-Hifz!",
                badge="ðŸ‘‘ Hafiz Milestone",
                celebration_message="All praise is due to Allah! You have completed your memorization target.",
                authentic_citation="Sunan Abi Dawud 1464: 'Recite and ascend in ranks, for your status is at the last verse you recite.'",
                next_objective="Begin the permanent Manzil revision cycle: recite your entire memorized portion weekly or biweekly.",
            )

        if progress_ratio >= 0.5:
            return MilestoneCelebration(
                achieved=True,
                milestone_title="Halfway Mark Achieved (50%)",
                badge="ðŸŒŸ Halfway Champion",
                celebration_message="You have memorized over half of your designated goal! Keep this momentum.",
                authentic_citation="Sahih Muslim 798: 'The one who recites the Quran skillfully is with the noble angels.'",
                next_objective="Conduct a complete revision of the first half before advancing further.",
            )

        return MilestoneCelebration(
            achieved=False,
            milestone_title="In Progress",
            badge="ðŸŒ± Active Learner",
            celebration_message=f"You have memorized {req.current_memorized_ayahs} ayahs so far. Every single letter holds ten rewards.",
            authentic_citation="Jami` at-Tirmidhi 2910: 'Whoever reads a letter from the Book of Allah gets a good deed multiplied tenfold.'",
            next_objective="Keep up your daily Sabaq and Sabqi revision.",
        )


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------


@router.post("/plan", response_model=GeneratePlanResponse)
async def generate_plan(request: GeneratePlanRequest) -> GeneratePlanResponse:
    """Generate a personalized Quran memorization plan with daily/weekly schedules and milestones."""
    return MemorizationPlannerEngine.generate_plan(request)


@router.get("/surahs")
async def list_surahs() -> list[dict[str, Any]]:
    """List all 114 Surahs with difficulty rating, page estimate, and Juz boundaries."""
    return [
        {
            "number": s.number,
            "name": s.name,
            "arabic_name": s.arabic_name,
            "ayah_count": s.ayah_count,
            "revelation_place": s.revelation_place,
            "difficulty": s.difficulty.value,
            "estimated_pages": s.estimated_pages,
            "juz_range": f"Juz {s.juz_start} - {s.juz_end}" if s.juz_start != s.juz_end else f"Juz {s.juz_start}",
            "notes": s.notes,
        }
        for s in _SURAH_DATA.values()
    ]


@router.post("/adjust", response_model=AdjustPlanResponse)
async def adjust_plan(request: AdjustPlanRequest) -> AdjustPlanResponse:
    """Dynamically adjust an ongoing memorization plan based on performance logs, missed days, or capacity changes."""
    return MemorizationPlannerEngine.adjust_plan(request)


@router.post("/milestone-check", response_model=MilestoneCelebration)
async def check_milestone(request: CheckMilestoneRequest) -> MilestoneCelebration:
    """Check milestone achievements and return prophetic celebrations, badges, and next steps."""
    return MemorizationPlannerEngine.check_milestone(request)


# ---------------------------------------------------------------------------
# Muraja'ah Revision Intelligence (#253)
# ---------------------------------------------------------------------------
# Spaced-repetition scheduler layered on the planner's Sabaq/Sabqi/Manzil
# pedagogy. Tracks per-ayah retention, error rate, and generates balanced
# revision sessions. Reuses AyahRange, UserCapacity, SessionLog,
# SessionPerformance, SurahInfo and get_surah_info from this module.


class RecallQuality(int, Enum):
    """SM-2 recall grade (0-5) for a single ayah."""

    BLACKOUT = 0
    WRONG = 1
    HARD = 2
    HESITANT = 3
    GOOD = 4
    PERFECT = 5


class RevisionOutcome(str, Enum):
    COMPLETED = "completed"
    PARTIAL = "partial"
    SKIPPED = "skipped"


MIN_EASE = 1.3
DEFAULT_EASE = 2.5
INITIAL_INTERVAL_DAYS = 1
SECOND_INTERVAL_DAYS = 3
EASY_BONUS = 1.3
HARD_PENALTY = 0.6


class AyahProgress(BaseModel):
    """Retention state for one ayah."""

    surah: int = Field(ge=1, le=114)
    ayah: int = Field(ge=1)
    ease: float = Field(default=DEFAULT_EASE, ge=MIN_EASE)
    interval_days: int = Field(default=INITIAL_INTERVAL_DAYS, ge=1)
    repetitions: int = Field(default=0, ge=0)
    lapses: int = Field(default=0, ge=0)
    error_count: int = Field(default=0, ge=0)
    review_count: int = Field(default=0, ge=0)
    last_reviewed: datetime | None = None
    next_review: date = Field(default_factory=date.today)

    @property
    def key(self) -> str:
        return f"{self.surah}:{self.ayah}"

    @property
    def error_rate(self) -> float:
        return self.error_count / self.review_count if self.review_count else 0.0

    @property
    def is_due(self) -> bool:
        return date.today() >= self.next_review

    @property
    def overdue_days(self) -> int:
        return max(0, (date.today() - self.next_review).days)

    @property
    def age_days(self) -> int:
        if not self.last_reviewed:
            return 0
        return (datetime.now() - self.last_reviewed).days


class RevisionRecordRequest(BaseModel):
    """Log one reviewed ayah."""

    surah: int = Field(ge=1, le=114)
    ayah: int = Field(ge=1)
    quality: RecallQuality
    outcome: RevisionOutcome = RevisionOutcome.COMPLETED


class RevisionSessionRequest(BaseModel):
    """Ask for a balanced revision session."""

    max_minutes: int = Field(default=30, ge=1, le=180)
    max_ayahs: int | None = Field(default=None, ge=1, le=500)


class RevisionSessionResponse(BaseModel):
    ayahs: list[AyahProgress]
    total_estimated_minutes: int
    breakdown: dict[str, int] = Field(description="Counts per tier: critical / weak / maintenance")


class RevisionReminderResponse(BaseModel):
    message: str | None
    due_today: int
    stats: dict[str, Any]


class RevisionStatsResponse(BaseModel):
    total_ayahs: int
    due_today: int
    avg_ease: float
    total_lapses: int
    overall_error_rate: float


def _forgetting_curve(days_since_review: float, stability: float = 1.0) -> float:
    """Exponential retention curve R = exp(-t / S)."""
    if days_since_review <= 0:
        return 1.0
    return math.exp(-days_since_review / max(0.1, stability))


def _priority_score(progress: AyahProgress) -> float:
    """Higher = more urgent. Combines retention decay, overdue, error, age."""
    if progress.last_reviewed is None:
        return 100.0
    now = datetime.now()
    days_since = (now - progress.last_reviewed).total_seconds() / 86400
    stability = progress.ease * max(1, progress.repetitions)
    urgency = 1.0 - _forgetting_curve(days_since, stability)
    overdue = min(progress.overdue_days, 30) / 30.0
    error = min(progress.error_rate, 1.0)
    age = min(progress.age_days, 365) / 365.0
    return (urgency * 3.0) + (overdue * 2.0) + (error * 2.0) + (age * 1.0)


def _sm2_update(progress: AyahProgress, quality: RecallQuality) -> AyahProgress:
    """Apply the SM-2 algorithm to one ayah's retention state."""
    q = int(quality)
    now = datetime.now()
    ease = progress.ease + (0.1 - (5 - q) * (0.08 + (5 - q) * 0.02))
    progress.ease = max(MIN_EASE, ease)

    if q < 3:
        progress.repetitions = 0
        progress.lapses += 1
        progress.error_count += 1
        progress.interval_days = 1
    else:
        progress.repetitions += 1
        if progress.repetitions == 1:
            progress.interval_days = INITIAL_INTERVAL_DAYS
        elif progress.repetitions == 2:
            progress.interval_days = SECOND_INTERVAL_DAYS
        else:
            progress.interval_days = round(progress.interval_days * progress.ease)
        if q == 5:
            progress.interval_days = round(progress.interval_days * EASY_BONUS)
        elif q == 3:
            progress.interval_days = max(1, round(progress.interval_days * HARD_PENALTY))

    progress.review_count += 1
    progress.last_reviewed = now
    progress.next_review = (now + timedelta(days=progress.interval_days)).date()
    return progress


class MurajaahEngine:
    """Stateless-class, per-call scheduler. Callers pass in prior progress."""

    @staticmethod
    def _rebuild(progress: list[AyahProgress]) -> dict[str, AyahProgress]:
        return {p.key: p.model_copy(deep=True) for p in progress}

    @classmethod
    def record(
        cls,
        prior: list[AyahProgress],
        req: RevisionRecordRequest,
    ) -> AyahProgress:
        store = cls._rebuild(prior)
        key = f"{req.surah}:{req.ayah}"
        p = store.get(key) or AyahProgress(surah=req.surah, ayah=req.ayah)
        if req.outcome == RevisionOutcome.SKIPPED:
            p.next_review = date.today() + timedelta(days=1)
            return p
        q = req.quality
        if req.outcome == RevisionOutcome.PARTIAL and int(q) > 2:
            q = RecallQuality.HARD
        return _sm2_update(p, q)

    @classmethod
    def build_session(
        cls,
        prior: list[AyahProgress],
        req: RevisionSessionRequest,
        seconds_per_ayah: int = 20,
    ) -> RevisionSessionResponse:
        store = cls._rebuild(prior)
        due = [p for p in store.values() if p.is_due]
        if not due:
            return RevisionSessionResponse(ayahs=[], total_estimated_minutes=0, breakdown={})

        budget = req.max_minutes * 60 // seconds_per_ayah
        if req.max_ayahs:
            budget = min(budget, req.max_ayahs)

        for p in due:
            p.__dict__["_priority"] = _priority_score(p)
        due.sort(key=lambda p: p.__dict__["_priority"], reverse=True)

        critical = [p for p in due if p.last_reviewed is None or p.overdue_days > 7]
        weak = [p for p in due if p not in critical and p.error_rate > 0.3]
        maintenance = [p for p in due if p not in critical and p not in weak]

        cap_c = max(1, int(budget * 0.5)) if critical else 0
        cap_w = max(1, int(budget * 0.3)) if weak else 0
        cap_m = max(0, budget - cap_c - cap_w)
        chosen = critical[:cap_c] + weak[:cap_w] + maintenance[:cap_m]

        if len(chosen) < budget:
            keys = {p.key for p in chosen}
            for p in due:
                if len(chosen) >= budget:
                    break
                if p.key not in keys:
                    chosen.append(p)

        for p in chosen:
            p.__dict__.pop("_priority", None)

        breakdown = {
            "critical": sum(1 for p in chosen if p in critical),
            "weak": sum(1 for p in chosen if p in weak),
            "maintenance": sum(1 for p in chosen if p in maintenance),
        }
        return RevisionSessionResponse(
            ayahs=chosen,
            total_estimated_minutes=(len(chosen) * seconds_per_ayah) // 60,
            breakdown=breakdown,
        )

    @classmethod
    def reminder(
        cls,
        prior: list[AyahProgress],
    ) -> RevisionReminderResponse:
        store = cls._rebuild(prior)
        due = [p for p in store.values() if p.is_due]
        stats = cls.stats(prior)
        if not due:
            return RevisionReminderResponse(message=None, due_today=0, stats=stats.model_dump())
        worst = max(due, key=_priority_score)
        msg = (
            f"{len(due)} ayah(s) due for Muraja'ah. "
            f"Start with {worst.surah}:{worst.ayah} "
            f"(error rate {worst.error_rate:.0%}, overdue {worst.overdue_days}d)."
        )
        return RevisionReminderResponse(message=msg, due_today=len(due), stats=stats.model_dump())

    @staticmethod
    def stats(prior: list[AyahProgress]) -> RevisionStatsResponse:
        total = len(prior)
        if not total:
            return RevisionStatsResponse(
                total_ayahs=0,
                due_today=0,
                avg_ease=0.0,
                total_lapses=0,
                overall_error_rate=0.0,
            )
        due = sum(1 for p in prior if p.is_due)
        avg_ease = sum(p.ease for p in prior) / total
        reviews = sum(p.review_count for p in prior)
        errors = sum(p.error_count for p in prior)
        return RevisionStatsResponse(
            total_ayahs=total,
            due_today=due,
            avg_ease=round(avg_ease, 2),
            total_lapses=sum(p.lapses for p in prior),
            overall_error_rate=round(errors / max(1, reviews), 3),
        )


@router.post("/revision/session", response_model=RevisionSessionResponse)
async def revision_session(request: RevisionSessionRequest) -> RevisionSessionResponse:
    """Build a balanced Muraja'ah session (critical / weak / maintenance tiers)."""
    return MurajaahEngine.build_session([], request)


@router.get("/revision/reminder", response_model=RevisionReminderResponse)
async def revision_reminder() -> RevisionReminderResponse:
    """Return a reminder for the most urgent due ayah, plus aggregate stats."""
    return MurajaahEngine.reminder([])


@router.get("/revision/stats", response_model=RevisionStatsResponse)
async def revision_stats() -> RevisionStatsResponse:
    """Return aggregate retention statistics."""
    return MurajaahEngine.stats([])
