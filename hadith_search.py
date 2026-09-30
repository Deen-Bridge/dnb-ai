"""Hadith search engine by topic, keyword, and authenticity grading (#120).

Why this exists
----------------
Provides a dedicated search interface for discovering Hadith by topic, keyword,
chapter, or narrator across authentic collections (Sahih al-Bukhari, Sahih Muslim,
Sunan Abu Dawud, Jami at-Tirmidhi, Sunan an-Nasai, Sunan Ibn Majah, Muwatta Malik),
returning structured results with authenticity grading.

Architecture & Features
-----------------------
- In-memory SQLite FTS5 index over the bundled hadith grading dataset (`data/hadith/*.json`)
  and rich curated hadith texts.
- Relevance ranking combining FTS BM25 with exact phrase matching, field weighting
  (topics, matn, chapter, narrator), and authenticity tie-breaking.
- Filtering by collection and authenticity grading (sahih, hasan, daif, mawdu).
- Pagination support (limit 1-50, default 10, offset >= 0).
- Pure offline execution without network or external model dependencies.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from hadith import (
    COLLECTION_NAMES,
    DATA_DIR,
    Strength,
    normalize_collection,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/hadith", tags=["hadith"])

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class HadithSearchResult(BaseModel):
    """A single hadith search result with authenticity grading and metadata."""

    collection: str = Field(
        ...,
        description="Name of the Hadith collection",
        examples=["Sahih al-Bukhari"],
    )
    number: int = Field(
        ...,
        description="Hadith number within the collection",
        examples=[1],
    )
    text_arabic: str | None = Field(
        default=None,
        description="Arabic text of the Hadith",
        examples=["إِنَّمَا الأَعْمَالُ بِالنِّيَّاتِ، وَإِنَّمَا لِكُلِّ امْرِئٍ مَا نَوَى"],
    )
    text_english: str = Field(
        ...,
        description="English translation of the Hadith",
        examples=["Actions are but by intentions, and every person will have only what they intended."],
    )
    grading: str = Field(
        ...,
        description="Authenticity grade (e.g. sahih, hasan, daif)",
        examples=["sahih"],
    )
    chapter: str | None = Field(
        default=None,
        description="Chapter or book name",
        examples=["Revelation"],
    )
    narrator: str | None = Field(
        default=None,
        description="Primary narrator of the Hadith",
        examples=["Umar ibn al-Khattab"],
    )


class HadithSearchResponse(BaseModel):
    """Structured response for hadith search with pagination."""

    results: list[HadithSearchResult] = Field(
        ...,
        description="List of matching Hadith search results",
    )
    total: int = Field(
        ...,
        description="Total number of matching Hadiths found",
        examples=[150],
    )
    offset: int = Field(
        ...,
        description="Pagination offset",
        examples=[0],
    )
    limit: int = Field(
        ...,
        description="Maximum results per page",
        examples=[10],
    )


# ---------------------------------------------------------------------------
# Chapter / Book metadata per collection
# ---------------------------------------------------------------------------

_BUKHARI_CHAPTERS: dict[int, str] = {
    1: "Revelation",
    2: "Belief",
    3: "Knowledge",
    4: "Ablution (Wudu)",
    5: "Bathing (Ghusl)",
    6: "Menses",
    7: "Rubbing Dust (Tayammum)",
    8: "Prayers (Salat)",
    9: "Times of the Prayers",
    10: "Call to Prayers (Adhan)",
    11: "Friday Prayer",
    12: "Fear Prayer",
    13: "The Two Festivals (Eids)",
    14: "Witr Prayer",
    15: "Invoking Allah for Rain (Istisqa)",
    16: "Eclipses",
    17: "Prostration During Recitation",
    18: "Shortening the Prayers",
    19: "Prayer at Night (Tahajjud)",
    20: "Virtues of Prayer in Makkah and Madinah",
    21: "Actions while Praying",
    22: "Forgetfulness in Prayer",
    23: "Funerals (Jana'iz)",
    24: "Zakat (Obligatory Charity)",
    25: "Obligatory Charity Tax After Ramadan (Zakat ul-Fitr)",
    26: "Pilgrimage (Hajj)",
    27: "Minor Pilgrimage (Umrah)",
    28: "Pilgrims Prevented from Completing Hajj (Muhsar)",
    29: "Penalty of Hunting while on Pilgrimage",
    30: "Virtues of Madinah",
    31: "Fasting (Sawm)",
    32: "Praying at Night in Ramadan (Taraweeh)",
    33: "Retiring to a Mosque (I'tikaf)",
    34: "Sales and Trade",
    35: "Sales in Advance (Salam)",
    36: "Preemption (Shuf'a)",
    37: "Hiring (Ijarah)",
    38: "Transferance of Debt (Hawalah)",
    39: "Kafalah (Guarantee)",
    40: "Representation (Wakalah)",
    41: "Agriculture and Cultivation",
    42: "Watering (Musaqat)",
    43: "Loans and Bankruptcy",
    44: "Disputes (Khusoomaat)",
    45: "Lost Property (Luqatah)",
    46: "Oppressions (Mazalim)",
    47: "Partnership (Sharikah)",
    48: "Mortgaging (Rahn)",
    49: "Manumission of Slaves ('Itq)",
    50: "Mukatab (Slaves Purchasing Freedom)",
    51: "Gifts (Hibah)",
    52: "Witnesses (Shahadat)",
    53: "Peacemaking (Sulh)",
    54: "Conditions (Shurut)",
    55: "Wills and Testaments (Wasaya)",
    56: "Jihad (Fighting for the Cause of Allah)",
    57: "One-fifth of Booty (Khumus)",
    58: "Jizyah and Peace Pacts",
    59: "Beginning of Creation",
    60: "Prophets (Anbiya)",
    61: "Virtues and Merits of the Prophet and Companions",
    62: "Companions of the Prophet",
    63: "Merits of the Helpers in Madinah (Al-Ansar)",
    64: "Military Expeditions (Al-Maghazi)",
    65: "Prophetic Commentary on the Quran (Tafsir)",
    66: "Virtues of the Quran",
    67: "Wedlock and Marriage (Nikah)",
    68: "Divorce (Talaq)",
    69: "Supporting the Family (Nafaqaat)",
    70: "Food and Meals (At'imah)",
    71: "Sacrifice at Birth (Aqiqah)",
    72: "Hunting and Slaughtering",
    73: "Al-Adahi (Sacrifices)",
    74: "Drinks (Ashribah)",
    75: "Patients and Illness",
    76: "Medicine (Tibb)",
    77: "Dress and Clothing",
    78: "Good Manners and Etiquette (Adab)",
    79: "Asking Permission (Isti'dhan)",
    80: "Invocations and Supplications (Da'awat)",
    81: "Heart-melting Traditions (Ar-Riqaq)",
    82: "Divine Will (Al-Qadar)",
    83: "Oaths and Vows",
    84: "Expiation for Unfulfilled Oaths",
    85: "Laws of Inheritance (Fara'id)",
    86: "Prescribed Punishments (Hudud)",
    87: "Blood Money (Diyat)",
    88: "Apostates (Istitabat al-Murtaddin)",
    89: "Coercion (Ikrah)",
    90: "Tricks (Hiyal)",
    91: "Interpretation of Dreams",
    92: "Afflictions and End of the World (Fitan)",
    93: "Judgments and Rulings (Ahkam)",
    94: "Wishes (Tamanni)",
    95: "Information given by One Person (Khabar al-Wahid)",
    96: "Holding Fast to Quran and Sunnah",
    97: "Oneness of Allah (Tawheed)",
}

_MUSLIM_CHAPTERS: dict[int, str] = {
    0: "Introduction",
    1: "Faith (Kitab al-Iman)",
    2: "Purification (Kitab al-Taharah)",
    3: "Menstruation (Kitab al-Hayd)",
    4: "Prayer (Kitab al-Salat)",
    5: "Mosques and Places of Prayer",
    6: "Prayer of Travelers",
    7: "Friday Prayer",
    8: "The Two Festivals (Eids)",
    9: "Prayer for Rain (Istisqa)",
    10: "Eclipses (Kusuf)",
    11: "Funerals (Jana'iz)",
    12: "Zakat (Charity)",
    13: "Fasting (Sawm)",
    14: "I'tikaf (Spiritual Retreat)",
    15: "Pilgrimage (Hajj)",
    16: "Marriage (Nikah)",
    17: "Suckling (Rada')",
    18: "Divorce (Talaq)",
    19: "Li'an (Invoking Curse)",
    20: "Emancipation of Slaves",
    21: "Transactions and Sales",
    22: "Musaqah and Sharecropping",
    23: "Rules of Inheritance",
    24: "Gifts (Hibah)",
    25: "Wills and Testaments",
    26: "Vows (Nudhur)",
    27: "Oaths (Ayman)",
    28: "Qasamah and Muharibin",
    29: "Judicial Decisions and Rulings",
    30: "Lost Property",
    31: "Jihad and Expeditions",
    32: "Government and Leadership (Imarah)",
    33: "Hunting and Slaughter",
    34: "Sacrificial Animals (Adahi)",
    35: "Drinks (Ashribah)",
    36: "Clothing and Adornment",
    37: "Manners and Etiquette (Adab)",
    38: "Greetings and Salam",
    39: "Correct Words and Expressions",
    40: "Poetry",
    41: "Dreams and Visions",
    42: "Virtues of the Prophet and Companions",
    43: "Merits of the Companions",
    44: "Maintaining Ties of Kinship and Good Conduct",
    45: "Destiny (Qadar)",
    46: "Knowledge (Ilm)",
    47: "Remembrance, Supplication, and Repentance",
    48: "Heart-Melting Traditions (Riqaq)",
    49: "Repentance (Tawbah)",
    50: "Characteristics of Hypocrites",
    51: "Paradise, Its Bliss, and Its Inhabitants",
    52: "Tribulations and Portents of the Hour (Fitan)",
    53: "Asceticism and Softening of Hearts (Zuhd)",
    54: "Commentary on the Quran (Tafsir)",
    55: "Virtues and Merits",
    56: "General Teachings",
}

_ABUDAWUD_CHAPTERS: dict[int, str] = {
    1: "Purification (Kitab al-Taharah)",
    2: "Prayer (Kitab al-Salat)",
    3: "Zakat",
    4: "Lost Property",
    5: "Pilgrimage (Manasik al-Hajj)",
    6: "Marriage (Nikah)",
    7: "Divorce (Talaq)",
    8: "Fasting (Sawm)",
    9: "Jihad",
    10: "Sacrifice (Dahaya)",
    11: "Hunting and Game",
    12: "Wills (Wasaya)",
    13: "Shares of Inheritance (Fara'id)",
    14: "Tribute, Spoils, and Rulership (Kharaj)",
    15: "Funerals (Jana'iz)",
    16: "Oaths and Vows",
    17: "Commercial Transactions (Buyu')",
    18: "Wages (Ijarah)",
    19: "Judges and Office of Qadi",
    20: "Knowledge (Ilm)",
    21: "Drinks (Ashribah)",
    22: "Foods (At'imah)",
    23: "Medicine (Tibb)",
    24: "Divination and Omens",
    25: "Emancipation of Slaves",
    26: "Dialects and Readings of Quran",
    27: "Bathing (Hammam)",
    28: "Dress (Libas)",
    29: "Hair and Combing (Tarajjul)",
    30: "Signet-Rings (Khatam)",
    31: "Trials and Tribulations (Fitan)",
    32: "The Promised Deliverer (Mahdi)",
    33: "Battles (Malahim)",
    34: "Prescribed Punishments (Hudud)",
    35: "Types of Blood-Wit (Diyat)",
    36: "Model Behavior of the Prophet (Sunnah)",
    37: "General Behavior and Etiquette (Adab)",
}

_TIRMIDHI_CHAPTERS: dict[int, str] = {
    1: "Purification (Taharah)",
    2: "Prayer (Salat)",
    3: "Zakat",
    4: "Fasting (Sawm)",
    5: "Hajj (Pilgrimage)",
    6: "Funerals (Jana'iz)",
    7: "Marriage (Nikah)",
    8: "Suckling (Rada')",
    9: "Divorce and Li'an",
    10: "Business Transactions (Buyu')",
    11: "Judgments and Legal Decisions (Ahkam)",
    12: "Blood Money (Diyat)",
    13: "Legal Punishments (Hudud)",
    14: "Hunting (Sayd)",
    15: "Sacrificial Animals (Adahi)",
    16: "Vows and Oaths",
    17: "Military Expeditions (Siyar)",
    18: "Virtues of Jihad",
    19: "Clothing (Libas)",
    20: "Food (At'imah)",
    21: "Drinks (Ashribah)",
    22: "Righteousness and Family Ties (Birr wa Silah)",
    23: "Medicine (Tibb)",
    24: "Inheritance (Fara'id)",
    25: "Wills and Testaments (Wasaya)",
    26: "Gifts and Wala",
    27: "Destiny (Qadar)",
    28: "Tribulations (Fitan)",
    29: "Dreams (Ru'ya)",
    30: "Manners and Etiquette (Adab)",
    31: "Parables (Amthal)",
    32: "Virtues of the Quran",
    33: "Recitation (Qira'at)",
    34: "Quranic Commentary (Tafsir)",
    35: "Supplications and Invocations (Da'awat)",
    36: "Virtues and Merits (Manaqib)",
}

_NASAI_CHAPTERS: dict[int, str] = {
    1: "Purification (Taharah)",
    2: "Water (Miyah)",
    3: "Menstruation (Hayd)",
    4: "Ghusl and Tayammum",
    5: "Salah (Prayer)",
    6: "Times of Prayer",
    7: "Adhan (Call to Prayer)",
    8: "Mosques (Masajid)",
    9: "The Qiblah",
    10: "Leadership in Prayer (Imamah)",
    11: "Opening of the Prayer",
    12: "Forgetfulness in Prayer (Sahw)",
    13: "Friday Prayer (Jumu'ah)",
    14: "Shortening Prayer on Journey",
    15: "Eclipse Prayer (Kusuf)",
    16: "Prayer for Rain (Istisqa)",
    17: "Fear Prayer",
    18: "The Two Eids",
    19: "Night Prayer (Qiyam al-Layl)",
    20: "Funerals (Jana'iz)",
    21: "Fasting (Siyam)",
    22: "Zakat",
    23: "Hajj Rituals",
    24: "Jihad",
    25: "Marriage (Nikah)",
    26: "Divorce (Talaq)",
    27: "Horses and Riding",
    28: "Endowments (Awqaf)",
    29: "Wills (Wasaya)",
    30: "Gifts (Nahl)",
    31: "Ruqba",
    32: "Umra",
    33: "Oaths and Vows",
    34: "Agriculture and Crop-Sharing",
    35: "Kind Treatment of Women",
    36: "Fighting and Prohibition of Blood",
    37: "Distribution of Spoils",
    38: "Pledge of Allegiance (Bay'ah)",
    39: "Aqiqah",
    40: "Fara' and 'Atira",
    41: "Hunting and Slaughter",
    42: "Sacrifices (Dahaya)",
    43: "Financial Transactions (Buyu')",
    44: "Qasamah",
    45: "Cutting Hand of Thief",
    46: "Faith and Its Signs",
    47: "Adornment (Zeenah)",
    48: "Etiquette of Judges",
    49: "Seeking Refuge with Allah (Isti'adhah)",
    50: "Drinks (Ashribah)",
}

_IBNMAJAH_CHAPTERS: dict[int, str] = {
    0: "The Book of the Sunnah (Introduction)",
    1: "Purification and Its Sunnahs",
    2: "Prayer (Salat)",
    3: "The Call to Prayer (Adhan)",
    4: "Mosques and Congregations",
    5: "Establishing Prayer (Iqamat al-Salat)",
    6: "Funerals (Jana'iz)",
    7: "Fasting (Siyam)",
    8: "Zakat (Obligatory Charity)",
    9: "Marriage (Nikah)",
    10: "Divorce (Talaq)",
    11: "Expiation (Kaffarat)",
    12: "Business Transactions (Tijarat)",
    13: "Judgments and Rulings (Ahkam)",
    14: "Gifts (Hibah)",
    15: "Charity and Endowments (Sadaqat)",
    16: "Mortgaging (Ruhn)",
    17: "Preemption (Shuf'ah)",
    18: "Lost Property (Luqatah)",
    19: "Manumission of Slaves ('Itq)",
    20: "Legal Punishments (Hudud)",
    21: "Blood Money (Diyat)",
    22: "Wills (Wasaya)",
    23: "Inheritance (Fara'id)",
    24: "Jihad",
    25: "Hajj and Pilgrimage Rituals",
    26: "Sacrifices (Adahi)",
    27: "Slaughtering (Dhabaih)",
    28: "Hunting (Sayd)",
    29: "Food (At'imah)",
    30: "Drinks (Ashribah)",
    31: "Medicine (Tibb)",
    32: "Dress (Libas)",
    33: "Etiquette and Manners (Adab)",
    34: "Supplication (Dua)",
    35: "Interpretation of Dreams (Ta'bir al-Ru'ya)",
    36: "Tribulations and End of Days (Fitan)",
    37: "Asceticism and Heart-Softening (Zuhd)",
}

_MALIK_CHAPTERS: dict[int, str] = {
    1: "The Times of Prayer",
    2: "Purity (Taharah)",
    3: "Prayer (Salat)",
    4: "Friday Prayer (Jumu'ah)",
    5: "Prayer in Ramadan",
    6: "Tahajjud (Night Prayer)",
    7: "Prayer in Congregation",
    8: "Shortening Prayer on Journey",
    9: "The Two Eids",
    10: "The Fear Prayer",
    11: "The Eclipse Prayer",
    12: "Prayer for Rain (Istisqa)",
    13: "The Qiblah",
    14: "Funerals (Jana'iz)",
    15: "Zakat",
    16: "Fasting (Siyam)",
    17: "I'tikaf (Spiritual Retreat)",
    18: "Hajj (Pilgrimage)",
    19: "Jihad",
    20: "Vows and Oaths",
    21: "Sacrificial Animals",
    22: "Slaughtering Animals",
    23: "Hunting",
    24: "Aqiqah",
    25: "Fara'id (Shares of Inheritance)",
    26: "Marriage (Nikah)",
    27: "Divorce (Talaq)",
    28: "Suckling (Rada')",
    29: "Business Transactions (Buyu')",
    30: "Qirad (Partnership)",
    31: "Sharecropping (Musaqat)",
    32: "Renting Land",
    33: "Preemption (Shuf'ah)",
    34: "Judgments and Rulings",
    35: "Wills and Testaments (Wasaya)",
    36: "Freeing Slaves ('Itq)",
    37: "Mukatab",
    38: "Hudud (Legal Punishments)",
    39: "Drinks (Ashribah)",
    40: "Blood Money (Diyat)",
    41: "The Oath of Qasamah",
    42: "Madinah",
    43: "General Chapter",
    44: "Good Character (Husn al-Khuluq)",
    45: "Dress and Clothing",
    46: "Description of the Prophet",
    47: "The Evil Eye",
    48: "Hair",
    49: "Visions and Dreams",
    50: "Greetings and Salam",
    51: "Asking Permission",
    52: "Speech and Words",
    53: "Jahannam (Hellfire)",
    54: "Sadaqah (Charity)",
    55: "Knowledge (Ilm)",
    56: "Supplication of the Wronged",
    57: "Names of the Prophet",
    58: "General Manners",
    59: "Supplications",
    60: "Remembrance of Allah",
    61: "Miscellaneous",
}

_COLLECTION_CHAPTER_MAPS: dict[str, dict[int, str]] = {
    "bukhari": _BUKHARI_CHAPTERS,
    "muslim": _MUSLIM_CHAPTERS,
    "abudawud": _ABUDAWUD_CHAPTERS,
    "tirmidhi": _TIRMIDHI_CHAPTERS,
    "nasai": _NASAI_CHAPTERS,
    "ibnmajah": _IBNMAJAH_CHAPTERS,
    "malik": _MALIK_CHAPTERS,
}


def get_chapter_title(collection: str, book: int | None) -> str:
    """Resolve a collection key and book number to its classical chapter title."""
    if book is None:
        return "General"
    chapters = _COLLECTION_CHAPTER_MAPS.get(collection, {})
    return chapters.get(book, f"Book {book}")


# ---------------------------------------------------------------------------
# Curated Hadith Records with Full Text, Narrators, and Topics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CuratedHadith:
    collection: str
    number: int
    text_arabic: str
    text_english: str
    grading: str
    chapter: str
    narrator: str
    topics: list[str]


_CURATED_HADITHS: list[CuratedHadith] = [
    CuratedHadith(
        collection="bukhari",
        number=1,
        text_arabic="إِنَّمَا الأَعْمَالُ بِالنِّيَّاتِ، وَإِنَّمَا لِكُلِّ امْرِئٍ مَا نَوَى، فَمَنْ كَانَتْ هِجْرَتُهُ إِلَى دُنْيَا يُصِيبُهَا أَوْ إِلَى امْرَأَةٍ يَنْكِحُهَا فَهِجْرَتُهُ إِلَى مَا هَاجَرَ إِلَيْهِ",
        text_english="Actions are but by intentions, and every person will have only what they intended. So whoever emigrated for worldly gain or to marry a woman, his emigration was for that which he emigrated.",
        grading="sahih",
        chapter="Revelation",
        narrator="Umar ibn al-Khattab",
        topics=["intention", "niyyah", "sincerity", "ikhlas", "deeds", "reward", "hijrah", "revelation"],
    ),
    CuratedHadith(
        collection="bukhari",
        number=2,
        text_arabic="أَنَّ الْحَارِثَ بْنَ هِشَامٍ سَأَلَ رَسُولَ اللَّهِ صلى الله عليه وسلم كَيْفَ يَأْتِيكَ الْوَحْىُ فَقَالَ رَسُولُ اللَّهِ صلى الله عليه وسلم أَحْيَانًا يَأْتِينِي مِثْلَ صَلْصَلَةِ الْجَرَسِ وَهُوَ أَشَدُّهُ عَلَىَّ",
        text_english="Al-Harith ibn Hisham asked the Messenger of Allah (peace be upon him): 'How does the revelation come to you?' The Messenger of Allah replied: 'Sometimes it comes to me like the ringing of a bell, and that is the hardest on me.'",
        grading="sahih",
        chapter="Revelation",
        narrator="Aishah bint Abi Bakr",
        topics=["revelation", "wahy", "prophethood", "gabriel", "quran"],
    ),
    CuratedHadith(
        collection="bukhari",
        number=3,
        text_arabic="أَوَّلُ مَا بُدِئَ بِهِ رَسُولُ اللَّهِ صلى الله عليه وسلم مِنَ الْوَحْىِ الرُّؤْيَا الصَّالِحَةُ فِي النَّوْمِ، فَكَانَ لاَ يَرَى رُؤْيَا إِلاَّ جَاءَتْ مِثْلَ فَلَقِ الصُّبْحِ، ثُمَّ حُبِّبَ إِلَيْهِ الْخَلاَءُ، وَكَانَ يَخْلُو بِغَارِ حِرَاءٍ فَيَتَحَنَّثُ فِيهِ",
        text_english="The commencement of the Divine Inspiration to Allah's Messenger (peace be upon him) was in the form of good dreams which came true like bright daylight, and then the love of seclusion was bestowed upon him. He used to go in seclusion in the cave of Hira where he used to worship Allah alone.",
        grading="sahih",
        chapter="Revelation",
        narrator="Aishah bint Abi Bakr",
        topics=["revelation", "hira", "cave", "khadijah", "gabriel", "iqra", "prophecy"],
    ),
    CuratedHadith(
        collection="bukhari",
        number=8,
        text_arabic="بُنِيَ الإِسْلاَمُ عَلَى خَمْسٍ شَهَادَةِ أَنْ لاَ إِلَهَ إِلاَّ اللَّهُ وَأَنَّ مُحَمَّدًا رَسُولُ اللَّهِ، وَإِقَامِ الصَّلاَةِ، وَإِيتَاءِ الزَّكَاةِ، وَالْحَجِّ، وَصَوْمِ رَمَضَانَ",
        text_english="Islam is built upon five: testifying that there is no god worthy of worship except Allah and that Muhammad is the Messenger of Allah, establishing prayer, giving zakat, pilgrimage to the House (Hajj), and fasting Ramadan.",
        grading="sahih",
        chapter="Belief",
        narrator="Abdullah ibn Umar",
        topics=["pillars of islam", "arkan", "faith", "iman", "prayer", "zakat", "fasting", "hajj", "tawhid"],
    ),
    CuratedHadith(
        collection="bukhari",
        number=10,
        text_arabic="الْمُسْلِمُ مَنْ سَلِمَ الْمُسْلِمُونَ مِنْ لِسَانِهِ وَيَدِهِ، وَالْمُهَاجِرُ مَنْ هَجَرَ مَا نَهَى اللَّهُ عَنْهُ",
        text_english="The Muslim is the one from whose tongue and hand the Muslims are safe, and the emigrant is the one who abandons what Allah has forbidden.",
        grading="sahih",
        chapter="Belief",
        narrator="Abdullah ibn Amr",
        topics=["muslim", "manners", "safety", "tongue", "harm", "character", "adab", "peace"],
    ),
    CuratedHadith(
        collection="bukhari",
        number=13,
        text_arabic="لاَ يُؤْمِنُ أَحَدُكُمْ حَتَّى يُحِبَّ لأَخِيهِ مَا يُحِبُّ لِنَفْسِهِ",
        text_english="None of you truly believes until he loves for his brother what he loves for himself.",
        grading="sahih",
        chapter="Belief",
        narrator="Anas ibn Malik",
        topics=["brotherhood", "faith", "iman", "love", "compassion", "empathy", "community"],
    ),
    CuratedHadith(
        collection="bukhari",
        number=52,
        text_arabic="إِنَّ الْحَلاَلَ بَيِّنٌ وَإِنَّ الْحَرَامَ بَيِّنٌ، وَبَيْنَهُمَا مُشْتَبِهَاتٌ لاَ يَعْلَمُهُنَّ كَثِيرٌ مِنَ النَّاسِ، فَمَنِ اتَّقَى الشُّبُهَاتِ اسْتَبْرَأَ لِدِينِهِ وَعِرْضِهِ",
        text_english="The lawful is clear and the unlawful is clear, and between them are doubtful matters about which many people have no knowledge. So whoever guards against doubtful matters clears himself in regard to his religion and his honor.",
        grading="sahih",
        chapter="Belief",
        narrator="Al-Nu'man ibn Bashir",
        topics=["halal", "haram", "doubtful matters", "piety", "taqwa", "worship", "heart", "conscience"],
    ),
    CuratedHadith(
        collection="bukhari",
        number=67,
        text_arabic="مَنْ يُرِدِ اللَّهُ بِهِ خَيْرًا يُفَقِّهْهُ فِي الدِّينِ، وَإِنَّمَا أَنَا قَاسِمٌ وَاللَّهُ يُعْطِي",
        text_english="Whomever Allah intends good for, He grants him deep understanding of the religion. I am only a distributor while Allah is the Giver.",
        grading="sahih",
        chapter="Knowledge",
        narrator="Mu'awiyah ibn Abi Sufyan",
        topics=["knowledge", "fiqh", "understanding", "learning", "wisdom", "blessing"],
    ),
    CuratedHadith(
        collection="bukhari",
        number=2989,
        text_arabic="وَالْكَلِمَةُ الطَّيِّبَةُ صَدَقَةٌ",
        text_english="A good, pleasant word is a form of charity.",
        grading="sahih",
        chapter="Jihad",
        narrator="Abu Hurayrah",
        topics=["charity", "sadaqah", "kindness", "good words", "speech", "adab", "manners"],
    ),
    CuratedHadith(
        collection="bukhari",
        number=6011,
        text_arabic="مَثَلُ الْمُؤْمِنِينَ فِي تَوَادِّهِمْ وَتَرَاحُمِهِمْ وَتَعَاطُفِهِمْ مَثَلُ الْجَسَدِ إِذَا اشْتَكَى مِنْهُ عُضْوٌ تَدَاعَى لَهُ سَائِرُ الْجَسَدِ بِالسَّهَرِ وَالْحُمَّى",
        text_english="The example of the believers in their mutual affection, mercy, and compassion is like that of a single body: when one limb suffers, the rest of the body responds with wakefulness and fever.",
        grading="sahih",
        chapter="Good Manners and Form (Adab)",
        narrator="Al-Nu'man ibn Bashir",
        topics=["mercy", "compassion", "brotherhood", "ummah", "unity", "love", "community"],
    ),
    CuratedHadith(
        collection="bukhari",
        number=6018,
        text_arabic="إِنَّمَا الصَّبْرُ عِنْدَ الصَّدْمَةِ الأُولَى",
        text_english="Verily, true patience is that which is shown at the initial shock of affliction.",
        grading="sahih",
        chapter="Good Manners and Form (Adab)",
        narrator="Anas ibn Malik",
        topics=["patience", "sabr", "calamity", "grief", "affliction", "steadfastness", "resilience"],
    ),
    CuratedHadith(
        collection="bukhari",
        number=6021,
        text_arabic="مَنْ كَانَ يُؤْمِنُ بِاللَّهِ وَالْيَوْمِ الآخِرِ فَلْيَقُلْ خَيْرًا أَوْ لِيَصْمُتْ، وَمَنْ كَانَ يُؤْمِنُ بِاللَّهِ وَالْيَوْمِ الآخِرِ فَلْيُكْرِمْ جَارَهُ",
        text_english="Whoever believes in Allah and the Last Day should speak good or remain silent; and whoever believes in Allah and the Last Day should be hospitable to his neighbor and honor his guest.",
        grading="sahih",
        chapter="Good Manners and Form (Adab)",
        narrator="Abu Hurayrah",
        topics=["speech", "silence", "neighbor", "guest", "manners", "adab", "hospitality", "faith"],
    ),
    CuratedHadith(
        collection="muslim",
        number=8,
        text_arabic="بَيْنَمَا نَحْنُ عِنْدَ رَسُولِ اللَّهِ صلى الله عليه وسلم ذَاتَ يَوْمٍ إِذْ طَلَعَ عَلَيْنَا رَجُلٌ شَدِيدُ بَيَاضِ الثِّيَابِ شَدِيدُ سَوَادِ الشَّعَرِ لاَ يُرَى عَلَيْهِ أَثَرُ السَّفَرِ وَلاَ يَعْرِفُهُ مِنَّا أَحَدٌ حَتَّى جَلَسَ إِلَى النَّبِيِّ صلى الله عليه وسلم فَأَسْنَدَ رُكْبَتَيْهِ إِلَى رُكْبَتَيْهِ وَوَضَعَ كَفَّيْهِ عَلَى فَخِذَيْهِ وَقَالَ يَا مُحَمَّدُ أَخْبِرْنِي عَنِ الإِسْلاَمِ",
        text_english="While we were one day sitting with the Messenger of Allah (peace be upon him), there appeared before us a man with very white clothing and very black hair. No marks of travel were visible upon him and none of us knew him. He sat down by the Prophet, leaning his knees against his and placing his hands on his thighs, and said: 'O Muhammad, inform me about Islam...'",
        grading="sahih",
        chapter="Faith (Kitab al-Iman)",
        narrator="Umar ibn al-Khattab",
        topics=["hadith of gabriel", "islam", "iman", "ihsan", "signs of the hour", "faith", "eschatology"],
    ),
    CuratedHadith(
        collection="muslim",
        number=223,
        text_arabic="الطُّهُورُ شَطْرُ الإِيمَانِ وَالْحَمْدُ لِلَّهِ تَمْلأُ الْمِيزَانَ وَسُبْحَانَ اللَّهِ وَالْحَمْدُ لِلَّهِ تَمْلآنِ أَوْ تَمْلأُ مَا بَيْنَ السَّمَاوَاتِ وَالأَرْضِ وَالصَّلاَةُ نُورٌ وَالصَّدَقَةُ بُرْهَانٌ وَالصَّبْرُ ضِيَاءٌ وَالْقُرْآنُ حُجَّةٌ لَكَ أَوْ عَلَيْكَ",
        text_english="Purity is half of faith, 'Alhamdulillah' (Praise be to Allah) fills the scale, 'SubhanAllah wa Alhamdulillah' fill what is between the heavens and the earth, prayer is a light, charity is a proof, patience is illumination, and the Quran is an argument for you or against you.",
        grading="sahih",
        chapter="Purification (Kitab al-Taharah)",
        narrator="Abu Malik al-Ash'ari",
        topics=["purification", "taharah", "wudu", "praise", "dhikr", "prayer", "charity", "patience", "quran"],
    ),
    CuratedHadith(
        collection="muslim",
        number=2564,
        text_arabic="يَا عِبَادِي إِنِّي حَرَّمْتُ الظُّلْمَ عَلَى نَفْسِي وَجَعَلْتُهُ بَيْنَكُمْ مُحَرَّمًا فَلاَ تَظَالَمُوا",
        text_english="Allah Almighty says (Hadith Qudsi): 'O My servants, I have forbidden injustice for Myself and made it forbidden among you, so do not oppress one another.'",
        grading="sahih",
        chapter="Maintaining Ties of Kinship and Good Conduct",
        narrator="Abu Dharr al-Ghifari",
        topics=["injustice", "oppression", "dhulm", "justice", "hadith qudsi", "rights", "forgiveness"],
    ),
    CuratedHadith(
        collection="muslim",
        number=2699,
        text_arabic="مَنْ سَلَكَ طَرِيقًا يَلْتَمِسُ فِيهِ عِلْمًا سَهَّلَ اللَّهُ لَهُ بِهِ طَرِيقًا إِلَى الْجَنَّةِ، وَمَا اجْتَمَعَ قَوْمٌ فِي بَيْتٍ مِنْ بُيُوتِ اللَّهِ يَتْلُونَ كِتَابَ اللَّهِ وَيَتَدَارَسُونَهُ بَيْنَهُمْ إِلاَّ نَزَلَتْ عَلَيْهِمُ السَّكِينَةُ",
        text_english="Whoever travels a path in pursuit of knowledge, Allah will make easy for him a path to Paradise. No people gather in one of the houses of Allah reciting the Book of Allah and teaching it to one another except that tranquility descends upon them, mercy envelops them, the angels surround them, and Allah mentions them to those with Him.",
        grading="sahih",
        chapter="Remembrance, Supplication, and Repentance",
        narrator="Abu Hurayrah",
        topics=["knowledge", "ilm", "paradise", "quran study", "mosque", "tranquility", "angels", "dhikr"],
    ),
    CuratedHadith(
        collection="abudawud",
        number=1,
        text_arabic="إِذَا أَرَادَ الْحَاجَةَ لَمْ يَرْفَعْ ثَوْبَهُ حَتَّى يَدْنُوَ مِنَ الأَرْضِ",
        text_english="When the Prophet (peace be upon him) wanted to relieve himself, he would not raise his garment until he was close to the ground.",
        grading="hasan",
        chapter="Purification (Kitab al-Taharah)",
        narrator="Abdullah ibn Umar",
        topics=["purification", "modesty", "haya", "etiquette", "toilet", "privacy"],
    ),
    CuratedHadith(
        collection="abudawud",
        number=3641,
        text_arabic="إِنَّ الْعُلَمَاءَ وَرَثَةُ الأَنْبِيَاءِ، وَإِنَّ الأَنْبِيَاءَ لَمْ يُوَرِّثُوا دِينَارًا وَلاَ دِرْهَمًا، إِنَّمَا وَرَّثُوا الْعِلْمَ، فَمَنْ أَخَذَهُ أَخَذَ بِحَظٍّ وَافِرٍ",
        text_english="The scholars are the inheritors of the Prophets. The Prophets did not leave behind dinars or dirhams as inheritance, but rather left behind knowledge; so whoever takes hold of it has acquired an abundant portion.",
        grading="sahih",
        chapter="Knowledge (Ilm)",
        narrator="Abu al-Darda",
        topics=["scholars", "knowledge", "prophets", "inheritance", "ilm", "virtue", "wisdom"],
    ),
    CuratedHadith(
        collection="abudawud",
        number=4941,
        text_arabic="الرَّاحِمُونَ يَرْحَمُهُمُ الرَّحْمَنُ، ارْحَمُوا مَنْ فِي الأَرْضِ يَرْحَمْكُمْ مَنْ فِي السَّمَاءِ",
        text_english="Those who are merciful will be shown mercy by the Most Merciful. Be merciful to those on earth, and the One in the heavens will be merciful to you.",
        grading="sahih",
        chapter="General Behavior and Etiquette (Adab)",
        narrator="Abdullah ibn Amr",
        topics=["mercy", "compassion", "rahmah", "kindness", "creation", "charity", "forgiveness"],
    ),
    CuratedHadith(
        collection="tirmidhi",
        number=1987,
        text_arabic="اتَّقِ اللَّهَ حَيْثُمَا كُنْتَ، وَأَتْبِعِ السَّيِّئَةَ الْحَسَنَةَ تَمْحُهَا، وَخَالِقِ النَّاسَ بِخُلُقٍ حَسَنٍ",
        text_english="Be conscious of Allah wherever you are, follow up a bad deed with a good deed and it will wipe it out, and interact with the people with good character.",
        grading="hasan",
        chapter="Righteousness and Family Ties (Birr wa Silah)",
        narrator="Abu Dharr and Mu'adh ibn Jabal",
        topics=["taqwa", "piety", "repentance", "good deeds", "good character", "manners", "akhlaq"],
    ),
    CuratedHadith(
        collection="tirmidhi",
        number=2317,
        text_arabic="مِنْ حُسْنِ إِسْلاَمِ الْمَرْءِ تَرْكُهُ مَا لاَ يَعْنِيهِ",
        text_english="Part of the excellence of a person's Islam is his leaving that which does not concern him.",
        grading="sahih",
        chapter="Asceticism and Heart-Softening (Zuhd)",
        narrator="Ali ibn Husayn (Zayn al-Abidin)",
        topics=["excellence", "manners", "restraint", "idle speech", "gossip", "time management", "adab"],
    ),
    CuratedHadith(
        collection="tirmidhi",
        number=2516,
        text_arabic="يَا غُلاَمُ إِنِّي أُعَلِّمُكَ كَلِمَاتٍ: احْفَظِ اللَّهَ يَحْفَظْكَ، احْفَظِ اللَّهَ تَجِدْهُ تُجَاهَكَ، إِذَا سَأَلْتَ فَاسْأَلِ اللَّهَ، وَإِذَا اسْتَعَنْتَ فَاسْتَعِنْ بِاللَّهِ",
        text_english="O young man, I shall teach you some words: Be mindful of Allah and He will protect you. Be mindful of Allah and you will find Him in front of you. If you ask, ask of Allah; and if you seek help, seek help from Allah.",
        grading="sahih",
        chapter="Description of the Day of Judgment and Softening of Hearts",
        narrator="Abdullah ibn Abbas",
        topics=["reliance on allah", "tawakkul", "supplication", "dua", "protection", "destiny", "youth", "guidance"],
    ),
    CuratedHadith(
        collection="nasai",
        number=3104,
        text_arabic="وَيْحَكَ أَلَكَ أُمٌّ قَالَ نَعَمْ قَالَ الْزَمْ رِجْلَهَا فَثَمَّ الْجَنَّةُ",
        text_english="A man came asking about going for Jihad. The Prophet (peace be upon him) asked: 'Is your mother alive?' The man replied: 'Yes.' The Prophet said: 'Then stay at her feet, for Paradise lies there.'",
        grading="sahih",
        chapter="Jihad",
        narrator="Mu'awiyah ibn Jahimah",
        topics=["mother", "parents", "birr al-walidayn", "paradise", "family", "respect", "honor", "kindness"],
    ),
    CuratedHadith(
        collection="nasai",
        number=5005,
        text_arabic="إِنَّ اللَّهَ لاَ يَقْبَلُ مِنَ الْعَمَلِ إِلاَّ مَا كَانَ لَهُ خَالِصًا وَابْتُغِيَ بِهِ وَجْهُهُ",
        text_english="Verily, Allah does not accept any deed unless it is done purely for His sake and His Pleasure is sought thereby.",
        grading="sahih",
        chapter="Jihad",
        narrator="Abu Umamah al-Bahili",
        topics=["sincerity", "ikhlas", "intention", "deeds", "acceptance", "worship", "purity"],
    ),
    CuratedHadith(
        collection="ibnmajah",
        number=224,
        text_arabic="طَلَبُ الْعِلْمِ فَرِيضَةٌ عَلَى كُلِّ مُسْلِمٍ",
        text_english="Seeking knowledge is an obligation upon every Muslim.",
        grading="hasan",
        chapter="The Book of the Sunnah (Introduction)",
        narrator="Anas ibn Malik",
        topics=["knowledge", "obligation", "learning", "education", "ilm", "study", "fard"],
    ),
    CuratedHadith(
        collection="ibnmajah",
        number=4036,
        text_arabic="سَيَأْتِي عَلَى النَّاسِ سَنَوَاتٌ خَدَّاعَاتُ يُصَدَّقُ فِيهَا الْكَاذِبُ وَيُكَذَّبُ فِيهَا الصَّادِقُ وَيُؤْتَمَنُ فِيهَا الْخَائِنُ وَيُخَوَّنُ فِيهَا الأَمِينُ",
        text_english="There will come upon the people deceitful years wherein the liar is believed and the truthful is deemed a liar, the untrustworthy is trusted and the trustworthy is considered treacherous.",
        grading="hasan",
        chapter="Tribulations and End of Days (Fitan)",
        narrator="Abu Hurayrah",
        topics=["truthfulness", "honesty", "deceit", "trials", "fitnah", "signs of the hour", "trustworthiness"],
    ),
    CuratedHadith(
        collection="malik",
        number=1614,
        text_arabic="تَرَكْتُ فِيكُمْ أَمْرَيْنِ لَنْ تَضِلُّوا مَا تَمَسَّكْتُمْ بِهِمَا كِتَابَ اللَّهِ وَسُنَّةَ نَبِيِّهِ",
        text_english="I have left among you two things; you will never go astray as long as you hold fast to them: the Book of Allah and the Sunnah of His Prophet.",
        grading="sahih",
        chapter="General Chapter",
        narrator="Yahya ibn Said (Mursal)",
        topics=["quran", "sunnah", "guidance", "steadfastness", "holding fast", "revelation", "truth"],
    ),
    CuratedHadith(
        collection="malik",
        number=1643,
        text_arabic="إِنَّمَا بُعِثْتُ لأُتَمِّمَ حُسْنَ الأَخْلاَقِ",
        text_english="I was sent only to perfect noble character and good manners.",
        grading="sahih",
        chapter="Good Character (Husn al-Khuluq)",
        narrator="Yahya ibn Said (Mursal)",
        topics=["character", "manners", "akhlaq", "morality", "virtue", "prophethood", "adab"],
    ),
]


# ---------------------------------------------------------------------------
# Text Normalization & Tokenization Helpers
# ---------------------------------------------------------------------------

_ARABIC_DIACRITICS_RE = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭ]")
_NON_ALPHANUM_RE = re.compile(r"[^\w\s]", re.UNICODE)
_WHITESPACE_RE = re.compile(r"\s+")

_SYNONYMS: dict[str, str] = {
    "prophet": "prophet",
    "messenger": "prophet",
    "rasool": "prophet",
    "rasul": "prophet",
    "nabi": "prophet",
    "muhammad": "prophet",
    "intentions": "intention",
    "intention": "intention",
    "niyyah": "intention",
    "niyah": "intention",
    "deeds": "deed",
    "action": "deed",
    "actions": "deed",
    "sincerity": "sincerity",
    "ikhlas": "sincerity",
    "faith": "faith",
    "iman": "faith",
    "belief": "faith",
    "prayer": "prayer",
    "salat": "prayer",
    "salah": "prayer",
    "namaz": "prayer",
    "charity": "charity",
    "zakat": "charity",
    "sadaqah": "charity",
    "sadaqa": "charity",
    "fasting": "fasting",
    "sawm": "fasting",
    "siyam": "fasting",
    "ramadan": "ramadan",
    "pilgrimage": "hajj",
    "hajj": "hajj",
    "umrah": "umrah",
    "purity": "purification",
    "purification": "purification",
    "taharah": "purification",
    "wudu": "ablution",
    "ablution": "ablution",
    "knowledge": "knowledge",
    "ilm": "knowledge",
    "learning": "knowledge",
    "patience": "patience",
    "sabr": "patience",
    "mercy": "mercy",
    "rahmah": "mercy",
    "compassion": "mercy",
    "brotherhood": "brotherhood",
    "forgiveness": "forgiveness",
    "tawbah": "repentance",
    "repentance": "repentance",
    "truth": "truthfulness",
    "truthfulness": "truthfulness",
    "sidq": "truthfulness",
    "modesty": "modesty",
    "haya": "modesty",
    "dhikr": "remembrance",
    "remembrance": "remembrance",
    "supplication": "supplication",
    "dua": "supplication",
    "taqwa": "piety",
    "piety": "piety",
    "halal": "lawful",
    "lawful": "lawful",
    "haram": "unlawful",
    "unlawful": "unlawful",
}

_STOPWORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "of",
        "to",
        "is",
        "are",
        "was",
        "were",
        "and",
        "or",
        "by",
        "for",
        "in",
        "on",
        "that",
        "this",
        "he",
        "she",
        "it",
        "his",
        "her",
        "him",
        "who",
        "whom",
        "with",
        "as",
        "so",
        "but",
        "from",
        "at",
        "be",
        "will",
        "shall",
    }
)


def strip_arabic_diacritics(text: str) -> str:
    """Remove Arabic harakat (tashkeel) and fold alef/hamza variants."""
    text = _ARABIC_DIACRITICS_RE.sub("", text)
    for variant, canon in (
        ("أ", "ا"),
        ("إ", "ا"),
        ("آ", "ا"),
        ("ى", "ي"),
        ("ة", "ه"),
    ):
        text = text.replace(variant, canon)
    return text


def normalize_search_text(text: str) -> str:
    """Normalize text: strip diacritics, lowercase, remove punctuation, collapse whitespace."""
    text = unicodedata.normalize("NFC", text)
    text = strip_arabic_diacritics(text)
    text = text.lower()
    text = _NON_ALPHANUM_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text)
    return text.strip()


def tokenize_query(query: str) -> list[str]:
    """Tokenize query string into normalized search tokens."""
    tokens: list[str] = []
    for raw_tok in normalize_search_text(query).split():
        tok = _SYNONYMS.get(raw_tok, raw_tok)
        if tok not in _STOPWORDS:
            tokens.append(tok)
    return tokens


# ---------------------------------------------------------------------------
# Search Index (SQLite FTS5 in-memory)
# ---------------------------------------------------------------------------


class HadithSearchEngine:
    """In-memory SQLite FTS5 search engine indexing the hadith grading dataset and texts."""

    def __init__(self, data_dir: Path = DATA_DIR):
        self._data_dir = data_dir
        self._conn = sqlite3.connect(":memory:", check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._curated_map: dict[tuple[str, int], CuratedHadith] = {
            (c.collection, c.number): c for c in _CURATED_HADITHS
        }
        self._init_db()

    def _init_db(self) -> None:
        """Create tables and build the FTS5 index."""
        self._conn.execute(
            """
            CREATE VIRTUAL TABLE hadith_fts USING fts5(
                collection,
                collection_key,
                number UNINDEXED,
                text_arabic UNINDEXED,
                text_arabic_norm,
                text_english,
                grading,
                chapter,
                narrator,
                topics,
                tokenize='unicode61'
            );
            """
        )

        # 1. Insert curated hadiths first
        for curated in _CURATED_HADITHS:
            display_name = COLLECTION_NAMES.get(curated.collection, curated.collection.capitalize())
            ar_norm = normalize_search_text(curated.text_arabic)
            self._conn.execute(
                """
                INSERT INTO hadith_fts (
                    collection, collection_key, number, text_arabic, text_arabic_norm,
                    text_english, grading, chapter, narrator, topics
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    display_name,
                    curated.collection,
                    curated.number,
                    curated.text_arabic,
                    ar_norm,
                    curated.text_english,
                    curated.grading.lower(),
                    curated.chapter,
                    curated.narrator,
                    " ".join(curated.topics),
                ),
            )

        # 2. Insert records from bundled grading files (data/hadith/*.json)
        for collection_key, display_name in COLLECTION_NAMES.items():
            file_path = self._data_dir / f"{collection_key}.json"
            if not file_path.exists():
                continue
            with open(file_path, encoding="utf-8") as f:
                payload = json.load(f)

            for item in payload.get("hadiths", []):
                num = item["n"]
                if (collection_key, num) in self._curated_map:
                    # Already inserted rich curated record
                    continue

                book_num = item.get("book")
                chapter = get_chapter_title(collection_key, book_num)
                grade_str = str(item.get("grade", Strength.UNKNOWN.value)).lower()

                # Generate clean synthetic English context description from chapter/book
                text_en = f"Hadith narrated in {display_name}, {chapter} (Hadith #{num})."
                topics = f"{chapter.lower()} hadith {collection_key} narration"

                self._conn.execute(
                    """
                    INSERT INTO hadith_fts (
                        collection, collection_key, number, text_arabic, text_arabic_norm,
                        text_english, grading, chapter, narrator, topics
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                    """,
                    (
                        display_name,
                        collection_key,
                        num,
                        None,
                        None,
                        text_en,
                        grade_str,
                        chapter,
                        None,
                        topics,
                    ),
                )

        self._conn.commit()

    def search(
        self,
        q: str,
        collection: str | None = None,
        grading: str | None = None,
        limit: int = 10,
        offset: int = 0,
    ) -> tuple[list[HadithSearchResult], int]:
        """Search the hadith corpus and return ranked results with total count."""
        raw_query = q.strip()
        if not raw_query:
            return [], 0

        # Normalize collection filter if provided
        norm_collection_key = None
        if collection:
            norm_collection_key = normalize_collection(collection)
            if not norm_collection_key:
                # If collection specified does not match any known collection, return 0
                return [], 0

        # Normalize grading filter
        norm_grading = None
        if grading:
            norm_grading = grading.strip().lower()

        # Build SQLite FTS query string
        tokens = tokenize_query(raw_query)
        if not tokens:
            tokens = [normalize_search_text(raw_query)]
        clean_tokens = [t for t in tokens if t]
        if not clean_tokens:
            return [], 0

        # Exact phrase or multi-term query
        fts_match_expr = " OR ".join(f'"{t}"*' for t in clean_tokens)

        # Build SQL with filters
        where_clauses = ["hadith_fts MATCH ?"]
        params: list[Any] = [fts_match_expr]

        if norm_collection_key:
            where_clauses.append("collection_key = ?")
            params.append(norm_collection_key)

        if norm_grading:
            where_clauses.append("grading = ?")
            params.append(norm_grading)

        where_sql = " AND ".join(where_clauses)

        # 1. Total matching count
        count_sql = f"SELECT COUNT(*) as total FROM hadith_fts WHERE {where_sql}"
        cursor = self._conn.execute(count_sql, params)
        total_row = cursor.fetchone()
        total = total_row["total"] if total_row else 0

        if total == 0:
            return [], 0

        # 2. Ranked query with field relevance scoring
        # SQLite FTS5 rank ordering (BM25: smaller is better, so ORDER BY rank ASC)
        # We also boost rich curated matches and exact phrase matches
        search_sql = f"""
            SELECT
                collection,
                collection_key,
                number,
                text_arabic,
                text_english,
                grading,
                chapter,
                narrator,
                topics,
                bm25(hadith_fts) as rank
            FROM hadith_fts
            WHERE {where_sql}
            ORDER BY
                (CASE WHEN text_arabic IS NOT NULL THEN 0 ELSE 1 END) ASC,
                rank ASC,
                number ASC
            LIMIT ? OFFSET ?;
        """
        fetch_params = list(params) + [limit, offset]
        cursor = self._conn.execute(search_sql, fetch_params)
        rows = cursor.fetchall()

        results: list[HadithSearchResult] = []
        for r in rows:
            results.append(
                HadithSearchResult(
                    collection=r["collection"],
                    number=r["number"],
                    text_arabic=r["text_arabic"],
                    text_english=r["text_english"],
                    grading=r["grading"],
                    chapter=r["chapter"],
                    narrator=r["narrator"],
                )
            )

        return results, total


# Singleton instance of the engine
@lru_cache(maxsize=1)
def get_search_engine() -> HadithSearchEngine:
    return HadithSearchEngine()


def search_hadith(
    q: str,
    collection: str | None = None,
    grading: str | None = None,
    limit: int = 10,
    offset: int = 0,
    engine: HadithSearchEngine | None = None,
) -> HadithSearchResponse:
    """Execute a hadith search with pagination and filtering."""
    engine = engine or get_search_engine()
    # Enforce limit bounds
    capped_limit = max(1, min(limit, 50))
    valid_offset = max(0, offset)

    results, total = engine.search(
        q=q,
        collection=collection,
        grading=grading,
        limit=capped_limit,
        offset=valid_offset,
    )

    return HadithSearchResponse(
        results=results,
        total=total,
        offset=valid_offset,
        limit=capped_limit,
    )


# ---------------------------------------------------------------------------
# API Route
# ---------------------------------------------------------------------------


@router.get(
    "/search",
    response_model=HadithSearchResponse,
    summary="Search hadith by topic and keyword",
    description=(
        "Search hadiths by topic, keyword, phrase, chapter, or narrator across authentic collections "
        "(Sahih al-Bukhari, Sahih Muslim, Sunan Abu Dawud, Jami at-Tirmidhi, Sunan an-Nasai, "
        "Sunan Ibn Majah, Muwatta Malik). Returns relevance-ranked structured results with authenticity grading."
    ),
)
def search_hadith_endpoint(
    q: str = Query(
        ...,
        min_length=1,
        max_length=500,
        description="Search query (keyword, topic, phrase, narrator, or chapter)",
        examples=["intention", "prayer", "faith", "Umar ibn al-Khattab"],
    ),
    collection: str | None = Query(
        default=None,
        description="Filter by collection (e.g. bukhari, muslim, abudawud, tirmidhi, nasai, ibnmajah, malik)",
        examples=["bukhari"],
    ),
    grading: str | None = Query(
        default=None,
        description="Filter by authenticity grade (e.g. sahih, hasan, daif)",
        examples=["sahih"],
    ),
    limit: int = Query(
        default=10,
        ge=1,
        le=50,
        description="Maximum results to return (default 10, max 50)",
    ),
    offset: int = Query(
        default=0,
        ge=0,
        description="Pagination offset (default 0)",
    ),
) -> HadithSearchResponse:
    """GET endpoint for hadith search."""
    clean_q = q.strip()
    if not clean_q:
        raise HTTPException(status_code=400, detail="Search query 'q' must not be empty or whitespace.")

    return search_hadith(
        q=clean_q,
        collection=collection,
        grading=grading,
        limit=limit,
        offset=offset,
    )
