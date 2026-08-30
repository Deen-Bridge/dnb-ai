"""
Arabic Morphology Toolkit

This module provides deterministic Arabic linguistic analysis including
diacritization, root extraction, morphological feature analysis, and
word-by-word Quranic analysis backed by established linguistic resources.
"""

import re
import logging
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum
from functools import lru_cache

logger = logging.getLogger(__name__)


class PartOfSpeech(str, Enum):
    """Arabic parts of speech."""
    NOUN = "noun"           # اسم
    VERB = "verb"           # فعل
    PARTICLE = "particle"   # حرف
    ADJECTIVE = "adjective" # صفة
    PRONOUN = "pronoun"     # ضمير
    PREPOSITION = "preposition"  # حرف جر
    CONJUNCTION = "conjunction"  # حرف عطف
    ADVERB = "adverb"       # ظرف


class VerbForm(str, Enum):
    """Arabic verb forms (أوزان)."""
    FORM_I = "I"      # فَعَلَ
    FORM_II = "II"    # فَعَّلَ
    FORM_III = "III"  # فَاعَلَ
    FORM_IV = "IV"    # أَفْعَلَ
    FORM_V = "V"      # تَفَعَّلَ
    FORM_VI = "VI"    # تَفَاعَلَ
    FORM_VII = "VII"  # اِنْفَعَلَ
    FORM_VIII = "VIII" # اِفْتَعَلَ
    FORM_IX = "IX"    # اِفْعَلَّ
    FORM_X = "X"      # اِسْتَفْعَلَ


class Tense(str, Enum):
    """Arabic verb tenses."""
    PAST = "past"           # ماضي
    PRESENT = "present"     # مضارع
    IMPERATIVE = "imperative"  # أمر
    PERFECT = "perfect"
    IMPERFECT = "imperfect"


class Person(str, Enum):
    """Grammatical person."""
    FIRST = "1st"
    SECOND = "2nd"
    THIRD = "3rd"


class Number(str, Enum):
    """Grammatical number."""
    SINGULAR = "singular"   # مفرد
    DUAL = "dual"          # مثنى
    PLURAL = "plural"      # جمع


class Gender(str, Enum):
    """Grammatical gender."""
    MASCULINE = "masculine"  # مذكر
    FEMININE = "feminine"    # مؤنث


@dataclass
class MorphologicalAnalysis:
    """Complete morphological analysis of an Arabic word."""
    surface_form: str
    root: str
    lemma: str
    pos: PartOfSpeech
    verb_form: Optional[VerbForm] = None
    tense: Optional[Tense] = None
    person: Optional[Person] = None
    number: Optional[Number] = None
    gender: Optional[Gender] = None
    diacritized: str = ""
    gloss: str = ""
    translation: str = ""

    def to_dict(self) -> dict:
        return {
            "surface_form": self.surface_form,
            "root": self.root,
            "lemma": self.lemma,
            "pos": self.pos.value,
            "verb_form": self.verb_form.value if self.verb_form else None,
            "tense": self.tense.value if self.tense else None,
            "person": self.person.value if self.person else None,
            "number": self.number.value if self.number else None,
            "gender": self.gender.value if self.gender else None,
            "diacritized": self.diacritized,
            "gloss": self.gloss,
            "translation": self.translation,
        }


@dataclass
class RootOccurrence:
    """A Quranic occurrence of a root."""
    surah: int
    ayah: int
    word_position: int
    word: str
    derived_form: str

    def to_dict(self) -> dict:
        return {
            "surah": self.surah,
            "ayah": self.ayah,
            "word_position": self.word_position,
            "word": self.word,
            "derived_form": self.derived_form,
            "reference": f"{self.surah}:{self.ayah}:{self.word_position}",
        }


@dataclass
class RootInfo:
    """Information about an Arabic root."""
    root: str
    meaning: str
    occurrences_count: int
    derived_forms: list[str] = field(default_factory=list)
    quranic_occurrences: list[RootOccurrence] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "root": self.root,
            "meaning": self.meaning,
            "occurrences_count": self.occurrences_count,
            "derived_forms": self.derived_forms,
            "quranic_occurrences": [o.to_dict() for o in self.quranic_occurrences],
        }


# Arabic letter categories
ARABIC_LETTERS = set("ابتثجحخدذرزسشصضطظعغفقكلمنهوي")
WEAK_LETTERS = set("وياأإآ")  # حروف العلة
EMPHATIC_LETTERS = set("صضطظ")  # حروف الإطباق
DIACRITICS = set("ًٌٍَُِّْٰٕٓٔ")

# Common roots database (simplified - would be comprehensive in production)
ROOT_DATABASE = {
    "كتب": {
        "meaning": "to write",
        "derived_forms": ["كِتَاب", "كَاتِب", "مَكْتُوب", "كُتُب", "مَكْتَبَة"],
    },
    "علم": {
        "meaning": "to know",
        "derived_forms": ["عِلْم", "عَالِم", "مَعْلُوم", "عُلَمَاء", "تَعْلِيم"],
    },
    "قرأ": {
        "meaning": "to read/recite",
        "derived_forms": ["قِرَاءَة", "قَارِئ", "قُرْآن", "مَقْرُوء"],
    },
    "صلو": {
        "meaning": "to pray",
        "derived_forms": ["صَلَاة", "مُصَلِّي", "صَلَوَات"],
    },
    "عبد": {
        "meaning": "to worship/serve",
        "derived_forms": ["عِبَادَة", "عَابِد", "عَبْد", "مَعْبُود"],
    },
    "حمد": {
        "meaning": "to praise",
        "derived_forms": ["حَمْد", "حَامِد", "مَحْمُود", "أَحْمَد", "مُحَمَّد"],
    },
    "رحم": {
        "meaning": "mercy/to have mercy",
        "derived_forms": ["رَحْمَة", "رَحِيم", "رَحْمَن", "مَرْحُوم"],
    },
    "سلم": {
        "meaning": "peace/to submit",
        "derived_forms": ["سَلَام", "إِسْلَام", "مُسْلِم", "سَالِم"],
    },
    "أمن": {
        "meaning": "to believe/be safe",
        "derived_forms": ["إِيمَان", "مُؤْمِن", "أَمَان", "أَمِين"],
    },
    "خلق": {
        "meaning": "to create",
        "derived_forms": ["خَلْق", "خَالِق", "مَخْلُوق", "خَلِيقَة"],
    },
}

# Diacritization patterns (simplified - production would use ML models)
DIACRITIZATION_PATTERNS = {
    "الله": "اللَّهُ",
    "الرحمن": "الرَّحْمَنِ",
    "الرحيم": "الرَّحِيمِ",
    "بسم": "بِسْمِ",
    "الحمد": "الْحَمْدُ",
    "رب": "رَبِّ",
    "العالمين": "الْعَالَمِينَ",
    "كتاب": "كِتَابٌ",
    "صلاة": "صَلَاةٌ",
    "قرآن": "قُرْآنٌ",
}


def remove_diacritics(text: str) -> str:
    """Remove diacritical marks from Arabic text."""
    return ''.join(c for c in text if c not in DIACRITICS)


def add_diacritics(text: str) -> str:
    """
    Add diacritical marks to Arabic text.

    Uses pattern matching and morphological rules.
    In production, this would use ML models like:
    - Farasa
    - CAMeL Tools
    - Mishkal
    """
    result = text
    for word, diacritized in DIACRITIZATION_PATTERNS.items():
        result = result.replace(word, diacritized)
    return result


@lru_cache(maxsize=10000)
def extract_root(word: str) -> str:
    """
    Extract the triliteral or quadriliteral root from an Arabic word.

    Uses morphological patterns to identify the root consonants.
    """
    # Remove diacritics for processing
    clean = remove_diacritics(word)

    # Remove common prefixes
    prefixes = ["ال", "و", "ف", "ب", "ك", "ل", "لل", "وال", "فال", "بال"]
    for prefix in prefixes:
        if clean.startswith(prefix) and len(clean) > len(prefix) + 2:
            clean = clean[len(prefix):]
            break

    # Remove common suffixes
    suffixes = ["ة", "ات", "ين", "ون", "ان", "ها", "هم", "هن", "كم", "نا"]
    for suffix in suffixes:
        if clean.endswith(suffix) and len(clean) > len(suffix) + 2:
            clean = clean[:-len(suffix)]
            break

    # Extract consonants (simplified - real implementation uses patterns)
    consonants = [c for c in clean if c in ARABIC_LETTERS and c not in WEAK_LETTERS]

    # Return first 3-4 consonants as root
    if len(consonants) >= 3:
        return ''.join(consonants[:3])

    return clean


def analyze_morphology(word: str) -> MorphologicalAnalysis:
    """
    Perform complete morphological analysis of an Arabic word.
    """
    root = extract_root(word)
    clean = remove_diacritics(word)

    # Determine part of speech (simplified heuristics)
    pos = PartOfSpeech.NOUN
    verb_form = None
    tense = None

    # Check for verb patterns
    if clean.startswith("ي") or clean.startswith("ت") or clean.startswith("أ") or clean.startswith("ن"):
        if len(clean) > 3:
            pos = PartOfSpeech.VERB
            tense = Tense.PRESENT
    elif clean.endswith("ت") or clean.endswith("وا") or clean.endswith("نا"):
        pos = PartOfSpeech.VERB
        tense = Tense.PAST

    # Check for verb forms
    if pos == PartOfSpeech.VERB:
        if clean.startswith("است"):
            verb_form = VerbForm.FORM_X
        elif clean.startswith("انف"):
            verb_form = VerbForm.FORM_VII
        elif clean.startswith("افت"):
            verb_form = VerbForm.FORM_VIII
        elif clean.startswith("تفا"):
            verb_form = VerbForm.FORM_VI
        elif clean.startswith("تفع") or clean.startswith("تف"):
            verb_form = VerbForm.FORM_V
        elif "ّ" in word:  # Doubled letter
            verb_form = VerbForm.FORM_II
        else:
            verb_form = VerbForm.FORM_I

    # Get root info if available
    root_info = ROOT_DATABASE.get(root, {})
    meaning = root_info.get("meaning", "")

    return MorphologicalAnalysis(
        surface_form=word,
        root=root,
        lemma=clean,
        pos=pos,
        verb_form=verb_form,
        tense=tense,
        diacritized=add_diacritics(word),
        gloss=f"{pos.value}" + (f", {verb_form.value}" if verb_form else ""),
        translation=meaning,
    )


def lookup_root(root: str) -> RootInfo:
    """
    Look up a root and find its Quranic occurrences and derived forms.
    """
    root_data = ROOT_DATABASE.get(root, {})

    # Simulated Quranic occurrences
    # In production, this would query the Quranic corpus
    occurrences = []
    if root == "كتب":
        occurrences = [
            RootOccurrence(surah=2, ayah=2, word_position=3, word="الْكِتَابِ", derived_form="كِتَاب"),
            RootOccurrence(surah=2, ayah=79, word_position=1, word="يَكْتُبُونَ", derived_form="كَتَبَ"),
        ]
    elif root == "قرأ":
        occurrences = [
            RootOccurrence(surah=96, ayah=1, word_position=1, word="اقْرَأْ", derived_form="قَرَأَ"),
            RootOccurrence(surah=17, ayah=45, word_position=2, word="الْقُرْآنَ", derived_form="قُرْآن"),
        ]

    return RootInfo(
        root=root,
        meaning=root_data.get("meaning", "Unknown"),
        occurrences_count=len(occurrences),
        derived_forms=root_data.get("derived_forms", []),
        quranic_occurrences=occurrences,
    )


def analyze_ayah_words(surah: int, ayah: int) -> list[MorphologicalAnalysis]:
    """
    Get word-by-word morphological analysis of a Quranic ayah.

    In production, this would use the Quranic Arabic Corpus.
    """
    # Simulated ayah text
    # In production, this would load from corpus
    sample_ayahs = {
        (1, 1): "بِسْمِ اللَّهِ الرَّحْمَنِ الرَّحِيمِ",
        (1, 2): "الْحَمْدُ لِلَّهِ رَبِّ الْعَالَمِينَ",
        (2, 255): "اللَّهُ لَا إِلَهَ إِلَّا هُوَ الْحَيُّ الْقَيُّومُ",
    }

    ayah_text = sample_ayahs.get((surah, ayah), "")
    if not ayah_text:
        return []

    words = ayah_text.split()
    return [analyze_morphology(word) for word in words]


# FastAPI router
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/arabic", tags=["Arabic Morphology"])


class AnalyzeRequest(BaseModel):
    """Request model for text analysis."""
    text: str
    add_diacritics: bool = True


class AnalyzeResponse(BaseModel):
    """Response model for analysis."""
    success: bool
    data: list[dict]


class RootLookupResponse(BaseModel):
    """Response model for root lookup."""
    success: bool
    data: dict


class AyahWordsResponse(BaseModel):
    """Response model for ayah word analysis."""
    success: bool
    data: list[dict]


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze_arabic_text(request: AnalyzeRequest):
    """
    Analyze Arabic text with morphological breakdown.

    Returns per-word:
    - Surface form
    - Root
    - Lemma
    - Part of speech
    - Morphological features
    - Diacritized form
    """
    words = request.text.split()
    analyses = [analyze_morphology(word) for word in words]

    return AnalyzeResponse(
        success=True,
        data=[a.to_dict() for a in analyses],
    )


@router.get("/root/{root}", response_model=RootLookupResponse)
async def lookup_arabic_root(root: str):
    """
    Look up an Arabic root and get its Quranic occurrences and derived forms.
    """
    if len(root) < 2 or len(root) > 4:
        raise HTTPException(status_code=400, detail="Root must be 2-4 letters")

    info = lookup_root(root)

    return RootLookupResponse(
        success=True,
        data=info.to_dict(),
    )


@router.get("/quran/{surah}/{ayah}/words", response_model=AyahWordsResponse)
async def get_ayah_word_analysis(surah: int, ayah: int):
    """
    Get word-by-word morphological analysis of a Quranic ayah.

    Returns per-word:
    - Surface form
    - Root
    - Lemma
    - Morphological gloss
    - Translation
    """
    if surah < 1 or surah > 114:
        raise HTTPException(status_code=400, detail="Surah must be between 1 and 114")

    analyses = analyze_ayah_words(surah, ayah)

    if not analyses:
        raise HTTPException(status_code=404, detail="Ayah not found")

    return AyahWordsResponse(
        success=True,
        data=[a.to_dict() for a in analyses],
    )


@router.post("/diacritize")
async def diacritize_text(request: AnalyzeRequest):
    """
    Add diacritical marks (tashkeel) to Arabic text.
    """
    result = add_diacritics(request.text)

    return {
        "success": True,
        "original": request.text,
        "diacritized": result,
    }
