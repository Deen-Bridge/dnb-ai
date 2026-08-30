"""Urdu Islamic-language processing utilities.

The processor is deterministic and offline.  It normalizes Urdu, Arabic, and
Persian Unicode variants without changing Quranic Arabic by default, recognizes
curated Islamic terminology, preserves multi-word terms during tokenization,
and supplies prompt guidance for an existing multilingual generation pipeline.
"""

from __future__ import annotations

import json
import re
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel, Field

router = APIRouter(prefix="/urdu", tags=["urdu"])

DATA_PATH = Path(__file__).resolve().parent / "data" / "urdu_islamic_terms.json"

# Presentation forms are compatibility-normalized by NFKC.  These mappings
# then fold Arabic/Persian keyboard variants to the standard Urdu code points.
_URDU_CHAR_MAP = str.maketrans(
    {
        "ي": "ی",
        "ى": "ی",
        "ئ": "ئ",
        "ك": "ک",
        "ة": "ۃ",
        "ۀ": "ہ",
        "ه": "ہ",
        "ھ": "ھ",
        "ؤ": "ؤ",
        "ـ": "",
        "\u200b": "",
        "\u200c": "",
        "\u200d": "",
        "\ufeff": "",
    }
)

# Arabic combining marks, Quranic annotation signs, and superscript alif.
_DIACRITICS_RE = re.compile("[\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06ed]")
_WHITESPACE_RE = re.compile(r"[\t\r\f\v ]+")
_URDU_CHAR_RE = re.compile(r"[\u0600-\u06ff\u0750-\u077f\u08a0-\u08ff]")
_LATIN_CHAR_RE = re.compile(r"[A-Za-z]")
_DEVANAGARI_CHAR_RE = re.compile(r"[\u0900-\u097f]")
_TOKEN_RE = re.compile(
    r"[\u0600-\u06ff\u0750-\u077f\u08a0-\u08ff]+(?:['’][\u0600-\u06ff]+)*"
    r"|[A-Za-z]+(?:['’-][A-Za-z]+)*|\d+(?:[.,:/-]\d+)*|[^\s]"
)

# Conservative character transliteration intended for discoverability and
# mixed-script search, not for replacing a scholarly transliteration standard.
_TRANSLITERATION_MAP = {
    "ا": "a",
    "آ": "aa",
    "أ": "a",
    "إ": "i",
    "ء": "'",
    "ؤ": "u",
    "ئ": "y",
    "ب": "b",
    "پ": "p",
    "ت": "t",
    "ٹ": "t",
    "ث": "th",
    "ج": "j",
    "چ": "ch",
    "ح": "h",
    "خ": "kh",
    "د": "d",
    "ڈ": "d",
    "ذ": "dh",
    "ر": "r",
    "ڑ": "r",
    "ز": "z",
    "ژ": "zh",
    "س": "s",
    "ش": "sh",
    "ص": "s",
    "ض": "d",
    "ط": "t",
    "ظ": "z",
    "ع": "'",
    "غ": "gh",
    "ف": "f",
    "ق": "q",
    "ک": "k",
    "گ": "g",
    "ل": "l",
    "م": "m",
    "ن": "n",
    "ں": "n",
    "و": "w",
    "ہ": "h",
    "ۃ": "h",
    "ھ": "h",
    "ی": "y",
    "ے": "e",
}


class UrduIslamicTerm(BaseModel):
    """A curated Islamic term used in Urdu scholarly and everyday writing."""

    id: str
    urdu_term: str
    arabic_original: str
    transliteration: str
    english_equivalent: str
    category: str
    definition_ur: str
    variants: list[str] = Field(default_factory=list)


class UrduToken(BaseModel):
    """A token with optional terminology metadata."""

    text: str
    normalized: str
    kind: Literal["islamic_term", "urdu", "latin", "number", "punctuation"]
    term_id: str | None = None


class ScriptProfile(BaseModel):
    """Character-level script statistics for code-switched input."""

    urdu_arabic_characters: int
    latin_characters: int
    devanagari_characters: int
    dominant_script: Literal["urdu_arabic", "latin", "devanagari", "mixed", "none"]
    mixed_script: bool


class UrduAnalysis(BaseModel):
    """Normalized text and linguistic signals for downstream retrieval/generation."""

    original_text: str
    normalized_text: str
    tokens: list[UrduToken]
    recognized_terms: list[UrduIslamicTerm]
    script_profile: ScriptProfile
    transliteration: str
    generation_guidance: str


class UrduProcessRequest(BaseModel):
    text: str = Field(min_length=1)
    preserve_diacritics: bool = True


class UrduTerminology:
    """Load and search the bundled Urdu Islamic terminology database."""

    def __init__(self, data_file: Path = DATA_PATH) -> None:
        self.terms: list[UrduIslamicTerm] = []
        self._lookup: dict[str, UrduIslamicTerm] = {}
        if data_file.exists():
            with data_file.open(encoding="utf-8") as source:
                payload = json.load(source)
            for raw in payload.get("terms", []):
                term = UrduIslamicTerm.model_validate(raw)
                self.terms.append(term)
                for form in (term.urdu_term, term.arabic_original, term.transliteration, *term.variants):
                    key = normalize_urdu(form, preserve_diacritics=False).casefold()
                    if key:
                        self._lookup[key] = term

    def lookup(self, text: str) -> UrduIslamicTerm | None:
        key = normalize_urdu(text, preserve_diacritics=False).casefold()
        return self._lookup.get(key)

    def search(self, query: str, limit: int = 20) -> list[UrduIslamicTerm]:
        key = normalize_urdu(query, preserve_diacritics=False).casefold()
        if not key:
            return self.terms[:limit]
        exact = self._lookup.get(key)
        matches = [
            term
            for term in self.terms
            if key in normalize_urdu(
                " ".join(
                    [term.urdu_term, term.arabic_original, term.transliteration, term.english_equivalent, *term.variants]
                ),
                preserve_diacritics=False,
            ).casefold()
        ]
        if exact is not None:
            matches = [exact, *(term for term in matches if term.id != exact.id)]
        return matches[:limit]

    @property
    def phrases(self) -> list[str]:
        forms = [form for form in self._lookup if " " in form]
        return sorted(forms, key=len, reverse=True)


@lru_cache(maxsize=1)
def get_terminology() -> UrduTerminology:
    return UrduTerminology()


def normalize_urdu(text: str, preserve_diacritics: bool = True) -> str:
    """Normalize presentation forms, keyboard variants, spacing, and punctuation.

    Diacritics are preserved by default because they can be semantically
    important in Arabic quotations.  Callers may remove them for retrieval keys.
    """
    normalized = unicodedata.normalize("NFKC", text).translate(_URDU_CHAR_MAP)
    if not preserve_diacritics:
        normalized = _DIACRITICS_RE.sub("", normalized)
    normalized = normalized.replace("?", "؟").replace(",", "،")
    lines = [_WHITESPACE_RE.sub(" ", line).strip() for line in normalized.split("\n")]
    return "\n".join(line for line in lines if line).strip()


def analyze_script(text: str) -> ScriptProfile:
    urdu_count = len(_URDU_CHAR_RE.findall(text))
    latin_count = len(_LATIN_CHAR_RE.findall(text))
    devanagari_count = len(_DEVANAGARI_CHAR_RE.findall(text))
    populated = sum(count > 0 for count in (urdu_count, latin_count, devanagari_count))
    if populated > 1:
        dominant = "mixed"
    elif urdu_count:
        dominant = "urdu_arabic"
    elif latin_count:
        dominant = "latin"
    elif devanagari_count:
        dominant = "devanagari"
    else:
        dominant = "none"
    return ScriptProfile(
        urdu_arabic_characters=urdu_count,
        latin_characters=latin_count,
        devanagari_characters=devanagari_count,
        dominant_script=dominant,
        mixed_script=populated > 1,
    )


def _protect_phrases(text: str, terminology: UrduTerminology) -> tuple[str, dict[str, str]]:
    protected = text
    replacements: dict[str, str] = {}
    for index, phrase in enumerate(terminology.phrases):
        pattern = re.compile(r"(?<![\w\u0600-\u06ff])" + re.escape(phrase) + r"(?![\w\u0600-\u06ff])", re.IGNORECASE)
        if pattern.search(protected):
            marker = f"ZZURDUTERM{index}ZZ"
            protected = pattern.sub(marker, protected)
            replacements[marker] = phrase
    return protected, replacements


def tokenize_urdu(text: str) -> list[UrduToken]:
    """Tokenize mixed Urdu text while keeping known multi-word terms intact."""
    terminology = get_terminology()
    normalized_text = normalize_urdu(text)
    protected, replacements = _protect_phrases(normalized_text, terminology)
    tokens: list[UrduToken] = []
    for raw in _TOKEN_RE.findall(protected):
        token_text = replacements.get(raw, raw)
        normalized = normalize_urdu(token_text, preserve_diacritics=False)
        term = terminology.lookup(normalized)
        if term is not None:
            kind: Literal["islamic_term", "urdu", "latin", "number", "punctuation"] = "islamic_term"
        elif normalized.replace(".", "").replace(",", "").isdigit():
            kind = "number"
        elif _URDU_CHAR_RE.search(normalized):
            kind = "urdu"
        elif _LATIN_CHAR_RE.search(normalized):
            kind = "latin"
        else:
            kind = "punctuation"
        tokens.append(
            UrduToken(
                text=token_text,
                normalized=normalized,
                kind=kind,
                term_id=term.id if term else None,
            )
        )
    return tokens


def transliterate_urdu(text: str) -> str:
    """Return a stable Latin transliteration, preferring curated term forms."""
    terminology = get_terminology()
    tokens = tokenize_urdu(text)
    rendered: list[str] = []
    for token in tokens:
        term = terminology.lookup(token.normalized)
        if term is not None:
            rendered.append(term.transliteration)
            continue
        value = "".join(_TRANSLITERATION_MAP.get(char, char) for char in token.normalized)
        rendered.append(value)

    result = " ".join(rendered)
    result = re.sub(r"\s+([،۔؟!,:;])", r"\1", result)
    result = result.replace("،", ",").replace("۔", ".").replace("؟", "?")
    return re.sub(r"\s+", " ", result).strip()


def extract_islamic_terms(text: str) -> list[UrduIslamicTerm]:
    """Recognize terms once, in textual order, across Urdu and Latin forms."""
    terminology = get_terminology()
    found: dict[str, UrduIslamicTerm] = {}
    for token in tokenize_urdu(text):
        term = terminology.lookup(token.normalized)
        if term is not None:
            found.setdefault(term.id, term)
    return list(found.values())


def build_generation_guidance(terms: list[UrduIslamicTerm], profile: ScriptProfile) -> str:
    """Create concise instructions for the existing multilingual model pipeline."""
    guidance = [
        "جواب واضح، باادب اور فطری اردو میں دیں۔",
        "قرآنی آیات اور احادیث کے اصل عربی متن کو تبدیل نہ کریں اور حوالہ واضح لکھیں۔",
        "فقہی اختلاف میں کسی ایک رائے کو بلا وضاحت قطعی یا واحد رائے نہ کہیں۔",
    ]
    if terms:
        labels = "، ".join(f"{term.urdu_term} ({term.transliteration})" for term in terms)
        guidance.append(f"سوال میں شناخت شدہ اسلامی اصطلاحات: {labels}۔ انہی مستند املا اور معانی کو ملحوظ رکھیں۔")
    if profile.mixed_script:
        guidance.append("مخلوط رسم الخط کو سمجھیں، مگر جواب کا بنیادی رسم الخط اردو رکھیں؛ ضروری اصل عربی محفوظ رکھیں۔")
    return " ".join(guidance)


def process_urdu(text: str, preserve_diacritics: bool = True) -> UrduAnalysis:
    """Run the complete Urdu preprocessing and query-understanding pipeline."""
    normalized = normalize_urdu(text, preserve_diacritics=preserve_diacritics)
    profile = analyze_script(normalized)
    terms = extract_islamic_terms(normalized)
    return UrduAnalysis(
        original_text=text,
        normalized_text=normalized,
        tokens=tokenize_urdu(normalized),
        recognized_terms=terms,
        script_profile=profile,
        transliteration=transliterate_urdu(normalized),
        generation_guidance=build_generation_guidance(terms, profile),
    )


@router.post("/process", response_model=UrduAnalysis)
def process_endpoint(request: UrduProcessRequest) -> UrduAnalysis:
    """Normalize and analyze an Urdu or Urdu-code-switched query."""
    return process_urdu(request.text, preserve_diacritics=request.preserve_diacritics)


@router.get("/terms", response_model=list[UrduIslamicTerm])
def search_terms(query: str = "", limit: int = 20) -> list[UrduIslamicTerm]:
    """Search curated terminology by Urdu, Arabic, transliteration, or English."""
    safe_limit = min(max(limit, 1), 100)
    return get_terminology().search(query, safe_limit)
