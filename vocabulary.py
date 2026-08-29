"""Quranic Vocabulary Analysis and Learning System (#168)

Provides root-word extraction, frequency statistics, morphological grouping,
search by root or meaning, and example-verse retrieval for Quranic Arabic.
"""

import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field

from corpus import corpus

logger = logging.getLogger(__name__)

router = APIRouter(tags=["vocabulary"])

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent / "data"
VOCAB_FILE = DATA_DIR / "quran_vocabulary.json"


def _load_vocabulary_data() -> dict[str, Any]:
    if VOCAB_FILE.exists():
        with open(VOCAB_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {}


VOCAB_DATA: dict[str, Any] = _load_vocabulary_data()

# ---------------------------------------------------------------------------
# Arabic root extraction (light stemmer)
# ---------------------------------------------------------------------------

# Common Arabic prefixes/suffixes stripped during light stemming
_PREFIXES = ["ال", "و", "ف", "ب", "ل", "ك", "س"]
_SUFFIXES = ["ه", "ها", "هم", "هن", "هما", "ون", "ين", "ان", "ات", "ية", "ي"]

_TASHKEEL = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06DC\u06DF-\u06E4\u06E7\u06E8\u06EA-\u06ED]")


def strip_tashkeel(word: str) -> str:
    return _TASHKEEL.sub("", word)


def extract_root(word: str) -> str:
    """Light Arabic root extraction: strip tashkeel, common prefixes/suffixes, and
    reduce consonant clusters to approximate the trilateral/quadrilateral root.

    This is a heuristic — for production, pair with a morphological analyser.
    """
    w = strip_tashkeel(word).strip()
    # strip common prefixes
    for p in sorted(_PREFIXES, key=len, reverse=True):
        if len(w) > 3 and w.startswith(p):
            candidate = w[len(p) :]
            if len(candidate) >= 2:
                w = candidate
                break
    # strip common suffixes
    for s in sorted(_SUFFIXES, key=len, reverse=True):
        if len(w) > 3 and w.endswith(s):
            candidate = w[: -len(s)]
            if len(candidate) >= 2:
                w = candidate
                break
    return w


# ---------------------------------------------------------------------------
# Vocabulary DB helpers
# ---------------------------------------------------------------------------


def get_all_entries() -> list[dict[str, Any]]:
    return VOCAB_DATA.get("entries", [])


def find_by_root(root: str) -> list[dict[str, Any]]:
    root = strip_tashkeel(root).strip()
    return [e for e in get_all_entries() if extract_root(e.get("word", "")) == root]


def find_by_meaning(query: str) -> list[dict[str, Any]]:
    q = query.lower()
    return [e for e in get_all_entries() if q in e.get("meaning", "").lower()]


def find_by_word(word: str) -> dict[str, Any] | None:
    w = strip_tashkeel(word).strip()
    for e in get_all_entries():
        if strip_tashkeel(e.get("word", "")) == w:
            return e
    return None


# ---------------------------------------------------------------------------
# Frequency analysis
# ---------------------------------------------------------------------------


def _count_word_in_corpus(word: str) -> int:
    """Count occurrences of a word across all ayat."""
    w = strip_tashkeel(word)
    count = 0
    for _key, ayah_data in corpus.ayat.items():
        text = strip_tashkeel(ayah_data.get("text", ""))
        count += text.count(w)
    return count


def compute_frequencies() -> list[dict[str, Any]]:
    """Return vocabulary entries enriched with corpus frequency counts."""
    results = []
    for entry in get_all_entries():
        freq = _count_word_in_corpus(entry.get("word", ""))
        results.append({**entry, "frequency": freq})
    return sorted(results, key=lambda e: -e["frequency"])


# ---------------------------------------------------------------------------
# Root family grouping
# ---------------------------------------------------------------------------


def group_by_root() -> dict[str, list[dict[str, Any]]]:
    """Group vocabulary entries by their extracted root."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in get_all_entries():
        root = extract_root(entry.get("word", ""))
        groups[root].append(entry)
    return dict(groups)


# ---------------------------------------------------------------------------
# Example verse retrieval
# ---------------------------------------------------------------------------


def get_example_verses(word: str, limit: int = 5) -> list[dict[str, Any]]:
    """Return up to *limit* ayat that contain *word*."""
    w = strip_tashkeel(word)
    results: list[dict[str, Any]] = []
    for key, ayah_data in corpus.ayat.items():
        text = ayah_data.get("text", "")
        if w in strip_tashkeel(text):
            parts = key.split(":")
            results.append(
                {
                    "surah": int(parts[0]),
                    "ayah": int(parts[1]),
                    "text": text,
                }
            )
            if len(results) >= limit:
                break
    return results


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class VocabularyEntry(BaseModel):
    word: str = Field(..., description="Quranic Arabic word")
    root: str = Field(default="", description="Extracted root")
    meaning: str = Field(default="", description="English meaning")
    frequency: int = Field(default=0, description="Occurrence count in Quran")
    verses: list[dict[str, Any]] = Field(default_factory=list, description="Example verses")


class RootFamily(BaseModel):
    root: str
    words: list[dict[str, Any]]


class VocabularySearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Root, word, or English meaning to search")
    search_type: str = Field(default="auto", description="'root', 'meaning', 'word', or 'auto'")


class VocabularySearchResponse(BaseModel):
    results: list[dict[str, Any]]
    total: int


class FrequencyResponse(BaseModel):
    entries: list[dict[str, Any]]
    total: int


class RootFamilyResponse(BaseModel):
    families: dict[str, list[dict[str, Any]]]
    total_roots: int


class ExampleVersesResponse(BaseModel):
    word: str
    verses: list[dict[str, Any]]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.get("/vocabulary/search", response_model=VocabularySearchResponse)
async def search_vocabulary(
    query: str = "",
    search_type: str = "auto",
) -> VocabularySearchResponse:
    """Search vocabulary by root, word form, or English meaning.

    Use ``search_type=root|word|meaning`` or ``auto`` (default) which tries
    all three and merges results.
    """
    results: list[dict[str, Any]] = []
    if search_type == "root":
        results = find_by_root(query)
    elif search_type == "meaning":
        results = find_by_meaning(query)
    elif search_type == "word":
        match = find_by_word(query)
        results = [match] if match else []
    else:
        # auto
        by_root = find_by_root(query)
        by_meaning = find_by_meaning(query)
        by_word = find_by_word(query)
        seen: set[str] = set()
        for entry in by_root + by_meaning:
            w = entry.get("word", "")
            if w not in seen:
                seen.add(w)
                results.append(entry)
        if by_word and by_word.get("word") not in seen:
            results.append(by_word)

    return VocabularySearchResponse(results=results, total=len(results))


@router.get("/vocabulary/frequencies", response_model=FrequencyResponse)
async def get_frequencies() -> FrequencyResponse:
    """Return all vocabulary entries sorted by Quranic frequency."""
    entries = compute_frequencies()
    return FrequencyResponse(entries=entries, total=len(entries))


@router.get("/vocabulary/roots", response_model=RootFamilyResponse)
async def get_root_families() -> RootFamilyResponse:
    """Group vocabulary by their Arabic root."""
    families = group_by_root()
    return RootFamilyResponse(families=families, total_roots=len(families))


@router.get("/vocabulary/verse-examples", response_model=ExampleVersesResponse)
async def verse_examples(word: str = "", limit: int = 5) -> ExampleVersesResponse:
    """Retrieve example ayat containing the given word."""
    verses = get_example_verses(word, limit=limit)
    return ExampleVersesResponse(word=word, verses=verses)


@router.get("/vocabulary/word/{word}", response_model=VocabularyEntry)
async def get_word_detail(word: str) -> VocabularyEntry:
    """Get full details for a single word including root, meaning, and examples."""
    entry = find_by_word(word)
    if not entry:
        from errors import APIException

        raise APIException(
            status_code=404,
            detail=f"Word '{word}' not found in vocabulary database.",
            hint="Try a different spelling or search by root/meaning using /vocabulary/search.",
        )
    root = extract_root(entry.get("word", ""))
    freq = _count_word_in_corpus(entry.get("word", ""))
    verses = get_example_verses(entry.get("word", ""), limit=5)
    return VocabularyEntry(
        word=entry.get("word", word),
        root=root,
        meaning=entry.get("meaning", ""),
        frequency=freq,
        verses=verses,
    )
