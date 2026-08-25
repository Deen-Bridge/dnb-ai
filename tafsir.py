"""Tafsir layer — grounded, attributed ayah explanations.

Why this exists
---------------
Asked "what does Surah al-'Asr mean?", a bare language model produces a
paraphrase from its own memory: no named mufassir, no way to see how Ibn Kathir
or al-Sa'di actually explained the ayah, and no line between the classical
reading and a modern gloss. Verse interpretation is where a fabricated or
flattened explanation does the most damage. This module replaces recall with
retrieval: real tafsir text is fetched for the ayah, every explanatory claim
carries the name of the work it came from, and where the mufassirun differ the
difference is surfaced rather than collapsed into one answer.

Attribution policy
------------------
The work's name, author, and language always come from the source's own
response for the resource that was fetched — never from this service's memory
of who wrote what. ``TAFSIR_REGISTRY`` below maps our stable keys to source
slugs and holds display names used only for "this tafsir is unavailable"
messages; the moment real text is returned, the attribution attached to it is
the source's.

Reference validation
--------------------
Surah and ayah bounds are checked offline against ``data/quran/surah_index.json``
before any network call, so ``2:300`` is a 400 with a clear message rather than
a request that gets answered with an invented verse.

Caching
-------
Tafsir text is immutable per ayah, so it is cached by exact key through
``semantic_cache.get_keyed_cache`` — the keyed sibling of the semantic response
cache, sharing that module's TTL and eviction configuration rather than
introducing a second cache system.
"""

from __future__ import annotations

import asyncio
import collections
import copy
import html
import json
import logging
import os
import re
from collections.abc import Awaitable, Callable
from functools import lru_cache
from pathlib import Path
from typing import Any, cast

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from semantic_cache import get_keyed_cache

logger = logging.getLogger(__name__)

router = APIRouter(tags=["tafsir"])

DATA_PATH = Path(__file__).resolve().parent / "data" / "quran" / "surah_index.json"

QURAN_API_BASE = os.getenv("QURAN_API_BASE", "https://api.quran.com/api/v4")
QURAN_API_TIMEOUT = float(os.getenv("QURAN_API_TIMEOUT", "15"))

# Quran.com translation resource ids, by language.
TRANSLATION_IDS: dict[str, int] = {
    "en": 20,  # Saheeh International
    "ur": 97,  # Maulana Fateh Muhammad Jalandhari
    "bn": 163,  # Taisirul Quran
    "ru": 79,  # Elmir Kuliev
}
DEFAULT_TRANSLATION_LANGUAGE = "en"

# Language codes used in TAFSIR_REGISTRY, spelled out for display.
LANGUAGE_NAMES: dict[str, str] = {
    "en": "english",
    "ar": "arabic",
    "ur": "urdu",
    "bn": "bengali",
    "ru": "russian",
}

MAX_AYAT_PER_REQUEST = int(os.getenv("TAFSIR_MAX_AYAT", "10"))
MAX_TAFSIRS_PER_REQUEST = 6
MAX_COMPARISON_TAFSIRS = 12
MIN_COMPARISON_TAFSIRS = 10
COMPARISON_REPRESENTATIVE_SENTENCES = 2
COMPARISON_REPRESENTATIVE_CHARS = 500
DEFAULT_SIMILARITY_THRESHOLD = 0.35

# Wall-clock budget for retrieving tafsir inside a /chat turn. Retrieval runs
# concurrently, but a slow upstream must not hold a chat turn open indefinitely:
# past this budget the turn proceeds ungrounded rather than stalling.
CHAT_RETRIEVAL_TIMEOUT = float(os.getenv("TAFSIR_CHAT_TIMEOUT", "20"))

# How much tafsir text is handed to the model when synthesizing a chat answer.
# Ibn Kathir on a single ayah can run to thousands of words; the cap keeps the
# prompt (and the bill) bounded without touching what /tafsir returns.
CHAT_EXCERPT_CHARS = int(os.getenv("TAFSIR_CHAT_EXCERPT_CHARS", "2500"))

DISCLAIMER = (
    "Tafsir text is retrieved verbatim from the works named above and is "
    "presented for study. Classical tafsir often assumes context this excerpt "
    "does not carry; consult a qualified scholar before acting on an "
    "interpretation."
)


# ---------------------------------------------------------------------------
# Tafsir registry
# ---------------------------------------------------------------------------


class TafsirWork(BaseModel):
    """A tafsir this service can retrieve, and where to get it per language.

    ``name`` and ``author`` are for *display before retrieval* — listing
    available works, and saying which work was unavailable for an ayah. Text
    that is actually returned is labelled from the source response instead.
    """

    key: str
    name: str
    author: str
    slugs: dict[str, str] = Field(..., description="Language code -> Quran.com tafsir slug")
    era: str = Field(default="unknown", description="Broad historical period of the work")
    death_year_ah: int | None = Field(default=None, description="Author's death year in the Hijri calendar, when known")
    methodologies: list[str] = Field(default_factory=list, description="Methodological descriptors for comparison")
    perspective: str = Field(default="not classified", description="Broad perspective label; not a sectarian ruling")

    def slug_for(self, language: str) -> str | None:
        return self.slugs.get(language)

    @property
    def languages(self) -> list[str]:
        return sorted(self.slugs)


TAFSIR_REGISTRY: dict[str, TafsirWork] = {
    work.key: work
    for work in [
        TafsirWork(
            key="ibn-kathir",
            name="Tafsir Ibn Kathir",
            author="Ibn Kathir (d. 774 AH)",
            slugs={
                "en": "en-tafisr-ibn-kathir",
                "ar": "ar-tafsir-ibn-kathir",
                "ur": "tafseer-ibn-e-kaseer-urdu",
                "bn": "bn-tafseer-ibn-e-kaseer",
            },
            era="classical",
            death_year_ah=774,
            methodologies=["narration", "hadith", "linguistic"],
            perspective="Sunni",
        ),
        TafsirWork(
            key="tabari",
            name="Jami' al-Bayan (Tafsir al-Tabari)",
            author="Ibn Jarir al-Tabari (d. 310 AH)",
            slugs={"ar": "ar-tafsir-al-tabari"},
            era="classical",
            death_year_ah=310,
            methodologies=["narration", "linguistic", "variant_readings"],
            perspective="Sunni",
        ),
        TafsirWork(
            key="qurtubi",
            name="Al-Jami' li-Ahkam al-Qur'an (Tafsir al-Qurtubi)",
            author="Al-Qurtubi (d. 671 AH)",
            slugs={"ar": "ar-tafseer-al-qurtubi"},
            era="classical",
            death_year_ah=671,
            methodologies=["juristic", "linguistic", "narration"],
            perspective="Sunni",
        ),
        TafsirWork(
            key="saadi",
            name="Taysir al-Karim al-Rahman (Tafsir al-Sa'di)",
            author="Abd al-Rahman al-Sa'di (d. 1376 AH)",
            slugs={"ar": "ar-tafseer-al-saddi", "ru": "ru-tafseer-al-saddi"},
            era="modern",
            death_year_ah=1376,
            methodologies=["thematic", "linguistic", "concise"],
            perspective="Sunni",
        ),
        TafsirWork(
            key="baghawi",
            name="Ma'alim al-Tanzil (Tafsir al-Baghawi)",
            author="Al-Baghawi (d. 516 AH)",
            slugs={"ar": "ar-tafsir-al-baghawi"},
            era="classical",
            death_year_ah=516,
            methodologies=["narration", "linguistic"],
            perspective="Sunni",
        ),
        TafsirWork(
            key="muyassar",
            name="Al-Tafsir al-Muyassar",
            author="King Fahd Complex scholarly committee",
            slugs={"ar": "ar-tafsir-muyassar"},
            era="contemporary",
            methodologies=["concise", "thematic"],
            perspective="institutional",
        ),
        TafsirWork(
            key="wasit",
            name="Al-Tafsir al-Wasit",
            author="Muhammad Sayyid Tantawi (d. 1431 AH)",
            slugs={"ar": "ar-tafsir-al-wasit"},
            era="contemporary",
            death_year_ah=1431,
            methodologies=["linguistic", "thematic", "juristic"],
            perspective="Sunni",
        ),
        TafsirWork(
            key="maarif-ul-quran",
            name="Ma'arif al-Qur'an",
            author="Mufti Muhammad Shafi (d. 1396 AH)",
            slugs={"en": "en-tafsir-maarif-ul-quran"},
            era="modern",
            death_year_ah=1396,
            methodologies=["juristic", "narration", "thematic"],
            perspective="Sunni",
        ),
        TafsirWork(
            key="bayan-ul-quran",
            name="Bayan ul Quran",
            author="Dr. Israr Ahmad (d. 1431 AH)",
            slugs={"ur": "tafsir-bayan-ul-quran"},
            era="contemporary",
            death_year_ah=1431,
            methodologies=["thematic", "linguistic"],
            perspective="Sunni",
        ),
        TafsirWork(
            key="fi-zilal",
            name="Fi Zilal al-Qur'an",
            author="Sayyid Qutb (d. 1386 AH)",
            slugs={"ur": "tafsir-fe-zalul-quran-syed-qatab"},
            era="modern",
            death_year_ah=1386,
            methodologies=["thematic", "literary", "social"],
            perspective="Sunni",
        ),
        TafsirWork(
            key="tazkirul-quran",
            name="Tazkirul Quran",
            author="Maulana Wahiduddin Khan (d. 1443 AH)",
            slugs={"en": "tazkirul-quran-en", "ur": "tazkiru-quran-ur"},
            era="contemporary",
            death_year_ah=1443,
            methodologies=["thematic", "linguistic", "concise"],
            perspective="Sunni",
        ),
        TafsirWork(
            key="ahsanul-bayaan",
            name="Tafsir Ahsanul Bayaan",
            author="Bayaan Foundation",
            slugs={"bn": "bn-tafsir-ahsanul-bayaan"},
            era="contemporary",
            methodologies=["concise", "thematic"],
            perspective="institutional",
        ),
    ]
}

# Four classical works spanning narration-based (Ibn Kathir, al-Tabari), legal
# (al-Qurtubi) and concise-summary (al-Sa'di) approaches — chosen so a default
# request already shows more than one methodology.
DEFAULT_TAFSIR_KEYS: tuple[str, ...] = ("ibn-kathir", "tabari", "qurtubi", "saadi")
DEFAULT_COMPARISON_TAFSIR_KEYS: tuple[str, ...] = tuple(TAFSIR_REGISTRY)

TAFSIR_ALIASES: dict[str, str] = {
    "ibnkathir": "ibn-kathir",
    "ibn kathir": "ibn-kathir",
    "ibn-katheer": "ibn-kathir",
    "kathir": "ibn-kathir",
    "al-tabari": "tabari",
    "at-tabari": "tabari",
    "jami-al-bayan": "tabari",
    "al-qurtubi": "qurtubi",
    "sadi": "saadi",
    "sa'di": "saadi",
    "as-sadi": "saadi",
    "al-saadi": "saadi",
    "saedi": "saadi",
    "al-baghawi": "baghawi",
    "maarif": "maarif-ul-quran",
    "maariful-quran": "maarif-ul-quran",
    "ma'arif al-qur'an": "maarif-ul-quran",
}


def normalize_tafsir_key(raw: str) -> str | None:
    """Map a user-supplied tafsir name to a registry key, or None."""
    cleaned = " ".join((raw or "").strip().casefold().split())
    if not cleaned:
        return None
    if cleaned in TAFSIR_REGISTRY:
        return cleaned
    if cleaned in TAFSIR_ALIASES:
        return TAFSIR_ALIASES[cleaned]
    hyphenated = cleaned.replace(" ", "-")
    if hyphenated in TAFSIR_REGISTRY:
        return hyphenated
    return TAFSIR_ALIASES.get(hyphenated)


# ---------------------------------------------------------------------------
# Surah index and ayah references
# ---------------------------------------------------------------------------


class Surah(BaseModel):
    number: int
    name: str
    arabic_name: str
    revelation_place: str
    ayah_count: int
    aliases: list[str] = []


@lru_cache(maxsize=1)
def load_surah_index() -> tuple[Surah, ...]:
    with DATA_PATH.open(encoding="utf-8") as f:
        return tuple(Surah(**row) for row in json.load(f))


@lru_cache(maxsize=1)
def _name_lookup() -> dict[str, int]:
    """Normalized surah name/alias -> surah number."""
    lookup: dict[str, int] = {}
    for surah in load_surah_index():
        for label in [surah.name, surah.arabic_name, *surah.aliases]:
            lookup.setdefault(_normalize_surah_name(label), surah.number)
    return lookup


# The definite article as it is actually transliterated. Before a "sun letter"
# the lām assimilates and the consonant doubles — At-Tawbah, Ash-Shams,
# Adh-Dhariyat — so the article cannot simply be matched as "al". Longest forms
# first, so "adh" is tried before "ad".
_SUN_ARTICLES: tuple[tuple[str, str], ...] = (
    ("ash", "sh"),
    ("adh", "dh"),
    ("ath", "th"),
    ("as", "s"),
    ("ad", "d"),
    ("an", "n"),
    ("ar", "r"),
    ("at", "t"),
    ("az", "z"),
)
_MOON_ARTICLES: tuple[str, ...] = ("al", "ul")


def _strip_article(collapsed: str) -> str:
    """Remove a leading definite article, assimilated or not.

    A sun-letter article is only stripped when the consonant it assimilated into
    actually doubles: "attawbah" → "tawbah", but "anfal" (Al-Anfal without its
    article) keeps its "an", because "f" is not "n" and the "an" is part of the
    name.
    """
    for article, consonant in _SUN_ARTICLES:
        remainder = collapsed[len(article) :]
        if collapsed.startswith(article) and remainder.startswith(consonant) and len(remainder) > 2:
            return remainder
    for article in _MOON_ARTICLES:
        if collapsed.startswith(article) and len(collapsed) > len(article) + 2:
            return collapsed[len(article) :]
    return collapsed


def _normalize_surah_name(name: str) -> str:
    """Fold the many spellings of a surah name onto one form.

    "Al-'Asr", "al asr", "Surat ul-Asr" and "AlAsr" all normalize to "asr", and
    "At-Tawbah", "at tawbah" and the bare "tawbah" all normalize to "tawbah":
    punctuation is dropped, case is folded, and the definite article is removed
    in whichever form it was written.
    """
    lowered = (name or "").casefold()
    lowered = re.sub(r"^surah?t?\s+", "", lowered)
    lowered = re.sub(r"[^\w؀-ۿ\s]", "", lowered)
    lowered = re.sub(r"^(al|ul|as|ash|adh|ath|ad|an|ar|at|az)\s+", r"\1", lowered)
    collapsed = re.sub(r"\s+", "", lowered)
    return _strip_article(collapsed)


def surah_by_number(number: int) -> Surah | None:
    index = load_surah_index()
    if 1 <= number <= len(index):
        return index[number - 1]
    return None


def surah_by_name(name: str) -> Surah | None:
    number = _name_lookup().get(_normalize_surah_name(name))
    return surah_by_number(number) if number else None


class AyahRef(BaseModel):
    """A validated single-ayah reference."""

    surah: int
    ayah: int

    @property
    def key(self) -> str:
        return f"{self.surah}:{self.ayah}"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.key


class InvalidReference(ValueError):
    """Raised when a reference names a surah or ayah that does not exist."""


REFERENCE_PATTERN = re.compile(
    r"^\s*(?P<surah>\d{1,3})\s*[:.\-]\s*(?P<start>\d{1,3})"
    r"(?:\s*(?:-|–|to)\s*(?P<end>\d{1,3}))?\s*$"
)


def validate_reference(surah: int, ayah: int) -> AyahRef:
    """Return a validated ``AyahRef`` or raise ``InvalidReference``.

    The message names the actual bound, so a caller who asked for 2:300 is told
    that Al-Baqarah has 286 ayat rather than being handed a made-up verse.
    """
    record = surah_by_number(surah)
    if record is None:
        raise InvalidReference(f"Surah {surah} does not exist. Surah numbers run from 1 to 114.")
    if ayah < 1 or ayah > record.ayah_count:
        raise InvalidReference(
            f"Ayah {ayah} does not exist in surah {surah} ({record.name}), which has {record.ayah_count} ayat."
        )
    return AyahRef(surah=surah, ayah=ayah)


def parse_reference(reference: str) -> list[AyahRef]:
    """Parse ``"103:1"``, ``"103:1-3"`` or ``"Al-Asr 1-3"`` into ayah refs.

    Raises ``InvalidReference`` for anything unparseable or out of bounds.
    """
    raw = (reference or "").strip()
    if not raw:
        raise InvalidReference("A surah:ayah reference is required, e.g. '103:1'.")

    numeric = REFERENCE_PATTERN.match(raw)
    parts = numeric.groupdict() if numeric else _match_named_reference(raw)
    if parts is None:
        raise InvalidReference(
            f"Could not parse '{reference}'. Use a surah:ayah reference such as '103:1' or a range such as '103:1-3'."
        )

    surah = int(cast(str, parts["surah"]))
    start = int(cast(str, parts["start"]))
    end = int(cast(str, parts["end"])) if parts.get("end") else start

    if end < start:
        raise InvalidReference(f"Invalid range {start}-{end}: the last ayah comes before the first.")

    first = validate_reference(surah, start)
    last = validate_reference(surah, end)
    span = last.ayah - first.ayah + 1
    if span > MAX_AYAT_PER_REQUEST:
        raise InvalidReference(f"Range covers {span} ayat; at most {MAX_AYAT_PER_REQUEST} may be requested at once.")
    return [AyahRef(surah=surah, ayah=n) for n in range(first.ayah, last.ayah + 1)]


def _match_named_reference(raw: str) -> dict[str, str | None] | None:
    """Match 'Al-Asr 1-3' / 'surah al-baqarah 255' by surah name."""
    named = re.match(
        r"^(?P<name>[^\d]+?)\s*[:,]?\s*(?P<start>\d{1,3})"
        r"(?:\s*(?:-|–|to)\s*(?P<end>\d{1,3}))?\s*$",
        raw,
    )
    if named is None:
        return None
    surah = surah_by_name(named["name"])
    if surah is None:
        return None
    return {
        "surah": str(surah.number),
        "start": named["start"],
        "end": named["end"],
    }


# ---------------------------------------------------------------------------
# Verse-explanation intent detection
# ---------------------------------------------------------------------------

# Reused by main.py's /chat handler. Kept here (next to reference parsing)
# rather than duplicated in the chat path.
EXPLANATION_CUES = (
    "tafsir",
    "tafseer",
    "explain",
    "explanation",
    "meaning",
    "what does",
    "what do",
    "interpret",
    "commentary",
    "mufassir",
    "asbab",
    "context of",
    "significance of",
    "why was",
    "revealed",
)

INLINE_REFERENCE_PATTERN = re.compile(
    r"\b(?P<surah>\d{1,3})\s*[:.]\s*(?P<start>\d{1,3})"
    r"(?:\s*(?:-|–)\s*(?P<end>\d{1,3}))?\b"
)

_AYAH_SUFFIX = (
    r"(?:\s*(?:,|:|ayah?|verse|aayah)?\s*(?P<start>\d{1,3})"
    r"(?:\s*(?:-|–)\s*(?P<end>\d{1,3}))?)?"
)

NAMED_SURAH_PATTERNS = (
    # "surah al-Baqarah 255", "surat Yusuf"
    re.compile(
        r"\bsurah?t?\s+(?:al-?|ul-?|as-?|ash-?|ad-?|an-?|ar-?|at-?|az-?)?"
        r"(?P<name>[\w'’\-]+)" + _AYAH_SUFFIX,
        re.IGNORECASE,
    ),
    # "Al-Ikhlas 1" — without the word "surah" both the definite article and an
    # explicit ayah number are required. A bare name that is also an ordinary
    # word, a person's name, or a name of Allah ("Muhammad", "Maryam",
    # "ar-Rahman") is then never mistaken for a surah reference.
    re.compile(
        r"\b(?:al|ul|as|ash|ad|an|ar|at|az)-(?P<name>[\w'’]+)"
        r"\s*(?:,|:|ayah?|verse|aayah)?\s*(?P<start>\d{1,3})"
        r"(?:\s*(?:-|–)\s*(?P<end>\d{1,3}))?",
        re.IGNORECASE,
    ),
)


def detect_ayah_references(prompt: str) -> list[AyahRef]:
    """Return ayah references a verse-explanation question is asking about.

    Empty when the prompt is not a verse-explanation question, or when it names
    no resolvable ayah — the caller then falls through to the normal chat path.
    An out-of-range reference is skipped rather than raised: a chat message is
    not the place to reject the whole turn over a stray number.
    """
    text = (prompt or "").strip()
    if not text:
        return []
    lowered = text.casefold()
    if not any(cue in lowered for cue in EXPLANATION_CUES):
        return []

    refs: list[AyahRef] = []
    seen: set[str] = set()

    def add(surah: int, start: int, end: int | None) -> None:
        last = end if end is not None else start
        if last < start or last - start + 1 > MAX_AYAT_PER_REQUEST:
            last = start
        for number in range(start, last + 1):
            try:
                ref = validate_reference(surah, number)
            except InvalidReference:
                continue
            if ref.key not in seen:
                seen.add(ref.key)
                refs.append(ref)

    for match in INLINE_REFERENCE_PATTERN.finditer(text):
        add(
            int(match["surah"]),
            int(match["start"]),
            int(match["end"]) if match["end"] else None,
        )

    for pattern in NAMED_SURAH_PATTERNS:
        for match in pattern.finditer(text):
            surah = surah_by_name(match["name"])
            if surah is None:
                continue
            if match["start"]:
                add(
                    surah.number,
                    int(match["start"]),
                    int(match["end"]) if match["end"] else None,
                )
            elif surah.ayah_count <= MAX_AYAT_PER_REQUEST:
                # A short surah named without an ayah number ("what does Surah
                # al-'Asr mean?") is a request for the whole surah.
                add(surah.number, 1, surah.ayah_count)
            else:
                add(surah.number, 1, None)

    return refs[:MAX_AYAT_PER_REQUEST]


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

_TAG_PATTERN = re.compile(r"<[^>]+>")
_BLOCK_END_PATTERN = re.compile(r"</(p|div|h[1-6]|li|br)\s*>|<br\s*/?>", re.IGNORECASE)
# Translations carry footnote markers as <sup foot_note="...">1</sup>. Dropping
# only the tags would leave a bare "1" glued to the end of the verse.
_FOOTNOTE_PATTERN = re.compile(r"<sup\b[^>]*>.*?</sup>", re.IGNORECASE | re.DOTALL)


def strip_html(raw: str) -> str:
    """Flatten the source's HTML tafsir into readable plain text."""
    if not raw:
        return ""
    text = _FOOTNOTE_PATTERN.sub("", raw)
    text = _BLOCK_END_PATTERN.sub("\n", text)
    text = _TAG_PATTERN.sub("", text)
    text = html.unescape(text)
    lines = [line.strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line).strip()


class TafsirText(BaseModel):
    """One tafsir's explanation of one ayah, as returned by the source."""

    key: str
    name: str
    author: str
    language: str
    text: str
    verse_range: str | None = Field(
        None,
        description=("Ayah range this passage covers when the tafsir treats several ayat together, e.g. '103:1-3'"),
    )


class TafsirUnavailable(BaseModel):
    key: str
    name: str
    author: str
    reason: str


class VerseText(BaseModel):
    arabic: str | None = None
    translation: str | None = None
    translation_language: str | None = None


class TafsirSource:
    """Retrieval seam. Tests substitute ``FakeTafsirSource`` for this."""

    async def fetch_tafsir(self, slug: str, verse_key: str) -> dict[str, Any] | None:
        raise NotImplementedError

    async def fetch_verse(self, verse_key: str, language: str) -> VerseText:
        raise NotImplementedError


class QuranComTafsirSource(TafsirSource):
    """Reads tafsir and ayah text from the Quran.com API (v4).

    Returns ``None`` for a tafsir the API does not have for that ayah instead
    of raising, so one missing work degrades to "unavailable" rather than
    failing the whole request.
    """

    def __init__(self, base_url: str = QURAN_API_BASE, timeout: float = QURAN_API_TIMEOUT):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    async def _get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any] | None:
        url = f"{self.base_url}/{path.lstrip('/')}"
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(url, params=params)
        except httpx.HTTPError as exc:
            logger.warning("Quran API request failed for %s: %s", path, exc)
            return None
        if response.status_code == 404:
            return None
        if response.status_code >= 400:
            logger.warning("Quran API returned %s for %s", response.status_code, path)
            return None
        try:
            return response.json()
        except ValueError:
            logger.warning("Quran API returned non-JSON for %s", path)
            return None

    async def fetch_tafsir(self, slug: str, verse_key: str) -> dict[str, Any] | None:
        payload = await self._get(f"tafsirs/{slug}/by_ayah/{verse_key}")
        if not payload:
            return None
        return payload.get("tafsir")

    async def fetch_verse(self, verse_key: str, language: str) -> VerseText:
        translation_id = TRANSLATION_IDS.get(language)
        params: dict[str, Any] = {"fields": "text_uthmani"}
        if translation_id is not None:
            params["translations"] = translation_id
        payload = await self._get(f"verses/by_key/{verse_key}", params)
        if not payload:
            return VerseText()
        verse = payload.get("verse") or {}
        translations = verse.get("translations") or []
        translation = strip_html(translations[0].get("text", "")) if translations else None
        return VerseText(
            arabic=verse.get("text_uthmani"),
            translation=translation or None,
            translation_language=language if translation else None,
        )


class FakeTafsirSource(TafsirSource):
    """Offline source for tests: serves canned payloads, records calls."""

    def __init__(
        self,
        tafsirs: dict[tuple[str, str], dict[str, Any]] | None = None,
        verses: dict[str, VerseText] | None = None,
    ) -> None:
        self.tafsirs = tafsirs or {}
        self.verses = verses or {}
        self.tafsir_calls: list[tuple[str, str]] = []
        self.verse_calls: list[tuple[str, str]] = []

    async def fetch_tafsir(self, slug: str, verse_key: str) -> dict[str, Any] | None:
        self.tafsir_calls.append((slug, verse_key))
        return self.tafsirs.get((slug, verse_key))

    async def fetch_verse(self, verse_key: str, language: str) -> VerseText:
        self.verse_calls.append((verse_key, language))
        return self.verses.get(verse_key, VerseText())


_source: TafsirSource = QuranComTafsirSource()


def get_source() -> TafsirSource:
    return _source


def set_source(source: TafsirSource) -> None:
    """Swap the retrieval backend (used by tests to stay offline)."""
    global _source
    _source = source


def _tafsir_cache():
    return get_keyed_cache("tafsir")


def parse_tafsir_payload(work: TafsirWork, language: str, payload: dict[str, Any]) -> TafsirText | None:
    """Turn a source payload into an attributed ``TafsirText``.

    The work's name comes from the payload, so the label on a passage is the
    source's own. The *language* is taken from ``language`` — the edition this
    service asked for — and deliberately not from the payload's
    ``translated_name.language_name``, which describes the language the work's
    *name* was translated into, not the language of the text. Trusting it would
    label al-Tabari's Arabic commentary "english" for an English-locale request.

    Returns None when the payload carries no usable text, so an empty entry
    never reaches a user dressed up as an explanation.
    """
    text = strip_html(payload.get("text") or "")
    if not text:
        return None

    translated = payload.get("translated_name") or {}
    name = translated.get("name") or payload.get("resource_name") or work.name
    source_language = LANGUAGE_NAMES.get(language, language)

    verses = payload.get("verses") or {}
    verse_range = None
    if len(verses) > 1:
        keys = sorted(
            verses,
            key=lambda k: tuple(int(part) for part in k.split(":")),
        )
        verse_range = f"{keys[0]}-{keys[-1].split(':')[1]}"
    elif len(verses) == 1:
        verse_range = next(iter(verses))

    return TafsirText(
        key=work.key,
        name=name,
        author=work.author,
        language=source_language,
        text=text,
        verse_range=verse_range,
    )


async def fetch_tafsirs_for_ayah(
    ref: AyahRef,
    keys: list[str],
    language: str,
    allow_language_fallback: bool = True,
    source: TafsirSource | None = None,
) -> tuple[list[TafsirText], list[TafsirUnavailable]]:
    """Retrieve each requested tafsir for one ayah.

    Returns ``(available, unavailable)``. A work that has no entry for the ayah,
    or is not published in a usable language, lands in ``unavailable`` with a
    reason — the rest of the response is still returned.
    """
    src = source or get_source()
    available: list[TafsirText] = []
    unavailable: list[TafsirUnavailable] = []

    # Resolve which edition of each work to fetch first, then fetch them
    # concurrently: four works fetched one after another would stack four
    # timeouts on a slow upstream, and they do not depend on each other.
    plans: list[tuple[TafsirWork, str, str]] = []
    for key in keys:
        work = TAFSIR_REGISTRY.get(key)
        if work is None:
            continue

        slug = work.slug_for(language)
        used_language = language
        if slug is None:
            if not allow_language_fallback:
                unavailable.append(
                    TafsirUnavailable(
                        key=work.key,
                        name=work.name,
                        author=work.author,
                        reason=(f"Not available in '{language}'. Available in: {', '.join(work.languages)}."),
                    )
                )
                continue
            used_language = work.languages[0]
            # `languages` are exactly the keys of `slugs`, so this always hits.
            slug = cast(str, work.slug_for(used_language))
        plans.append((work, used_language, slug))

    payloads = await asyncio.gather(*(_fetch_tafsir_cached(src, slug, ref) for _, _, slug in plans))

    for (work, used_language, _), payload in zip(plans, payloads, strict=True):
        if payload is None:
            unavailable.append(
                TafsirUnavailable(
                    key=work.key,
                    name=work.name,
                    author=work.author,
                    reason=f"No entry for {ref.key} in this tafsir.",
                )
            )
            continue

        parsed = parse_tafsir_payload(work, used_language, payload)
        if parsed is None:
            unavailable.append(
                TafsirUnavailable(
                    key=work.key,
                    name=work.name,
                    author=work.author,
                    reason=f"This tafsir returned no commentary text for {ref.key}.",
                )
            )
            continue
        available.append(parsed)

    return available, unavailable


async def _fetch_tafsir_cached(src: TafsirSource, slug: str, ref: AyahRef) -> dict[str, Any] | None:
    """One tafsir payload, from the keyed cache when it is already there."""
    cache = _tafsir_cache()
    cache_key = f"{slug}|{ref.key}"
    payload = cache.get(cache_key)
    if payload is not None:
        return payload
    payload = await src.fetch_tafsir(slug, ref.key)
    if payload is not None:
        # Tafsir text is immutable per ayah, so this never goes stale within a
        # TTL window.
        cache.put(cache_key, payload)
    return payload


async def fetch_verse_text(ref: AyahRef, language: str, source: TafsirSource | None = None) -> VerseText:
    """Ayah text plus translation, cached per ayah (both are immutable)."""
    src = source or get_source()
    cache = _tafsir_cache()
    cache_key = f"verse|{ref.key}|{language}"
    cached = cache.get(cache_key)
    if cached is not None:
        return VerseText(**cached)
    verse = await src.fetch_verse(ref.key, language)
    if verse.arabic or verse.translation:
        cache.put(cache_key, verse.model_dump())
    return verse


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class TafsirRequest(BaseModel):
    reference: str = Field(
        ...,
        description="Ayah reference: '103:1', a range '103:1-3', or 'Al-Asr 1-3'",
        json_schema_extra={"examples": ["103:1-3", "2:255", "Al-Fatihah 1"]},
    )
    tafsirs: list[str] | None = Field(
        None,
        description=(f"Tafsir keys to include (see GET /tafsir/sources). Defaults to {list(DEFAULT_TAFSIR_KEYS)}."),
    )
    language: str = Field(
        DEFAULT_TRANSLATION_LANGUAGE,
        description="Preferred language code for tafsir text and translation",
    )
    allow_language_fallback: bool = Field(
        True,
        description=(
            "When a tafsir is not published in the requested language, return "
            "it in its original language (labelled) instead of omitting it"
        ),
    )


class AyahTafsir(BaseModel):
    ayah: str
    surah_name: str
    arabic: str | None = None
    translation: str | None = None
    translation_language: str | None = None
    tafsirs: list[TafsirText]
    unavailable: list[TafsirUnavailable] = []


class TafsirResponse(BaseModel):
    reference: str
    language: str
    ayat: list[AyahTafsir]
    disclaimer: str = DISCLAIMER


class ComparisonSource(BaseModel):
    """Metadata and retrieved text for one work in a comparison."""

    key: str
    name: str
    author: str
    language: str
    era: str
    death_year_ah: int | None = None
    methodologies: list[str] = Field(default_factory=list)
    perspective: str
    text: str
    verse_range: str | None = None


class InterpretationCluster(BaseModel):
    """A reproducible semantic grouping of similar source passages."""

    id: str
    label: str
    source_keys: list[str]
    representative: str
    similarity_threshold: float


class ComparisonConsensus(BaseModel):
    """A consensus claim supported by the listed retrieved works."""

    statement: str = Field(..., min_length=1)
    source_keys: list[str] = Field(..., min_length=2)
    support_ratio: float = Field(..., ge=0.0, le=1.0)


class ComparisonPosition(BaseModel):
    statement: str = Field(..., min_length=1)
    source_keys: list[str] = Field(..., min_length=1)


class ComparisonDivergence(BaseModel):
    """A difference between interpretations, with source attribution."""

    topic: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)
    positions: list[ComparisonPosition] = Field(..., min_length=2)


class ComparisonInsight(BaseModel):
    """A distinctive observation attributed to one or more works."""

    statement: str = Field(..., min_length=1)
    source_keys: list[str] = Field(..., min_length=1)


class ComparisonSynthesis(BaseModel):
    """A source-aware synthesis suitable for display to a learner."""

    summary: str = Field(..., min_length=1)
    evolution: str = Field(..., min_length=1)
    methodological_notes: str = Field(..., min_length=1)
    balance_note: str = Field(..., min_length=1)
    cited_source_keys: list[str] = Field(default_factory=list)


class GeneratedComparisonAnalysis(BaseModel):
    """Schema-validated model analysis before source-key guardrails."""

    consensus: list[ComparisonConsensus] = Field(default_factory=list)
    divergences: list[ComparisonDivergence] = Field(default_factory=list)
    unique_insights: list[ComparisonInsight] = Field(default_factory=list)
    synthesis: ComparisonSynthesis


class TafsirComparisonRequest(BaseModel):
    reference: str = Field(
        ...,
        description="One ayah reference, such as '103:2', '2:255', or 'Al-Fatihah 1'",
        json_schema_extra={"examples": ["103:2", "2:255", "Al-Fatihah 1"]},
    )
    tafsirs: list[str] | None = Field(
        None,
        min_length=MIN_COMPARISON_TAFSIRS,
        max_length=MAX_COMPARISON_TAFSIRS,
        description=(
            f"At least {MIN_COMPARISON_TAFSIRS} and at most {MAX_COMPARISON_TAFSIRS} tafsir keys. "
            "Defaults to every registered work."
        ),
    )
    language: str = Field(DEFAULT_TRANSLATION_LANGUAGE, description="Preferred language for retrieved tafsir text")
    allow_language_fallback: bool = Field(
        True,
        description="Use a work's original language when the preferred language is unavailable",
    )
    similarity_threshold: float = Field(
        DEFAULT_SIMILARITY_THRESHOLD,
        ge=0.0,
        le=1.0,
        description="Token similarity threshold used to group interpretation passages",
    )


class TafsirComparisonResponse(BaseModel):
    reference: str
    language: str
    ayah: AyahTafsir
    sources: list[ComparisonSource]
    unavailable: list[TafsirUnavailable]
    clusters: list[InterpretationCluster]
    consensus: list[ComparisonConsensus]
    divergences: list[ComparisonDivergence]
    unique_insights: list[ComparisonInsight]
    synthesis: ComparisonSynthesis
    disclaimer: str = DISCLAIMER


class TafsirSourceInfo(BaseModel):
    key: str
    name: str
    author: str
    languages: list[str]
    era: str
    death_year_ah: int | None = None
    methodologies: list[str] = Field(default_factory=list)
    perspective: str


def resolve_requested_tafsirs(requested: list[str] | None) -> list[str]:
    """Normalize requested tafsir keys, or fall back to the default set.

    Raises ``InvalidReference`` when nothing requested is recognized — silently
    substituting a different tafsir would be a misattribution waiting to happen.
    """
    if not requested:
        return list(DEFAULT_TAFSIR_KEYS)

    resolved: list[str] = []
    unknown: list[str] = []
    for raw in requested[:MAX_TAFSIRS_PER_REQUEST]:
        key = normalize_tafsir_key(raw)
        if key is None:
            unknown.append(raw)
        elif key not in resolved:
            resolved.append(key)

    if not resolved:
        raise InvalidReference(
            f"Unknown tafsir(s): {', '.join(unknown)}. Available: {', '.join(sorted(TAFSIR_REGISTRY))}."
        )
    if unknown:
        logger.info("Ignoring unknown tafsir(s): %s", ", ".join(unknown))
    return resolved


def resolve_comparison_tafsirs(requested: list[str] | None) -> list[str]:
    """Resolve a comparison's source list without the ordinary six-work cap."""
    if not requested:
        return list(DEFAULT_COMPARISON_TAFSIR_KEYS)

    resolved: list[str] = []
    unknown: list[str] = []
    for raw in requested:
        key = normalize_tafsir_key(raw)
        if key is None:
            unknown.append(raw)
        elif key not in resolved:
            resolved.append(key)

    if unknown:
        logger.info("Ignoring unknown comparison tafsir(s): %s", ", ".join(unknown))
    if not resolved:
        raise InvalidReference(
            f"Unknown tafsir(s): {', '.join(unknown)}. Available: {', '.join(sorted(TAFSIR_REGISTRY))}."
        )
    if len(resolved) < MIN_COMPARISON_TAFSIRS:
        raise InvalidReference(
            f"A comparison needs at least {MIN_COMPARISON_TAFSIRS} distinct tafsir works; "
            f"{len(resolved)} were recognized."
        )
    if len(resolved) > MAX_COMPARISON_TAFSIRS:
        raise InvalidReference(f"A comparison can include at most {MAX_COMPARISON_TAFSIRS} tafsir works.")
    return resolved


async def assemble_ayah(
    ref: AyahRef,
    keys: list[str],
    language: str,
    allow_language_fallback: bool = True,
    source: TafsirSource | None = None,
) -> AyahTafsir:
    """Verse text plus every requested tafsir for one ayah.

    Shared by ``/tafsir`` and the chat path so both assemble an ayah — and
    degrade on a missing work — identically. The verse text and the tafsir
    lookups are independent, so they run concurrently.
    """
    verse, (available, unavailable) = await asyncio.gather(
        fetch_verse_text(ref, language, source),
        fetch_tafsirs_for_ayah(
            ref,
            keys,
            language,
            allow_language_fallback=allow_language_fallback,
            source=source,
        ),
    )
    surah = surah_by_number(ref.surah)
    return AyahTafsir(
        ayah=ref.key,
        surah_name=surah.name if surah else str(ref.surah),
        arabic=verse.arabic,
        translation=verse.translation,
        translation_language=verse.translation_language,
        tafsirs=available,
        unavailable=unavailable,
    )


async def build_tafsir_response(request: TafsirRequest, source: TafsirSource | None = None) -> TafsirResponse:
    """Assemble the /tafsir response. Raises ``InvalidReference`` on bad input."""
    refs = parse_reference(request.reference)
    keys = resolve_requested_tafsirs(request.tafsirs)
    language = (request.language or DEFAULT_TRANSLATION_LANGUAGE).strip().casefold()

    ayat = await asyncio.gather(
        *(
            assemble_ayah(
                ref,
                keys,
                language,
                allow_language_fallback=request.allow_language_fallback,
                source=source,
            )
            for ref in refs
        )
    )

    return TafsirResponse(reference=request.reference, language=language, ayat=list(ayat))


# ---------------------------------------------------------------------------
# Comparative analysis
# ---------------------------------------------------------------------------


_WORD_PATTERN = re.compile(r"[^\W_]+", re.UNICODE)
_COMPARISON_STOPWORDS = frozenset(
    {
        "a",
        "about",
        "after",
        "all",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "he",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "their",
        "this",
        "to",
        "was",
        "which",
        "with",
    }
)


def _comparison_tokens(text: str) -> collections.Counter[str]:
    """Return normalized content-word counts for deterministic similarity."""
    tokens = (
        token
        for token in _WORD_PATTERN.findall(text.casefold())
        if len(token) > 2 and token not in _COMPARISON_STOPWORDS
    )
    return collections.Counter(tokens)


def _semantic_similarity(left: str, right: str) -> float:
    """Approximate semantic similarity using weighted content-word cosine."""
    left_counts = _comparison_tokens(left)
    right_counts = _comparison_tokens(right)
    if not left_counts or not right_counts:
        return 0.0
    vocabulary = left_counts.keys() | right_counts.keys()
    dot = sum(left_counts[token] * right_counts[token] for token in vocabulary)
    left_norm = sum(value * value for value in left_counts.values()) ** 0.5
    right_norm = sum(value * value for value in right_counts.values()) ** 0.5
    return dot / (left_norm * right_norm)


def _comparison_sentence(text: str) -> str:
    """Keep evidence labels concise while preserving the source's wording."""
    sentences = re.split(r"(?<=[.!?؟])\s+", text.strip())
    selected = [sentence.strip() for sentence in sentences if sentence.strip()]
    if not selected:
        representative = text.strip()
    else:
        representative = " ".join(selected[:COMPARISON_REPRESENTATIVE_SENTENCES])
    if len(representative) > COMPARISON_REPRESENTATIVE_CHARS:
        return representative[:COMPARISON_REPRESENTATIVE_CHARS].rstrip() + "..."
    return representative


def _cluster_comparison_sources(
    sources: list[ComparisonSource],
    threshold: float,
) -> list[InterpretationCluster]:
    """Group passages by similarity to a stable cluster representative."""
    clusters: list[InterpretationCluster] = []
    for source in sources:
        best_cluster: InterpretationCluster | None = None
        best_score = 0.0
        for cluster in clusters:
            score = _semantic_similarity(source.text, cluster.representative)
            if score >= threshold and score > best_score:
                best_cluster = cluster
                best_score = score
        if best_cluster is None:
            clusters.append(
                InterpretationCluster(
                    id=f"cluster-{len(clusters) + 1}",
                    label=f"Interpretive grouping {len(clusters) + 1}",
                    source_keys=[source.key],
                    representative=_comparison_sentence(source.text),
                    similarity_threshold=threshold,
                )
            )
        else:
            best_cluster.source_keys.append(source.key)
    return clusters


def _build_comparison_consensus(
    clusters: list[InterpretationCluster],
    sources_by_key: dict[str, ComparisonSource],
) -> list[ComparisonConsensus]:
    """Treat a high-support cluster as consensus without overstating certainty."""
    total = len(sources_by_key)
    if not total:
        return []
    minimum_support = max(2, int(total * 0.75 + 0.999))
    consensus: list[ComparisonConsensus] = []
    for cluster in clusters:
        if len(cluster.source_keys) < minimum_support:
            continue
        consensus.append(
            ComparisonConsensus(
                statement=cluster.representative,
                source_keys=list(cluster.source_keys),
                support_ratio=round(len(cluster.source_keys) / total, 4),
            )
        )
    return consensus


def _build_comparison_divergences(
    clusters: list[InterpretationCluster],
) -> list[ComparisonDivergence]:
    """Expose competing clusters as distinct attributed positions."""
    if len(clusters) < 2:
        return []
    ordered = sorted(clusters, key=lambda cluster: len(cluster.source_keys), reverse=True)
    return [
        ComparisonDivergence(
            topic="Interpretive emphasis",
            description="The retrieved works cluster around distinct readings of the verse.",
            positions=[
                ComparisonPosition(statement=cluster.representative, source_keys=list(cluster.source_keys))
                for cluster in ordered
            ],
        )
    ]


def _build_comparison_insights(
    clusters: list[InterpretationCluster],
) -> list[ComparisonInsight]:
    """Mark singleton clusters as distinctive observations, not consensus."""
    return [
        ComparisonInsight(statement=cluster.representative, source_keys=list(cluster.source_keys))
        for cluster in clusters
        if len(cluster.source_keys) == 1
    ]


def _comparison_evolution(sources: list[ComparisonSource]) -> str:
    """Summarize the chronological distribution without asserting causation."""
    by_era: dict[str, list[str]] = {}
    for source in sorted(sources, key=lambda item: item.death_year_ah or 9999):
        by_era.setdefault(source.era, []).append(source.name)
    parts = [f"{era}: {len(names)} work(s)" for era, names in by_era.items()]
    return "Chronological coverage represented in the retrieved set: " + "; ".join(parts) + "."


def _comparison_methodologies(sources: list[ComparisonSource]) -> str:
    counts: dict[str, int] = {}
    for source in sources:
        for methodology in source.methodologies:
            counts[methodology] = counts.get(methodology, 0) + 1
    labels = ", ".join(f"{method} ({count})" for method, count in sorted(counts.items()))
    return f"Methodologies represented: {labels or 'not classified'}."


def _comparison_balance_note(sources: list[ComparisonSource], unavailable: list[TafsirUnavailable]) -> str:
    perspectives = sorted({source.perspective for source in sources})
    note = f"Retrieved perspectives: {', '.join(perspectives)}."
    if unavailable:
        note += f" {len(unavailable)} requested work(s) were unavailable and were not treated as evidence."
    return note


def _fallback_comparison_synthesis(
    sources: list[ComparisonSource],
    unavailable: list[TafsirUnavailable],
    consensus: list[ComparisonConsensus],
    divergences: list[ComparisonDivergence],
    insights: list[ComparisonInsight],
) -> ComparisonSynthesis:
    """Build a bounded synthesis when no model is configured or available."""
    consensus_note = (
        f"{len(consensus)} high-support interpretation grouping(s) were identified."
        if consensus
        else "No high-support grouping met the comparison threshold."
    )
    divergence_note = (
        f"{len(divergences)} divergence grouping(s) remain visible as separate positions."
        if divergences
        else "The retrieved passages did not form multiple distinct groupings."
    )
    unique_note = (
        f"{len(insights)} distinctive source observation(s) were retained separately."
        if insights
        else "No singleton interpretation grouping was identified."
    )
    return ComparisonSynthesis(
        summary=f"Compared {len(sources)} retrieved tafsir works. {consensus_note} {divergence_note} {unique_note}",
        evolution=_comparison_evolution(sources),
        methodological_notes=_comparison_methodologies(sources),
        balance_note=_comparison_balance_note(sources, unavailable),
        cited_source_keys=[source.key for source in sources],
    )


def _comparison_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Return a Gemini-compatible schema for a simple Pydantic response."""
    schema = model.model_json_schema()
    definitions = schema.pop("$defs", {})

    def normalize(value: Any) -> Any:
        if isinstance(value, dict):
            if "$ref" in value:
                key = value["$ref"].split("/")[-1]
                return normalize(copy.deepcopy(definitions[key]))
            return {key: normalize(item) for key, item in value.items() if key != "title"}
        if isinstance(value, list):
            return [normalize(item) for item in value]
        return value

    return normalize(schema)


class ComparisonGenerator:
    """Synthesis seam; tests can inject a recorded generator."""

    def generate(self, prompt: str, schema: dict[str, Any]) -> str:
        raise NotImplementedError


class GeminiComparisonGenerator(ComparisonGenerator):
    """Generate only the prose synthesis; evidence remains deterministic."""

    def __init__(self, model_name: str = "gemini-2.5-flash") -> None:
        self.model_name = model_name

    def generate(self, prompt: str, schema: dict[str, Any]) -> str:
        import google.generativeai as genai

        model = genai.GenerativeModel(self.model_name)
        response = model.generate_content(
            prompt,
            generation_config={
                "temperature": 0.2,
                "response_mime_type": "application/json",
                "response_schema": schema,
            },
        )
        return response.text


class FakeComparisonGenerator(ComparisonGenerator):
    """Offline generator for tests and local demos."""

    def __init__(self, response: str) -> None:
        self.response = response
        self.call_count = 0

    def generate(self, prompt: str, schema: dict[str, Any]) -> str:
        self.call_count += 1
        return self.response


def _comparison_prompt(
    ayah: AyahTafsir,
    sources: list[ComparisonSource],
    consensus: list[ComparisonConsensus],
    divergences: list[ComparisonDivergence],
    insights: list[ComparisonInsight],
) -> str:
    evidence = "\n\n".join(
        f"[{source.key}] {source.name} — {source.author}; era={source.era}; "
        f"methodologies={','.join(source.methodologies)}; perspective={source.perspective}\n{source.text[:1600]}"
        for source in sources
    )
    return f"""You are a careful comparative tafsir editor.
Analyze only the retrieved evidence below. Do not introduce an interpretation
from memory, claim that a view is universal, or erase a disagreement. Use the
deterministic similarity results as supporting evidence, while also comparing
cross-language meanings that token matching may miss. Every consensus claim,
position, and unique insight must cite only source keys present in the evidence.
Consensus requires support from at least two retrieved works. A divergence must
contain at least two distinct attributed positions. Do not infer a sectarian
perspective beyond the supplied metadata.
Return only JSON matching the supplied schema.

Ayah: {ayah.ayah} ({ayah.surah_name})
Translation: {ayah.translation or "unavailable"}

Retrieved evidence:
{evidence}

Consensus evidence:
{json.dumps([item.model_dump() for item in consensus], ensure_ascii=False)}

Divergence evidence:
{json.dumps([item.model_dump() for item in divergences], ensure_ascii=False)}

Unique observations:
{json.dumps([item.model_dump() for item in insights], ensure_ascii=False)}
"""


def _validate_generated_analysis(
    analysis: GeneratedComparisonAnalysis,
    source_keys: set[str],
) -> GeneratedComparisonAnalysis:
    """Reject generated claims whose attribution is absent from retrieval."""

    def validate_keys(keys: list[str], minimum: int) -> list[str]:
        unique = list(dict.fromkeys(keys))
        if len(unique) < minimum or not set(unique) <= source_keys:
            raise ValueError("analysis cited an unknown or insufficient source set")
        return unique

    consensus = [
        item.model_copy(
            update={
                "source_keys": validate_keys(item.source_keys, 2),
                "support_ratio": round(len(set(item.source_keys)) / len(source_keys), 4),
            }
        )
        for item in analysis.consensus
    ]
    divergences: list[ComparisonDivergence] = []
    for item in analysis.divergences:
        positions = [
            position.model_copy(update={"source_keys": validate_keys(position.source_keys, 1)})
            for position in item.positions
        ]
        if len({tuple(position.source_keys) for position in positions}) < 2:
            raise ValueError("divergence positions must cite distinct source sets")
        divergences.append(item.model_copy(update={"positions": positions}))
    unique_insights = [
        item.model_copy(update={"source_keys": validate_keys(item.source_keys, 1)}) for item in analysis.unique_insights
    ]
    cited = validate_keys(analysis.synthesis.cited_source_keys, 1)
    synthesis = analysis.synthesis.model_copy(update={"cited_source_keys": cited})
    return GeneratedComparisonAnalysis(
        consensus=consensus,
        divergences=divergences,
        unique_insights=unique_insights,
        synthesis=synthesis,
    )


async def _generate_comparison_analysis(
    generator: ComparisonGenerator | None,
    prompt: str,
    fallback: GeneratedComparisonAnalysis,
    source_keys: set[str],
) -> GeneratedComparisonAnalysis:
    """Use a model when available, but keep comparison evidence independently validated."""
    if generator is None or not source_keys:
        return fallback
    try:
        raw = await asyncio.to_thread(generator.generate, prompt, _comparison_schema(GeneratedComparisonAnalysis))
        parsed = GeneratedComparisonAnalysis.model_validate_json(raw)
        return _validate_generated_analysis(parsed, source_keys)
    except Exception as exc:  # noqa: BLE001 - deterministic synthesis remains available
        logger.warning("Tafsir comparison analysis failed; using deterministic evidence: %s", exc)
        return fallback


def _get_comparison_generator() -> ComparisonGenerator | None:
    """Return the configured synthesis model, or None for offline operation."""
    if not os.getenv("GEMINI_API_KEY"):
        return None
    return GeminiComparisonGenerator(os.getenv("MODEL_NAME", "gemini-2.5-flash"))


async def build_tafsir_comparison(
    request: TafsirComparisonRequest,
    source: TafsirSource | None = None,
    generator: ComparisonGenerator | None = None,
) -> TafsirComparisonResponse:
    """Retrieve, cluster, attribute, and synthesize interpretations for one ayah."""
    refs = parse_reference(request.reference)
    if len(refs) != 1:
        raise InvalidReference("Tafsir comparison accepts exactly one ayah reference, not a range.")
    keys = resolve_comparison_tafsirs(request.tafsirs)
    language = (request.language or DEFAULT_TRANSLATION_LANGUAGE).strip().casefold()
    ayah = await assemble_ayah(
        refs[0],
        keys,
        language,
        allow_language_fallback=request.allow_language_fallback,
        source=source,
    )
    sources = sorted(
        [
            ComparisonSource(
                key=tafsir.key,
                name=tafsir.name,
                author=tafsir.author,
                language=tafsir.language,
                era=TAFSIR_REGISTRY[tafsir.key].era,
                death_year_ah=TAFSIR_REGISTRY[tafsir.key].death_year_ah,
                methodologies=list(TAFSIR_REGISTRY[tafsir.key].methodologies),
                perspective=TAFSIR_REGISTRY[tafsir.key].perspective,
                text=tafsir.text,
                verse_range=tafsir.verse_range,
            )
            for tafsir in ayah.tafsirs
        ],
        key=lambda item: (item.death_year_ah is None, item.death_year_ah or 0, item.key),
    )
    sources_by_key = {item.key: item for item in sources}
    clusters = _cluster_comparison_sources(sources, request.similarity_threshold)
    consensus = _build_comparison_consensus(clusters, sources_by_key)
    divergences = _build_comparison_divergences(clusters)
    insights = _build_comparison_insights(clusters)
    fallback = GeneratedComparisonAnalysis(
        consensus=consensus,
        divergences=divergences,
        unique_insights=insights,
        synthesis=_fallback_comparison_synthesis(sources, ayah.unavailable, consensus, divergences, insights),
    )
    analysis = await _generate_comparison_analysis(
        generator,
        _comparison_prompt(ayah, sources, consensus, divergences, insights),
        fallback,
        set(sources_by_key),
    )
    return TafsirComparisonResponse(
        reference=request.reference,
        language=language,
        ayah=ayah,
        sources=sources,
        unavailable=ayah.unavailable,
        clusters=clusters,
        consensus=analysis.consensus,
        divergences=analysis.divergences,
        unique_insights=analysis.unique_insights,
        synthesis=analysis.synthesis,
    )


# ---------------------------------------------------------------------------
# Chat integration
# ---------------------------------------------------------------------------

TAFSIR_SYNTHESIS_CONTEXT = """

TAFSIR GROUNDING (verse-explanation question):
Retrieved tafsir passages are provided below. They are the only permitted basis
for explaining these ayat. Follow these rules exactly:

1. Attribute every explanatory claim to the named work — "Ibn Kathir explains…",
   "al-Sa'di adds…". Never write "Islam says" or "scholars say" for something
   that came from one named tafsir.
2. Do not add interpretation from your own memory. If the passages do not cover
   part of the question, say so plainly instead of filling the gap.
3. Where the mufassirun differ — on asbab al-nuzul, a legal implication, or a
   linguistic reading — present both readings with attribution. Never merge
   differing views into a single unattributed reading, and never rank one
   mufassir above another.
4. Paraphrase for clarity, but never present your paraphrase as a tafsir's exact
   words. Quote directly only when reproducing the passage faithfully.
5. Note when a work was unavailable for an ayah rather than implying it was
   consulted.
6. Each passage is labelled with the language it is in. When a passage is not in
   the user's language, render it faithfully into their language and say which
   work it is a rendering of — never treat a rendering as the work's exact words.
"""

NO_TAFSIR_NOTE = (
    "\n\nNote: no tafsir text could be retrieved for this ayah right now. "
    "Explain only what the translation itself states, say plainly that named "
    "tafsir was unavailable, and point the user to consult a tafsir or a "
    "qualified scholar."
)


class TafsirContext(BaseModel):
    """Retrieved tafsir for a chat turn, plus the prompt block built from it."""

    references: list[str]
    prompt_block: str
    ayat: list[AyahTafsir]

    @property
    def has_tafsir(self) -> bool:
        return any(ayah.tafsirs for ayah in self.ayat)


def build_tafsir_prompt_block(ayat: list[AyahTafsir], excerpt_chars: int = CHAT_EXCERPT_CHARS) -> str:
    """Render retrieved tafsir as an attributed block for the model prompt."""
    sections: list[str] = []
    for ayah in ayat:
        lines = [f"--- Ayah {ayah.ayah} (Surah {ayah.surah_name}) ---"]
        if ayah.arabic:
            lines.append(f"Arabic: {ayah.arabic}")
        if ayah.translation:
            lines.append(f"Translation: {ayah.translation}")
        for tafsir in ayah.tafsirs:
            excerpt = tafsir.text
            if len(excerpt) > excerpt_chars:
                excerpt = excerpt[:excerpt_chars].rstrip() + " […excerpt truncated]"
            covers = f" (passage covers {tafsir.verse_range})" if tafsir.verse_range else ""
            lines.append(f"\n[{tafsir.name} — {tafsir.author}, in {tafsir.language}]{covers}\n{excerpt}")
        for missing in ayah.unavailable:
            lines.append(f"\n[UNAVAILABLE — {missing.name}]: {missing.reason}")
        sections.append("\n".join(lines))
    return "\n\n".join(sections)


async def build_chat_tafsir_context(
    prompt: str,
    language: str = DEFAULT_TRANSLATION_LANGUAGE,
    source: TafsirSource | None = None,
    timeout: float | None = CHAT_RETRIEVAL_TIMEOUT,
) -> TafsirContext | None:
    """Retrieve tafsir for a chat prompt, or None if it isn't a tafsir question.

    Returns None if retrieval exceeds *timeout*, so a slow upstream costs the
    turn its grounding but never its response. Pass ``timeout=None`` to wait
    indefinitely (offline tests do, since their source is instant).
    """
    if timeout is None:
        return await _build_chat_tafsir_context(prompt, language, source)
    try:
        return await asyncio.wait_for(_build_chat_tafsir_context(prompt, language, source), timeout)
    except (TimeoutError, asyncio.TimeoutError):  # noqa: UP041 - Python 3.10 compatibility
        logger.warning("Tafsir retrieval exceeded %ss; answering without it", timeout)
        return None


async def _build_chat_tafsir_context(
    prompt: str,
    language: str = DEFAULT_TRANSLATION_LANGUAGE,
    source: TafsirSource | None = None,
) -> TafsirContext | None:
    refs = detect_ayah_references(prompt)
    if not refs:
        return None

    keys = list(DEFAULT_TAFSIR_KEYS)
    ayat = list(
        await asyncio.gather(
            *(assemble_ayah(ref, keys, language, allow_language_fallback=True, source=source) for ref in refs)
        )
    )

    context = TafsirContext(
        references=[ayah.ayah for ayah in ayat],
        prompt_block=build_tafsir_prompt_block(ayat),
        ayat=ayat,
    )
    return context


class TafsirInfo(BaseModel):
    """Which tafsir text actually backed a verse-explanation chat answer."""

    references: list[str]
    works_cited: list[str]
    unavailable: list[str] = []
    grounded: bool


def summarize_tafsir_context(context: TafsirContext) -> TafsirInfo:
    """Report the works whose text was retrieved, not the ones that were asked for."""
    works_cited: list[str] = []
    unavailable: list[str] = []
    for ayah in context.ayat:
        for tafsir in ayah.tafsirs:
            label = f"{tafsir.name} — {tafsir.author}"
            if label not in works_cited:
                works_cited.append(label)
        for missing in ayah.unavailable:
            label = f"{missing.name} ({missing.reason})"
            if label not in unavailable:
                unavailable.append(label)
    return TafsirInfo(
        references=context.references,
        works_cited=works_cited,
        unavailable=unavailable,
        grounded=context.has_tafsir,
    )


def tafsir_system_context(context: TafsirContext) -> str:
    """System-prompt addition for a chat turn that has retrieved tafsir."""
    block = TAFSIR_SYNTHESIS_CONTEXT
    if not context.has_tafsir:
        block += NO_TAFSIR_NOTE
    return f"{block}\nRETRIEVED TAFSIR PASSAGES:\n{context.prompt_block}\n"


# Type alias for the chat handler's retrieval hook, so main.py can inject a
# stub in tests without importing httpx machinery.
TafsirRetriever = Callable[[str], Awaitable[TafsirContext | None]]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/tafsir/sources", response_model=list[TafsirSourceInfo])
async def list_tafsir_sources() -> list[TafsirSourceInfo]:
    """Tafsir works this service can retrieve, and their languages."""
    return [
        TafsirSourceInfo(
            key=work.key,
            name=work.name,
            author=work.author,
            languages=work.languages,
            era=work.era,
            death_year_ah=work.death_year_ah,
            methodologies=list(work.methodologies),
            perspective=work.perspective,
        )
        for work in TAFSIR_REGISTRY.values()
    ]


@router.post("/tafsir", response_model=TafsirResponse)
async def get_tafsir(request: TafsirRequest) -> TafsirResponse:
    """Explain an ayah (or a short range) from named classical tafsir works."""
    try:
        response = await build_tafsir_response(request)
    except InvalidReference as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    logger.info(
        "Tafsir lookup %s (%s) -> %d ayat",
        request.reference,
        request.language,
        len(response.ayat),
    )
    return response


@router.post("/tafsir/compare", response_model=TafsirComparisonResponse)
async def compare_tafsir(request: TafsirComparisonRequest) -> TafsirComparisonResponse:
    """Compare at least ten attributed tafsir works for one ayah."""
    try:
        response = await build_tafsir_comparison(request, generator=_get_comparison_generator())
    except InvalidReference as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    logger.info(
        "Tafsir comparison %s (%s) -> %d available works, %d unavailable",
        request.reference,
        request.language,
        len(response.sources),
        len(response.unavailable),
    )
    return response
