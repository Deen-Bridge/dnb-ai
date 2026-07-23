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

import html
import json
import logging
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

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
TRANSLATION_IDS: Dict[str, int] = {
    "en": 20,   # Saheeh International
    "ur": 97,   # Maulana Fateh Muhammad Jalandhari
    "bn": 163,  # Taisirul Quran
    "ru": 79,   # Elmir Kuliev
}
DEFAULT_TRANSLATION_LANGUAGE = "en"

# Language codes used in TAFSIR_REGISTRY, spelled out for display.
LANGUAGE_NAMES: Dict[str, str] = {
    "en": "english",
    "ar": "arabic",
    "ur": "urdu",
    "bn": "bengali",
    "ru": "russian",
}

MAX_AYAT_PER_REQUEST = int(os.getenv("TAFSIR_MAX_AYAT", "10"))
MAX_TAFSIRS_PER_REQUEST = 6

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
    slugs: Dict[str, str] = Field(
        ..., description="Language code -> Quran.com tafsir slug"
    )

    def slug_for(self, language: str) -> Optional[str]:
        return self.slugs.get(language)

    @property
    def languages(self) -> List[str]:
        return sorted(self.slugs)


TAFSIR_REGISTRY: Dict[str, TafsirWork] = {
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
        ),
        TafsirWork(
            key="tabari",
            name="Jami' al-Bayan (Tafsir al-Tabari)",
            author="Ibn Jarir al-Tabari (d. 310 AH)",
            slugs={"ar": "ar-tafsir-al-tabari"},
        ),
        TafsirWork(
            key="qurtubi",
            name="Al-Jami' li-Ahkam al-Qur'an (Tafsir al-Qurtubi)",
            author="Al-Qurtubi (d. 671 AH)",
            slugs={"ar": "ar-tafseer-al-qurtubi"},
        ),
        TafsirWork(
            key="saadi",
            name="Taysir al-Karim al-Rahman (Tafsir al-Sa'di)",
            author="Abd al-Rahman al-Sa'di (d. 1376 AH)",
            slugs={"ar": "ar-tafseer-al-saddi", "ru": "ru-tafseer-al-saddi"},
        ),
        TafsirWork(
            key="baghawi",
            name="Ma'alim al-Tanzil (Tafsir al-Baghawi)",
            author="Al-Baghawi (d. 516 AH)",
            slugs={"ar": "ar-tafsir-al-baghawi"},
        ),
        TafsirWork(
            key="muyassar",
            name="Al-Tafsir al-Muyassar",
            author="King Fahd Complex scholarly committee",
            slugs={"ar": "ar-tafsir-muyassar"},
        ),
        TafsirWork(
            key="wasit",
            name="Al-Tafsir al-Wasit",
            author="Muhammad Sayyid Tantawi (d. 1431 AH)",
            slugs={"ar": "ar-tafsir-al-wasit"},
        ),
        TafsirWork(
            key="maarif-ul-quran",
            name="Ma'arif al-Qur'an",
            author="Mufti Muhammad Shafi (d. 1396 AH)",
            slugs={"en": "en-tafsir-maarif-ul-quran"},
        ),
        TafsirWork(
            key="bayan-ul-quran",
            name="Bayan ul Quran",
            author="Dr. Israr Ahmad (d. 1431 AH)",
            slugs={"ur": "tafsir-bayan-ul-quran"},
        ),
        TafsirWork(
            key="fi-zilal",
            name="Fi Zilal al-Qur'an",
            author="Sayyid Qutb (d. 1386 AH)",
            slugs={"ur": "tafsir-fe-zalul-quran-syed-qatab"},
        ),
        TafsirWork(
            key="tazkirul-quran",
            name="Tazkirul Quran",
            author="Maulana Wahiduddin Khan (d. 1443 AH)",
            slugs={"en": "tazkirul-quran-en", "ur": "tazkiru-quran-ur"},
        ),
        TafsirWork(
            key="ahsanul-bayaan",
            name="Tafsir Ahsanul Bayaan",
            author="Bayaan Foundation",
            slugs={"bn": "bn-tafsir-ahsanul-bayaan"},
        ),
    ]
}

# Four classical works spanning narration-based (Ibn Kathir, al-Tabari), legal
# (al-Qurtubi) and concise-summary (al-Sa'di) approaches — chosen so a default
# request already shows more than one methodology.
DEFAULT_TAFSIR_KEYS: Tuple[str, ...] = ("ibn-kathir", "tabari", "qurtubi", "saadi")

TAFSIR_ALIASES: Dict[str, str] = {
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


def normalize_tafsir_key(raw: str) -> Optional[str]:
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
    aliases: List[str] = []


@lru_cache(maxsize=1)
def load_surah_index() -> Tuple[Surah, ...]:
    with DATA_PATH.open(encoding="utf-8") as f:
        return tuple(Surah(**row) for row in json.load(f))


@lru_cache(maxsize=1)
def _name_lookup() -> Dict[str, int]:
    """Normalized surah name/alias -> surah number."""
    lookup: Dict[str, int] = {}
    for surah in load_surah_index():
        for label in [surah.name, surah.arabic_name, *surah.aliases]:
            lookup.setdefault(_normalize_surah_name(label), surah.number)
    return lookup


def _normalize_surah_name(name: str) -> str:
    """Fold the many spellings of a surah name onto one form.

    "Al-'Asr", "al asr", "Surat ul-Asr" and "AlAsr" all normalize to "asr":
    punctuation is dropped, the article is stripped, and case is folded.
    """
    lowered = (name or "").casefold()
    lowered = re.sub(r"^surah?t?\s+", "", lowered)
    lowered = re.sub(r"[^\w؀-ۿ\s]", "", lowered)
    lowered = re.sub(r"^(al|al-|as|ash|ad|an|ar|at|az)\s+", "", lowered)
    collapsed = re.sub(r"\s+", "", lowered)
    for article in ("al", "ul"):
        if collapsed.startswith(article) and len(collapsed) > len(article) + 2:
            candidate = collapsed[len(article):]
            if candidate:
                collapsed = candidate
                break
    return collapsed


def surah_by_number(number: int) -> Optional[Surah]:
    index = load_surah_index()
    if 1 <= number <= len(index):
        return index[number - 1]
    return None


def surah_by_name(name: str) -> Optional[Surah]:
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
        raise InvalidReference(
            f"Surah {surah} does not exist. Surah numbers run from 1 to 114."
        )
    if ayah < 1 or ayah > record.ayah_count:
        raise InvalidReference(
            f"Ayah {ayah} does not exist in surah {surah} ({record.name}), "
            f"which has {record.ayah_count} ayat."
        )
    return AyahRef(surah=surah, ayah=ayah)


def parse_reference(reference: str) -> List[AyahRef]:
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
            f"Could not parse '{reference}'. Use a surah:ayah reference such as "
            "'103:1' or a range such as '103:1-3'."
        )

    surah = int(parts["surah"])
    start = int(parts["start"])
    end = int(parts["end"]) if parts.get("end") else start

    if end < start:
        raise InvalidReference(
            f"Invalid range {start}-{end}: the last ayah comes before the first."
        )

    first = validate_reference(surah, start)
    last = validate_reference(surah, end)
    span = last.ayah - first.ayah + 1
    if span > MAX_AYAT_PER_REQUEST:
        raise InvalidReference(
            f"Range covers {span} ayat; at most {MAX_AYAT_PER_REQUEST} may be "
            "requested at once."
        )
    return [AyahRef(surah=surah, ayah=n) for n in range(first.ayah, last.ayah + 1)]


def _match_named_reference(raw: str) -> Optional[Dict[str, Optional[str]]]:
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
    "tafsir", "tafseer", "explain", "explanation", "meaning", "what does",
    "what do", "interpret", "commentary", "mufassir", "asbab", "context of",
    "significance of", "why was", "revealed",
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


def detect_ayah_references(prompt: str) -> List[AyahRef]:
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

    refs: List[AyahRef] = []
    seen: set[str] = set()

    def add(surah: int, start: int, end: Optional[int]) -> None:
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
    verse_range: Optional[str] = Field(
        None,
        description=(
            "Ayah range this passage covers when the tafsir treats several "
            "ayat together, e.g. '103:1-3'"
        ),
    )


class TafsirUnavailable(BaseModel):
    key: str
    name: str
    author: str
    reason: str


class VerseText(BaseModel):
    arabic: Optional[str] = None
    translation: Optional[str] = None
    translation_language: Optional[str] = None


class TafsirSource:
    """Retrieval seam. Tests substitute ``FakeTafsirSource`` for this."""

    async def fetch_tafsir(self, slug: str, verse_key: str) -> Optional[Dict[str, Any]]:
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

    async def _get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
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
            logger.warning(
                "Quran API returned %s for %s", response.status_code, path
            )
            return None
        try:
            return response.json()
        except ValueError:
            logger.warning("Quran API returned non-JSON for %s", path)
            return None

    async def fetch_tafsir(self, slug: str, verse_key: str) -> Optional[Dict[str, Any]]:
        payload = await self._get(f"tafsirs/{slug}/by_ayah/{verse_key}")
        if not payload:
            return None
        return payload.get("tafsir")

    async def fetch_verse(self, verse_key: str, language: str) -> VerseText:
        translation_id = TRANSLATION_IDS.get(language)
        params: Dict[str, Any] = {"fields": "text_uthmani"}
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
        tafsirs: Optional[Dict[Tuple[str, str], Dict[str, Any]]] = None,
        verses: Optional[Dict[str, VerseText]] = None,
    ) -> None:
        self.tafsirs = tafsirs or {}
        self.verses = verses or {}
        self.tafsir_calls: List[Tuple[str, str]] = []
        self.verse_calls: List[Tuple[str, str]] = []

    async def fetch_tafsir(self, slug: str, verse_key: str) -> Optional[Dict[str, Any]]:
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


def parse_tafsir_payload(
    work: TafsirWork, language: str, payload: Dict[str, Any]
) -> Optional[TafsirText]:
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
    keys: List[str],
    language: str,
    allow_language_fallback: bool = True,
    source: Optional[TafsirSource] = None,
) -> Tuple[List[TafsirText], List[TafsirUnavailable]]:
    """Retrieve each requested tafsir for one ayah.

    Returns ``(available, unavailable)``. A work that has no entry for the ayah,
    or is not published in a usable language, lands in ``unavailable`` with a
    reason — the rest of the response is still returned.
    """
    src = source or get_source()
    available: List[TafsirText] = []
    unavailable: List[TafsirUnavailable] = []

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
                        reason=(
                            f"Not available in '{language}'. Available in: "
                            f"{', '.join(work.languages)}."
                        ),
                    )
                )
                continue
            used_language = work.languages[0]
            slug = work.slug_for(used_language)

        cache_key = f"{slug}|{ref.key}"
        cache = _tafsir_cache()
        payload = cache.get(cache_key)
        if payload is None:
            payload = await src.fetch_tafsir(slug, ref.key)
            if payload is not None:
                # Tafsir text is immutable per ayah, so this never goes stale
                # within a TTL window.
                cache.put(cache_key, payload)

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


async def fetch_verse_text(
    ref: AyahRef, language: str, source: Optional[TafsirSource] = None
) -> VerseText:
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
    tafsirs: Optional[List[str]] = Field(
        None,
        description=(
            "Tafsir keys to include (see GET /tafsir/sources). "
            f"Defaults to {list(DEFAULT_TAFSIR_KEYS)}."
        ),
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
    arabic: Optional[str] = None
    translation: Optional[str] = None
    translation_language: Optional[str] = None
    tafsirs: List[TafsirText]
    unavailable: List[TafsirUnavailable] = []


class TafsirResponse(BaseModel):
    reference: str
    language: str
    ayat: List[AyahTafsir]
    disclaimer: str = DISCLAIMER


class TafsirSourceInfo(BaseModel):
    key: str
    name: str
    author: str
    languages: List[str]


def resolve_requested_tafsirs(requested: Optional[List[str]]) -> List[str]:
    """Normalize requested tafsir keys, or fall back to the default set.

    Raises ``InvalidReference`` when nothing requested is recognized — silently
    substituting a different tafsir would be a misattribution waiting to happen.
    """
    if not requested:
        return list(DEFAULT_TAFSIR_KEYS)

    resolved: List[str] = []
    unknown: List[str] = []
    for raw in requested[:MAX_TAFSIRS_PER_REQUEST]:
        key = normalize_tafsir_key(raw)
        if key is None:
            unknown.append(raw)
        elif key not in resolved:
            resolved.append(key)

    if not resolved:
        raise InvalidReference(
            f"Unknown tafsir(s): {', '.join(unknown)}. "
            f"Available: {', '.join(sorted(TAFSIR_REGISTRY))}."
        )
    if unknown:
        logger.info("Ignoring unknown tafsir(s): %s", ", ".join(unknown))
    return resolved


async def build_tafsir_response(
    request: TafsirRequest, source: Optional[TafsirSource] = None
) -> TafsirResponse:
    """Assemble the /tafsir response. Raises ``InvalidReference`` on bad input."""
    refs = parse_reference(request.reference)
    keys = resolve_requested_tafsirs(request.tafsirs)
    language = (request.language or DEFAULT_TRANSLATION_LANGUAGE).strip().casefold()

    ayat: List[AyahTafsir] = []
    for ref in refs:
        verse = await fetch_verse_text(ref, language, source)
        available, unavailable = await fetch_tafsirs_for_ayah(
            ref,
            keys,
            language,
            allow_language_fallback=request.allow_language_fallback,
            source=source,
        )
        surah = surah_by_number(ref.surah)
        ayat.append(
            AyahTafsir(
                ayah=ref.key,
                surah_name=surah.name if surah else str(ref.surah),
                arabic=verse.arabic,
                translation=verse.translation,
                translation_language=verse.translation_language,
                tafsirs=available,
                unavailable=unavailable,
            )
        )

    return TafsirResponse(reference=request.reference, language=language, ayat=ayat)


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

    references: List[str]
    prompt_block: str
    ayat: List[AyahTafsir]

    @property
    def has_tafsir(self) -> bool:
        return any(ayah.tafsirs for ayah in self.ayat)


def build_tafsir_prompt_block(ayat: List[AyahTafsir], excerpt_chars: int = CHAT_EXCERPT_CHARS) -> str:
    """Render retrieved tafsir as an attributed block for the model prompt."""
    sections: List[str] = []
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
            lines.append(
                f"\n[{tafsir.name} — {tafsir.author}, in {tafsir.language}]{covers}\n{excerpt}"
            )
        for missing in ayah.unavailable:
            lines.append(f"\n[UNAVAILABLE — {missing.name}]: {missing.reason}")
        sections.append("\n".join(lines))
    return "\n\n".join(sections)


async def build_chat_tafsir_context(
    prompt: str,
    language: str = DEFAULT_TRANSLATION_LANGUAGE,
    source: Optional[TafsirSource] = None,
) -> Optional[TafsirContext]:
    """Retrieve tafsir for a chat prompt, or None if it isn't a tafsir question."""
    refs = detect_ayah_references(prompt)
    if not refs:
        return None

    keys = list(DEFAULT_TAFSIR_KEYS)
    ayat: List[AyahTafsir] = []
    for ref in refs:
        verse = await fetch_verse_text(ref, language, source)
        available, unavailable = await fetch_tafsirs_for_ayah(
            ref, keys, language, allow_language_fallback=True, source=source
        )
        surah = surah_by_number(ref.surah)
        ayat.append(
            AyahTafsir(
                ayah=ref.key,
                surah_name=surah.name if surah else str(ref.surah),
                arabic=verse.arabic,
                translation=verse.translation,
                translation_language=verse.translation_language,
                tafsirs=available,
                unavailable=unavailable,
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

    references: List[str]
    works_cited: List[str]
    unavailable: List[str] = []
    grounded: bool


def summarize_tafsir_context(context: TafsirContext) -> TafsirInfo:
    """Report the works whose text was retrieved, not the ones that were asked for."""
    works_cited: List[str] = []
    unavailable: List[str] = []
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
TafsirRetriever = Callable[[str], Awaitable[Optional[TafsirContext]]]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/tafsir/sources", response_model=List[TafsirSourceInfo])
async def list_tafsir_sources() -> List[TafsirSourceInfo]:
    """Tafsir works this service can retrieve, and their languages."""
    return [
        TafsirSourceInfo(
            key=work.key,
            name=work.name,
            author=work.author,
            languages=work.languages,
        )
        for work in TAFSIR_REGISTRY.values()
    ]


@router.post("/tafsir", response_model=TafsirResponse)
async def get_tafsir(request: TafsirRequest) -> TafsirResponse:
    """Explain an ayah (or a short range) from named classical tafsir works."""
    try:
        response = await build_tafsir_response(request)
    except InvalidReference as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    logger.info(
        "Tafsir lookup %s (%s) -> %d ayat",
        request.reference,
        request.language,
        len(response.ayat),
    )
    return response
