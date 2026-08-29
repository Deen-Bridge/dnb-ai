"""Structured Quran and Hadith citations for chat answers (#15).

Why this exists
---------------
An answer that says "the Qur'an counsels patience" is unusable to a client that
wants to link the ayah, and unverifiable by anyone reading it. This module turns
the references a model already writes in prose into typed, bounds-checked
objects on the response, so a caller receives 2:153 as a surah number and an
ayah number rather than a substring it has to parse for itself.

How the model is asked
----------------------
Whole-response JSON was rejected deliberately: it degrades prose quality, and a
single malformed brace loses the entire answer. Instead the model appends one
delimited block after its normal prose, and that block is parsed off and never
shown to the user.

Parsing is total. A malformed, truncated, or absent block yields an empty
citation list and the prose is still returned. Nothing in this module can fail
a chat turn.

Validation, and where the data comes from
-----------------------------------------
Quran references are bounds-checked against ``data/quran/surah_index.json``
through ``tafsir.surah_by_number`` and ``tafsir.surah_by_name`` -- the same
114-surah index the tafsir layer validates against, so the two cannot drift
apart. ``surah_name`` is always taken from that index and never from the model,
which is what makes the field trustworthy.

Hadith collections are normalized with ``hadith.normalize_collection``, reusing
that module's alias table rather than introducing a second one, and gradings
are read from the same bundled grading dataset that powers the authenticity
caution notes. A citation naming an unrecognized collection is rejected rather
than echoed back.

``corpus.py`` is deliberately not used here. Its backing file
``data/quran_uthmani.json`` currently contains only surahs 1, 2 and 114, so
``corpus.get_ayah_count(3)`` returns None and a bounds check against it would
report valid citations as fabricated.

The confidence signal
---------------------
``CitationExtraction.score`` is the share of attempted citations that
validated, which is precisely the ``citation_verification`` signal that
``confidence.build_signals`` already reserves at weight 0.30. It is None when
an answer cites nothing, so an uncited answer is not penalised -- the signal
simply drops out of the weighted average.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field

from hadith import COLLECTION_NAMES, get_default_source, normalize_collection
from tafsir import surah_by_name, surah_by_number

logger = logging.getLogger(__name__)

CITATION_BLOCK_START = "<<<CITATIONS>>>"
CITATION_BLOCK_END = "<<<END_CITATIONS>>>"

# A model that ignores the format and emits a hundred citations should not be
# able to turn one answer into a hundred grading lookups.
MAX_CITATIONS = 24

_BLOCK_PATTERN = re.compile(
    re.escape(CITATION_BLOCK_START) + r"\s*(?P<payload>.*?)\s*" + re.escape(CITATION_BLOCK_END),
    re.DOTALL,
)

# A block cut off by max_output_tokens never reaches its end marker. Strip it
# from the prose anyway: showing the user half a JSON object is worse than
# showing them no citations at all.
_UNTERMINATED_PATTERN = re.compile(re.escape(CITATION_BLOCK_START) + r".*\Z", re.DOTALL)


class QuranCitation(BaseModel):
    """A validated Quran reference. ``surah_name`` comes from the index."""

    type: Literal["quran"] = "quran"
    surah: int = Field(..., ge=1, le=114)
    ayah_start: int = Field(..., ge=1)
    ayah_end: int | None = Field(None, ge=1)
    surah_name: str

    @property
    def reference(self) -> str:
        if self.ayah_end and self.ayah_end != self.ayah_start:
            return f"{self.surah}:{self.ayah_start}-{self.ayah_end}"
        return f"{self.surah}:{self.ayah_start}"


class HadithCitation(BaseModel):
    """A hadith reference with a canonical collection name.

    ``grading`` is filled from the bundled grading dataset whenever the
    reference can be found there, in preference to whatever the model claimed.
    """

    type: Literal["hadith"] = "hadith"
    collection: str
    number: str | None = None
    grading: str | None = None


class ScholarlyReference(BaseModel):
    """A named work that is neither Quran nor hadith.

    ``volume``, ``pages``, ``edition``, and ``publisher`` are optional
    bibliographic fields that the advanced verification layer
    (``citation_verification.py``) checks for format consistency and
    cross-references against known editions.
    """

    type: Literal["scholarly"] = "scholarly"
    work: str
    author: str | None = None
    detail: str | None = None
    volume: str | None = None
    pages: str | None = None
    edition: str | None = None
    publisher: str | None = None


Citation = Annotated[
    QuranCitation | HadithCitation | ScholarlyReference,
    Field(discriminator="type"),
]


class CitationExtraction(BaseModel):
    """Citations recovered from one answer, plus what was discarded.

    ``rejected`` exists so that a rejection is diagnosable from logs rather
    than silent. It is not part of the chat response.
    """

    citations: list[Citation] = []
    attempted: int = 0
    rejected: list[str] = []
    verification: dict[str, Any] = Field(default_factory=dict)

    @property
    def score(self) -> float | None:
        """Share of attempted citations that validated, or None if none tried."""
        if self.attempted <= 0:
            return None
        return round(len(self.citations) / self.attempted, 4)


def _coerce_int(value: Any) -> int | None:
    """Accept 2 and "2" alike; reject everything else without raising."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def _clean_str(value: Any) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _lookup_grading(collection_key: str, number: str | None) -> str | None:
    """Grade this reference from the bundled dataset, or return None."""
    if not number:
        return None
    try:
        numeric = int(number)
    except (TypeError, ValueError):
        return None
    try:
        record = get_default_source().get(collection_key, numeric)
    except Exception as exc:  # noqa: BLE001 - grading is best-effort
        logger.warning("Hadith grading lookup failed for %s %s: %s", collection_key, numeric, exc)
        return None
    if record is None:
        return None
    return record.grade.value.lower()


def _build_quran(raw: dict[str, Any]) -> tuple[QuranCitation | None, str | None]:
    surah_value = raw.get("surah")
    record = None

    number = _coerce_int(surah_value)
    if number is not None:
        record = surah_by_number(number)
    if record is None and isinstance(surah_value, str):
        record = surah_by_name(surah_value)
    if record is None:
        named = _clean_str(raw.get("surah_name"))
        if named:
            record = surah_by_name(named)
    if record is None:
        return None, f"unknown surah {surah_value!r}"

    start = _coerce_int(raw.get("ayah_start"))
    if start is None:
        start = _coerce_int(raw.get("ayah"))
    if start is None:
        return None, f"missing ayah_start for surah {record.number}"
    if start < 1 or start > record.ayah_count:
        return None, f"{record.name} has {record.ayah_count} ayat; ayah {start} does not exist"

    # An unusable range is dropped to the single opening ayah rather than
    # rejecting a citation whose start is perfectly valid.
    end = _coerce_int(raw.get("ayah_end"))
    if end is not None and (end < start or end > record.ayah_count):
        end = None

    citation = QuranCitation(
        surah=record.number,
        ayah_start=start,
        ayah_end=end,
        surah_name=record.name,
    )
    return citation, None


def _build_hadith(raw: dict[str, Any]) -> tuple[HadithCitation | None, str | None]:
    name = _clean_str(raw.get("collection"))
    if not name:
        return None, "hadith citation without a collection"
    key = normalize_collection(name)
    if key is None:
        return None, f"unrecognized hadith collection {name!r}"

    number_raw = raw.get("number")
    number = None
    if number_raw is not None:
        number = str(number_raw).strip() or None

    claimed = _clean_str(raw.get("grading"))
    graded = _lookup_grading(key, number)

    citation = HadithCitation(
        collection=COLLECTION_NAMES.get(key, name),
        number=number,
        grading=graded or (claimed.lower() if claimed else None),
    )
    return citation, None


def _build_scholarly(raw: dict[str, Any]) -> tuple[ScholarlyReference | None, str | None]:
    work = _clean_str(raw.get("work")) or _clean_str(raw.get("title"))
    if not work:
        return None, "scholarly reference without a work"
    citation = ScholarlyReference(
        work=work,
        author=_clean_str(raw.get("author")),
        detail=_clean_str(raw.get("detail")) or _clean_str(raw.get("note")),
        volume=_clean_str(raw.get("volume")),
        pages=_clean_str(raw.get("pages")) or _clean_str(raw.get("page")),
        edition=_clean_str(raw.get("edition")),
        publisher=_clean_str(raw.get("publisher")),
    )
    return citation, None


_BUILDERS = {
    "quran": _build_quran,
    "hadith": _build_hadith,
    "scholarly": _build_scholarly,
}


def _infer_type(entry: dict[str, Any]) -> str:
    """Guess the citation type when the model omitted it."""
    if "surah" in entry or "ayah_start" in entry or "ayah" in entry:
        return "quran"
    if "collection" in entry:
        return "hadith"
    if "work" in entry or "title" in entry:
        return "scholarly"
    return ""


def parse_citations(payload: str | None) -> CitationExtraction:
    """Parse the JSON payload of a citation block. Never raises."""
    extraction = CitationExtraction()
    if not payload or not payload.strip():
        return extraction

    try:
        data = json.loads(payload)
    except (TypeError, ValueError) as exc:
        logger.info("Citation block was not valid JSON; returning no citations: %s", exc)
        return extraction

    if isinstance(data, dict):
        items = data.get("citations")
    elif isinstance(data, list):
        items = data
    else:
        items = None
    if not isinstance(items, list):
        return extraction

    seen = set()
    for entry in items[:MAX_CITATIONS]:
        extraction.attempted += 1
        if not isinstance(entry, dict):
            extraction.rejected.append("citation entry was not an object")
            continue

        kind = _clean_str(entry.get("type"))
        kind = kind.lower() if kind else _infer_type(entry)
        builder = _BUILDERS.get(kind)
        if builder is None:
            extraction.rejected.append(f"unknown citation type {kind!r}")
            continue

        try:
            citation, reason = builder(entry)
        except Exception as exc:  # noqa: BLE001 - a bad citation is never fatal
            citation, reason = None, f"citation rejected: {exc}"

        if citation is None:
            extraction.rejected.append(reason or "citation rejected")
            continue

        fingerprint = json.dumps(citation.model_dump(), sort_keys=True)
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        extraction.citations.append(citation)

    # Run advanced verification (format, completeness, cross-reference, drift).
    # Imported lazily to avoid a circular import: citation_verification imports
    # the Citation models from this module at module load time.
    try:
        from citation_verification import verify_citations

        extraction.verification = verify_citations(extraction).model_dump()
    except Exception as exc:  # noqa: BLE001 - verification is best-effort
        logger.warning("Citation verification failed; skipping: %s", exc)

    return extraction


def extract_citations(text: str | None) -> tuple[str, CitationExtraction]:
    """Split *text* into (prose, citations). Never raises.

    Text containing no citation block is returned unchanged, so this is safe to
    run over every answer.
    """
    if not text:
        return "", CitationExtraction()

    match = _BLOCK_PATTERN.search(text)
    if match:
        prose = (text[: match.start()] + text[match.end() :]).strip()
        return prose, parse_citations(match.group("payload"))

    truncated = _UNTERMINATED_PATTERN.search(text)
    if truncated:
        prose = text[: truncated.start()].strip()
        payload = text[truncated.start() + len(CITATION_BLOCK_START) :]
        return prose, parse_citations(payload)

    return text, CitationExtraction()


class CitationStreamFilter:
    """Keep the citation block out of SSE deltas.

    A streamed answer must not flash raw block markup at the user before the
    terminal event arrives. ``feed`` withholds any text that could still turn
    out to be the start of the block; ``finish`` returns whatever prose is left
    along with the parsed citations for the done event.
    """

    def __init__(self) -> None:
        self._pending = ""
        self._tail = ""
        self._in_block = False

    def feed(self, chunk: str) -> str:
        """Consume a streamed chunk and return the text safe to emit now."""
        if not chunk:
            return ""
        if self._in_block:
            self._tail += chunk
            return ""

        self._pending += chunk
        index = self._pending.find(CITATION_BLOCK_START)
        if index != -1:
            emit = self._pending[:index]
            self._tail = self._pending[index:]
            self._pending = ""
            self._in_block = True
            return emit

        # Hold back enough characters that a marker split across two chunks is
        # still recognised once the rest of it arrives.
        hold = len(CITATION_BLOCK_START) - 1
        if len(self._pending) > hold:
            emit = self._pending[:-hold]
            self._pending = self._pending[-hold:]
            return emit
        return ""

    def finish(self) -> tuple[str, CitationExtraction]:
        """Return (remaining prose to emit, citations)."""
        if self._in_block:
            _, extraction = extract_citations(self._tail)
            remainder = self._pending
            self._pending = ""
            return remainder, extraction

        prose, extraction = extract_citations(self._pending)
        self._pending = ""
        return prose, extraction


CITATION_BLOCK_CONTEXT = """

STRUCTURED CITATIONS:
After your normal answer, and only if you referenced the Qur'an, a hadith, or a
named scholarly work, append exactly one block in this format, with nothing
after it:

<<<CITATIONS>>>
{"citations": [
  {"type": "quran", "surah": 2, "ayah_start": 153, "ayah_end": null},
  {"type": "hadith", "collection": "Sahih al-Bukhari", "number": "1"},
  {"type": "scholarly", "work": "Al-Muwafaqat", "author": "Al-Shatibi"}
]}
<<<END_CITATIONS>>>

Rules for the block:
1. It is read by a machine. Emit valid JSON only, with no commentary inside it.
2. Never invent a reference to populate it. Cite only what your prose cites.
3. Omit the block entirely if you cited nothing. An empty block is worse than none.
4. Write your answer normally first. The block is in addition to your answer,
   never a replacement for it, and never a substitute for citing sources in prose.
"""
