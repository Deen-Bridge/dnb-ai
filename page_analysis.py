"""Islamic book page layout analysis (#222).

Why this exists
---------------
A scanned page of a classical Islamic work is not a flat stream of words. It is
a layout: a heading, a body of main text, a band of footnotes at the foot of the
page, marginal commentary (hashiya), and citations to the Qur'an and hadith
threaded through all of it. An OCR engine reads the glyphs but throws that
structure away - it hands back a bag of text blocks with pixel coordinates and
no notion of which block is a heading, which is a footnote, or what order a
human reads them in. This module rebuilds that structure.

Why structured input rather than pixels
---------------------------------------
It deliberately operates on the *output* of an OCR/layout engine - a list of
positioned text blocks - not on raw image pixels. That keeps the whole thing
deterministic and dependency-free: no OpenCV, no ML model, no image decoding.
Everything here is geometry and text heuristics over ``PageInput``, so it runs
in the same lightweight FastAPI process as the rest of the service and is fully
unit-testable without fixtures of scanned images.

What it produces
----------------
Given a ``PageInput`` (page dimensions plus positioned ``PageBlock`` objects) the
analyzer returns, for the page:

* an **element classification** for every block (heading, body text, footnote,
  commentary, reference citation, or decorative), from font size, vertical
  position, centering, and text-pattern heuristics;
* a **reading order** that respects multi-column layout - columns are detected
  by clustering block x-positions - and Arabic right-to-left column ordering;
* **references** - Qur'an ``surah:ayah`` citations and hadith-source mentions -
  extracted and linked back to the block they came from;
* a **semantic structure**: the ordered blocks grouped into regions by role
  (main text, footnotes, commentary, references), the "semantically structured
  digital text" the layout is turned into.

Citation contract (#854)
------------------------
Reference extraction runs in two separate stages. The first finds *candidates*
in block text and records exactly where each came from (block index, matched
substring, character offsets). The second checks each candidate against the
repository's canonical data: Qur'an surah/ayah bounds through
``tafsir.surah_by_number`` / ``surah_by_name`` (the same 114-surah index
``citations.py`` uses), and hadith collections through
``hadith.find_collection_mentions`` (the alias table behind
``hadith.normalize_collection``) plus the bundled hadith number index. There is
no second alias or bounds table here.

A ``validated`` reference means only that the citation is well formed and
points at something that exists in that canonical data. It does not mean the
OCR read the page correctly, that the quoted text matches the ayah, or that a
hadith is authentic; grading is never assessed here. Anything weaker stays
``unverified``, and a citation that cannot exist (surah 115, ayah 200 of a
seven-ayah surah, a backwards range) is moved to ``rejected_references`` with
a reason instead of being returned as a reference. OCR confidence is carried
from the source block when the engine supplied one and is reported as
``unavailable`` otherwise; the analyzer never manufactures a score from a
regex match. The full contract is in ``docs/PAGE_ANALYSIS_CONTRACT.md``.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from enum import Enum

from fastapi import APIRouter
from pydantic import BaseModel, Field, model_validator

from hadith import COLLECTION_NAMES, find_collection_mentions, get_default_source
from media_upload_limits import (
    MAX_PAGE_BLOCK_TEXT_CHARS,
    MAX_PAGE_BLOCKS,
    MAX_PAGE_DIMENSION,
    MAX_PAGE_TEXT_CHARS,
    evaluate_confidence_and_fallback,
    validate_page_layout_input,
)
from tafsir import is_canonical_surah_name, surah_by_name, surah_by_number

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/page-analysis", tags=["page-analysis"])

# Bumped when the meaning of a response field changes, so clients can tell the
# citation semantics they are reading.
CONTRACT_VERSION = "2"

# A real page cites a few dozen sources at most. The cap bounds the response a
# maximal 200,000-character page can produce; overflow is reported, not hidden.
MAX_REFERENCES_PER_PAGE = 1000


class ElementType(str, Enum):
    """The role a block plays in the page layout."""

    HEADING = "heading"
    BODY_TEXT = "body-text"
    FOOTNOTE = "footnote"
    COMMENTARY = "commentary"
    REFERENCE_CITATION = "reference-citation"
    DECORATIVE = "decorative"


# ---------------------------------------------------------------------------
# Input models
# ---------------------------------------------------------------------------


class PageBlock(BaseModel):
    """One positioned text block, as an OCR/layout engine would emit it.

    Coordinates are in the page's own units with the origin at the top-left, so
    ``y`` grows downward. ``font_size`` is optional because not every engine
    reports it; the classifier falls back to text and position heuristics when
    it is absent.
    """

    x: float = Field(
        ..., ge=0, le=MAX_PAGE_DIMENSION, allow_inf_nan=False, description="Left edge of the bounding box."
    )
    y: float = Field(..., ge=0, le=MAX_PAGE_DIMENSION, allow_inf_nan=False, description="Top edge of the bounding box.")
    width: float = Field(..., ge=0, le=MAX_PAGE_DIMENSION, allow_inf_nan=False)
    height: float = Field(..., ge=0, le=MAX_PAGE_DIMENSION, allow_inf_nan=False)
    text: str = Field("", max_length=MAX_PAGE_BLOCK_TEXT_CHARS, description="OCR text of the block.")
    font_size: float | None = Field(None, gt=0, le=MAX_PAGE_DIMENSION, allow_inf_nan=False)
    confidence: float | None = Field(
        None,
        ge=0.0,
        le=1.0,
        allow_inf_nan=False,
        description=(
            "Recognition confidence the OCR engine reported for this block, if any. "
            "Omit it when the engine reports none; the analyzer never invents one."
        ),
    )

    @property
    def center_x(self) -> float:
        return self.x + self.width / 2

    @property
    def bottom(self) -> float:
        return self.y + self.height


class PageInput(BaseModel):
    """A single scanned page: its dimensions and the blocks found on it.

    ``rtl`` marks a right-to-left page (Arabic, Urdu, Persian) so that columns
    are read right-to-left. It defaults to True because the primary corpus this
    serves is Arabic Islamic texts.
    """

    width: float = Field(
        ..., gt=0, le=MAX_PAGE_DIMENSION, allow_inf_nan=False, description="Page width in units (max 20000)."
    )
    height: float = Field(
        ..., gt=0, le=MAX_PAGE_DIMENSION, allow_inf_nan=False, description="Page height in units (max 20000)."
    )
    blocks: list[PageBlock] = Field(
        default_factory=list, max_length=MAX_PAGE_BLOCKS, description="Blocks on page (max 5000)."
    )
    rtl: bool = True
    consent_to_process: bool | None = Field(
        None,
        description=(
            "Required by POST /page-analysis/analyze and must be true. An omitted or false value is "
            "rejected with 400 before any analysis runs."
        ),
    )

    @model_validator(mode="after")
    def _bound_total_text(self) -> PageInput:
        total = sum(len(block.text) for block in self.blocks)
        if total > MAX_PAGE_TEXT_CHARS:
            raise ValueError(f"Page text totals {total} characters; the maximum is {MAX_PAGE_TEXT_CHARS}.")
        return self


# ---------------------------------------------------------------------------
# Output models
# ---------------------------------------------------------------------------


class ReferenceStatus(str, Enum):
    """How far a reference was checked. Neither value asserts OCR accuracy or authenticity."""

    VALIDATED = "validated"
    """Well formed and points at something that exists in the canonical data."""

    UNVERIFIED = "unverified"
    """A plausible candidate that could not be checked to the validated standard."""


class OCRConfidenceBand(str, Enum):
    """Band of the source block's upstream OCR confidence."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNAVAILABLE = "unavailable"
    """The OCR engine reported no confidence for the block."""


class ExtractedReference(BaseModel):
    """A Qur'an or hadith reference recovered from a block's text.

    ``type``, ``reference``, ``text``, ``block_index``, ``surah``, ``ayah`` and
    ``collection`` keep their original meaning; everything else was added by the
    v2 citation contract (#854).
    """

    type: str = Field(..., description="'quran' or 'hadith'.")
    reference: str = Field(
        ...,
        description="Normalized reference: 'surah:ayah' (range start) for the Qur'an, collection name for hadith.",
    )
    text: str = Field(..., description="The exact substring matched in the source block.")
    block_index: int = Field(..., ge=0, description="Index of the source block in the input.")
    surah: int | None = None
    ayah: int | None = None
    collection: str | None = None
    start: int = Field(..., ge=0, description="Character offset of the match in the source block's text.")
    end: int = Field(..., ge=0, description="End offset (exclusive): block.text[start:end] == text.")
    ayah_end: int | None = Field(None, description="Last ayah of a validated range, e.g. 256 for '2:255-256'.")
    surah_name: str | None = Field(None, description="Canonical surah name from the surah index.")
    collection_key: str | None = Field(None, description="Canonical hadith collection key, e.g. 'bukhari'.")
    hadith_number: int | None = Field(None, description="Hadith number cited after the collection name, if any.")
    book: int | None = Field(None, description="Book number cited before the hadith number, if any.")
    status: ReferenceStatus = Field(..., description="'validated' or 'unverified'; see the contract doc.")
    status_reason: str = Field(..., description="Why the reference has this status.")
    ocr_confidence: float | None = Field(
        None, ge=0.0, le=1.0, description="Source block's OCR confidence; null when the engine gave none."
    )
    ocr_confidence_band: OCRConfidenceBand = Field(..., description="Band of ocr_confidence, or 'unavailable'.")
    requires_review: bool = Field(
        ...,
        description="True unless the reference is validated and its block has medium or high OCR confidence.",
    )


class RejectedReference(BaseModel):
    """A citation-shaped match that cannot point at real data, with the reason."""

    type: str = Field(..., description="'quran' or 'hadith'.")
    text: str = Field(..., description="The exact substring matched in the source block.")
    block_index: int = Field(..., ge=0)
    start: int = Field(..., ge=0)
    end: int = Field(..., ge=0)
    reason: str


class ClassifiedElement(BaseModel):
    """A block paired with the role the analyzer assigned it."""

    block_index: int = Field(..., ge=0)
    element_type: ElementType
    text: str
    reading_order: int = Field(..., ge=0, description="0-based position in the reading sequence.")
    column: int = Field(..., ge=0, description="0-based column index in reading order.")


class PageRegion(BaseModel):
    """A group of same-role elements, in reading order."""

    element_type: ElementType
    element_indices: list[int] = Field(default_factory=list, description="Indices into the input blocks.")
    text: str = Field("", description="The region's text, blocks joined in reading order.")


class StructuredPage(BaseModel):
    """The semantically structured page assembled from the classified blocks."""

    columns: int = Field(..., ge=0)
    rtl: bool
    regions: list[PageRegion] = Field(default_factory=list)


class PageAnalysisResponse(BaseModel):
    """Everything the analyzer derives from one page."""

    columns: int = Field(..., ge=0)
    rtl: bool
    elements: list[ClassifiedElement] = Field(default_factory=list)
    reading_order: list[int] = Field(
        default_factory=list,
        description="Block indices in reading order.",
    )
    references: list[ExtractedReference] = Field(default_factory=list)
    structure: StructuredPage
    rejected_references: list[RejectedReference] = Field(
        default_factory=list,
        description="Citation-shaped matches that were dropped because they cannot exist.",
    )
    warnings: list[str] = Field(default_factory=list, description="Page-level notes on uncertainty.")
    contract_version: str = Field(CONTRACT_VERSION, description="Version of the citation contract in use.")


class ElementTypeInfo(BaseModel):
    """A documented element type for the discovery endpoint."""

    type: ElementType
    description: str


# ---------------------------------------------------------------------------
# Reference extraction, stage 1: candidates with provenance
# ---------------------------------------------------------------------------

# "2:153", "Surah 2:153", "Q 2:255-256", "سورة ٢:١٥٣": a colon-separated
# surah/ayah pair with an optional range end. ``\d`` also matches Arabic-Indic
# and Extended Arabic-Indic digits, and ``int()`` reads them. Without a marker
# the pair must start on a boundary and must not sit inside a longer run such
# as a clock time "10:30:15". The one-letter markers must be a capital Q or S
# standing alone, so the "s" of "it's 10:30" is not read as "Surah".
_QURAN_NUMERIC_PATTERN = re.compile(
    r"(?:(?P<marker>\b(?:Qur[’']?an|Surah|Sura|Surat)\.?|(?<![\w’'])(?-i:[QS])\.?|سورة|سوره)\s*)?"
    r"(?(marker)|(?<![\w:.]))"
    r"(?P<surah>\d{1,3})\s?:\s?(?P<ayah>\d{1,3})"
    r"(?:\s?[-–]\s?(?P<end>\d{1,3}))?(?![\d:])",
    re.IGNORECASE,
)

# "[البقرة: ١٥٣]", "(Al-Baqarah: 255-256)", "[Surah Yusuf, 4]": the bracketed
# surah-name citation classical Arabic works use. The name is resolved through
# tafsir.surah_by_name; a bracket whose "name" is not a surah is ignored. Only
# the canonical or Arabic name, or an explicit "Surah"/"سورة" prefix, counts as
# a marker: aliases include ordinary words, so "(Time: 3)" stays unverified.
_QURAN_NAMED_PATTERN = re.compile(
    r"[\[(]\s*(?:(?P<prefix>سورة|سوره|Surah|Sura|Surat)\s+)?"
    r"(?P<name>[^\[\]():،,\d]{2,40}?)\s*[:،,]\s*(?:(?:الآية|آية|ayah|ayat|verse)\s*)?"
    r"(?P<ayah>\d{1,3})(?:\s?[-–]\s?(?P<end>\d{1,3}))?\s*[\])]",
    re.IGNORECASE,
)

# What may follow a collection name for the number to belong to it: "1",
# "#1", "no. 1", "hadith 1", "Book 2, Hadith 3". Kept tight on purpose: OCR text
# is prose, and a number several words later is not a hadith number.
_HADITH_NUMBER_AFTER = re.compile(
    r"\s*[,:]?\s*(?:(?:book|kitab|vol\.?)\s*(?P<book>\d{1,4})\s*[,:]?\s*)?"
    r"(?:(?:hadith|hadeeth|no\.?|number|#)\s*[:#.]?\s*)?(?P<number>\d{1,5})(?!\d|\s?:\s?\d)",
    re.IGNORECASE,
)

# Single words that are far more often ordinary prose ("a Muslim", "Imam
# Malik said") than a citation. They count only with a hadith number or as
# part of a longer alias such as "Sahih Muslim".
_AMBIGUOUS_BARE_ALIASES = frozenset({"muslim", "malik"})

# How far back an opening bracket may sit for "[Al-Baqarah 2:153]" to count as
# a bracketed (marked) numeric citation.
_BRACKET_LOOKBACK = 40


@dataclass(frozen=True)
class ReferenceCandidate:
    """A citation-shaped match, before any check against canonical data."""

    type: str
    block_index: int
    start: int
    end: int
    text: str
    marked: bool = False
    surah: int | None = None
    named_surah: bool = False
    ayah: int | None = None
    ayah_end: int | None = None
    collection_key: str | None = None
    hadith_number: int | None = None
    book: int | None = None


def _is_bracketed(text: str, start: int, end: int) -> bool:
    """True when [start, end) sits inside an open bracket that closes right after it."""
    tail = text[end:].lstrip()
    if not tail or tail[0] not in "])":
        return False
    for char in reversed(text[max(0, start - _BRACKET_LOOKBACK) : start]):
        if char in "[(":
            return True
        if char in "])":
            return False
    return False


def _overlaps(span: tuple[int, int], taken: list[tuple[int, int]]) -> bool:
    return any(span[0] < other[1] and other[0] < span[1] for other in taken)


def _quran_candidates(index: int, text: str) -> list[ReferenceCandidate]:
    found: list[ReferenceCandidate] = []
    taken: list[tuple[int, int]] = []

    for match in _QURAN_NAMED_PATTERN.finditer(text):
        name = match.group("name").strip()
        record = surah_by_name(name)
        if record is None:
            continue
        end_group = match.group("end")
        found.append(
            ReferenceCandidate(
                type="quran",
                block_index=index,
                start=match.start(),
                end=match.end(),
                text=match.group(0),
                marked=bool(match.group("prefix")) or is_canonical_surah_name(name, record),
                surah=record.number,
                named_surah=True,
                ayah=int(match.group("ayah")),
                ayah_end=int(end_group) if end_group else None,
            )
        )
        taken.append(match.span())

    for match in _QURAN_NUMERIC_PATTERN.finditer(text):
        if _overlaps(match.span(), taken):
            continue
        end_group = match.group("end")
        found.append(
            ReferenceCandidate(
                type="quran",
                block_index=index,
                start=match.start(),
                end=match.end(),
                text=match.group(0),
                marked=bool(match.group("marker")) or _is_bracketed(text, match.start(), match.end()),
                surah=int(match.group("surah")),
                ayah=int(match.group("ayah")),
                ayah_end=int(end_group) if end_group else None,
            )
        )
    return found


def _hadith_candidates(index: int, text: str) -> list[ReferenceCandidate]:
    found: list[ReferenceCandidate] = []
    for key, matched, (start, end) in find_collection_mentions(text):
        number_match = _HADITH_NUMBER_AFTER.match(text, end)
        if number_match is not None:
            book_group = number_match.group("book")
            found.append(
                ReferenceCandidate(
                    type="hadith",
                    block_index=index,
                    start=start,
                    end=number_match.end(),
                    text=text[start : number_match.end()],
                    marked=True,
                    collection_key=key,
                    hadith_number=int(number_match.group("number")),
                    book=int(book_group) if book_group else None,
                )
            )
            continue
        if matched.casefold() in _AMBIGUOUS_BARE_ALIASES:
            continue
        found.append(
            ReferenceCandidate(type="hadith", block_index=index, start=start, end=end, text=matched, collection_key=key)
        )
    return found


def find_reference_candidates(blocks: list[PageBlock]) -> list[ReferenceCandidate]:
    """Stage 1: every citation-shaped match, in block order then left to right."""
    candidates: list[ReferenceCandidate] = []
    for index, block in enumerate(blocks):
        text = block.text or ""
        if not text.strip():
            continue
        per_block = _quran_candidates(index, text) + _hadith_candidates(index, text)
        per_block.sort(key=lambda candidate: (candidate.start, candidate.end))
        candidates.extend(per_block)
    return candidates


# ---------------------------------------------------------------------------
# Reference extraction, stage 2: check candidates against canonical data
# ---------------------------------------------------------------------------


def _confidence_band(confidence: float | None) -> OCRConfidenceBand:
    if confidence is None:
        return OCRConfidenceBand.UNAVAILABLE
    band, _fallback, _review = evaluate_confidence_and_fallback(confidence)
    return OCRConfidenceBand(band.value)


def _rejected(candidate: ReferenceCandidate, reason: str) -> RejectedReference:
    return RejectedReference(
        type=candidate.type,
        text=candidate.text,
        block_index=candidate.block_index,
        start=candidate.start,
        end=candidate.end,
        reason=reason,
    )


def _accepted(
    candidate: ReferenceCandidate,
    block: PageBlock,
    *,
    reference: str,
    status: ReferenceStatus,
    reason: str,
    surah: int | None = None,
    ayah: int | None = None,
    ayah_end: int | None = None,
    surah_name: str | None = None,
    collection: str | None = None,
    collection_key: str | None = None,
) -> ExtractedReference:
    band = _confidence_band(block.confidence)
    requires_review = status != ReferenceStatus.VALIDATED or band in (
        OCRConfidenceBand.LOW,
        OCRConfidenceBand.UNAVAILABLE,
    )
    return ExtractedReference(
        type=candidate.type,
        reference=reference,
        text=candidate.text,
        block_index=candidate.block_index,
        start=candidate.start,
        end=candidate.end,
        surah=surah,
        ayah=ayah,
        ayah_end=ayah_end,
        surah_name=surah_name,
        collection=collection,
        collection_key=collection_key,
        hadith_number=candidate.hadith_number,
        book=candidate.book,
        status=status,
        status_reason=reason,
        ocr_confidence=block.confidence,
        ocr_confidence_band=band,
        requires_review=requires_review,
    )


def _verify_quran(candidate: ReferenceCandidate, block: PageBlock) -> ExtractedReference | RejectedReference:
    surah, ayah, ayah_end = candidate.surah or 0, candidate.ayah or 0, candidate.ayah_end
    record = surah_by_number(surah)
    if record is None:
        return _rejected(candidate, f"Surah {surah} does not exist; surahs are numbered 1-114.")
    if not 1 <= ayah <= record.ayah_count:
        return _rejected(candidate, f"{record.name} has {record.ayah_count} ayat; ayah {ayah} does not exist.")
    if ayah_end is not None:
        if ayah_end < ayah or ayah_end > record.ayah_count:
            return _rejected(
                candidate,
                f"Range {ayah}-{ayah_end} is not valid within {record.name} ({record.ayah_count} ayat).",
            )
        if ayah_end == ayah:
            ayah_end = None

    if candidate.marked:
        status = ReferenceStatus.VALIDATED
        reason = (
            "Qur'an citation marker present and the surah/ayah exist in the surah index. "
            "The OCR reading itself is not verified."
        )
    elif candidate.named_surah:
        status = ReferenceStatus.UNVERIFIED
        reason = (
            "Surah matched only through an alias that can also be an ordinary word; "
            "confirm the bracket is a Qur'an citation."
        )
    else:
        status = ReferenceStatus.UNVERIFIED
        reason = (
            "Bare number pair with no Qur'an marker; it exists in the surah index but may be "
            "a time, a ratio, or another text's verse."
        )
    return _accepted(
        candidate,
        block,
        reference=f"{record.number}:{ayah}",
        status=status,
        reason=reason,
        surah=record.number,
        ayah=ayah,
        ayah_end=ayah_end,
        surah_name=record.name,
    )


def _verify_hadith(candidate: ReferenceCandidate, block: PageBlock) -> ExtractedReference:
    key = candidate.collection_key or ""
    display = COLLECTION_NAMES.get(key, key)
    if candidate.hadith_number is None:
        status = ReferenceStatus.UNVERIFIED
        reason = "Collection named without a hadith number; nothing specific can be checked."
    elif get_default_source().get(key, candidate.hadith_number, candidate.book) is not None:
        status = ReferenceStatus.VALIDATED
        reason = (
            "Collection recognized and the hadith number exists in the bundled hadith index. "
            "Authenticity and the narration text are not assessed here."
        )
    else:
        status = ReferenceStatus.UNVERIFIED
        reason = (
            "Collection recognized, but this number was not found in the bundled hadith index; "
            "numbering schemes differ between editions."
        )
    return _accepted(
        candidate,
        block,
        reference=display,
        status=status,
        reason=reason,
        collection=display,
        collection_key=key,
    )


def verify_reference_candidate(
    candidate: ReferenceCandidate, blocks: list[PageBlock]
) -> ExtractedReference | RejectedReference:
    """Stage 2: check one candidate, returning an accepted or a rejected reference."""
    block = blocks[candidate.block_index]
    if candidate.type == "quran":
        return _verify_quran(candidate, block)
    return _verify_hadith(candidate, block)


def extract_and_verify_references(
    blocks: list[PageBlock],
) -> tuple[list[ExtractedReference], list[RejectedReference]]:
    """Run both stages: accepted references and rejected citation-shaped matches.

    Never raises on block content. Order is stable: blocks in input order,
    matches left to right within a block.
    """
    accepted: list[ExtractedReference] = []
    rejected: list[RejectedReference] = []
    for candidate in find_reference_candidates(blocks):
        outcome = verify_reference_candidate(candidate, blocks)
        if isinstance(outcome, RejectedReference):
            rejected.append(outcome)
        else:
            accepted.append(outcome)
    return accepted, rejected


def extract_references(blocks: list[PageBlock]) -> list[ExtractedReference]:
    """Accepted Qur'an and hadith references from every block, linked to their source.

    Kept for existing callers; ``extract_and_verify_references`` also returns
    what was rejected and why.
    """
    return extract_and_verify_references(blocks)[0]


# ---------------------------------------------------------------------------
# Column detection and reading order
# ---------------------------------------------------------------------------


def detect_columns(page: PageInput) -> list[list[int]]:
    """Group block indices into columns by clustering their horizontal centers.

    Blocks are sorted by center-x and split wherever a gap between consecutive
    centers exceeds a fraction of the page width, so a two-column spread splits
    at the gutter while a single justified column stays whole. Each returned
    column lists its block indices sorted top-to-bottom.
    """
    if not page.blocks:
        return []

    order = sorted(range(len(page.blocks)), key=lambda i: page.blocks[i].center_x)
    gap_threshold = page.width * 0.15

    columns: list[list[int]] = [[order[0]]]
    for prev, cur in zip(order, order[1:], strict=False):
        if page.blocks[cur].center_x - page.blocks[prev].center_x > gap_threshold:
            columns.append([cur])
        else:
            columns[-1].append(cur)

    for column in columns:
        column.sort(key=lambda i: page.blocks[i].y)

    return columns


def determine_reading_order(page: PageInput) -> tuple[list[int], dict[int, int]]:
    """Return block indices in reading order, plus a block-index -> column map.

    Columns are read left-to-right for LTR pages and right-to-left for RTL
    pages (Arabic script); within each column blocks run top to bottom.
    """
    columns = detect_columns(page)

    ordered_columns = list(reversed(columns)) if page.rtl else columns

    reading_order: list[int] = []
    column_of: dict[int, int] = {}
    for column_number, column in enumerate(ordered_columns):
        for block_index in column:
            reading_order.append(block_index)
            column_of[block_index] = column_number

    return reading_order, column_of


# ---------------------------------------------------------------------------
# Element classification
# ---------------------------------------------------------------------------

# A leading footnote marker: "1.", "(2)", "[3]", or a superscript-style digit
# opening a small block near the foot of the page.
_FOOTNOTE_MARKER = re.compile(r"^\s*[\(\[]?\d{1,3}[\)\].\-:]")

# Decorative separators and ornament lines carry no readable words.
_DECORATIVE = re.compile(r"^[\s\*\-─-╿•\.٭۞~=_]+$")


def _median_font_size(page: PageInput) -> float | None:
    sizes = sorted(b.font_size for b in page.blocks if b.font_size is not None)
    if not sizes:
        return None
    mid = len(sizes) // 2
    if len(sizes) % 2:
        return sizes[mid]
    return (sizes[mid - 1] + sizes[mid]) / 2


def classify_block(
    block: PageBlock,
    page: PageInput,
    median_font: float | None,
    has_reference: bool,
) -> ElementType:
    """Assign one block its layout role from geometry and text heuristics.

    The order of the checks is the priority: a purely ornamental line is
    decorative first; a small block low on the page is a footnote; a large or
    centered block high on the page is a heading; a block dominated by a
    citation is a reference; anything in a side column is commentary; the
    remainder is body text.
    """
    text = (block.text or "").strip()

    if not text or _DECORATIVE.match(text):
        return ElementType.DECORATIVE

    rel_y = block.y / page.height if page.height else 0.0
    rel_bottom = block.bottom / page.height if page.height else 0.0
    font = block.font_size
    is_small = median_font is not None and font is not None and font < median_font * 0.85
    is_large = median_font is not None and font is not None and font > median_font * 1.15

    # Footnotes sit in the bottom fifth of the page and read smaller, often
    # opening with a numeric marker.
    if rel_y >= 0.8 and (is_small or _FOOTNOTE_MARKER.match(text)):
        return ElementType.FOOTNOTE

    # Headings are short, high on the page, and either larger or centered.
    is_centered = abs(block.center_x - page.width / 2) < page.width * 0.15
    is_short = len(text) <= 80
    if rel_bottom <= 0.25 and is_short and (is_large or (is_centered and not _FOOTNOTE_MARKER.match(text))):
        return ElementType.HEADING

    # A short block whose content is essentially a citation is a reference line.
    if has_reference and is_short:
        return ElementType.REFERENCE_CITATION

    return ElementType.BODY_TEXT


def _is_side_column(block: PageBlock, page: PageInput, columns: int) -> bool:
    """A narrow block hugging a page margin in a multi-column layout is marginalia."""
    if columns < 2:
        return False
    narrow = block.width < page.width * 0.35
    near_margin = block.x < page.width * 0.1 or block.bottom > 0 and (block.x + block.width) > page.width * 0.9
    return narrow and near_margin


# ---------------------------------------------------------------------------
# Top-level analysis
# ---------------------------------------------------------------------------


def analyze_page(page: PageInput) -> PageAnalysisResponse:
    """Classify, order, extract references from, and structure a page. Never raises."""
    reading_order, column_of = determine_reading_order(page)
    columns = max(column_of.values(), default=-1) + 1
    references, rejected = extract_and_verify_references(page.blocks)
    truncated = len(references) > MAX_REFERENCES_PER_PAGE or len(rejected) > MAX_REFERENCES_PER_PAGE
    references = references[:MAX_REFERENCES_PER_PAGE]
    rejected = rejected[:MAX_REFERENCES_PER_PAGE]
    # Only validated citations shape layout classification; an unverified
    # "10:30" must not turn a block into a reference line.
    referenced_blocks = {ref.block_index for ref in references if ref.status == ReferenceStatus.VALIDATED}
    median_font = _median_font_size(page)

    position_in_reading = {block_index: pos for pos, block_index in enumerate(reading_order)}

    elements: list[ClassifiedElement] = []
    for index, block in enumerate(page.blocks):
        element_type = classify_block(block, page, median_font, index in referenced_blocks)
        if element_type == ElementType.BODY_TEXT and _is_side_column(block, page, columns):
            element_type = ElementType.COMMENTARY
        elements.append(
            ClassifiedElement(
                block_index=index,
                element_type=element_type,
                text=(block.text or "").strip(),
                reading_order=position_in_reading.get(index, 0),
                column=column_of.get(index, 0),
            )
        )

    structure = build_structure(page, elements, reading_order)

    return PageAnalysisResponse(
        columns=columns,
        rtl=page.rtl,
        elements=elements,
        reading_order=reading_order,
        references=references,
        structure=structure,
        rejected_references=rejected,
        warnings=_page_warnings(page, references, rejected, truncated),
    )


def _page_warnings(
    page: PageInput,
    references: list[ExtractedReference],
    rejected: list[RejectedReference],
    truncated: bool = False,
) -> list[str]:
    """Page-level uncertainty notes. Counts only; never echoes page text."""
    warnings: list[str] = []
    if truncated:
        warnings.append(
            f"More than {MAX_REFERENCES_PER_PAGE} references or rejections were found; "
            "each list was cut to the first entries in block order."
        )
    if not any((block.text or "").strip() for block in page.blocks):
        warnings.append("No text was found on the page; nothing was extracted.")
    unavailable = sum(1 for ref in references if ref.ocr_confidence_band == OCRConfidenceBand.UNAVAILABLE)
    if unavailable:
        warnings.append(
            f"{unavailable} reference(s) come from blocks with no OCR confidence; "
            "confirm them against the page image before citing."
        )
    low = sum(1 for ref in references if ref.ocr_confidence_band == OCRConfidenceBand.LOW)
    if low:
        warnings.append(f"{low} reference(s) come from low-confidence OCR blocks and may be misread.")
    unverified = sum(1 for ref in references if ref.status == ReferenceStatus.UNVERIFIED)
    if unverified:
        warnings.append(f"{unverified} reference(s) are unverified candidates, not checked citations.")
    if rejected:
        warnings.append(
            f"{len(rejected)} citation-shaped match(es) were rejected because they cannot exist; "
            "see rejected_references."
        )
    return warnings


# The order regions appear in the structured page: main text leads, then the
# supporting apparatus.
_REGION_ORDER: list[ElementType] = [
    ElementType.HEADING,
    ElementType.BODY_TEXT,
    ElementType.COMMENTARY,
    ElementType.FOOTNOTE,
    ElementType.REFERENCE_CITATION,
    ElementType.DECORATIVE,
]


def build_structure(
    page: PageInput,
    elements: list[ClassifiedElement],
    reading_order: list[int],
) -> StructuredPage:
    """Group classified blocks into regions by role, each in reading order."""
    by_index = {element.block_index: element for element in elements}
    columns = max((element.column for element in elements), default=-1) + 1

    grouped: dict[ElementType, list[int]] = {element_type: [] for element_type in _REGION_ORDER}
    for block_index in reading_order:
        element = by_index.get(block_index)
        if element is not None:
            grouped[element.element_type].append(block_index)

    regions: list[PageRegion] = []
    for element_type in _REGION_ORDER:
        indices = grouped[element_type]
        if not indices:
            continue
        text = "\n".join((page.blocks[i].text or "").strip() for i in indices).strip()
        regions.append(PageRegion(element_type=element_type, element_indices=indices, text=text))

    return StructuredPage(columns=columns, rtl=page.rtl, regions=regions)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

_ELEMENT_TYPE_DESCRIPTIONS: dict[ElementType, str] = {
    ElementType.HEADING: "A short title or section header, high on the page and larger or centered.",
    ElementType.BODY_TEXT: "The main running text of the page.",
    ElementType.FOOTNOTE: "A small annotation near the foot of the page, often numbered.",
    ElementType.COMMENTARY: "Marginal or side-column commentary (hashiya) accompanying the main text.",
    ElementType.REFERENCE_CITATION: "A block dominated by a Qur'an or hadith citation.",
    ElementType.DECORATIVE: "An ornament, separator, or non-textual line.",
}


@router.post("/analyze", response_model=PageAnalysisResponse)
async def analyze(page: PageInput) -> PageAnalysisResponse:
    """Analyze one scanned Islamic book page from its OCR block layout.

    Returns per-block classification, a layout-aware reading order, extracted
    Qur'an and hadith references with their verification status, rejected
    citation-shaped matches, and a semantically structured page.

    ``consent_to_process`` must be true: omitted or false is rejected with 400
    before analysis. Limit violations (dimensions, block count, block or page
    text length) are rejected with 422 by request validation. Nothing is
    stored; see docs/PAGE_ANALYSIS_CONTRACT.md.
    """
    validate_page_layout_input(page.width, page.height, len(page.blocks), consent=page.consent_to_process is True)
    started = time.perf_counter()
    result = analyze_page(page)
    logger.info(
        "page_analysis completed: %s",
        {
            "blocks": len(page.blocks),
            "text_chars": sum(len(block.text) for block in page.blocks),
            "validated": sum(1 for ref in result.references if ref.status == ReferenceStatus.VALIDATED),
            "unverified": sum(1 for ref in result.references if ref.status == ReferenceStatus.UNVERIFIED),
            "rejected": len(result.rejected_references),
            "requires_review": sum(1 for ref in result.references if ref.requires_review),
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
        },
    )
    return result


@router.get("/element-types", response_model=list[ElementTypeInfo])
async def element_types() -> list[ElementTypeInfo]:
    """List the element types the analyzer can assign, with descriptions."""
    return [
        ElementTypeInfo(type=element_type, description=description)
        for element_type, description in _ELEMENT_TYPE_DESCRIPTIONS.items()
    ]
