"""Islamic book page layout analysis (#222).

Why this exists
---------------
A scanned page of a classical Islamic work is not a flat stream of words. It is
a layout: a heading, a body of main text, a band of footnotes at the foot of the
page, marginal commentary (hashiya), and citations to the Qur'an and hadith
threaded through all of it. An OCR engine reads the glyphs but throws that
structure away — it hands back a bag of text blocks with pixel coordinates and
no notion of which block is a heading, which is a footnote, or what order a
human reads them in. This module rebuilds that structure.

Why structured input rather than pixels
---------------------------------------
It deliberately operates on the *output* of an OCR/layout engine — a list of
positioned text blocks — not on raw image pixels. That keeps the whole thing
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
* a **reading order** that respects multi-column layout — columns are detected
  by clustering block x-positions — and Arabic right-to-left column ordering;
* **references** — Qur'an ``surah:ayah`` citations and hadith-source mentions —
  extracted and linked back to the block they came from;
* a **semantic structure**: the ordered blocks grouped into regions by role
  (main text, footnotes, commentary, references), the "semantically structured
  digital text" the layout is turned into.

Reuse
-----
Reference extraction mirrors the conventions of ``citations.py`` — the same
notion of typed Qur'an and hadith references — but works on positioned OCR text
rather than a model's delimited citation block, so the two do not share code.
"""

from __future__ import annotations

import re
from enum import Enum

from fastapi import APIRouter
from pydantic import BaseModel, Field

router = APIRouter(prefix="/page-analysis", tags=["page-analysis"])


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

    x: float = Field(..., ge=0, description="Left edge of the block's bounding box.")
    y: float = Field(..., ge=0, description="Top edge of the block's bounding box.")
    width: float = Field(..., ge=0)
    height: float = Field(..., ge=0)
    text: str = ""
    font_size: float | None = Field(None, gt=0)

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

    width: float = Field(..., gt=0)
    height: float = Field(..., gt=0)
    blocks: list[PageBlock] = Field(default_factory=list)
    rtl: bool = True


# ---------------------------------------------------------------------------
# Output models
# ---------------------------------------------------------------------------


class ExtractedReference(BaseModel):
    """A Qur'an or hadith reference recovered from a block's text."""

    type: str = Field(..., description="'quran' or 'hadith'.")
    reference: str = Field(..., description="Normalized reference, e.g. '2:153'.")
    text: str = Field(..., description="The exact substring matched in the source.")
    block_index: int = Field(..., ge=0, description="Index of the source block in the input.")
    surah: int | None = None
    ayah: int | None = None
    collection: str | None = None


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


class ElementTypeInfo(BaseModel):
    """A documented element type for the discovery endpoint."""

    type: ElementType
    description: str


# ---------------------------------------------------------------------------
# Reference extraction
# ---------------------------------------------------------------------------

# "2:153", "Surah 2:153", "Q 2:255-256" — a colon-separated surah/ayah pair,
# optionally with a range end that we keep only as the matched text.
_QURAN_PATTERN = re.compile(
    r"\b(?:Q(?:ur[’']?an)?\.?\s*|S(?:urah|ura|urat)?\.?\s*)?"
    r"(?P<surah>\d{1,3}):(?P<ayah>\d{1,3})(?:-\d{1,3})?\b",
    re.IGNORECASE,
)

# The canonical hadith collections, matched case-insensitively as whole words.
_HADITH_COLLECTIONS: dict[str, str] = {
    "bukhari": "Sahih al-Bukhari",
    "muslim": "Sahih Muslim",
    "abu dawud": "Sunan Abi Dawud",
    "abu dawood": "Sunan Abi Dawud",
    "tirmidhi": "Jami' at-Tirmidhi",
    "nasai": "Sunan an-Nasa'i",
    "nasa'i": "Sunan an-Nasa'i",
    "ibn majah": "Sunan Ibn Majah",
    "muwatta": "Muwatta Malik",
    "ahmad": "Musnad Ahmad",
}

_HADITH_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(name) for name in _HADITH_COLLECTIONS) + r")\b",
    re.IGNORECASE,
)


def extract_references(blocks: list[PageBlock]) -> list[ExtractedReference]:
    """Pull Qur'an and hadith references out of every block, linked to its index.

    Never raises: a block whose text matches nothing simply contributes no
    references. Order is stable — blocks in input order, matches left to right.
    """
    references: list[ExtractedReference] = []
    for index, block in enumerate(blocks):
        text = block.text or ""

        for match in _QURAN_PATTERN.finditer(text):
            surah = int(match.group("surah"))
            ayah = int(match.group("ayah"))
            if not (1 <= surah <= 114) or ayah < 1:
                continue
            references.append(
                ExtractedReference(
                    type="quran",
                    reference=f"{surah}:{ayah}",
                    text=match.group(0).strip(),
                    block_index=index,
                    surah=surah,
                    ayah=ayah,
                )
            )

        for match in _HADITH_PATTERN.finditer(text):
            canonical = _HADITH_COLLECTIONS[match.group(1).lower()]
            references.append(
                ExtractedReference(
                    type="hadith",
                    reference=canonical,
                    text=match.group(0).strip(),
                    block_index=index,
                    collection=canonical,
                )
            )

    return references


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
    references = extract_references(page.blocks)
    referenced_blocks = {ref.block_index for ref in references}
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
    )


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
    Qur'an and hadith references, and a semantically structured page.
    """
    return analyze_page(page)


@router.get("/element-types", response_model=list[ElementTypeInfo])
async def element_types() -> list[ElementTypeInfo]:
    """List the element types the analyzer can assign, with descriptions."""
    return [
        ElementTypeInfo(type=element_type, description=description)
        for element_type, description in _ELEMENT_TYPE_DESCRIPTIONS.items()
    ]
