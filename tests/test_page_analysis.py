"""Tests for the Islamic book page layout analyzer (#222).

All tests run offline against synthetic OCR block layouts — no scanned images,
no GEMINI_API_KEY, and only page_analysis is imported so the suite needs none
of the app's heavy dependencies.
"""

from page_analysis import (
    ElementType,
    ExtractedReference,
    PageBlock,
    PageInput,
    analyze_page,
    build_structure,
    classify_block,
    detect_columns,
    determine_reading_order,
    extract_references,
)


def _single_column_page() -> PageInput:
    """A simple LTR page: heading on top, a body paragraph, a footnote at foot."""
    return PageInput(
        width=100.0,
        height=100.0,
        rtl=False,
        blocks=[
            PageBlock(x=30, y=2, width=40, height=6, text="Chapter on Patience", font_size=20.0),
            PageBlock(
                x=10,
                y=20,
                width=80,
                height=40,
                text="The believers are those who persevere through hardship with faith.",
                font_size=12.0,
            ),
            PageBlock(x=10, y=90, width=80, height=6, text="1. See Surah 2:153 on patience.", font_size=8.0),
        ],
    )


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def test_heading_body_footnote_classification() -> None:
    page = _single_column_page()
    result = analyze_page(page)
    types = {element.block_index: element.element_type for element in result.elements}

    assert types[0] == ElementType.HEADING
    assert types[1] == ElementType.BODY_TEXT
    assert types[2] == ElementType.FOOTNOTE


def test_decorative_block_is_detected() -> None:
    page = PageInput(
        width=100.0,
        height=100.0,
        blocks=[PageBlock(x=10, y=50, width=80, height=2, text="* * *")],
    )
    result = analyze_page(page)
    assert result.elements[0].element_type == ElementType.DECORATIVE


def test_footnote_near_bottom_by_marker_and_size() -> None:
    block = PageBlock(x=10, y=92, width=80, height=5, text="2. A note.", font_size=8.0)
    page = PageInput(width=100.0, height=100.0, blocks=[block])
    element_type = classify_block(block, page, median_font=12.0, has_reference=False)
    assert element_type == ElementType.FOOTNOTE


# ---------------------------------------------------------------------------
# Reading order + columns
# ---------------------------------------------------------------------------


def _two_column_page(rtl: bool) -> PageInput:
    """Two columns, each with a top and a bottom block."""
    return PageInput(
        width=100.0,
        height=100.0,
        rtl=rtl,
        blocks=[
            PageBlock(x=5, y=10, width=35, height=20, text="left-top", font_size=12.0),
            PageBlock(x=5, y=50, width=35, height=20, text="left-bottom", font_size=12.0),
            PageBlock(x=60, y=10, width=35, height=20, text="right-top", font_size=12.0),
            PageBlock(x=60, y=50, width=35, height=20, text="right-bottom", font_size=12.0),
        ],
    )


def test_detect_two_columns() -> None:
    columns = detect_columns(_two_column_page(rtl=False))
    assert len(columns) == 2
    # Each column is ordered top-to-bottom.
    assert columns[0] == [0, 1]
    assert columns[1] == [2, 3]


def test_reading_order_ltr_left_column_first() -> None:
    order, column_of = determine_reading_order(_two_column_page(rtl=False))
    assert order == [0, 1, 2, 3]
    assert column_of[0] == 0
    assert column_of[2] == 1


def test_reading_order_rtl_right_column_first() -> None:
    order, column_of = determine_reading_order(_two_column_page(rtl=True))
    # RTL: the right-hand column (blocks 2, 3) is read before the left one.
    assert order == [2, 3, 0, 1]
    assert column_of[2] == 0
    assert column_of[0] == 1


# ---------------------------------------------------------------------------
# Reference extraction
# ---------------------------------------------------------------------------


def test_extract_quran_reference() -> None:
    blocks = [PageBlock(x=0, y=0, width=10, height=5, text="As stated in Surah 2:153, be patient.")]
    refs = extract_references(blocks)
    quran = [r for r in refs if r.type == "quran"]
    assert len(quran) == 1
    assert quran[0].reference == "2:153"
    assert quran[0].surah == 2
    assert quran[0].ayah == 153
    assert quran[0].block_index == 0


def test_extract_hadith_reference() -> None:
    blocks = [PageBlock(x=0, y=0, width=10, height=5, text="Narrated in Sahih al-Bukhari.")]
    refs = extract_references(blocks)
    hadith = [r for r in refs if r.type == "hadith"]
    assert len(hadith) == 1
    assert hadith[0].collection == "Sahih al-Bukhari"


def test_invalid_surah_is_rejected() -> None:
    blocks = [PageBlock(x=0, y=0, width=10, height=5, text="see 200:5 nowhere")]
    refs = extract_references(blocks)
    assert [r for r in refs if r.type == "quran"] == []


def test_analyze_page_surfaces_reference() -> None:
    result = analyze_page(_single_column_page())
    assert any(r.reference == "2:153" for r in result.references)
    assert isinstance(result.references[0], ExtractedReference)


# ---------------------------------------------------------------------------
# Semantic structuring
# ---------------------------------------------------------------------------


def test_structure_groups_blocks_by_type() -> None:
    page = _single_column_page()
    result = analyze_page(page)
    regions = {region.element_type: region for region in result.structure.regions}

    assert ElementType.HEADING in regions
    assert ElementType.BODY_TEXT in regions
    assert ElementType.FOOTNOTE in regions
    assert regions[ElementType.HEADING].element_indices == [0]
    assert regions[ElementType.BODY_TEXT].element_indices == [1]
    assert regions[ElementType.FOOTNOTE].element_indices == [2]
    assert "Patience" in regions[ElementType.HEADING].text


def test_structure_preserves_rtl_reading_order() -> None:
    page = _two_column_page(rtl=True)
    result = analyze_page(page)
    body = next(r for r in result.structure.regions if r.element_type == ElementType.BODY_TEXT)
    # Right column read first under RTL.
    assert body.element_indices == [2, 3, 0, 1]


def test_build_structure_is_reading_order_stable() -> None:
    page = _single_column_page()
    result = analyze_page(page)
    structure = build_structure(page, result.elements, result.reading_order)
    assert structure.columns == result.columns
    assert structure.rtl is False


def test_empty_page_does_not_raise() -> None:
    result = analyze_page(PageInput(width=100.0, height=100.0, blocks=[]))
    assert result.columns == 0
    assert result.elements == []
    assert result.reading_order == []
    assert result.references == []
