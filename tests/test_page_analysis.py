"""Tests for the Islamic book page layout analyzer (#222) and its citation contract (#854).

All tests run offline against synthetic OCR block layouts — no scanned images,
no GEMINI_API_KEY, no network, and the FastAPI app is never imported. Citation
checks read only the bundled surah index and hadith number index.
"""

import pytest
from pydantic import ValidationError

from media_upload_limits import MAX_PAGE_BLOCK_TEXT_CHARS, MAX_PAGE_BLOCKS, MAX_PAGE_TEXT_CHARS
from page_analysis import (
    CONTRACT_VERSION,
    MAX_REFERENCES_PER_PAGE,
    ElementType,
    ExtractedReference,
    OCRConfidenceBand,
    PageBlock,
    PageInput,
    ReferenceStatus,
    analyze_page,
    build_structure,
    classify_block,
    detect_columns,
    determine_reading_order,
    extract_and_verify_references,
    extract_references,
    find_reference_candidates,
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


# ---------------------------------------------------------------------------
# Citation contract (#854): helpers
# ---------------------------------------------------------------------------


def _block(text: str, confidence: float | None = None, **geometry: float) -> PageBlock:
    return PageBlock(
        x=geometry.get("x", 10),
        y=geometry.get("y", 40),
        width=geometry.get("width", 80),
        height=geometry.get("height", 10),
        text=text,
        confidence=confidence,
    )


def _refs(text: str, confidence: float | None = None) -> list[ExtractedReference]:
    return extract_references([_block(text, confidence)])


# ---------------------------------------------------------------------------
# Qur'an references: Arabic, English, mixed; bounds and ranges
# ---------------------------------------------------------------------------


def test_marked_quran_reference_is_validated_with_surah_name() -> None:
    [ref] = _refs("As stated in Surah 2:153, be patient.")
    assert ref.status == ReferenceStatus.VALIDATED
    assert ref.surah_name == "Al-Baqarah"
    assert ref.ayah_end is None


def test_arabic_indic_digits_are_read() -> None:
    [ref] = _refs("قال تعالى سورة ٢:١٥٣ في الصبر")
    assert (ref.surah, ref.ayah) == (2, 153)
    assert ref.status == ReferenceStatus.VALIDATED


def test_arabic_bracketed_surah_name_citation() -> None:
    [ref] = _refs("﴿يا أيها الذين آمنوا استعينوا بالصبر والصلاة﴾ [البقرة: ١٥٣]")
    assert ref.reference == "2:153"
    assert ref.text == "[البقرة: ١٥٣]"
    assert ref.status == ReferenceStatus.VALIDATED


def test_english_bracketed_surah_name_range() -> None:
    [ref] = _refs("Ayat al-Kursi (Al-Baqarah: 255-256) is recited often.")
    assert ref.reference == "2:255"
    assert ref.ayah_end == 256
    assert ref.status == ReferenceStatus.VALIDATED


def test_mixed_arabic_english_block_keeps_both_references_in_order() -> None:
    refs = _refs("See Q 3:200 and [آل عمران: ١٩٠] for reflection.")
    assert [(r.surah, r.ayah) for r in refs] == [(3, 200), (3, 190)]
    assert all(r.status == ReferenceStatus.VALIDATED for r in refs)


@pytest.mark.parametrize("text", ["(Time: 3)", "(Light, 35)", "[cow: 5]"])
def test_bracketed_alias_that_is_an_ordinary_word_stays_unverified(text: str) -> None:
    [ref] = _refs(text)
    assert ref.status == ReferenceStatus.UNVERIFIED
    assert "alias" in ref.status_reason


@pytest.mark.parametrize("text", ["(Surah Light, 35)", "[An-Nur: 35]", "[النور: ٣٥]", "(Nur: 35)"])
def test_bracketed_canonical_or_prefixed_name_is_validated(text: str) -> None:
    [ref] = _refs(text)
    assert ref.reference == "24:35"
    assert ref.status == ReferenceStatus.VALIDATED


def test_bracket_naming_a_non_surah_is_ignored() -> None:
    assert _refs("(page: 12)") == []
    assert [r.type for r in _refs("(Muslim, 3)")] == ["hadith"]


def test_ayah_beyond_surah_length_is_rejected_with_reason() -> None:
    accepted, rejected = extract_and_verify_references([_block("Surah 1:200")])
    assert accepted == []
    assert len(rejected) == 1
    assert "Al-Fatihah has 7 ayat" in rejected[0].reason


@pytest.mark.parametrize("text", ["Surah 0:5", "Surah 115:1", "Surah 2:0"])
def test_impossible_quran_numbers_are_rejected(text: str) -> None:
    accepted, rejected = extract_and_verify_references([_block(text)])
    assert accepted == []
    assert [r.text for r in rejected] == [text]


def test_valid_range_is_kept_and_invalid_ranges_are_rejected() -> None:
    [ok] = _refs("Q 2:255-257")
    assert (ok.ayah, ok.ayah_end) == (255, 257)

    for bad in ("Q 2:257-255", "Q 2:285-290"):
        accepted, rejected = extract_and_verify_references([_block(bad)])
        assert accepted == []
        assert "not valid" in rejected[0].reason


def test_degenerate_range_collapses_to_single_ayah() -> None:
    [ref] = _refs("Q 2:5-5")
    assert ref.ayah_end is None


def test_bare_number_pair_is_unverified_not_validated() -> None:
    [ref] = _refs("The lecture starts at 10:30 sharp.")
    assert ref.status == ReferenceStatus.UNVERIFIED
    assert "no Qur'an marker" in ref.status_reason
    assert ref.requires_review is True


@pytest.mark.parametrize("text", ["it's 10:30 now", "John's 3:16", "as 2:153 says"])
def test_letters_inside_words_are_not_surah_markers(text: str) -> None:
    [ref] = _refs(text)
    assert ref.status == ReferenceStatus.UNVERIFIED


@pytest.mark.parametrize("text", ["Q 2:255", "Q2:255", "S. 2:255", "Quran 2:255", "Qur’an 2:255", "Surat 2:255"])
def test_explicit_markers_validate(text: str) -> None:
    [ref] = _refs(text)
    assert ref.reference == "2:255"
    assert ref.status == ReferenceStatus.VALIDATED


def test_bracketed_bare_number_pair_counts_as_marked() -> None:
    [ref] = _refs("patience is commanded [2:153].")
    assert ref.status == ReferenceStatus.VALIDATED


def test_clock_time_with_seconds_is_not_a_candidate() -> None:
    assert _refs("recorded 10:30:15 on the scan") == []


def test_ocr_noise_inside_numbers_produces_no_reference() -> None:
    assert _refs("Surah 2:l53 and S urah") == []


# ---------------------------------------------------------------------------
# Hadith references: recognized, numbered, ambiguous, unknown
# ---------------------------------------------------------------------------


def test_numbered_hadith_found_in_index_is_validated_without_grading() -> None:
    [ref] = _refs("Narrated in Bukhari #1 on intentions.")
    assert ref.collection_key == "bukhari"
    assert ref.collection == "Sahih al-Bukhari"
    assert ref.hadith_number == 1
    assert ref.status == ReferenceStatus.VALIDATED
    assert "Authenticity" in ref.status_reason
    assert "grade" not in ref.model_dump()


def test_numbered_hadith_missing_from_index_is_unverified() -> None:
    [ref] = _refs("Abu Dawud no. 99999")
    assert ref.collection == "Sunan Abu Dawud"
    assert ref.status == ReferenceStatus.UNVERIFIED


def test_bare_collection_name_is_an_unverified_mention() -> None:
    [ref] = _refs("Narrated in Sahih al-Bukhari.")
    assert ref.hadith_number is None
    assert ref.status == ReferenceStatus.UNVERIFIED


@pytest.mark.parametrize("text", ["A Muslim should be patient.", "Imam Malik said so."])
def test_ambiguous_single_word_names_are_not_citations(text: str) -> None:
    assert _refs(text) == []


def test_ambiguous_name_with_number_is_a_citation() -> None:
    [ref] = _refs("Muslim 2564")
    assert ref.collection_key == "muslim"
    assert ref.hadith_number == 2564


def test_number_far_from_collection_name_is_not_attached() -> None:
    [ref] = _refs("Sahih al-Bukhari reports that he prayed, said 40 times.")
    assert ref.hadith_number is None


def test_unknown_collection_is_not_extracted() -> None:
    assert [r for r in _refs("Musnad Ahmad 123") if r.type == "hadith"] == []


# ---------------------------------------------------------------------------
# Provenance and ordering
# ---------------------------------------------------------------------------


def test_offsets_slice_back_to_the_matched_text() -> None:
    blocks = [_block("intro"), _block("First Surah 2:153 then Bukhari 1 then Surah 2:153 again.")]
    refs = extract_references(blocks)
    assert [r.block_index for r in refs] == [1, 1, 1]
    for ref in refs:
        assert blocks[ref.block_index].text[ref.start : ref.end] == ref.text
    # The repeated citation is reported twice, at different offsets.
    assert refs[0].reference == refs[2].reference
    assert refs[0].start != refs[2].start


def test_references_point_to_source_blocks_on_rtl_page() -> None:
    page = PageInput(
        width=100.0,
        height=100.0,
        rtl=True,
        blocks=[
            PageBlock(x=5, y=10, width=35, height=20, text="left column, no citation", font_size=12.0),
            PageBlock(x=60, y=10, width=35, height=20, text="right column [البقرة: ١٥٣]", font_size=12.0),
        ],
    )
    result = analyze_page(page)
    assert result.reading_order == [1, 0]
    assert [r.block_index for r in result.references] == [1]


def test_candidates_are_found_before_verification() -> None:
    candidates = find_reference_candidates([_block("Surah 1:200 and Surah 2:153")])
    assert [c.text for c in candidates] == ["Surah 1:200", "Surah 2:153"]


# ---------------------------------------------------------------------------
# Confidence and fallback
# ---------------------------------------------------------------------------


def test_high_confidence_validated_reference_needs_no_review() -> None:
    [ref] = _refs("Surah 2:153", confidence=0.93)
    assert ref.ocr_confidence == 0.93
    assert ref.ocr_confidence_band == OCRConfidenceBand.HIGH
    assert ref.requires_review is False


def test_low_confidence_block_forces_review_and_warns() -> None:
    page = PageInput(width=100.0, height=100.0, blocks=[_block("Surah 2:153", confidence=0.2)])
    result = analyze_page(page)
    [ref] = result.references
    assert ref.ocr_confidence_band == OCRConfidenceBand.LOW
    assert ref.requires_review is True
    assert any("low-confidence" in w for w in result.warnings)


def test_missing_confidence_is_unavailable_not_invented() -> None:
    page = PageInput(width=100.0, height=100.0, blocks=[_block("Surah 2:153")])
    result = analyze_page(page)
    [ref] = result.references
    assert ref.ocr_confidence is None
    assert ref.ocr_confidence_band == OCRConfidenceBand.UNAVAILABLE
    assert ref.requires_review is True
    assert any("no OCR confidence" in w for w in result.warnings)


def test_unverified_reference_does_not_make_a_reference_citation_block() -> None:
    page = PageInput(width=100.0, height=100.0, rtl=False, blocks=[_block("Meet at 10:30")])
    result = analyze_page(page)
    assert result.elements[0].element_type == ElementType.BODY_TEXT
    assert result.references[0].status == ReferenceStatus.UNVERIFIED


def test_rejections_surface_in_response_and_warnings() -> None:
    page = PageInput(width=100.0, height=100.0, blocks=[_block("Surah 1:200")])
    result = analyze_page(page)
    assert result.references == []
    assert len(result.rejected_references) == 1
    assert any("rejected" in w for w in result.warnings)
    assert result.contract_version == CONTRACT_VERSION


# ---------------------------------------------------------------------------
# Empty, blank, malformed and oversized input
# ---------------------------------------------------------------------------


def test_blank_and_whitespace_blocks_are_decorative_and_warned() -> None:
    page = PageInput(width=100.0, height=100.0, blocks=[_block(""), _block("   \n ")])
    result = analyze_page(page)
    assert [e.element_type for e in result.elements] == [ElementType.DECORATIVE, ElementType.DECORATIVE]
    assert result.references == []
    assert any("No text" in w for w in result.warnings)


def test_block_text_over_limit_is_rejected_by_validation() -> None:
    with pytest.raises(ValidationError):
        _block("x" * (MAX_PAGE_BLOCK_TEXT_CHARS + 1))


def test_total_page_text_over_limit_is_rejected_by_validation() -> None:
    per_block = MAX_PAGE_BLOCK_TEXT_CHARS
    count = MAX_PAGE_TEXT_CHARS // per_block + 1
    with pytest.raises(ValidationError, match="Page text totals"):
        PageInput(width=100.0, height=100.0, blocks=[_block("x" * per_block) for _ in range(count)])


def test_too_many_blocks_is_rejected_by_validation() -> None:
    with pytest.raises(ValidationError):
        PageInput(width=100.0, height=100.0, blocks=[_block("a")] * (MAX_PAGE_BLOCKS + 1))


@pytest.mark.parametrize(
    "fields",
    [
        {"x": -1.0},
        {"width": float("nan")},
        {"height": float("inf")},
        {"y": 20001.0},
        {"confidence": 1.5},
        {"confidence": -0.1},
    ],
)
def test_malformed_block_geometry_or_confidence_is_rejected(fields: dict[str, float]) -> None:
    base: dict[str, float | str] = {"x": 0, "y": 0, "width": 10, "height": 10, "text": "a"}
    base.update(fields)
    with pytest.raises(ValidationError):
        PageBlock(**base)  # type: ignore[arg-type]


@pytest.mark.parametrize("dims", [(0, 100), (100, -5), (20001, 100), (100, float("inf"))])
def test_invalid_page_dimensions_are_rejected(dims: tuple[float, float]) -> None:
    with pytest.raises(ValidationError):
        PageInput(width=dims[0], height=dims[1])


def test_consent_defaults_to_absent_not_true() -> None:
    assert PageInput(width=100.0, height=100.0).consent_to_process is None


def test_reference_lists_are_capped_with_a_warning() -> None:
    dense = "Surah 2:153 " * 1000  # 12,000 characters, 1,000 citations per block
    page = PageInput(width=100.0, height=100.0, blocks=[_block(dense), _block(dense)])
    result = analyze_page(page)
    assert len(result.references) == MAX_REFERENCES_PER_PAGE
    assert any("cut to the first" in w for w in result.warnings)
