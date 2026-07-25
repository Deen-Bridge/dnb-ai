"""Tests for the hadith grading module — no live API calls, no network."""

from typing import Dict, Optional, Tuple

import pytest

from hadith import (
    ChainType,
    GradeRecord,
    GradingSource,
    HADITH_ADAB_CONTEXT,
    Strength,
    aggregate_chain_type,
    aggregate_strength,
    annotate,
    build_caution_note,
    get_default_source,
    normalize_collection,
    parse_grade_string,
    parse_references,
)


# ---------------------------------------------------------------------------
# Grade-string tokenizer
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("raw,expected_strength,expected_chain", [
    ("Sahih", Strength.SAHIH, ChainType.MARFU),
    ("Daif", Strength.DAIF, ChainType.MARFU),
    ("Hasan", Strength.HASAN, ChainType.MARFU),
    ("Mawdu", Strength.MAWDU, ChainType.MARFU),
    ("Batil", Strength.MAWDU, ChainType.MARFU),
    # Composite strings resolve to the weaker word.
    ("Hasan Sahih", Strength.HASAN, ChainType.MARFU),
    ("Sahih Lighairihi", Strength.SAHIH, ChainType.MARFU),
    ("Very Daif", Strength.DAIF, ChainType.MARFU),
    ("Isnaad Hasan", Strength.HASAN, ChainType.MARFU),
    ("Isnaad Sahih", Strength.SAHIH, ChainType.MARFU),
    # Munkar and Shadh are weak-hadith subtypes.
    ("Munkar", Strength.DAIF, ChainType.MARFU),
    ("Shadh", Strength.DAIF, ChainType.MARFU),
    # Cross-references to the two Sahihs still mean "authentic".
    ("Sahih Bukhari (142) Sahih Muslim (375)", Strength.SAHIH, ChainType.MARFU),
    ("Sahih Muslim (1480)", Strength.SAHIH, ChainType.MARFU),
    # Chain-type keywords are independent of strength.
    ("Mauquf Sahih", Strength.SAHIH, ChainType.MAUQUF),
    ("Mauquf Daif", Strength.DAIF, ChainType.MAUQUF),
    ("Sahih Muquf", Strength.SAHIH, ChainType.MAUQUF),
    ("Maqtu Sahih", Strength.SAHIH, ChainType.MAQTU),
    ("Sahih Isnaad Mursal", Strength.SAHIH, ChainType.MURSAL),
    # Unrecognized / blank.
    ("-", Strength.UNKNOWN, ChainType.MARFU),
    ("", Strength.UNKNOWN, ChainType.MARFU),
])
def test_parse_grade_string(raw, expected_strength, expected_chain):
    strength, chain = parse_grade_string(raw)
    assert strength == expected_strength
    assert chain == expected_chain


def test_parse_grade_string_is_case_insensitive():
    assert parse_grade_string("SAHIH") == (Strength.SAHIH, ChainType.MARFU)
    assert parse_grade_string("daif") == (Strength.DAIF, ChainType.MARFU)


# ---------------------------------------------------------------------------
# Aggregation across multiple graders
# ---------------------------------------------------------------------------


def test_aggregate_strength_weakest_wins():
    assert aggregate_strength([Strength.SAHIH, Strength.DAIF]) == Strength.DAIF
    assert aggregate_strength([Strength.SAHIH, Strength.HASAN]) == Strength.HASAN
    assert aggregate_strength([Strength.SAHIH, Strength.MAWDU]) == Strength.MAWDU
    assert aggregate_strength([Strength.SAHIH, Strength.SAHIH]) == Strength.SAHIH


def test_aggregate_strength_ignores_unknown_unless_all_unknown():
    assert aggregate_strength([Strength.UNKNOWN, Strength.SAHIH]) == Strength.SAHIH
    assert aggregate_strength([Strength.UNKNOWN, Strength.UNKNOWN]) == Strength.UNKNOWN
    assert aggregate_strength([]) == Strength.UNKNOWN


def test_aggregate_chain_type_any_non_marfu_wins():
    assert aggregate_chain_type([ChainType.MARFU, ChainType.MAUQUF]) == ChainType.MAUQUF
    assert aggregate_chain_type([ChainType.MARFU, ChainType.MARFU]) == ChainType.MARFU
    assert aggregate_chain_type([ChainType.MAQTU, ChainType.MURSAL]) == ChainType.MAQTU


# ---------------------------------------------------------------------------
# Collection alias normalization
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("raw,expected", [
    ("Bukhari", "bukhari"),
    ("bukhari", "bukhari"),
    ("Sahih al-Bukhari", "bukhari"),
    ("Sahih Al Bukhari", "bukhari"),
    ("al-Bukhari", "bukhari"),
    ("Muslim", "muslim"),
    ("Sahih Muslim", "muslim"),
    ("Abu Dawud", "abudawud"),
    ("Abu Dawood", "abudawud"),
    ("Sunan Abu Dawud", "abudawud"),
    ("Tirmidhi", "tirmidhi"),
    ("Jami' at-Tirmidhi", "tirmidhi"),
    ("An-Nasai", "nasai"),
    ("Sunan an-Nasa'i", "nasai"),
    ("Ibn Majah", "ibnmajah"),
    ("Sunan Ibn Majah", "ibnmajah"),
    ("Malik", "malik"),
    ("Muwatta Malik", "malik"),
    ("Muwatta Imam Malik", "malik"),
])
def test_normalize_collection(raw, expected):
    assert normalize_collection(raw) == expected


@pytest.mark.parametrize("raw", [None, "", "Sunan Ibn Kathir", "Not A Collection"])
def test_normalize_collection_unknown(raw):
    assert normalize_collection(raw) is None


# ---------------------------------------------------------------------------
# Reference parsing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text,expected_collection,expected_number", [
    ("Sahih al-Bukhari 1 states...", "bukhari", 1),
    ("as narrated in Bukhari #1", "bukhari", 1),
    ("See Sahih Muslim, Hadith 55.", "muslim", 55),
    ("(Abu Dawud 3)", "abudawud", 3),
    ("Sunan Ibn Majah, Book 1, Hadith 4", "ibnmajah", 4),
])
def test_parse_references_common_phrasings(text, expected_collection, expected_number):
    refs = parse_references(text)
    assert len(refs) == 1
    assert refs[0].collection == expected_collection
    assert refs[0].number == expected_number


def test_parse_references_multiple_in_one_text():
    text = "Compare Sahih al-Bukhari 1 with Sahih Muslim 2 on this topic."
    refs = parse_references(text)
    assert [(r.collection, r.number) for r in refs] == [("bukhari", 1), ("muslim", 2)]


def test_parse_references_no_citation():
    assert parse_references("Prayer five times a day is obligatory.") == []


def test_parse_references_collection_without_number_is_ignored():
    assert parse_references("Sahih al-Bukhari is a reliable collection.") == []


# ---------------------------------------------------------------------------
# annotate() + build_caution_note() against a fake grading source
# ---------------------------------------------------------------------------


class FakeGradingSource(GradingSource):
    def __init__(self, records: Dict[Tuple[str, int], GradeRecord]):
        self._records = records

    def get(self, collection: str, number: int, book: Optional[int] = None) -> Optional[GradeRecord]:
        return self._records.get((collection, number))


def _record(collection, number, grade, chain=ChainType.MARFU, grader="Al-Albani"):
    return GradeRecord(
        collection=collection,
        hadith_number=number,
        book=1,
        book_number=number,
        grade=grade,
        chain_type=chain,
        graders=[{"g": grader, "raw": grade.value, "s": grade.value, "c": chain.value}],
    )


FAKE_SOURCE = FakeGradingSource({
    ("bukhari", 1): _record("bukhari", 1, Strength.SAHIH, grader="Scholarly consensus (Bukhari and Muslim)"),
    ("abudawud", 3): _record("abudawud", 3, Strength.DAIF),
    ("abudawud", 99): _record("abudawud", 99, Strength.MAWDU),
    ("abudawud", 5): _record("abudawud", 5, Strength.SAHIH, chain=ChainType.MAUQUF),
})


def test_annotate_sahih_is_verified_and_not_flagged():
    refs = annotate("See Sahih al-Bukhari 1 for this.", source=FAKE_SOURCE)
    assert len(refs) == 1
    assert refs[0].grade == "SAHIH"
    assert refs[0].verified is True
    assert refs[0].flagged is False


def test_annotate_weak_hadith_as_unqualified_evidence_is_flagged():
    refs = annotate("This is proven by Sunan Abu Dawud 3.", source=FAKE_SOURCE)
    assert refs[0].grade == "DAIF"
    assert refs[0].flagged is True


def test_annotate_weak_hadith_with_nearby_caveat_is_not_flagged():
    text = "This weak (da'if) narration, Sunan Abu Dawud 3, is mentioned only for encouragement."
    refs = annotate(text, source=FAKE_SOURCE)
    assert refs[0].grade == "DAIF"
    assert refs[0].flagged is False


def test_annotate_mawdu_hadith_is_flagged():
    refs = annotate("Some claim Sunan Abu Dawud 99 supports this.", source=FAKE_SOURCE)
    assert refs[0].grade == "MAWDU"
    assert refs[0].flagged is True


def test_annotate_mauquf_hadith_is_flagged_even_if_strength_is_sahih():
    refs = annotate("Sunan Abu Dawud 5 is cited as a Prophetic hadith.", source=FAKE_SOURCE)
    assert refs[0].grade == "SAHIH"
    assert refs[0].chain_type == "MAUQUF"
    assert refs[0].flagged is True


def test_annotate_unknown_reference_is_unverified_not_flagged_alone():
    refs = annotate("Narrated in Jami at-Tirmidhi 9999.", source=FAKE_SOURCE)
    assert refs[0].verified is False
    assert refs[0].grade == "UNKNOWN"
    assert refs[0].note == "grading unverified"
    # annotate() itself never implies authenticity for unverified refs, but
    # whether it's a policy violation depends on whether it's sole support —
    # that's build_caution_note's job, not annotate()'s.
    assert refs[0].flagged is False


def test_build_caution_note_none_when_all_sahih():
    refs = annotate("Sahih al-Bukhari 1 confirms this.", source=FAKE_SOURCE)
    assert build_caution_note("Sahih al-Bukhari 1 confirms this.", refs) is None


def test_build_caution_note_flags_unqualified_weak_evidence():
    text = "This is proven by Sunan Abu Dawud 3."
    refs = annotate(text, source=FAKE_SOURCE)
    note = build_caution_note(text, refs)
    assert note is not None
    assert "Sunan Abu Dawud 3" in note


def test_build_caution_note_none_when_weak_is_labeled_for_targhib():
    text = "This weak (da'if) narration, Sunan Abu Dawud 3, is mentioned only for encouragement."
    refs = annotate(text, source=FAKE_SOURCE)
    assert build_caution_note(text, refs) is None


def test_build_caution_note_flags_sole_unverified_reference():
    text = "Narrated in Jami at-Tirmidhi 9999."
    refs = annotate(text, source=FAKE_SOURCE)
    note = build_caution_note(text, refs)
    assert note is not None
    assert "only hadith cited" in note


def test_build_caution_note_does_not_flag_unverified_when_not_sole_support():
    text = "Sahih al-Bukhari 1 and also Jami at-Tirmidhi 9999 touch on this."
    refs = annotate(text, source=FAKE_SOURCE)
    # The Bukhari reference alone is enough evidentiary support; the second,
    # unverified reference is still labeled unverified in its own record but
    # doesn't trigger the "sole support" escalation.
    assert not any(r.flagged for r in refs)
    assert build_caution_note(text, refs) is None


# ---------------------------------------------------------------------------
# Real bundled dataset (data/hadith/*.json) — integration smoke tests
# ---------------------------------------------------------------------------


def test_default_source_bukhari_hadith_one_is_sahih_by_consensus():
    source = get_default_source()
    record = source.get("bukhari", 1)
    assert record is not None
    assert record.grade == Strength.SAHIH
    assert record.graders[0]["g"] == "Scholarly consensus (Bukhari and Muslim)"


def test_default_source_known_daif_abu_dawud_hadith():
    source = get_default_source()
    record = source.get("abudawud", 3)
    assert record is not None
    assert record.grade == Strength.DAIF


def test_default_source_unknown_hadith_number_returns_none():
    source = get_default_source()
    assert source.get("abudawud", 10_000_000) is None


def test_annotate_against_real_dataset_end_to_end():
    text = "Sahih al-Bukhari 1 is authentic; Sunan Abu Dawud 3 is weak."
    refs = annotate(text)
    assert refs[0].grade == "SAHIH"
    assert refs[0].verified is True
    assert refs[1].grade == "DAIF"


# ---------------------------------------------------------------------------
# Prompt context block
# ---------------------------------------------------------------------------


def test_hadith_adab_context_mentions_grading_rules():
    assert "collection" in HADITH_ADAB_CONTEXT.lower()
    assert "grade" in HADITH_ADAB_CONTEXT.lower()
    assert "sahih al-bukhari" in HADITH_ADAB_CONTEXT.lower()
