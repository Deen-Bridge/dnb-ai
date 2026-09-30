"""Tests for structured citation extraction (#15).

The governing requirement is that this layer can never fail a chat turn, so
most of these tests feed it broken input and assert that it degrades to an
empty citation list with the prose left intact.
"""

import json

import pytest

from citations import (
    CITATION_BLOCK_END,
    CITATION_BLOCK_START,
    CitationExtraction,
    CitationStreamFilter,
    HadithCitation,
    QuranCitation,
    ScholarlyReference,
    extract_citations,
    parse_citations,
)


def block(payload: str) -> str:
    return f"{CITATION_BLOCK_START}\n{payload}\n{CITATION_BLOCK_END}"


def answer_with(payload: str, prose: str = "Patience is enjoined.") -> str:
    return f"{prose}\n\n{block(payload)}"


class TestBlockSplitting:
    def test_prose_is_returned_without_the_block(self):
        prose, _ = extract_citations(answer_with('{"citations": []}'))
        assert prose == "Patience is enjoined."
        assert CITATION_BLOCK_START not in prose

    def test_text_without_a_block_is_unchanged(self):
        text = "Just an answer, no citations at all."
        prose, extraction = extract_citations(text)
        assert prose == text
        assert extraction.citations == []
        assert extraction.score is None

    def test_truncated_block_is_still_stripped(self):
        # max_output_tokens cut the answer off mid-block.
        text = "Patience is enjoined.\n\n" + CITATION_BLOCK_START + '\n{"citations": [{"type":'
        prose, extraction = extract_citations(text)
        assert prose == "Patience is enjoined."
        assert CITATION_BLOCK_START not in prose
        assert extraction.citations == []

    def test_empty_and_none_input(self):
        assert extract_citations("")[0] == ""
        assert extract_citations(None)[0] == ""


class TestMalformedInput:
    @pytest.mark.parametrize(
        "payload",
        [
            "not json at all",
            "{",
            '{"citations": "not a list"}',
            "[]",
            "null",
            '{"other_key": []}',
        ],
    )
    def test_never_raises_and_yields_no_citations(self, payload):
        prose, extraction = extract_citations(answer_with(payload))
        assert prose == "Patience is enjoined."
        assert extraction.citations == []

    def test_garbage_entries_are_counted_as_attempts(self):
        extraction = parse_citations('{"citations": [1, 2, 3]}')
        assert extraction.attempted == 3
        assert extraction.citations == []
        assert extraction.score == 0.0


class TestQuranValidation:
    def test_valid_reference_is_accepted_and_named(self):
        extraction = parse_citations('{"citations": [{"type": "quran", "surah": 2, "ayah_start": 255}]}')
        assert len(extraction.citations) == 1
        citation = extraction.citations[0]
        assert isinstance(citation, QuranCitation)
        assert citation.surah == 2
        assert citation.ayah_start == 255
        # The name comes from the index, never from the model.
        assert citation.surah_name == "Al-Baqarah"
        assert citation.reference == "2:255"

    def test_surah_name_from_the_model_is_overridden_by_the_index(self):
        extraction = parse_citations(
            '{"citations": [{"type": "quran", "surah": 2, "ayah_start": 1, "surah_name": "The Cow Chapter"}]}'
        )
        assert extraction.citations[0].surah_name == "Al-Baqarah"

    def test_surah_can_be_given_by_name(self):
        extraction = parse_citations('{"citations": [{"type": "quran", "surah": "Al-Baqarah", "ayah_start": 255}]}')
        assert extraction.citations[0].surah == 2

    @pytest.mark.parametrize("surah", [0, 115, 999, -1])
    def test_surah_out_of_range_is_rejected(self, surah):
        extraction = parse_citations(json.dumps({"citations": [{"type": "quran", "surah": surah, "ayah_start": 1}]}))
        assert extraction.citations == []
        assert extraction.attempted == 1
        assert extraction.score == 0.0

    def test_ayah_beyond_the_surah_is_rejected_with_the_real_bound(self):
        extraction = parse_citations('{"citations": [{"type": "quran", "surah": 2, "ayah_start": 300}]}')
        assert extraction.citations == []
        assert "286" in extraction.rejected[0]

    def test_ayah_range_is_kept_when_valid(self):
        extraction = parse_citations('{"citations": [{"type": "quran", "surah": 103, "ayah_start": 1, "ayah_end": 3}]}')
        citation = extraction.citations[0]
        assert citation.ayah_end == 3
        assert citation.reference == "103:1-3"

    def test_impossible_range_degrades_to_the_single_ayah(self):
        extraction = parse_citations('{"citations": [{"type": "quran", "surah": 103, "ayah_start": 2, "ayah_end": 1}]}')
        citation = extraction.citations[0]
        assert citation.ayah_start == 2
        assert citation.ayah_end is None

    def test_string_numbers_are_accepted(self):
        extraction = parse_citations('{"citations": [{"type": "quran", "surah": "2", "ayah_start": "255"}]}')
        assert extraction.citations[0].surah == 2


class TestHadithValidation:
    def test_collection_alias_is_canonicalised(self):
        extraction = parse_citations('{"citations": [{"type": "hadith", "collection": "bukhari", "number": "1"}]}')
        citation = extraction.citations[0]
        assert isinstance(citation, HadithCitation)
        assert citation.collection == "Sahih al-Bukhari"

    @pytest.mark.parametrize("alias", ["Sahih al-Bukhari", "sahih bukhari", "Bukhari", "al-bukhari"])
    def test_aliases_all_resolve(self, alias):
        extraction = parse_citations(
            json.dumps({"citations": [{"type": "hadith", "collection": alias, "number": "1"}]})
        )
        assert extraction.citations[0].collection == "Sahih al-Bukhari"

    def test_unknown_collection_is_rejected(self):
        extraction = parse_citations('{"citations": [{"type": "hadith", "collection": "Book of Made Up Things"}]}')
        assert extraction.citations == []
        assert extraction.attempted == 1

    def test_collection_without_a_number_is_still_accepted(self):
        extraction = parse_citations('{"citations": [{"type": "hadith", "collection": "Sahih Muslim"}]}')
        assert extraction.citations[0].number is None


class TestScholarlyReference:
    def test_work_is_required(self):
        extraction = parse_citations('{"citations": [{"type": "scholarly", "author": "X"}]}')
        assert extraction.citations == []

    def test_valid_reference(self):
        extraction = parse_citations(
            '{"citations": [{"type": "scholarly", "work": "Al-Muwafaqat", "author": "Al-Shatibi"}]}'
        )
        citation = extraction.citations[0]
        assert isinstance(citation, ScholarlyReference)
        assert citation.work == "Al-Muwafaqat"


class TestTypeInference:
    def test_missing_type_is_inferred(self):
        extraction = parse_citations(
            '{"citations": [{"surah": 1, "ayah_start": 1},'
            ' {"collection": "Muslim", "number": "1"},'
            ' {"work": "Al-Risala"}]}'
        )
        assert len(extraction.citations) == 3
        assert [c.type for c in extraction.citations] == ["quran", "hadith", "scholarly"]

    def test_unrecognisable_entry_is_rejected(self):
        extraction = parse_citations('{"citations": [{"nonsense": true}]}')
        assert extraction.citations == []


class TestScoring:
    def test_score_is_none_when_nothing_was_cited(self):
        assert CitationExtraction().score is None

    def test_score_is_the_valid_share(self):
        extraction = parse_citations(
            '{"citations": [{"type": "quran", "surah": 2, "ayah_start": 255},'
            ' {"type": "quran", "surah": 2, "ayah_start": 9999}]}'
        )
        assert extraction.attempted == 2
        assert len(extraction.citations) == 1
        assert extraction.score == 0.5

    def test_all_valid_scores_one(self):
        extraction = parse_citations('{"citations": [{"type": "quran", "surah": 1, "ayah_start": 1}]}')
        assert extraction.score == 1.0


class TestDeduplication:
    def test_identical_citations_collapse(self):
        extraction = parse_citations(
            '{"citations": [{"type": "quran", "surah": 2, "ayah_start": 255},'
            ' {"type": "quran", "surah": 2, "ayah_start": 255}]}'
        )
        assert len(extraction.citations) == 1


class TestCaps:
    def test_citation_count_is_capped(self):
        many = [{"type": "quran", "surah": 2, "ayah_start": n} for n in range(1, 60)]
        extraction = parse_citations(json.dumps({"citations": many}))
        assert extraction.attempted <= 24


class TestStreamFilter:
    def feed_all(self, chunks):
        stream = CitationStreamFilter()
        emitted = "".join(stream.feed(c) for c in chunks)
        remainder, extraction = stream.finish()
        return emitted + remainder, extraction

    def test_marker_never_reaches_the_client(self):
        text = answer_with('{"citations": [{"type": "quran", "surah": 1, "ayah_start": 1}]}')
        chunks = [text[i : i + 7] for i in range(0, len(text), 7)]
        emitted, extraction = self.feed_all(chunks)
        assert CITATION_BLOCK_START not in emitted
        assert CITATION_BLOCK_END not in emitted
        assert "Patience is enjoined." in emitted
        assert len(extraction.citations) == 1

    def test_marker_split_across_chunks_is_caught(self):
        text = answer_with('{"citations": []}')
        # One character at a time is the worst case for a split marker.
        emitted, _ = self.feed_all(list(text))
        assert CITATION_BLOCK_START not in emitted

    def test_stream_without_citations_passes_prose_through(self):
        text = "A plain answer with no block."
        emitted, extraction = self.feed_all([text])
        assert emitted == text
        assert extraction.citations == []

    def test_stream_truncated_mid_block(self):
        text = "Answer.\n\n" + CITATION_BLOCK_START + '\n{"citations": ['
        emitted, extraction = self.feed_all([text])
        assert CITATION_BLOCK_START not in emitted
        assert extraction.citations == []
