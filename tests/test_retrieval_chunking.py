"""Tests for retrieval.chunking — deterministic, content-type-aware splitting.

Fully offline: chunking never embeds or hits the network.
"""

import json
from pathlib import Path

import pytest

from retrieval.chunking import (
    Chunk,
    chunk_ayah,
    chunk_hadith,
    chunk_prose,
    chunk_surah,
    content_hash,
    iter_corpus_chunks,
    iter_hadith_chunks,
    iter_quran_chunks,
    iter_surah_index_chunks,
    make_chunk,
)

# ---------------------------------------------------------------------------
# content_hash + Chunk model
# ---------------------------------------------------------------------------


def test_content_hash_is_deterministic():
    assert content_hash("the same text") == content_hash("the same text")


def test_content_hash_ignores_whitespace_reflow_only():
    assert content_hash("a  b\tc\n d") == content_hash("a b c d")


def test_content_hash_changes_with_content():
    assert content_hash("2:255") != content_hash("2:256")


def test_content_hash_preserves_case_and_script():
    assert content_hash("Allah") != content_hash("allah")


def test_chunk_roundtrips_through_dict():
    chunk = make_chunk(
        source="quran",
        source_id="quran:2:255",
        text="Ayat al-Kursi",
        metadata={"surah": 2, "ayah": 255},
    )
    restored = Chunk.from_dict(json.loads(json.dumps(chunk.to_dict())))
    assert restored == chunk


def test_make_chunk_atomic_id_is_source_id():
    chunk = make_chunk(source="quran", source_id="quran:1:1", text="x")
    assert chunk.chunk_id == "quran:1:1"
    assert chunk.part == 0 and chunk.part_count == 1


def test_make_chunk_carries_scope_and_publish_fields():
    chunk = make_chunk(source="x", source_id="x:1", text="t", scope="private", published=False)
    assert chunk.scope == "private"
    assert chunk.published is False


# ---------------------------------------------------------------------------
# Atomic record chunkers
# ---------------------------------------------------------------------------


def test_chunk_ayah_is_atomic_with_reference_metadata():
    chunk = chunk_ayah(2, 255, {"arabic": "اللَّهُ", "english": "Allah - there is no deity except Him"})
    assert chunk.source == "quran"
    assert chunk.source_id == "quran:2:255"
    assert chunk.part_count == 1
    assert chunk.metadata["reference"] == "2:255"
    assert "2:255" in chunk.text


def test_chunk_hadith_renders_grading_record():
    chunk = chunk_hadith("bukhari", {"n": 1, "book": 1, "grade": "SAHIH", "chain": "MARFU"})
    assert chunk is not None
    assert chunk.source_id == "hadith:bukhari:1"
    assert chunk.metadata["grade"] == "SAHIH"
    assert "bukhari" in chunk.text


def test_chunk_hadith_skips_record_without_number():
    assert chunk_hadith("bukhari", {"grade": "SAHIH"}) is None


def test_chunk_surah_is_atomic():
    chunk = chunk_surah(
        {
            "number": 1,
            "name": "Al-Fatihah",
            "arabic_name": "الفاتحة",
            "revelation_place": "meccan",
            "ayah_count": 7,
            "aliases": ["fatiha", "opening"],
        }
    )
    assert chunk.source_id == "surah:1"
    assert "Al-Fatihah" in chunk.text
    assert chunk.metadata["ayah_count"] == 7


# ---------------------------------------------------------------------------
# Prose chunking (token budget + overlap)
# ---------------------------------------------------------------------------


def test_short_prose_is_a_single_atomic_chunk():
    chunks = chunk_prose("doc", "doc:1", "one two three", max_tokens=10, overlap=2)
    assert len(chunks) == 1
    assert chunks[0].chunk_id == "doc:1"
    assert chunks[0].part_count == 1


def test_long_prose_splits_with_overlap():
    text = " ".join(f"t{i}" for i in range(10))
    chunks = chunk_prose("doc", "doc:1", text, max_tokens=4, overlap=1)
    assert len(chunks) == 3
    # Stable, distinct chunk ids under the same source_id.
    assert [c.chunk_id for c in chunks] == ["doc:1#0", "doc:1#1", "doc:1#2"]
    assert all(c.source_id == "doc:1" for c in chunks)
    assert all(c.part_count == 3 for c in chunks)
    # Overlap: the last token of window 0 reappears as the first of window 1.
    assert chunks[0].text.split()[-1] == chunks[1].text.split()[0]


def test_prose_split_is_deterministic():
    text = " ".join(f"w{i}" for i in range(50))
    first = [c.content_hash for c in chunk_prose("d", "d:1", text, max_tokens=8, overlap=2)]
    second = [c.content_hash for c in chunk_prose("d", "d:1", text, max_tokens=8, overlap=2)]
    assert first == second


def test_prose_no_window_exceeds_budget():
    text = " ".join(f"w{i}" for i in range(37))
    for chunk in chunk_prose("d", "d:1", text, max_tokens=8, overlap=3):
        assert len(chunk.text.split()) <= 8


@pytest.mark.parametrize("overlap", [-1, 4, 5])
def test_prose_rejects_bad_overlap(overlap):
    with pytest.raises(ValueError):
        chunk_prose("d", "d:1", "a b c", max_tokens=4, overlap=overlap)


def test_prose_rejects_nonpositive_budget():
    with pytest.raises(ValueError):
        chunk_prose("d", "d:1", "a b c", max_tokens=0, overlap=0)


# ---------------------------------------------------------------------------
# Corpus iterators (deterministic order over a fixture corpus)
# ---------------------------------------------------------------------------


@pytest.fixture
def fixture_corpus(tmp_path: Path) -> Path:
    data = tmp_path / "data"
    (data / "quran").mkdir(parents=True)
    (data / "hadith").mkdir(parents=True)
    (data / "quran_uthmani.json").write_text(
        json.dumps(
            {
                "surahs": {"1": {"ayahs_count": 7, "name": "Al-Fatiha"}},
                "ayat": {
                    "1:2": {"arabic": "الْحَمْدُ", "english": "All praise"},
                    "1:1": {"arabic": "بِسْمِ", "english": "In the name"},
                },
            }
        ),
        encoding="utf-8",
    )
    (data / "quran" / "surah_index.json").write_text(
        json.dumps([{"number": 1, "name": "Al-Fatihah", "arabic_name": "الفاتحة", "ayah_count": 7}]),
        encoding="utf-8",
    )
    (data / "hadith" / "bukhari.json").write_text(
        json.dumps(
            {
                "collection": "bukhari",
                "hadiths": [
                    {"n": 2, "grade": "SAHIH", "chain": "MARFU"},
                    {"n": 1, "grade": "SAHIH", "chain": "MARFU"},
                ],
            }
        ),
        encoding="utf-8",
    )
    return data


def test_iter_quran_chunks_sorted_by_reference(fixture_corpus: Path):
    chunks = list(iter_quran_chunks(fixture_corpus / "quran_uthmani.json"))
    assert [c.source_id for c in chunks] == ["quran:1:1", "quran:1:2"]


def test_iter_hadith_chunks_sorted_by_number(fixture_corpus: Path):
    chunks = list(iter_hadith_chunks(fixture_corpus / "hadith"))
    assert [c.metadata["number"] for c in chunks] == [1, 2]


def test_iter_surah_index_chunks(fixture_corpus: Path):
    chunks = list(iter_surah_index_chunks(fixture_corpus / "quran" / "surah_index.json"))
    assert [c.source_id for c in chunks] == ["surah:1"]


def test_iter_corpus_chunks_is_deterministic(fixture_corpus: Path):
    first = [c.chunk_id for c in iter_corpus_chunks(fixture_corpus)]
    second = [c.chunk_id for c in iter_corpus_chunks(fixture_corpus)]
    assert first == second
    # Every source family is represented.
    assert {c.split(":")[0] for c in first} == {"surah", "quran", "hadith"}


def test_iter_corpus_chunks_missing_files_yield_nothing(tmp_path: Path):
    assert list(iter_corpus_chunks(tmp_path)) == []
