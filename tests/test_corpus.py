"""Tests for the Quran corpus loader, lookups, and empty/missing-entry handling.

Every test runs offline against temporary fixtures or the small bundled
``data/quran_uthmani.json``; no network access is required.
"""

import json
from pathlib import Path

from corpus import DATA_PATH, QuranCorpus


def _write_json(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_bundled_corpus_loads_surahs_and_ayat():
    corpus = QuranCorpus(data_file=DATA_PATH, relationship_file=DATA_PATH.with_name("missing.json"))

    assert corpus.get_surah_info(1) == {"name": "Al-Fatiha", "ayahs_count": 7}
    assert corpus.get_ayah_count(2) == 286
    assert corpus.get_ayah(1, 1)["english"].startswith("In the name of Allah")
    assert corpus.has_hadith_corpus() is False


def test_unknown_surah_and_ayah_return_none():
    corpus = QuranCorpus(data_file=DATA_PATH, relationship_file=DATA_PATH.with_name("missing.json"))

    assert corpus.get_surah_info(999) is None
    assert corpus.get_ayah_count(999) is None
    assert corpus.get_ayah(1, 999) is None


def test_missing_corpus_file_leaves_empty_corpus(tmp_path):
    corpus = QuranCorpus(data_file=tmp_path / "absent.json", relationship_file=tmp_path / "relationships.json")

    assert corpus.surahs == {}
    assert corpus.ayat == {}
    assert corpus.get_surah_info(1) is None
    assert corpus.get_ayah(1, 1) is None


def test_empty_json_object_is_treated_as_empty_corpus(tmp_path):
    data_file = _write_json(tmp_path / "empty.json", {})

    corpus = QuranCorpus(data_file=data_file, relationship_file=tmp_path / "relationships.json")

    assert corpus.surahs == {}
    assert corpus.ayat == {}
    assert corpus.get_ayah_count(1) is None


def test_missing_relationship_file_yields_empty_graph(tmp_path):
    corpus = QuranCorpus(data_file=DATA_PATH, relationship_file=tmp_path / "relationships.json")

    assert corpus.relationships == {}
    assert corpus.get_related_ayah(1, 1) == []
    assert corpus.get_relationship_graph() == {}


def test_relationships_are_loaded_and_filtered(tmp_path):
    relationship_file = _write_json(
        tmp_path / "relationships.json",
        {
            "1:1": [
                {"target": "1:2", "type": "parallel", "strength": 0.9},
                {"target": "1:3", "type": "elaboration", "strength": 0.4},
            ]
        },
    )
    corpus = QuranCorpus(data_file=DATA_PATH, relationship_file=relationship_file)

    everything = corpus.get_related_ayah(1, 1)
    assert {entry["target"] for entry in everything} == {"1:2", "1:3"}
    assert all(entry["source"] == "1:1" for entry in everything)

    assert [entry["target"] for entry in corpus.get_related_ayah(1, 1, relationship_type="parallel")] == ["1:2"]
    assert [entry["target"] for entry in corpus.get_related_ayah(1, 1, min_strength=0.5)] == ["1:2"]
    assert corpus.get_related_ayah(1, 1, relationship_type="parallel", min_strength=0.95) == []


def test_build_relationships_writes_symmetric_edges(tmp_path):
    data_file = _write_json(
        tmp_path / "corpus.json",
        {
            "surahs": {},
            "ayat": {
                "1:1": {"text": "salat zakat rahmah"},
                "1:2": {"text": "salat zakat iman"},
            },
        },
    )
    relationship_file = tmp_path / "relationships.json"
    corpus = QuranCorpus(data_file=data_file, relationship_file=relationship_file)

    corpus.build_relationships(threshold=0.2)

    saved = json.loads(relationship_file.read_text(encoding="utf-8"))
    assert saved["1:1"] == [{"target": "1:2", "type": "parallel", "strength": 0.5}]
    assert saved["1:2"] == [{"target": "1:1", "type": "parallel", "strength": 0.5}]


def test_scholarly_notes_are_per_ayah_and_default_to_empty():
    corpus = QuranCorpus(data_file=DATA_PATH, relationship_file=DATA_PATH.with_name("missing.json"))

    assert corpus.get_scholarly_notes(1, 1) == []

    corpus.add_scholarly_note(1, 1, "First note")
    corpus.add_scholarly_note(1, 2, "Other ayah")

    assert corpus.get_scholarly_notes(1, 1) == ["First note"]
    assert corpus.get_scholarly_notes(1, 2) == ["Other ayah"]
    assert corpus.get_scholarly_notes(2, 255) == []
