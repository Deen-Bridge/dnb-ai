"""Tests for scripts/build_index.py — the idempotent backfill job.

Fully offline via ``--fake-embeddings`` / the deterministic embedder, so the
backfill runs in CI with no network access and no API key.
"""

import json
from pathlib import Path

import numpy as np

from retrieval.index import SQLiteVectorStore
from scripts.build_index import build_index, deterministic_embedding, main


def _write_corpus(root: Path) -> Path:
    data = root / "data"
    (data / "quran").mkdir(parents=True)
    (data / "hadith").mkdir(parents=True)
    (data / "quran_uthmani.json").write_text(
        json.dumps(
            {
                "surahs": {},
                "ayat": {"1:1": {"arabic": "بِسْمِ", "english": "In the name"}},
            }
        ),
        encoding="utf-8",
    )
    (data / "quran" / "surah_index.json").write_text(
        json.dumps([{"number": 1, "name": "Al-Fatihah", "ayah_count": 7}]),
        encoding="utf-8",
    )
    (data / "hadith" / "bukhari.json").write_text(
        json.dumps(
            {
                "collection": "bukhari",
                "hadiths": [{"n": 1, "grade": "SAHIH", "chain": "MARFU"}],
            }
        ),
        encoding="utf-8",
    )
    return data


def test_deterministic_embedding_is_stable_and_shaped():
    a = deterministic_embedding("some text")
    b = deterministic_embedding("some text")
    assert np.array_equal(a, b)
    assert a.dtype == np.float32
    assert deterministic_embedding("some text").shape == deterministic_embedding("other").shape
    assert not np.array_equal(deterministic_embedding("x"), deterministic_embedding("y"))


def test_backfill_indexes_every_corpus(tmp_path: Path):
    data = _write_corpus(tmp_path)
    store = SQLiteVectorStore(tmp_path / "idx.db")
    result = build_index(store, data_dir=data, fake_embeddings=True)
    # 1 surah + 1 ayah + 1 hadith = 3 chunks.
    assert result.added == 3
    assert store.count() == 3
    assert store.source_ids() == {"surah:1", "quran:1:1", "hadith:bukhari:1"}


def test_backfill_is_idempotent(tmp_path: Path):
    """AC: running the backfill twice yields identical contents and counts."""
    data = _write_corpus(tmp_path)
    store = SQLiteVectorStore(tmp_path / "idx.db")

    first = build_index(store, data_dir=data, fake_embeddings=True)
    snapshot = sorted(store.hashes().items())
    count = store.count()

    second = build_index(store, data_dir=data, fake_embeddings=True)
    assert first.added == second.unchanged  # everything now unchanged
    assert second.embedded == 0
    assert not second.changed
    assert store.count() == count
    assert sorted(store.hashes().items()) == snapshot


def test_backfill_two_fresh_stores_match(tmp_path: Path):
    """Idempotency across processes: two clean builds produce the same index."""
    data = _write_corpus(tmp_path)
    store_a = SQLiteVectorStore(tmp_path / "a.db")
    store_b = SQLiteVectorStore(tmp_path / "b.db")
    build_index(store_a, data_dir=data, fake_embeddings=True)
    build_index(store_b, data_dir=data, fake_embeddings=True)
    assert store_a.hashes() == store_b.hashes()
    assert store_a.count() == store_b.count()


def test_backfill_survives_restart(tmp_path: Path):
    """AC: the persisted index survives a process restart."""
    data = _write_corpus(tmp_path)
    db = tmp_path / "idx.db"
    build_index(SQLiteVectorStore(db), data_dir=data, fake_embeddings=True)

    # New store object on the same file == a restart.
    reopened = SQLiteVectorStore(db)
    assert reopened.count() == 3


def test_main_entrypoint_runs_offline(tmp_path: Path, capsys):
    data = _write_corpus(tmp_path)
    db = tmp_path / "idx.db"
    rc = main(["--index-path", str(db), "--data-dir", str(data), "--fake-embeddings"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Backfill complete" in out
    assert SQLiteVectorStore(db).count() == 3
