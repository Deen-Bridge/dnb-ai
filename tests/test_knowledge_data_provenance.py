"""Provenance and integrity of the bundled public-knowledge datasets (#848).

Every public-knowledge output must be traceable to versioned data. These
checks fail when a dataset loses its version or source metadata, when a second
copy of reference data drifts from the canonical one, or when text is stored
mis-encoded. Offline: they read files under data/ and never import the app.
"""

from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path

import pytest

from memorization_planner import _SURAH_DATA
from tafsir import load_surah_index, surah_by_name

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

# UTF-8 text decoded as Latin-1/cp1252 ("Ø§Ù„..." for Arabic, "ðŸ..." for emoji)
# plus stray C1 control characters, which never belong in user-facing text.
_MOJIBAKE = re.compile("[\u0080-\u009f]|Ã[\u0080-¿]|Ø[§-¿]|Ù[\u0080-\u008a]|ðŸ|â€")
_ARABIC_NAME = re.compile(r"^[؀-ۿ\s]+$")

KNOWLEDGE_MODULES = (
    "hadith.py",
    "hadith_context.py",
    "hadith_research.py",
    "tafsir.py",
    "concordance.py",
    "faraid.py",
    "study.py",
    "study_store.py",
    "memorization_planner.py",
    "public_knowledge.py",
)


def _load(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


# ---------------------------------------------------------------------------
# Canonical Qur'an reference data
# ---------------------------------------------------------------------------


def test_surah_index_is_complete() -> None:
    index = load_surah_index()
    assert [s.number for s in index] == list(range(1, 115))
    assert sum(s.ayah_count for s in index) == 6236
    assert all(_ARABIC_NAME.match(s.arabic_name) for s in index)


def test_memorization_planner_surahs_match_the_canonical_index() -> None:
    index = {s.number: s for s in load_surah_index()}
    assert sorted(_SURAH_DATA) == sorted(index)
    for number, info in _SURAH_DATA.items():
        canonical = index[number]
        assert info.arabic_name == canonical.arabic_name, number
        assert info.ayah_count == canonical.ayah_count, number
        assert info.revelation_place == canonical.revelation_place, number
        resolved = surah_by_name(info.name)
        assert resolved is not None and resolved.number == number, info.name


# ---------------------------------------------------------------------------
# Dataset version and source metadata
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", sorted((DATA / "hadith").glob("*.json")), ids=lambda p: p.name)
def test_hadith_datasets_carry_versioned_provenance(path: Path) -> None:
    payload = _load(path)
    if path.name == "variants.json":
        assert payload.get("version") and payload.get("provenance")
        return
    source = payload["source"]
    assert source["repo"] == "fawazahmed0/hadith-api"
    assert source["ref"], "upstream ref must be pinned"
    date.fromisoformat(source["fetched"])


def test_curated_datasets_declare_a_version_and_source() -> None:
    themes = _load(DATA / "theme_verses.meta.json")
    assert themes["dataset_version"] and themes["source"]

    adhkar = _load(DATA / "adhkar.json")
    assert adhkar["version"] and adhkar["source_description"]
    date.fromisoformat(adhkar["last_updated"])

    rijal = _load(DATA / "rijal" / "narrators.json")
    assert rijal["version"] and rijal["provenance"]


@pytest.mark.parametrize(
    "path",
    [DATA / "hadith" / "PROVENANCE.md", DATA / "quran" / "PROVENANCE.md", DATA / "adhkar" / "PROVENANCE.md"],
    ids=lambda p: str(p.relative_to(DATA)),
)
def test_provenance_documents_exist(path: Path) -> None:
    assert path.read_text(encoding="utf-8").strip()


# ---------------------------------------------------------------------------
# Encoding integrity of Arabic and other user-facing text
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path",
    sorted(p for p in DATA.rglob("*") if p.suffix in {".json", ".md"} and "eval" not in p.parts),
    ids=lambda p: str(p.relative_to(DATA)),
)
def test_bundled_data_has_no_mojibake(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    assert not _MOJIBAKE.search(text), path


@pytest.mark.parametrize("module", KNOWLEDGE_MODULES)
def test_knowledge_modules_have_no_mojibake(module: str) -> None:
    text = (ROOT / module).read_text(encoding="utf-8")
    assert not _MOJIBAKE.search(text), module
