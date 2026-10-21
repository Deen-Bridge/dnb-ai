"""Unit tests for Quranic Vocabulary Analysis module (vocabulary.py)."""

from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import vocabulary
from vocabulary import (
    compute_frequencies,
    extract_root,
    find_by_meaning,
    find_by_root,
    find_by_word,
    get_all_entries,
    get_example_verses,
    group_by_root,
    router,
    strip_tashkeel,
)

# Setup test app with vocabulary router
app = FastAPI()
app.include_router(router)
client = TestClient(app)

MOCK_ENTRIES = [
    {"word": "كِتَابٌ", "meaning": "Book", "root": "كتب"},
    {"word": "الرَّحْمَنِ", "meaning": "The Most Gracious", "root": "رحم"},
    {"word": "عَلِمُوا", "meaning": "They knew", "root": "علم"},
]

MOCK_AYAT = {
    "1:1": {"text": "بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ"},
    "2:2": {"text": "ذَٰلِكَ الْكِتَابُ لَا رَيْبَ ۛ فِيهِ"},
    "2:3": {"text": "الَّذِينَ يُؤْمِنُونَ بِالْغَيْبِ"},
}


class TestTashkeelAndRoot:
    def test_strip_tashkeel_with_diacritics(self):
        text = "بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ"
        stripped = strip_tashkeel(text)
        assert stripped == "بسم الله الرحمن الرحيم"

    def test_strip_tashkeel_plain_text(self):
        assert strip_tashkeel("كتاب") == "كتاب"
        assert strip_tashkeel("") == ""

    def test_extract_root_prefixes(self):
        # Strip definite article "ال"
        assert extract_root("الكتاب") == "كتاب"
        # Strip prefix "و" leaving remaining string
        assert extract_root("والشمس") == "الشمس"

    def test_extract_root_suffixes(self):
        # Strip pronouns and plural markers
        assert extract_root("قلوبهم") == "قلوب"
        assert extract_root("يعلمون") == "يعلم"
        assert extract_root("صابرين") == "صابر"

    def test_extract_root_with_tashkeel(self):
        assert extract_root("الْكِتَابُ") == "كتاب"

    def test_extract_root_short_word(self):
        # Short words should not have characters stripped
        assert extract_root("من") == "من"
        assert extract_root("في") == "في"


class TestVocabularyHelpers:
    @pytest.fixture(autouse=True)
    def mock_vocab_data(self):
        with patch.dict(vocabulary.VOCAB_DATA, {"entries": MOCK_ENTRIES}, clear=True):
            yield

    def test_get_all_entries(self):
        entries = get_all_entries()
        assert len(entries) == 3
        assert entries[0]["meaning"] == "Book"

    def test_find_by_word_found(self):
        entry = find_by_word("كتاب")
        assert entry is not None
        assert entry["meaning"] == "Book"

    def test_find_by_word_with_tashkeel(self):
        entry = find_by_word("كِتَابٌ")
        assert entry is not None
        assert entry["meaning"] == "Book"

    def test_find_by_word_not_found(self):
        assert find_by_word("كلمة_غير_موجودة") is None

    def test_find_by_meaning(self):
        results = find_by_meaning("book")
        assert len(results) == 1
        assert results[0]["word"] == "كِتَابٌ"

        # Case insensitive partial match
        results_gracious = find_by_meaning("GRACIOUS")
        assert len(results_gracious) == 1
        assert results_gracious[0]["word"] == "الرَّحْمَنِ"

    def test_find_by_root(self):
        # extract_root("الرَّحْمَنِ") -> "رحمن"
        results = find_by_root("رحمن")
        assert len(results) >= 1
        assert results[0]["meaning"] == "The Most Gracious"

    def test_group_by_root(self):
        families = group_by_root()
        assert isinstance(families, dict)
        assert len(families) > 0


class TestCorpusHelpers:
    def test_count_word_in_corpus(self):
        with patch.object(vocabulary.corpus, "ayat", MOCK_AYAT):
            count = vocabulary._count_word_in_corpus("الرحمن")
            assert count >= 1

    def test_compute_frequencies(self):
        with (
            patch.dict(vocabulary.VOCAB_DATA, {"entries": MOCK_ENTRIES}, clear=True),
            patch.object(vocabulary.corpus, "ayat", MOCK_AYAT),
        ):
            freqs = compute_frequencies()
            assert len(freqs) == 3
            assert all("frequency" in e for e in freqs)

    def test_get_example_verses(self):
        with patch.object(vocabulary.corpus, "ayat", MOCK_AYAT):
            verses = get_example_verses("الكتاب", limit=5)
            assert len(verses) >= 1
            assert verses[0]["surah"] == 2
            assert verses[0]["ayah"] == 2
            assert "الكتاب" in strip_tashkeel(verses[0]["text"])

    def test_get_example_verses_limit(self):
        with patch.object(vocabulary.corpus, "ayat", MOCK_AYAT):
            verses = get_example_verses("الله", limit=1)
            assert len(verses) <= 1


class TestAPIRoutes:
    @pytest.fixture(autouse=True)
    def mock_vocab_and_corpus(self):
        with (
            patch.dict(vocabulary.VOCAB_DATA, {"entries": MOCK_ENTRIES}, clear=True),
            patch.object(vocabulary.corpus, "ayat", MOCK_AYAT),
        ):
            yield

    def test_search_auto(self):
        res = client.get("/vocabulary/search?query=book&search_type=auto")
        assert res.status_code == 200
        data = res.json()
        assert data["total"] == 1
        assert data["results"][0]["meaning"] == "Book"

    def test_search_meaning(self):
        res = client.get("/vocabulary/search?query=gracious&search_type=meaning")
        assert res.status_code == 200
        data = res.json()
        assert data["total"] == 1
        assert data["results"][0]["word"] == "الرَّحْمَنِ"

    def test_search_word(self):
        res = client.get("/vocabulary/search?query=كتاب&search_type=word")
        assert res.status_code == 200
        data = res.json()
        assert data["total"] == 1

    def test_search_root(self):
        res = client.get("/vocabulary/search?query=رحمن&search_type=root")
        assert res.status_code == 200
        data = res.json()
        assert data["total"] >= 1

    def test_frequencies_endpoint(self):
        res = client.get("/vocabulary/frequencies")
        assert res.status_code == 200
        data = res.json()
        assert data["total"] == 3
        assert len(data["entries"]) == 3

    def test_roots_endpoint(self):
        res = client.get("/vocabulary/roots")
        assert res.status_code == 200
        data = res.json()
        assert data["total_roots"] > 0
        assert isinstance(data["families"], dict)

    def test_verse_examples_endpoint(self):
        res = client.get("/vocabulary/verse-examples?word=الكتاب&limit=2")
        assert res.status_code == 200
        data = data = res.json()
        assert data["word"] == "الكتاب"
        assert len(data["verses"]) >= 1

    def test_get_word_detail_success(self):
        res = client.get("/vocabulary/word/كتاب")
        assert res.status_code == 200
        data = res.json()
        assert data["meaning"] == "Book"
        assert "frequency" in data
        assert "verses" in data

    def test_get_word_detail_not_found(self):
        res = client.get("/vocabulary/word/unknown_word_xyz")
        assert res.status_code == 404
