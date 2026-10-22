"""Tests for the Quranic vocabulary module."""

import asyncio

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import vocabulary
from errors import APIException
from vocabulary import (
    extract_root,
    find_by_meaning,
    find_by_root,
    find_by_word,
    get_word_detail,
    router,
    strip_tashkeel,
)

_KITAB = "\u0643\u062a\u0627\u0628"
_KATABA = "\u0643\u062a\u0628"

_SAMPLE_VOCAB = {
    "entries": [
        {"word": _KATABA, "meaning": "to write", "root": _KATABA},
        {"word": _KITAB, "meaning": "book", "root": _KATABA},
    ]
}


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setattr(vocabulary, "VOCAB_DATA", _SAMPLE_VOCAB)
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def test_strip_tashkeel_removes_diacritics() -> None:
    vowelled = "\u0643\u064e\u062a\u064e\u0628\u064e"

    assert strip_tashkeel(vowelled) == _KATABA


def test_extract_root_keeps_a_triliteral_root() -> None:
    assert extract_root("\u0643\u064e\u062a\u064e\u0628\u064e") == _KATABA


def test_find_by_root_returns_matching_entries(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(vocabulary, "VOCAB_DATA", _SAMPLE_VOCAB)

    results = find_by_root(_KATABA)

    assert len(results) == 1
    assert results[0]["word"] == _KATABA
    assert find_by_root("\u0636\u0631\u0628") == []


def test_find_by_meaning_and_word(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(vocabulary, "VOCAB_DATA", _SAMPLE_VOCAB)

    assert find_by_meaning("book")[0]["word"] == _KITAB
    assert find_by_meaning("BOOK")[0]["word"] == _KITAB
    match = find_by_word(_KITAB)
    assert match is not None
    assert match["meaning"] == "book"
    assert find_by_word("missing") is None


def test_word_detail_raises_for_unknown_word(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(vocabulary, "VOCAB_DATA", _SAMPLE_VOCAB)

    with pytest.raises(APIException):
        asyncio.run(get_word_detail("missing"))


def test_search_endpoint_filters_by_meaning(client: TestClient) -> None:
    response = client.get("/vocabulary/search", params={"query": "book", "search_type": "meaning"})

    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 1
    assert body["results"][0]["word"] == _KITAB


def test_search_endpoint_auto_merges_word_and_root(client: TestClient) -> None:
    response = client.get("/vocabulary/search", params={"query": _KATABA})

    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 1
    assert body["results"][0]["word"] == _KATABA


def test_frequencies_endpoint_returns_sorted_entries(client: TestClient) -> None:
    response = client.get("/vocabulary/frequencies")

    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 2
    frequencies = [entry["frequency"] for entry in body["entries"]]
    assert frequencies == sorted(frequencies, reverse=True)


def test_roots_endpoint_groups_entries(client: TestClient) -> None:
    response = client.get("/vocabulary/roots")

    assert response.status_code == 200
    body = response.json()
    assert body["total_roots"] >= 1
    assert _KATABA in body["families"]


def test_verse_examples_endpoint_handles_unknown_word(client: TestClient) -> None:
    response = client.get("/vocabulary/verse-examples", params={"word": "zzzz"})

    assert response.status_code == 200
    assert response.json()["verses"] == []


def test_word_detail_endpoint_returns_entry(client: TestClient) -> None:
    response = client.get(f"/vocabulary/word/{_KATABA}")

    assert response.status_code == 200
    body = response.json()
    assert body["word"] == _KATABA
    assert body["root"] == _KATABA
    assert body["meaning"] == "to write"
