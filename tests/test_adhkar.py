import pytest
from fastapi.testclient import TestClient

from adhkar import AdhkarCorpus
from main import app

client = TestClient(app)


@pytest.fixture
def authenticated_corpus(monkeypatch):
    entries = [
        {
            "id": "morning_1",
            "category": "morning",
            "transliteration": "Asbahna",
            "translation": "We have reached the morning.",
            "source": "Hisnul Muslim",
            "grading": "authentic",
        },
        {
            "id": "distress_1",
            "category": "distress",
            "transliteration": "Allahumma inni",
            "translation": "I seek refuge from anxiety and grief.",
            "source": "Sahih al-Bukhari",
            "grading": "authentic",
        },
        {
            "id": "evening_1",
            "category": "evening",
            "transliteration": "Amsayna",
            "translation": "We have reached the evening.",
            "source": "Hisnul Muslim",
            "grading": "authentic",
        },
    ]
    corpus = AdhkarCorpus.__new__(AdhkarCorpus)
    corpus.data_file = None
    corpus.entries = entries
    monkeypatch.setattr("main.adhkar_corpus", corpus)
    return corpus


def test_search_matches_category_without_case_sensitivity(authenticated_corpus):
    matches = authenticated_corpus.search(category="MORNING")

    assert [entry["id"] for entry in matches] == ["morning_1"]


def test_search_combines_category_with_normalized_free_text(authenticated_corpus):
    matches = authenticated_corpus.search(category="distress", query="  ANXIETY AND GRIEF  ")

    assert [entry["id"] for entry in matches] == ["distress_1"]


def test_search_finds_authenticated_entry_by_transliteration(authenticated_corpus):
    matches = authenticated_corpus.search(query="asbahna")

    assert [entry["id"] for entry in matches] == ["morning_1"]
    assert all(entry["grading"] == "authentic" for entry in matches)


def test_search_with_empty_context_returns_all_authenticated_entries(authenticated_corpus):
    matches = authenticated_corpus.search(category="  ", query="  ")

    assert [entry["id"] for entry in matches] == ["morning_1", "distress_1", "evening_1"]


def test_search_unknown_context_returns_no_recommendations(authenticated_corpus):
    assert authenticated_corpus.search(category="travel") == []
    assert authenticated_corpus.search(query="a phrase absent from the corpus") == []


def test_recommend_endpoint_returns_matching_authenticated_entries(authenticated_corpus):
    response = client.post("/adhkar/recommend", json={"category": "morning"})

    assert response.status_code == 200
    payload = response.json()
    assert [entry["id"] for entry in payload["matches"]] == ["morning_1"]
    assert payload["matches"][0]["source"] == "Hisnul Muslim"
    assert payload["message"] == "Found 1 authenticated supplication(s) matching your request."


def test_recommend_endpoint_handles_unknown_context_without_fabricating_results(authenticated_corpus):
    response = client.post(
        "/adhkar/recommend",
        json={"category": "travel", "query": "unknown context"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["matches"] == []
    assert payload["message"] == "No authenticated supplication found matching your request."
