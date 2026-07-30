from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_recommends_by_category():
    response = client.post(
        "/adhkar/recommend",
        json={"category": "morning"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["matches"]
    assert payload["matches"][0]["category"] == "morning"
    assert payload["matches"][0]["source"] == "Hisnul Muslim"


def test_recommends_by_free_text_query():
    response = client.post(
        "/adhkar/recommend",
        json={"query": "anxiety and grief"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["matches"]
    assert payload["matches"][0]["category"] == "distress"


def test_returns_no_match_message_when_nothing_matches():
    response = client.post(
        "/adhkar/recommend",
        json={"category": "travel", "query": "nothing here"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["matches"] == []
    assert "No authenticated supplication" in payload["message"]
