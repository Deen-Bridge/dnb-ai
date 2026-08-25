import uuid
from unittest.mock import AsyncMock, MagicMock

from fastapi.testclient import TestClient

import main
from main import CHAT_CONTEXT_MAX_LENGTH, CHAT_PROMPT_MAX_LENGTH, ChatRequest, app

client = TestClient(app)


def test_empty_and_whitespace_prompts_are_rejected_before_gemini(monkeypatch):
    get_model = MagicMock()
    monkeypatch.setattr(main, "get_model", get_model)

    for prompt in ("", "   \t\n"):
        response = client.post("/chat", json={"prompt": prompt})

        assert response.status_code == 422
    get_model.assert_not_called()


def test_prompt_whitespace_is_stripped_before_handler_use():
    request = ChatRequest(prompt="  A valid question  ")

    assert request.prompt == "A valid question"


def test_prompt_over_limit_is_rejected_before_gemini(monkeypatch):
    get_model = MagicMock()
    monkeypatch.setattr(main, "get_model", get_model)

    response = client.post("/chat", json={"prompt": "x" * (CHAT_PROMPT_MAX_LENGTH + 1)})

    assert response.status_code == 422
    get_model.assert_not_called()


def test_context_over_limit_is_rejected():
    response = client.post(
        "/chat",
        json={"prompt": "A valid question", "context": "x" * (CHAT_CONTEXT_MAX_LENGTH + 1)},
    )

    assert response.status_code == 422


def test_malformed_chat_id_is_rejected_before_gemini(monkeypatch):
    get_model = MagicMock()
    monkeypatch.setattr(main, "get_model", get_model)

    response = client.post("/chat", json={"prompt": "A valid question", "chat_id": "not-a-uuid"})

    assert response.status_code == 422
    get_model.assert_not_called()


def test_delete_unknown_chat_returns_404():
    response = client.delete(f"/chat/{uuid.uuid4()}")

    assert response.status_code == 404
    assert response.json() == {"detail": "Chat session not found"}


def test_delete_existing_in_memory_chat_returns_200():
    chat_id = str(uuid.uuid4())
    main.active_chats[chat_id] = object()

    response = client.delete(f"/chat/{chat_id}")

    assert response.status_code == 200
    assert response.json() == {"message": "Chat session deleted successfully"}
    assert chat_id not in main.active_chats


def test_delete_existing_persisted_chat_returns_200(monkeypatch):
    chat_id = uuid.uuid4()
    delete_session = AsyncMock(return_value=True)
    monkeypatch.setattr(main.session_store, "delete_session", delete_session)

    response = client.delete(f"/chat/{chat_id}")

    assert response.status_code == 200
    assert response.json() == {"message": "Chat session deleted successfully"}
    delete_session.assert_awaited_once_with(str(chat_id))


def test_openapi_documents_chat_request_constraints():
    schema = app.openapi()["components"]["schemas"]["ChatRequest"]["properties"]

    assert schema["prompt"]["minLength"] == 1
    assert schema["prompt"]["maxLength"] == CHAT_PROMPT_MAX_LENGTH
    assert schema["prompt"]["description"]
    assert schema["prompt"]["examples"]
    assert schema["chat_id"]["anyOf"][0]["format"] == "uuid"
    assert schema["chat_id"]["description"]
    assert schema["chat_id"]["examples"]
    assert schema["context"]["anyOf"][0]["maxLength"] == CHAT_CONTEXT_MAX_LENGTH
    assert schema["context"]["description"]
    assert schema["context"]["examples"]
