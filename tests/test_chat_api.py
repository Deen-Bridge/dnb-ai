from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

import main
from main import app

client = TestClient(app)


@dataclass
class FakePart:
    text: str


@dataclass
class FakeContent:
    role: str
    text: str

    @property
    def parts(self) -> list[FakePart]:
        return [FakePart(self.text)]


class FakeChatSession:
    def __init__(self) -> None:
        self.history: list[FakeContent] = []
        self.messages: list[str] = []

    async def send_message_async(self, message: str, **kwargs) -> SimpleNamespace:
        self.messages.append(message)
        answer = f"Mock answer {len(self.messages)}"
        self.history.extend(
            [
                FakeContent(role="user", text=message),
                FakeContent(role="model", text=answer),
            ]
        )
        return SimpleNamespace(
            text=answer,
            candidates=[SimpleNamespace(finish_reason="STOP")],
            prompt_feedback=None,
        )


class FakeModel:
    def __init__(self) -> None:
        self.sessions: list[FakeChatSession] = []

    def start_chat(self, history=None) -> FakeChatSession:
        session = FakeChatSession()
        if history:
            session.history.extend(
                FakeContent(
                    role=content.role,
                    text=content.parts[0].text if content.parts else "",
                )
                for content in history
            )
        self.sessions.append(session)
        return session


@pytest.fixture
def fake_model(monkeypatch):
    model = FakeModel()
    monkeypatch.setattr(main, "get_model", lambda: model)
    monkeypatch.setenv("SAFETY_PIPELINE_ENABLED", "false")
    monkeypatch.setattr(main, "active_chats", {})
    monkeypatch.setattr(main, "SEMANTIC_CACHE_ENABLED", False)
    monkeypatch.setattr(main, "tafsir_retriever", _empty_retriever)
    monkeypatch.setattr(main, "zakat_retriever", _empty_retriever)
    monkeypatch.setattr(main, "purchase_retriever", _empty_retriever)
    monkeypatch.setattr(main, "personal_context_retriever", _empty_retriever)
    monkeypatch.setattr(main, "enqueue_for_review", _empty_enqueue)
    return model


async def _empty_retriever(*args, **kwargs):
    return None


async def _empty_enqueue(*args, **kwargs):
    return None


def test_ping_returns_ok():
    response = client.get("/ping")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_chat_happy_path_preserves_history_for_same_session(fake_model):
    chat_id = str(uuid4())
    first = client.post("/chat", json={"prompt": "What is salah?", "chat_id": chat_id})
    second = client.post("/chat", json={"prompt": "How many prayers are there?", "chat_id": chat_id})

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["response"].startswith("Mock answer 1")
    assert second.json()["response"].startswith("Mock answer 2")
    assert len(first.json()["history"]) == 2
    assert len(second.json()["history"]) == 4
    assert len(fake_model.sessions) == 1
    assert len(fake_model.sessions[0].messages) == 2
    assert "How many prayers are there?" in fake_model.sessions[0].messages[1]


def test_chat_generates_new_id_and_reuses_existing_session(fake_model):
    first = client.post("/chat", json={"prompt": "First question"})
    first_id = first.json()["chat_id"]

    second = client.post("/chat", json={"prompt": "Follow up", "chat_id": first_id})
    third = client.post("/chat", json={"prompt": "Separate question"})

    assert first.status_code == 200
    assert second.status_code == 200
    assert third.status_code == 200
    assert second.json()["chat_id"] == first_id
    assert third.json()["chat_id"] != first_id
    assert len(fake_model.sessions) == 2


def test_delete_chat_removes_existing_session(fake_model):
    response = client.post("/chat", json={"prompt": "Start a session"})
    chat_id = response.json()["chat_id"]

    assert chat_id in main.active_chats

    deleted = client.delete(f"/chat/{chat_id}")

    assert deleted.status_code == 200
    assert deleted.json() == {"message": "Chat session deleted successfully"}
    assert chat_id not in main.active_chats


def test_chat_rejects_empty_prompt_and_missing_prompt_without_calling_gemini(monkeypatch):
    get_model = MagicMock()
    monkeypatch.setattr(main, "get_model", get_model)

    empty_prompt = client.post("/chat", json={"prompt": ""})
    missing_prompt = client.post("/chat", json={})

    assert empty_prompt.status_code == 422
    assert missing_prompt.status_code == 422
    get_model.assert_not_called()


def test_get_model_uses_islamic_prompt_and_safety_settings(monkeypatch):
    generative_model = MagicMock()
    monkeypatch.setattr(main.genai, "GenerativeModel", generative_model)

    main.get_model()

    generative_model.assert_called_once_with(
        model_name=main.settings.model_name,
        system_instruction=main.ISLAMIC_CONTEXT,
        safety_settings=main.get_safety_settings(),
    )
