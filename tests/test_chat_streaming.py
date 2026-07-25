import pytest
from fastapi.testclient import TestClient

import main


class FakeChunk:
    def __init__(self, text: str):
        self.text = text


class FakeChatSession:
    def __init__(self):
        self.history = []

    async def send_message_async(self, message, stream=True):
        self.history.append({"role": "user", "text": message})

        async def generator():
            yield FakeChunk("Hello")
            yield FakeChunk(" world")

        return generator()


class FailingChatSession(FakeChatSession):
    async def send_message_async(self, message, stream=True):
        self.history.append({"role": "user", "text": message})

        async def generator():
            yield FakeChunk("Hello")
            raise RuntimeError("upstream failed")

        return generator()


@pytest.fixture(autouse=True)
def reset_app_state(monkeypatch):
    main.sessions.clear()
    monkeypatch.setattr(main, "GEMINI_API_KEY", "fake-key")
    monkeypatch.setattr(main, "CITATION_VERIFY_MODE", "off")
    monkeypatch.setattr(main, "get_model", lambda: None)


def test_chat_stream_endpoint_emits_sse_events(monkeypatch):
    chat_session = FakeChatSession()
    monkeypatch.setattr(main, "get_model", lambda: type("Model", (), {"start_chat": lambda self, history=[]: chat_session})())

    client = TestClient(main.app)
    response = client.post("/chat/stream", json={"message": "Hello", "chat_id": "stream-test"})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "event: delta" in response.text
    assert "Hello" in response.text
    assert "event: done" in response.text
    assert '"chat_id": "stream-test"' in response.text


def test_chat_stream_endpoint_emits_error_event_on_upstream_failure(monkeypatch):
    chat_session = FailingChatSession()
    monkeypatch.setattr(main, "get_model", lambda: type("Model", (), {"start_chat": lambda self, history=[]: chat_session})())

    client = TestClient(main.app)
    response = client.post("/chat/stream", json={"message": "Hello", "chat_id": "broken-stream"})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "event: error" in response.text
    assert "upstream failed" in response.text
