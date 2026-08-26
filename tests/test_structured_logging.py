"""Issue #11: structured JSON logs, request correlation, and prompt redaction.

Everything here goes through the real ASGI stack — the app's own middleware,
the real /chat and /chat/stream handlers — because the properties under test
(one id per request, no prompt text, a traceback on failure) only mean anything
end to end. Records are captured by attaching the production formatter and
filter to the root logger, so what is asserted is the exact bytes a deployment
would write to stdout.
"""

import io
import json
import logging
from dataclasses import dataclass
from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

import logging_config
import main
from logging_config import REQUEST_ID_HEADER
from main import app

client = TestClient(app)

# Distinctive enough that a substring search for it is conclusive.
SECRET_PROMPT = "Is my inheritance of 40 grams of gold zakatable this year"


# ---------------------------------------------------------------------------
# Log capture
# ---------------------------------------------------------------------------
class LogCapture:
    def __init__(self, stream: io.StringIO) -> None:
        self._stream = stream

    @property
    def text(self) -> str:
        return self._stream.getvalue()

    def lines(self) -> list[str]:
        return [line for line in self._stream.getvalue().splitlines() if line.strip()]

    def records(self) -> list[dict]:
        """Every captured line, parsed. Fails loudly on non-JSON output."""
        parsed = []
        for line in self.lines():
            try:
                parsed.append(json.loads(line))
            except json.JSONDecodeError as exc:  # pragma: no cover - assertion aid
                raise AssertionError(f"log line is not valid JSON: {line!r}") from exc
        return parsed

    def request_records(self) -> list[dict]:
        """Records emitted while serving a request (id bound), server logs aside."""
        return [r for r in self.records() if r["request_id"] != logging_config.NO_REQUEST_ID]

    def find(self, message: str) -> dict:
        matches = [r for r in self.records() if r["message"] == message]
        assert matches, f"no {message!r} record in: {[r['message'] for r in self.records()]}"
        return matches[0]


@pytest.fixture
def json_logs():
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging_config.JsonFormatter())
    handler.addFilter(logging_config.RequestIdFilter())

    root = logging.getLogger()
    previous_level = root.level
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    # httpx narrates the TestClient's own call from outside the request; it is
    # not part of what the service emits.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    try:
        yield LogCapture(stream)
    finally:
        root.removeHandler(handler)
        root.setLevel(previous_level)


# ---------------------------------------------------------------------------
# A model that answers without touching the network
# ---------------------------------------------------------------------------
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

    async def send_message_async(self, message: str, **kwargs):
        answer = "Zakat is due at 2.5% once the nisab is met."
        self.history.extend([FakeContent("user", message), FakeContent("model", answer)])
        if kwargs.get("stream"):

            async def chunks():
                for piece in ("Zakat is due ", "at 2.5% once ", "the nisab is met."):
                    yield SimpleNamespace(text=piece)

            return chunks()
        return SimpleNamespace(
            text=answer,
            candidates=[SimpleNamespace(finish_reason="STOP")],
            prompt_feedback=None,
        )


class FakeModel:
    def start_chat(self, history=None) -> FakeChatSession:
        return FakeChatSession()


async def _empty(*args, **kwargs):
    return None


@pytest.fixture
def fake_model(monkeypatch):
    model = FakeModel()
    monkeypatch.setattr(main, "get_model", lambda: model)
    monkeypatch.setenv("SAFETY_PIPELINE_ENABLED", "false")
    monkeypatch.setattr(main, "active_chats", {})
    monkeypatch.setattr(main, "SEMANTIC_CACHE_ENABLED", False)
    for name in (
        "tafsir_retriever",
        "zakat_retriever",
        "purchase_retriever",
        "personal_context_retriever",
        "enqueue_for_review",
    ):
        monkeypatch.setattr(main, name, _empty)
    return model


# ---------------------------------------------------------------------------
# JSON output
# ---------------------------------------------------------------------------
def test_every_log_line_is_valid_json_with_the_expected_envelope(json_logs, fake_model):
    response = client.post("/chat", json={"prompt": SECRET_PROMPT})

    assert response.status_code == 200
    records = json_logs.records()
    assert records, "the request should have logged something"
    for record in records:
        assert set(record) >= {"timestamp", "level", "logger", "message", "request_id"}
        assert record["level"] in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        assert isinstance(record["message"], str)


def test_extra_fields_are_promoted_to_top_level_keys(json_logs, fake_model):
    client.post("/chat", json={"prompt": SECRET_PROMPT})

    stage = json_logs.find("stage completed")
    assert stage["stage"]
    assert isinstance(stage["duration_ms"], (int, float))


# ---------------------------------------------------------------------------
# Request correlation
# ---------------------------------------------------------------------------
def test_one_request_id_spans_every_record_and_the_response_header(json_logs, fake_model):
    response = client.post("/chat", json={"prompt": SECRET_PROMPT})

    request_id = response.headers[REQUEST_ID_HEADER]
    assert request_id

    ids = {record["request_id"] for record in json_logs.request_records()}
    assert ids == {request_id}
    # Records span more than one module, which is the point of correlating.
    loggers = {record["logger"] for record in json_logs.request_records()}
    assert len(loggers) > 1


def test_incoming_request_id_is_honoured(json_logs, fake_model):
    supplied = "backend-req-0f2c19"

    response = client.post("/chat", json={"prompt": SECRET_PROMPT}, headers={REQUEST_ID_HEADER: supplied})

    assert response.headers[REQUEST_ID_HEADER] == supplied
    assert {r["request_id"] for r in json_logs.request_records()} == {supplied}


@pytest.mark.parametrize(
    "hostile",
    ["bad id with spaces", "x" * 200, "abc\r\nX-Injected: 1", "  ", ""],
)
def test_unusable_incoming_request_id_is_replaced_not_echoed(json_logs, fake_model, hostile):
    """An id is echoed into a response header and into logs, so it is validated."""
    response = client.post("/chat", json={"prompt": SECRET_PROMPT}, headers={REQUEST_ID_HEADER: hostile})

    echoed = response.headers[REQUEST_ID_HEADER]
    assert echoed != hostile
    assert echoed.isalnum()
    assert "X-Injected" not in json_logs.text
    if hostile.strip():
        assert hostile not in json_logs.text


def test_every_endpoint_returns_a_request_id_header():
    for method, path in [("get", "/ping"), ("get", "/health"), ("delete", f"/chat/{uuid4()}")]:
        response = getattr(client, method)(path)
        assert response.headers.get(REQUEST_ID_HEADER), f"{method.upper()} {path} lost the request id"


def test_request_id_is_the_telemetry_trace_id(json_logs, fake_model):
    """One id, not two: #69's X-Trace-Id and this request id are the same value."""
    response = client.post("/chat", json={"prompt": SECRET_PROMPT})

    assert response.headers["X-Trace-Id"] == response.headers[REQUEST_ID_HEADER]
    assert json_logs.find("stage completed")["trace_id"] == response.headers[REQUEST_ID_HEADER]


# ---------------------------------------------------------------------------
# Prompt redaction
# ---------------------------------------------------------------------------
def test_prompt_text_never_reaches_the_logs_by_default(json_logs, fake_model):
    client.post("/chat", json={"prompt": SECRET_PROMPT, "context": "context-probe-string"})

    assert SECRET_PROMPT not in json_logs.text
    assert "context-probe-string" not in json_logs.text

    received = json_logs.find("chat request received")
    assert received["prompt_chars"] == len(SECRET_PROMPT)
    assert received["context_chars"] == len("context-probe-string")
    assert "prompt" not in received


def test_streaming_path_is_redacted_too(json_logs, fake_model):
    with client.stream("POST", "/chat/stream", json={"prompt": SECRET_PROMPT}) as response:
        assert response.status_code == 200
        body = "".join(response.iter_text())

    assert "data:" in body
    assert SECRET_PROMPT not in json_logs.text
    assert json_logs.find("streaming chat request received")["prompt_chars"] == len(SECRET_PROMPT)


def test_raw_prompt_logging_requires_an_explicit_opt_in(json_logs, fake_model, monkeypatch):
    monkeypatch.setenv("LOG_PROMPTS", "true")

    client.post("/chat", json={"prompt": SECRET_PROMPT})

    assert json_logs.find("chat request received")["prompt"] == SECRET_PROMPT


# ---------------------------------------------------------------------------
# Stack traces
# ---------------------------------------------------------------------------
def test_forced_failure_in_chat_logs_a_json_record_with_a_full_traceback(json_logs, fake_model, monkeypatch):
    def exploding_model():
        raise RuntimeError("gemini exploded")

    monkeypatch.setattr(main, "get_model", exploding_model)

    response = client.post("/chat", json={"prompt": SECRET_PROMPT})

    assert response.status_code == 500

    failure = json_logs.find("unexpected error in /chat handler")
    assert failure["level"] == "ERROR"
    assert failure["exception"]["type"] == "RuntimeError"
    assert failure["exception"]["message"] == "gemini exploded"
    traceback_text = failure["exception"]["traceback"]
    assert traceback_text.startswith("Traceback (most recent call last):")
    assert "exploding_model" in traceback_text
    assert 'raise RuntimeError("gemini exploded")' in traceback_text
    # The trace is correlatable and still leaks no prompt.
    assert failure["request_id"] == response.headers[REQUEST_ID_HEADER]
    assert SECRET_PROMPT not in json_logs.text


# ---------------------------------------------------------------------------
# Request completion
# ---------------------------------------------------------------------------
def test_chat_emits_one_completion_record_with_duration_and_status(json_logs, fake_model):
    response = client.post("/chat", json={"prompt": SECRET_PROMPT})

    completions = [r for r in json_logs.records() if r["message"] == "request completed"]
    assert len(completions) == 1

    completed = completions[0]
    assert completed["status_code"] == response.status_code == 200
    assert completed["http_method"] == "POST"
    assert completed["path"] == "/chat"
    assert completed["duration_ms"] > 0
    assert completed["request_id"] == response.headers[REQUEST_ID_HEADER]


def test_completion_record_reports_the_real_status_on_a_failure(json_logs, fake_model, monkeypatch):
    monkeypatch.setattr(main, "get_model", lambda: (_ for _ in ()).throw(RuntimeError("boom")))

    client.post("/chat", json={"prompt": SECRET_PROMPT})

    assert json_logs.find("request completed")["status_code"] == 500


def test_streaming_completion_is_logged_after_the_last_chunk(json_logs, fake_model):
    with client.stream("POST", "/chat/stream", json={"prompt": SECRET_PROMPT}) as response:
        assert response.status_code == 200
        chunks = list(response.iter_text())

    assert chunks
    completed = json_logs.find("request completed")
    assert completed["path"] == "/chat/stream"
    assert completed["status_code"] == 200
    # Emitted once the body was fully sent, so it times the whole stream.
    assert completed["duration_ms"] > 0


# ---------------------------------------------------------------------------
# Middleware hygiene
# ---------------------------------------------------------------------------
async def test_middleware_streams_chunks_through_instead_of_buffering():
    """/chat/stream is Server-Sent Events, so the middleware must not collect
    the body before forwarding it. Asserted at the ASGI level: each chunk the
    app sends must arrive as its own message, still flagged more_body."""

    async def streaming_app(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": [(b"content-type", b"text/event-stream")]})
        for index in range(3):
            await send({"type": "http.response.body", "body": f"data: {index}\n\n".encode(), "more_body": True})
        await send({"type": "http.response.body", "body": b"", "more_body": False})

    sent: list[dict] = []

    async def record(message):
        sent.append(message)

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    middleware = logging_config.RequestContextMiddleware(streaming_app)
    await middleware(
        {"type": "http", "method": "POST", "path": "/chat/stream", "headers": []},
        receive,
        record,
    )

    body_messages = [m for m in sent if m["type"] == "http.response.body"]
    assert [m["body"] for m in body_messages] == [b"data: 0\n\n", b"data: 1\n\n", b"data: 2\n\n", b""]
    assert [m["more_body"] for m in body_messages] == [True, True, True, False]

    start = next(m for m in sent if m["type"] == "http.response.start")
    assert (REQUEST_ID_HEADER.lower().encode(), start["headers"][-1][1]) == start["headers"][-1]


async def test_middleware_passes_non_http_scopes_through_untouched():
    seen: list[str] = []

    async def inner(scope, receive, send):
        seen.append(scope["type"])

    await logging_config.RequestContextMiddleware(inner)({"type": "lifespan"}, None, None)

    assert seen == ["lifespan"]


# ---------------------------------------------------------------------------
# Thread boundaries
# ---------------------------------------------------------------------------
def test_request_id_survives_the_threadpool_hop(json_logs, fake_model, monkeypatch):
    """Blocking store I/O runs via run_in_threadpool. Correlation would have a
    hole in it if records emitted inside that worker thread lost the id, so
    /feedback is driven end to end with the store logging from the worker."""
    answered = client.post("/chat", json={"prompt": SECRET_PROMPT})
    chat_id = answered.json()["chat_id"]
    message_id = answered.json()["message_id"]

    worker_logger = logging.getLogger("feedback.store")

    def upsert_from_worker(record):
        worker_logger.info("stored from worker thread", extra={"feedback_id": record.feedback_id})

    monkeypatch.setattr(main.feedback_store, "upsert", upsert_from_worker)

    response = client.post(
        "/feedback",
        json={"chat_id": chat_id, "message_id": message_id, "rating": "up"},
    )

    assert response.status_code == 200
    request_id = response.headers[REQUEST_ID_HEADER]

    from_worker = json_logs.find("stored from worker thread")
    assert from_worker["request_id"] == request_id
    assert from_worker["feedback_id"]

    # And the whole /feedback request still correlates on one id.
    feedback_records = [r for r in json_logs.records() if r["request_id"] == request_id]
    assert {r["message"] for r in feedback_records} >= {"stored from worker thread", "feedback stored"}


# ---------------------------------------------------------------------------
# JSON validity under hostile values
# ---------------------------------------------------------------------------
def _strict_loads(line: str) -> dict:
    """Parse rejecting NaN/Infinity — json.loads accepts them by default, so a
    lax parse would hide exactly the bug this guards against."""

    def reject(token):
        raise AssertionError(f"line contains the non-JSON token {token!r}: {line}")

    return json.loads(line, parse_constant=reject)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_floats_do_not_produce_invalid_json(bad):
    """LLM_PRICE_TABLE can be configured with non-finite prices, which reach
    the telemetry record's cost_usd. json.dumps would write bare NaN/Infinity,
    which is not JSON and breaks every downstream log parser."""
    record = logging.LogRecord("telemetry", logging.INFO, __file__, 1, "model call completed", (), None)
    record.cost_usd = bad
    record.nested = {"prices": [{"input": bad}]}

    line = logging_config.JsonFormatter().format(record)
    parsed = _strict_loads(line)

    assert parsed["cost_usd"] == str(bad)
    assert parsed["nested"]["prices"][0]["input"] == str(bad)


def test_finite_floats_are_still_emitted_as_numbers():
    record = logging.LogRecord("telemetry", logging.INFO, __file__, 1, "model call completed", (), None)
    record.cost_usd = 0.00019
    record.latency_ms = 1843.2

    parsed = _strict_loads(logging_config.JsonFormatter().format(record))

    assert parsed["cost_usd"] == 0.00019
    assert parsed["latency_ms"] == 1843.2


def test_an_unserializable_extra_still_yields_one_valid_json_line():
    """A formatter that raises drops the record and prints a traceback to
    stderr. The envelope must survive whatever a call site attaches."""

    class Hostile:
        def __repr__(self):
            raise RuntimeError("repr exploded")

        def __str__(self):
            raise RuntimeError("str exploded")

    record = logging.LogRecord("main", logging.INFO, __file__, 1, "chat request received", (), None)
    record.weird = Hostile()

    parsed = _strict_loads(logging_config.JsonFormatter().format(record))

    assert parsed["message"] == "chat request received"
    assert parsed["logger"] == "main"
    assert parsed["log_serialization_error"] is True


def test_live_request_lines_survive_a_strict_json_parser(json_logs, fake_model):
    """The end-to-end version: every line a real request emits parses under a
    parser that rejects NaN/Infinity, not just under json.loads' defaults."""
    client.post("/chat", json={"prompt": SECRET_PROMPT})

    lines = json_logs.lines()
    assert lines
    for line in lines:
        _strict_loads(line)
