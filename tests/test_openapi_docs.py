"""Issue #18: the interactive docs are the integration reference, so the
schema they are generated from is asserted here rather than eyeballed.

Two things are checked, and the second matters as much as the first:

1. The OpenAPI document actually carries the metadata — tags, summaries,
   descriptions, per-field descriptions and examples, typed responses.
2. The documentation does not lie. Documented examples are validated against
   the real models, and documented error bodies are compared with what the
   live endpoints return. A docs change that quietly altered a response would
   fail here.
"""

import inspect
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

import main
from main import ChatRequest, app

client = TestClient(app)


@pytest.fixture(scope="module")
def spec() -> dict:
    response = client.get("/openapi.json")
    assert response.status_code == 200
    return response.json()


# ---------------------------------------------------------------------------
# App-level metadata
# ---------------------------------------------------------------------------
def test_app_metadata_is_populated(spec):
    info = spec["info"]

    assert info["title"] == "DeenBridge AI API"
    assert info["version"] == main.API_VERSION
    assert len(info["description"]) > 200
    # The session contract is the thing integrators get wrong; it belongs in
    # the front-page description, not only on the route.
    assert "chat_id" in info["description"]

    tags = {tag["name"]: tag["description"] for tag in spec["tags"]}
    assert set(tags) == {"chat", "health"}
    assert all(description for description in tags.values())


def test_docs_page_is_served(spec):
    docs = client.get("/docs")

    assert docs.status_code == 200
    assert "text/html" in docs.headers["content-type"]
    assert spec["openapi"].startswith("3.")


# ---------------------------------------------------------------------------
# Route documentation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("method", "path", "tag"),
    [
        ("post", "/chat", "chat"),
        ("post", "/chat/stream", "chat"),
        ("get", "/chat/{chat_id}/history", "chat"),
        ("delete", "/chat/{chat_id}", "chat"),
        ("get", "/user/{user_id}/chats", "chat"),
        ("get", "/ping", "health"),
        ("get", "/health", "health"),
    ],
)
def test_routes_are_tagged_and_summarized(spec, method, path, tag):
    operation = spec["paths"][path][method]

    assert operation["tags"] == [tag]
    assert operation["summary"]


@pytest.mark.parametrize(
    ("method", "path"),
    [("post", "/chat"), ("delete", "/chat/{chat_id}"), ("get", "/ping")],
)
def test_session_routes_explain_themselves(spec, method, path):
    """The three routes the issue names carry a prose description."""
    description = spec["paths"][path][method]["description"]

    assert len(description) > 80
    assert "chat_id" in description or "status" in description


def test_chat_documents_success_and_error_responses(spec):
    responses = spec["paths"]["/chat"]["post"]["responses"]

    assert set(responses) >= {"200", "422", "500"}
    assert responses["200"]["content"]["application/json"]["schema"]["$ref"].endswith("/ChatResponse")

    # Every status the handler can raise, so /docs is not a partial list.
    for code in ("400", "422", "429", "500", "503", "504"):
        example = responses[code]["content"]["application/json"]["example"]
        assert example["detail"], f"{code} needs an example body"
        assert responses[code]["description"]


@pytest.mark.parametrize(
    ("code", "detail"),
    [
        ("400", "Invalid request parameters."),
        ("429", "Rate limit exceeded. Please try again later."),
        ("500", "AI service error"),
        ("503", "AI service temporarily unavailable."),
        ("504", "AI service timed out."),
    ],
)
def test_documented_chat_error_bodies_match_the_handler(spec, code, detail):
    """Each documented example is the exact `detail` the handler raises, so /docs
    cannot drift into describing errors the service never returns.

    The route is a thin locking wrapper over ``_chat``, which is where the
    HTTPExceptions are actually raised — so that is what gets inspected.
    """
    documented = spec["paths"]["/chat"]["post"]["responses"][code]["content"]["application/json"]["example"]

    assert documented == {"detail": detail}
    assert f'detail="{detail}"' in inspect.getsource(main._chat)


def test_auth_token_documents_every_use_of_the_credential(spec):
    """A bearer credential's documented data use must match the code path."""
    description = spec["components"]["schemas"]["ChatRequest"]["properties"]["auth_token"]["description"]
    # The chat path fans its retrievals out through this helper, so both uses
    # of the token are visible in one place.
    source = inspect.getsource(main.retrieve_chat_contexts)

    assert "purchase history" in description
    # The token reaches purchase history and, unconditionally, personal-context
    # retrieval — not only when `transactions` is absent.
    assert "purchase_retriever(request.prompt, request.transactions, request.auth_token)" in source
    assert "personal_context_retriever(request.prompt, request.user_id, request.auth_token)" in source
    assert "personal-context" in description


def test_delete_and_ping_declare_typed_response_schemas(spec):
    delete_schema = spec["paths"]["/chat/{chat_id}"]["delete"]["responses"]["200"]
    ping_schema = spec["paths"]["/ping"]["get"]["responses"]["200"]

    assert delete_schema["content"]["application/json"]["schema"]["$ref"].endswith("/DeleteChatResponse")
    assert ping_schema["content"]["application/json"]["schema"]["$ref"].endswith("/PingResponse")

    assert spec["components"]["schemas"]["DeleteChatResponse"]["properties"]["message"]["type"] == "string"
    assert spec["components"]["schemas"]["PingResponse"]["properties"]["status"]["type"] == "string"


# ---------------------------------------------------------------------------
# Model documentation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("model", ["ChatRequest", "ChatResponse", "Message", "PingResponse", "DeleteChatResponse"])
def test_every_field_has_a_description_and_an_example(spec, model):
    properties = spec["components"]["schemas"][model]["properties"]

    assert properties
    undocumented = [
        name for name, field in properties.items() if not field.get("description") or not field.get("examples")
    ]
    assert undocumented == [], f"{model} fields missing description/examples: {undocumented}"


def test_chat_request_carries_a_full_payload_example_that_actually_validates(spec):
    """The request example must be a plain payload a caller can paste, and it
    must survive the real validators — otherwise /docs teaches a 422."""
    example = spec["components"]["schemas"]["ChatRequest"]["example"]

    assert set(example) >= {"prompt", "chat_id"}
    # A plain payload, not an OpenAPI Example Object wrapper.
    assert "value" not in example and "summary" not in example

    parsed = ChatRequest.model_validate(example)
    assert parsed.prompt == example["prompt"]
    assert str(parsed.chat_id) == example["chat_id"]


def test_chat_id_documentation_states_the_session_contract(spec):
    description = spec["components"]["schemas"]["ChatRequest"]["properties"]["chat_id"]["description"]

    assert "Omit it to start a new conversation" in description
    # Unknown ids start a fresh session rather than 404-ing; say so instead of
    # leaving callers to discover it.
    assert "does not recognise" in description


# ---------------------------------------------------------------------------
# Documentation vs. reality
# ---------------------------------------------------------------------------
def test_ping_response_matches_its_documented_schema():
    response = client.get("/ping")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_delete_unknown_session_returns_the_documented_404_body(spec):
    documented = spec["paths"]["/chat/{chat_id}"]["delete"]["responses"]["404"]["content"]["application/json"][
        "example"
    ]

    response = client.delete(f"/chat/{uuid4()}")

    assert response.status_code == 404
    assert response.json() == documented


def test_empty_prompt_returns_the_documented_422_shape(spec):
    documented = spec["paths"]["/chat"]["post"]["responses"]["422"]["content"]["application/json"]["example"]

    response = client.post("/chat", json={"prompt": ""})

    assert response.status_code == 422
    body = response.json()
    assert isinstance(body["detail"], list)
    assert set(documented["detail"][0]) <= set(body["detail"][0])
    assert body["detail"][0]["loc"] == documented["detail"][0]["loc"]
