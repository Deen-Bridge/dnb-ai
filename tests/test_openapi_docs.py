"""Tests for the OpenAPI documentation metadata.

These assert the contract the interactive docs at ``/docs`` present to
integrators: tagged and summarized routes, documented error responses, typed
response schemas, and a description plus example on every request/response
field. Documentation rots silently, so it is checked like any other behaviour.

``main`` imports and configures the Gemini SDK at module scope, so the SDK is
stubbed before import — these tests make no model calls and need no API key.
"""

import sys
import types

import pytest


def _install_gemini_stub() -> None:
    """Satisfy main.py's import-time genai.configure() without a real key."""
    if "google.generativeai" in sys.modules:
        return
    google = types.ModuleType("google")
    google.__path__ = []  # mark as a package
    genai = types.ModuleType("google.generativeai")

    class _StubModel:
        def __init__(self, *args, **kwargs):
            pass

    genai.GenerativeModel = _StubModel
    genai.configure = lambda **kwargs: None
    sys.modules.setdefault("google", google)
    sys.modules["google.generativeai"] = genai


@pytest.fixture(scope="module")
def spec():
    _install_gemini_stub()
    import os

    os.environ.setdefault("GEMINI_API_KEY", "test-key-not-used")
    import main

    return main.app.openapi()


@pytest.fixture(scope="module")
def schemas(spec):
    return spec["components"]["schemas"]


# ---------------------------------------------------------------------------
# Application metadata
# ---------------------------------------------------------------------------


class TestAppMetadata:
    def test_title_description_and_version(self, spec):
        info = spec["info"]
        assert info["title"] == "DeenBridge AI API"
        assert info["version"]
        assert len(info["description"]) > 200

    def test_description_explains_session_semantics(self, spec):
        """The one thing an integrator must know before their first call."""
        description = spec["info"]["description"].lower()
        assert "omit" in description and "chat_id" in description
        assert "continue" in description

    def test_tags_are_declared_with_descriptions(self, spec):
        tags = {tag["name"]: tag for tag in spec["tags"]}
        assert {"chat", "health"} <= set(tags)
        for tag in tags.values():
            assert tag["description"].strip()

    def test_every_route_is_tagged_and_summarized(self, spec):
        for path, operations in spec["paths"].items():
            for verb, operation in operations.items():
                assert operation.get("tags"), f"{verb.upper()} {path} has no tag"
                assert operation.get("summary"), f"{verb.upper()} {path} has no summary"

    def test_declared_tags_cover_every_route(self, spec):
        declared = {tag["name"] for tag in spec["tags"]}
        used = {
            tag
            for operations in spec["paths"].values()
            for operation in operations.values()
            for tag in operation.get("tags", [])
        }
        assert used <= declared, f"undeclared tags in use: {used - declared}"


# ---------------------------------------------------------------------------
# Route documentation
# ---------------------------------------------------------------------------


class TestChatRoute:
    def test_documents_success_and_both_error_paths(self, spec):
        responses = spec["paths"]["/chat"]["post"]["responses"]
        assert {"200", "422", "500"} <= set(responses)

    @pytest.mark.parametrize("status", ["200", "422", "500"])
    def test_each_documented_response_has_an_example(self, spec, status):
        content = spec["paths"]["/chat"]["post"]["responses"][status]["content"]
        schema = content["application/json"]
        assert "example" in schema or "examples" in schema

    def test_has_a_description_covering_sessions(self, spec):
        description = spec["paths"]["/chat"]["post"]["description"].lower()
        assert "chat_id" in description
        assert "omit" in description

    def test_request_body_carries_labelled_examples(self, spec):
        """Labelled variants belong on the operation, as OpenAPI Example Objects."""
        content = spec["paths"]["/chat"]["post"]["requestBody"]["content"]
        examples = content["application/json"]["examples"]
        summaries = [example["summary"] for example in examples.values()]
        assert any("new" in s.lower() for s in summaries)
        assert any("continue" in s.lower() for s in summaries)

    def test_start_example_really_starts_a_session(self, spec):
        """An example that contradicts its own caption is worse than none."""
        content = spec["paths"]["/chat"]["post"]["requestBody"]["content"]
        examples = content["application/json"]["examples"]
        start = next(
            e for e in examples.values() if "new" in e["summary"].lower()
        )
        assert "chat_id" not in start["value"]
        assert start["value"]["prompt"]

    def test_continue_example_sends_a_chat_id(self, spec):
        content = spec["paths"]["/chat"]["post"]["requestBody"]["content"]
        examples = content["application/json"]["examples"]
        cont = next(
            e for e in examples.values() if "continue" in e["summary"].lower()
        )
        assert cont["value"]["chat_id"]

    def test_schema_examples_are_plain_model_instances(self, schemas):
        """JSON Schema `examples` take instances of the model itself.

        Wrapping them in {summary, description, value} is the OpenAPI Example
        Object shape and would show the wrapper in the schema view instead of
        a usable request body.
        """
        examples = schemas["ChatRequest"]["examples"]
        properties = set(schemas["ChatRequest"]["properties"])
        for example in examples:
            assert not {"summary", "value", "description"} & set(example)
            assert set(example) <= properties
            assert example.get("prompt")


class TestOtherRoutes:
    def test_delete_declares_a_typed_response(self, spec, schemas):
        responses = spec["paths"]["/chat/{chat_id}"]["delete"]["responses"]
        schema = responses["200"]["content"]["application/json"]["schema"]
        assert schema["$ref"].endswith("/DeleteChatResponse")
        assert "message" in schemas["DeleteChatResponse"]["properties"]

    def test_delete_documents_the_idempotent_case(self, spec):
        """Deleting an unknown session returns 200, which must be documented."""
        responses = spec["paths"]["/chat/{chat_id}"]["delete"]["responses"]
        examples = responses["200"]["content"]["application/json"]["examples"]
        assert "not_found" in examples

    def test_delete_path_parameter_is_described(self, spec):
        parameters = spec["paths"]["/chat/{chat_id}"]["delete"]["parameters"]
        chat_id = next(p for p in parameters if p["name"] == "chat_id")
        assert chat_id.get("description")

    def test_ping_declares_a_typed_response(self, spec):
        schema = spec["paths"]["/ping"]["get"]["responses"]["200"]["content"][
            "application/json"
        ]["schema"]
        assert schema["$ref"].endswith("/PingResponse")

    def test_ping_schema_matches_what_ping_actually_returns(self, schemas):
        """The schema documents the real payload rather than a tidier one."""
        assert schemas["PingResponse"]["type"] == "array"


# ---------------------------------------------------------------------------
# Model field documentation
# ---------------------------------------------------------------------------


DOCUMENTED_MODELS = ["ChatRequest", "ChatResponse", "Message"]


class TestFieldDocumentation:
    @pytest.mark.parametrize("model", DOCUMENTED_MODELS)
    def test_model_has_a_description(self, schemas, model):
        assert schemas[model].get("description", "").strip()

    @pytest.mark.parametrize("model", DOCUMENTED_MODELS)
    def test_every_field_is_described(self, schemas, model):
        for name, field in schemas[model]["properties"].items():
            assert _description_of(field), f"{model}.{name} has no description"

    @pytest.mark.parametrize("model", DOCUMENTED_MODELS)
    def test_every_scalar_field_has_an_example(self, schemas, model):
        """Object and array fields document themselves through their own schema."""
        for name, field in schemas[model]["properties"].items():
            if _is_scalar(field):
                assert field.get("examples"), f"{model}.{name} has no example"

    def test_context_field_explains_what_it_is_for(self, schemas):
        description = _description_of(schemas["ChatRequest"]["properties"]["context"])
        assert "context" in description.lower()

    def test_chat_id_field_explains_session_semantics(self, schemas):
        description = _description_of(
            schemas["ChatRequest"]["properties"]["chat_id"]
        ).lower()
        assert "omit" in description
        assert "new session" in description


def _description_of(field: dict) -> str:
    """Field description, whether inline or attached beside a $ref."""
    if field.get("description"):
        return field["description"]
    for key in ("allOf", "anyOf", "oneOf"):
        for member in field.get(key, []):
            if member.get("description"):
                return member["description"]
    return ""


def _is_scalar(field: dict) -> bool:
    """True for plain string/number/bool fields.

    Fields that are (or may be) another model or a collection are excluded:
    they document themselves through the schema they reference, so requiring a
    duplicate inline example there would be noise rather than help.
    """
    members = [field, *field.get("anyOf", []), *field.get("oneOf", []), *field.get("allOf", [])]
    if any("$ref" in member for member in members):
        return False
    types_present = {member.get("type") for member in members}
    if types_present & {"array", "object"}:
        return False
    return bool(types_present - {None})
