"""Exercise real chat/stream handlers with offline candidate and stable models."""

import asyncio
import json
import sqlite3
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

import main
from confidence import UNCERTAINTY_NOTE
from experiments import ExperimentManager
from prompts.registry import PromptRegistry
from prompts.template import PromptTemplate


@dataclass
class Content:
    role: str
    text: str

    @property
    def parts(self):
        return [SimpleNamespace(text=self.text)]


@pytest.fixture
def runtime(monkeypatch, tmp_path):
    registry = PromptRegistry()
    registry.register(PromptTemplate("stable", "1.0.0", main.ISLAMIC_CONTEXT))
    registry.register(PromptTemplate("candidate", "1.0.0", "CANDIDATE {response_language}", ("response_language",)))
    manager = ExperimentManager(
        registry,
        "stable-model",
        str(tmp_path / "experiments.sqlite3"),
        default_generation_config=main.GENERATION_CONFIG,
    )
    monkeypatch.setattr(main, "experiment_harness", manager)
    monkeypatch.setattr(main, "active_chats", {})
    monkeypatch.setattr(main, "experimental_chats", set())
    monkeypatch.setattr(main, "ADMIN_TOKEN", "test-admin")
    monkeypatch.setattr(main.settings, "model_name", "stable-model")
    monkeypatch.setenv("SAFETY_PIPELINE_ENABLED", "false")
    monkeypatch.delenv("MOCK_UPSTREAMS", raising=False)
    monkeypatch.setattr(main, "SEMANTIC_CACHE_ENABLED", True)
    for name in ["tafsir_retriever", "zakat_retriever", "purchase_retriever", "personal_context_retriever"]:
        monkeypatch.setattr(main, name, AsyncMock(return_value=None))
    monkeypatch.setattr(main, "enqueue_for_review", AsyncMock())
    monkeypatch.setattr(main, "embed_for_cache", AsyncMock(side_effect=AssertionError("experiment reached cache")))
    state = SimpleNamespace(
        fail_candidate=False, fail_stable=False, empty_candidate=False, delay=0, calls=[], started=None
    )

    class Session:
        def __init__(self, name, history):
            self.name = name
            self.history = history

        async def send_message_async(self, message, **kwargs):
            state.calls.append((self.name, message, kwargs, list(self.history)))
            if state.started is not None:
                state.started.set()
            if self.name == "candidate-model" and state.delay:
                await asyncio.sleep(state.delay)
            if (self.name == "candidate-model" and state.fail_candidate) or (
                self.name == "stable-model" and state.fail_stable
            ):
                self.history.append(Content("user", "FAILED PARTIAL TURN"))
                raise ValueError("upstream rejected configuration")
            text = "" if self.name == "candidate-model" and state.empty_candidate else f"{self.name} answer"
            self.history.extend([Content("user", message), Content("model", text)])
            return SimpleNamespace(text=text, candidates=[SimpleNamespace(finish_reason="STOP")], prompt_feedback=None)

    class Model:
        def __init__(self, model_name, **_kwargs):
            self.name = model_name

        def start_chat(self, history=None):
            return Session(self.name, list(history or []))

    monkeypatch.setattr(main.genai, "GenerativeModel", Model)
    client = TestClient(main.app)
    headers = {"X-Admin-Token": "test-admin"}
    payload = {
        "experiment_id": "runtime",
        "control_template": "stable",
        "control_model_name": "stable-model",
        "variants": [
            {
                "name": "candidate",
                "template_name": "candidate",
                "model_name": "candidate-model",
                "generation_config": {"temperature": 0.2},
            }
        ],
        "traffic_percent": 100,
        "rollback_policy": {"min_samples": 1, "window_size": 3, "max_error_rate": 0, "max_p95_latency_ms": None},
        "activate": True,
    }
    response = client.post("/experiments", json=payload, headers=headers)
    assert response.status_code == 200, response.text
    return SimpleNamespace(client=client, manager=manager, state=state, headers=headers, payload=payload)


def chat(runtime, chat_id=None, stream=False):
    return runtime.client.post(
        "/chat/stream" if stream else "/chat",
        json={"prompt": "Private question", "chat_id": chat_id or str(uuid4()), "remember": False},
    )


def expected_answer(model):
    # Experiment routing must retain the handler's existing confidence policy.
    return f"{model} answer\n\n{UNCERTAINTY_NOTE}"


def test_model_prompt_and_generation_variant_are_used_and_measured(runtime):
    response = chat(runtime)
    assert response.status_code == 200
    assert response.json()["response"] == expected_answer("candidate-model")
    assert response.headers["X-Experiment-Variant"] == "candidate"
    assert response.headers["X-Semantic-Cache"] == "bypass"
    name, prompt, kwargs, _ = runtime.state.calls[0]
    assert name == "candidate-model" and "CANDIDATE auto" in prompt
    assert kwargs["generation_config"]["temperature"] == 0.2
    metrics = runtime.manager.metrics("runtime")
    assert metrics[0]["window_attempts"] == 1 and metrics[0]["errors"] == 0
    assert "Private question" not in json.dumps(metrics)
    assert "Private question" not in json.dumps(runtime.manager.events("runtime"))


@pytest.mark.parametrize("failure", ["fail_candidate", "empty_candidate"])
def test_failed_treatment_falls_back_without_losing_or_duplicating_history(runtime, failure):
    chat_id = str(uuid4())
    assert chat(runtime, chat_id).status_code == 200
    setattr(runtime.state, failure, True)
    response = chat(runtime, chat_id)
    assert response.status_code == 200
    assert response.json()["response"] == expected_answer("stable-model")
    assert len(response.json()["history"]) == 4
    assert "FAILED PARTIAL TURN" not in json.dumps(response.json())
    assert response.headers["X-Experiment-Rolled-Back"] == "true"
    assert runtime.manager.choose(chat_id).is_control
    assert runtime.manager.events("runtime")[-1]["reason"] == "error_rate"
    # Subsequent requests never retry the rejected candidate.
    assert chat(runtime, chat_id).status_code == 200
    assert [call[0] for call in runtime.state.calls] == [
        "candidate-model",
        "candidate-model",
        "stable-model",
        "stable-model",
    ]


def test_latency_threshold_automatically_routes_subsequent_requests_to_stable(runtime):
    runtime.payload["rollback_policy"] = {"min_samples": 1, "max_error_rate": None, "max_p95_latency_ms": 1}
    assert runtime.client.post("/experiments", json=runtime.payload, headers=runtime.headers).status_code == 200
    runtime.state.delay = 0.02
    assert chat(runtime).json()["response"] == expected_answer("candidate-model")
    assert chat(runtime).json()["response"] == expected_answer("stable-model")
    assert runtime.manager.events("runtime")[-1]["reason"] == "p95_latency"


def test_manual_rollback_deactivation_and_session_deletion_preserve_contracts(runtime):
    chat_id = str(uuid4())
    assert chat(runtime, chat_id).status_code == 200
    assert runtime.client.post("/experiments/runtime/rollback", headers=runtime.headers).status_code == 200
    assert chat(runtime, chat_id).json()["response"] == expected_answer("stable-model")
    assert runtime.client.post("/experiments/runtime/deactivate", headers=runtime.headers).status_code == 200
    response = chat(runtime, chat_id)
    assert response.status_code == 200 and len(response.json()["history"]) == 6
    assert "X-Experiment-Id" not in response.headers
    assert chat_id not in main.experimental_chats
    assert runtime.client.delete(f"/chat/{chat_id}").status_code == 200
    assert chat_id not in main.active_chats


def test_stream_buffers_failed_candidate_and_emits_only_stable_content(runtime):
    runtime.state.fail_candidate = True
    response = chat(runtime, stream=True)
    events = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]
    assert [event["type"] for event in events][0] == "metadata"
    assert events[-1]["type"] == "done"
    assert "".join(event["delta"] for event in events if event["type"] == "content") == expected_answer("stable-model")
    assert events[-1]["experiment"]["rolled_back"] is True
    assert len(events[-1]["history"]) == 2


def test_both_models_failing_yields_one_sse_error_and_no_partial_content(runtime):
    runtime.state.fail_candidate = runtime.state.fail_stable = True
    response = chat(runtime, stream=True)
    events = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]
    assert [event["type"] for event in events] == ["metadata", "error"]
    assert "upstream rejected configuration" not in response.text
    assert main.active_chats == {}


def test_configuration_updates_are_staged_and_rollback_blocks_unsafe_resume(runtime):
    assert runtime.client.post("/experiments/runtime/rollback", headers=runtime.headers).status_code == 200
    assert runtime.client.post("/experiments/runtime/resume", headers=runtime.headers).status_code == 409
    assert runtime.client.post("/experiments/runtime/activate", headers=runtime.headers).status_code == 409
    runtime.payload.update(activate=False, control_model_name="new-stable")
    assert runtime.client.post("/experiments", json=runtime.payload, headers=runtime.headers).json()["version"] == 2
    assert runtime.manager.choose("session").is_control
    assert (
        runtime.client.post("/experiments/runtime/stable", json={"version": 2}, headers=runtime.headers).status_code
        == 200
    )
    assert runtime.client.post("/experiments/runtime/activate", headers=runtime.headers).status_code == 200
    assert runtime.client.post("/experiments/runtime/rollback", headers=runtime.headers).status_code == 200
    assert runtime.manager.choose("session").variant.model_name == "new-stable"
    versions = runtime.client.get("/experiments/runtime/versions", headers=runtime.headers).json()["versions"]
    assert [version["version"] for version in versions] == [1, 2]
    assert versions[0]["config"]["control"]["model_name"] == "stable-model"


@pytest.mark.parametrize("route", ["activate", "deactivate", "kill", "resume", "rollback", "stable"])
def test_all_mutation_routes_require_admin(runtime, route):
    response = runtime.client.post(f"/experiments/runtime/{route}", json={"version": 1})
    assert response.status_code in (401, 403)


@pytest.mark.parametrize("route", ["versions", "metrics", "events"])
def test_operational_metadata_requires_admin(runtime, route):
    assert runtime.client.get(f"/experiments/runtime/{route}").status_code in (401, 403)
    assert runtime.client.get(f"/experiments/runtime/{route}", headers=runtime.headers).status_code == 200


@pytest.mark.parametrize("change", ["template", "duplicate", "policy", "generation", "percentage"])
def test_invalid_configuration_returns_422_without_changing_active_version(runtime, change):
    if change == "template":
        runtime.payload["control_template"] = "missing"
    elif change == "duplicate":
        runtime.payload["variants"][0]["name"] = "control"
    elif change == "policy":
        runtime.payload["rollback_policy"] = {"min_samples": 3, "window_size": 2}
    elif change == "generation":
        runtime.payload["variants"][0]["generation_config"] = {"unexpected_option": 1}
    else:
        runtime.payload["traffic_percent"] = -1
    assert runtime.client.post("/experiments", json=runtime.payload, headers=runtime.headers).status_code == 422
    assert runtime.manager.choose("session").version == 1


def test_unknown_experiment_and_version_return_404(runtime):
    assert runtime.client.post("/experiments/missing/rollback", headers=runtime.headers).status_code == 404
    assert runtime.client.get("/experiments/missing/versions", headers=runtime.headers).status_code == 404
    assert (
        runtime.client.post("/experiments/runtime/stable", json={"version": 99}, headers=runtime.headers).status_code
        == 404
    )


def test_control_plane_read_failure_falls_back_to_default_without_losing_history(runtime, monkeypatch):
    chat_id = str(uuid4())
    assert chat(runtime, chat_id).status_code == 200

    def unavailable(_session_id):
        raise sqlite3.OperationalError("database unavailable")

    monkeypatch.setattr(runtime.manager, "choose", unavailable)
    response = chat(runtime, chat_id)
    assert response.status_code == 200
    assert response.json()["response"] == expected_answer("stable-model")
    assert len(response.json()["history"]) == 4


def test_deletion_during_candidate_failure_uses_captured_stable_snapshot(runtime, monkeypatch):
    original = main.send_message_with_retry

    async def remove_before_failure(session, message, **kwargs):
        if session.name == "candidate-model":
            runtime.manager.unregister_experiment("runtime")
            raise ValueError("candidate failed after deletion")
        return await original(session, message, **kwargs)

    monkeypatch.setattr(main, "send_message_with_retry", remove_before_failure)
    response = chat(runtime)
    assert response.status_code == 200
    assert response.json()["response"] == expected_answer("stable-model")
    assert len(response.json()["history"]) == 2
    assert runtime.manager.choose("session") is None


def test_safety_refusal_never_invokes_experimental_model(runtime, monkeypatch):
    from safety.pipeline import SafetyResult

    monkeypatch.setenv("SAFETY_PIPELINE_ENABLED", "true")
    refusal = SafetyResult(
        text="I cannot assist with this request.",
        action="refuse",
        category_id="test_policy",
        confidence=1.0,
        stages_fired=["input_gate"],
        latency_ms=0,
        generator_called=False,
    )
    monkeypatch.setattr(main.safety_pipeline, "run_async", AsyncMock(return_value=refusal))
    response = chat(runtime, stream=True)
    events = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]
    assert [event["type"] for event in events] == ["metadata", "error"]
    assert runtime.state.calls == []
    assert runtime.manager.metrics("runtime") == []


def test_admin_endpoints_fail_closed_when_token_is_not_configured(runtime, monkeypatch):
    monkeypatch.setattr(main, "ADMIN_TOKEN", "")
    assert runtime.client.post("/experiments/runtime/rollback", headers=runtime.headers).status_code == 503
    assert not runtime.manager.choose("session").is_control


@pytest.mark.asyncio
async def test_cancelled_candidate_does_not_commit_history_or_trigger_rollback(runtime):
    runtime.state.delay = 10
    runtime.state.started = asyncio.Event()
    decision = runtime.manager.choose("session")
    task = asyncio.create_task(
        main.generate_experiment_response(decision, "session", [], main.ISLAMIC_CONTEXT, {"response_language": "en"})
    )
    await runtime.state.started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert main.active_chats == {}
    assert runtime.manager.metrics("runtime") == []
    assert not runtime.manager.choose("session").is_control
