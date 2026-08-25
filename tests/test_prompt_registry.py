"""Tests for the prompt template registry and A/B experimentation harness.

Covers: rendering, variable substitution, sticky assignment, kill switch,
metadata, validation, and the API endpoints.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from prompts import (
    ExperimentAssignment,
    ExperimentConfig,
    ExperimentHarness,
    PromptRegistry,
    PromptTemplate,
    Variant,
    get_registry,
    register_defaults,
)
from prompts.registry import _stable_hash

# -----------------------------------------------------------------------
# PromptTemplate rendering
# -----------------------------------------------------------------------


class TestPromptTemplate:
    def test_render_without_variables(self):
        t = PromptTemplate(name="greet", version="1.0.0", body="Hello, world!")
        assert t.render() == "Hello, world!"

    def test_render_with_variables(self):
        t = PromptTemplate(
            name="greet",
            version="1.0.0",
            body="Hello, {name}!",
            variables=("name",),
        )
        assert t.render(name="Alice") == "Hello, Alice!"

    def test_render_missing_variable_raises_key_error(self):
        t = PromptTemplate(
            name="greet",
            version="1.0.0",
            body="Hello, {name}!",
            variables=("name",),
        )
        with pytest.raises(KeyError, match="Missing template variables"):
            t.render()

    def test_render_multiple_variables(self):
        t = PromptTemplate(
            name="multi",
            version="1.0.0",
            body="{a} and {b}",
            variables=("a", "b"),
        )
        assert t.render(a="X", b="Y") == "X and Y"

    def test_variables_used(self):
        t = PromptTemplate(
            name="t",
            version="1.0.0",
            body="{x} {y} {x}",
            variables=("x", "y"),
        )
        assert t.variables_used() == {"x", "y"}

    def test_validate_clean(self):
        t = PromptTemplate(
            name="t",
            version="1.0.0",
            body="{x}",
            variables=("x",),
        )
        assert t.validate() == []

    def test_validate_declared_but_unused(self):
        t = PromptTemplate(
            name="t",
            version="1.0.0",
            body="no placeholder",
            variables=("x",),
        )
        errors = t.validate()
        assert len(errors) == 1
        assert "unused" in errors[0]

    def test_validate_undeclared_in_body(self):
        t = PromptTemplate(
            name="t",
            version="1.0.0",
            body="{x} {y}",
            variables=("x",),
        )
        errors = t.validate()
        assert len(errors) == 1
        assert "Undeclared" in errors[0]


# -----------------------------------------------------------------------
# PromptRegistry
# -----------------------------------------------------------------------


class TestPromptRegistry:
    def test_register_and_get(self):
        reg = PromptRegistry()
        t = PromptTemplate(name="a", version="1.0.0", body="A")
        reg.register(t)
        assert reg.get("a") is t
        assert reg.get("a", "1.0.0") is t

    def test_get_nonexistent_returns_none(self):
        reg = PromptRegistry()
        assert reg.get("nope") is None
        assert reg.get("nope", "1.0.0") is None

    def test_latest_returns_highest_version(self):
        reg = PromptRegistry()
        t1 = PromptTemplate(name="a", version="1.0.0", body="v1")
        t2 = PromptTemplate(name="a", version="2.0.0", body="v2")
        reg.register(t1)
        reg.register(t2)
        assert reg.latest("a") is t2
        assert reg.get("a", "1.0.0") is t1

    def test_list_templates(self):
        reg = PromptRegistry()
        reg.register(PromptTemplate(name="x", version="1.0.0", body="X"))
        reg.register(PromptTemplate(name="y", version="2.0.0", body="Y"))
        assert reg.list_templates() == {"x": "1.0.0", "y": "2.0.0"}

    def test_register_invalid_template_raises(self):
        reg = PromptRegistry()
        t = PromptTemplate(
            name="bad",
            version="1.0.0",
            body="no var here",
            variables=("missing",),
        )
        with pytest.raises(ValueError, match="validation errors"):
            reg.register(t)


# -----------------------------------------------------------------------
# Default templates
# -----------------------------------------------------------------------


class TestDefaultTemplates:
    def test_register_defaults_populates_registry(self):
        register_defaults()
        reg = get_registry()
        templates = reg.list_templates()
        assert "islamic_context" in templates
        assert "language_instructions" in templates

    def test_islamic_context_renders(self):
        register_defaults()
        reg = get_registry()
        t = reg.get("islamic_context")
        rendered = t.render()
        assert "Deen Bridge" in rendered
        assert "Islamic education" in rendered

    def test_language_instructions_render(self):
        register_defaults()
        reg = get_registry()
        t = reg.get("language_instructions")
        rendered = t.render(response_language="ar")
        assert "response_language: ar" in rendered

    def test_idempotent_registration(self):
        register_defaults()
        register_defaults()
        reg = get_registry()
        assert len(reg._templates.get("islamic_context", [])) == 1


# -----------------------------------------------------------------------
# ExperimentHarness — sticky assignment
# -----------------------------------------------------------------------


class TestExperimentHarness:
    @pytest.fixture()
    def harness(self):
        register_defaults()
        reg = get_registry()
        h = ExperimentHarness(reg)
        cfg = ExperimentConfig(
            experiment_id="test_exp",
            control=Variant(name="control", template_name="islamic_context", weight=1.0),
            variants=[
                Variant(name="variant_a", template_name="islamic_context", weight=1.0),
            ],
        )
        h.register_experiment(cfg)
        return h

    def test_sticky_assignment(self, harness):
        a1 = harness.assign("test_exp", "session-42")
        a2 = harness.assign("test_exp", "session-42")
        assert a1.variant_name == a2.variant_name

    def test_different_sessions_can_get_different_variants(self, harness):
        results = set()
        for i in range(200):
            a = harness.assign("test_exp", f"session-{i}")
            results.add(a.variant_name)
        assert len(results) == 2

    def test_unknown_experiment_raises(self, harness):
        with pytest.raises(KeyError, match="Unknown experiment"):
            harness.assign("nonexistent", "s1")

    def test_kill_switch_returns_control(self, harness):
        harness._experiments["test_exp"].kill_switch = True
        a = harness.assign("test_exp", "session-42")
        assert a.variant_name == "control"
        assert a.kill_switch_active is True

    def test_active_experiments_excludes_killed(self, harness):
        assert "test_exp" in harness.active_experiments()
        harness._experiments["test_exp"].kill_switch = True
        assert "test_exp" not in harness.active_experiments()

    def test_unregister_experiment(self, harness):
        harness.unregister_experiment("test_exp")
        assert harness.active_experiments() == []
        with pytest.raises(KeyError):
            harness.assign("test_exp", "s1")

    def test_resolve_template_renders(self, harness):
        rendered, assignment = harness.resolve_template(
            "test_exp",
            "session-42",
            response_language="en",
        )
        assert "Deen Bridge" in rendered
        assert assignment.experiment_id == "test_exp"


# -----------------------------------------------------------------------
# Stable hash
# -----------------------------------------------------------------------


class TestStableHash:
    def test_deterministic(self):
        assert _stable_hash("foo") == _stable_hash("foo")

    def test_different_inputs_different_hashes(self):
        assert _stable_hash("foo") != _stable_hash("bar")

    def test_in_zero_one_range(self):
        for i in range(100):
            h = _stable_hash(f"test-{i}")
            assert 0.0 <= h < 1.0


# -----------------------------------------------------------------------
# ExperimentAssignment model
# -----------------------------------------------------------------------


class TestExperimentAssignment:
    def test_model_dump(self):
        a = ExperimentAssignment(
            experiment_id="exp1",
            variant_name="control",
            kill_switch_active=False,
        )
        d = a.model_dump()
        assert d["experiment_id"] == "exp1"
        assert d["variant_name"] == "control"
        assert d["kill_switch_active"] is False


# -----------------------------------------------------------------------
# API endpoints (via TestClient)
# -----------------------------------------------------------------------


class TestPromptAPIEndpoints:
    @pytest.fixture(autouse=True)
    def _setup(self):
        register_defaults()

    def test_list_templates(self):
        from main import app

        client = TestClient(app)
        resp = client.get("/prompts")
        assert resp.status_code == 200
        data = resp.json()
        assert "islamic_context" in data

    def test_get_template(self):
        from main import app

        client = TestClient(app)
        resp = client.get("/prompts/islamic_context")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "islamic_context"
        assert data["version"] == "1.0.0"
        assert "body" in data

    def test_get_template_not_found(self):
        from main import app

        client = TestClient(app)
        resp = client.get("/prompts/nonexistent")
        assert resp.status_code == 404

    def test_list_experiments(self):
        from main import app

        client = TestClient(app)
        resp = client.get("/experiments")
        assert resp.status_code == 200
        assert "experiments" in resp.json()


class TestExperimentAPIEndpoints:
    @pytest.fixture(autouse=True)
    def _setup(self):
        register_defaults()
        import os

        os.environ["ADMIN_TOKEN"] = "test-admin-token"
        # Re-read the module-level ADMIN_TOKEN in main.py
        import main

        main.ADMIN_TOKEN = "test-admin-token"

    def _admin_headers(self):
        return {"X-Admin-Token": "test-admin-token"}

    def test_create_experiment(self):
        from main import app

        client = TestClient(app)
        resp = client.post(
            "/experiments",
            json={
                "experiment_id": "e1",
                "control_template": "islamic_context",
                "variants": [],
            },
            headers=self._admin_headers(),
        )
        assert resp.status_code == 200
        assert resp.json()["experiment_id"] == "e1"

    def test_create_experiment_unauthorized(self):
        from main import app

        client = TestClient(app)
        resp = client.post(
            "/experiments",
            json={
                "experiment_id": "e1",
                "control_template": "islamic_context",
                "variants": [],
            },
        )
        # Either 401, 403, or 503 (ADMIN_TOKEN not configured)
        assert resp.status_code in (401, 403, 503)

    def test_kill_and_resume_experiment(self):
        from main import app

        client = TestClient(app)
        headers = self._admin_headers()
        # Create
        client.post(
            "/experiments",
            json={
                "experiment_id": "e2",
                "control_template": "islamic_context",
                "variants": [],
            },
            headers=headers,
        )
        # Kill
        resp = client.post("/experiments/e2/kill", headers=headers)
        assert resp.status_code == 200
        assert resp.json()["kill_switch"] is True
        # Resume
        resp = client.post("/experiments/e2/resume", headers=headers)
        assert resp.status_code == 200
        assert resp.json()["kill_switch"] is False

    def test_delete_experiment(self):
        from main import app

        client = TestClient(app)
        headers = self._admin_headers()
        client.post(
            "/experiments",
            json={
                "experiment_id": "e3",
                "control_template": "islamic_context",
                "variants": [],
            },
            headers=headers,
        )
        resp = client.delete("/experiments/e3", headers=headers)
        assert resp.status_code == 200
