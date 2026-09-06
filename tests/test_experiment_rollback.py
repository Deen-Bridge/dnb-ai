"""Comprehensive test suite for experiment rollback, canary deployments, and feature flags (#269).

Validates:
- Feature flag percentage rollouts and user/role targeting
- A/B testing across models, prompts, and configurations
- Canary deployments with percentage-based routing
- Automated rollback triggers (error rate, P95 latency, consecutive errors)
- Preserving seamless user experience during rollbacks (zero user-visible failures)
- Configuration version history and instant rollback
- Audit trail logging of all state changes
- Admin REST API endpoints with authentication
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from experiment_rollback import (
    AuditAction,
    ExperimentStatus,
    ExperimentTargetType,
    ExperimentVariant,
    FeatureFlag,
    RollbackTriggerConfig,
    reset_rollback_manager,
)
from experiment_rollback_api import router as experiment_rollback_router

test_app = FastAPI()
test_app.include_router(experiment_rollback_router)


@pytest.fixture(autouse=True)
def clean_rollback_manager():
    """Ensure a fresh manager instance for every test."""
    mgr = reset_rollback_manager()
    yield mgr
    reset_rollback_manager()


class TestFeatureFlags:
    def test_disabled_flag_blocks_all(self, clean_rollback_manager):
        flag = FeatureFlag(
            flag_id="new_stream_engine",
            name="Stream Engine V2",
            enabled=False,
            rollout_percentage=100.0,
        )
        clean_rollback_manager.register_flag(flag)
        assert not flag.is_enabled_for("user-123")

    def test_user_whitelisting(self, clean_rollback_manager):
        flag = FeatureFlag(
            flag_id="beta_search",
            name="Beta Search",
            enabled=True,
            rollout_percentage=0.0,
            allowed_users=["vip_user_1", "vip_user_2"],
        )
        clean_rollback_manager.register_flag(flag)
        assert flag.is_enabled_for("vip_user_1")
        assert not flag.is_enabled_for("random_user")

    def test_role_targeting(self, clean_rollback_manager):
        flag = FeatureFlag(
            flag_id="admin_tools",
            name="Admin Tools",
            enabled=True,
            rollout_percentage=0.0,
            allowed_roles=["admin", "scholar"],
        )
        clean_rollback_manager.register_flag(flag)
        assert flag.is_enabled_for("user-1", role="admin")
        assert not flag.is_enabled_for("user-1", role="standard")

    def test_gradual_percentage_rollout(self, clean_rollback_manager):
        flag = FeatureFlag(
            flag_id="gradual_feature",
            name="Gradual Feature",
            enabled=True,
            rollout_percentage=50.0,
        )
        clean_rollback_manager.register_flag(flag)

        # Evaluate across 100 deterministic session keys
        enabled_count = sum(1 for i in range(100) if flag.is_enabled_for(f"session-{i}"))
        # 50% rollout should yield approximately 40-60 enabled sessions
        assert 35 <= enabled_count <= 65

    def test_update_flag_audit(self, clean_rollback_manager):
        flag = FeatureFlag(
            flag_id="flag_audit_test",
            name="Audit Test",
            enabled=False,
        )
        clean_rollback_manager.register_flag(flag)
        updated = clean_rollback_manager.update_flag(
            flag_id="flag_audit_test",
            enabled=True,
            rollout_percentage=25.0,
            author="lead_engineer",
            reason="Canary launch",
        )
        assert updated.enabled is True
        assert updated.rollout_percentage == 25.0

        audit_entries = clean_rollback_manager.get_audit_trail(experiment_id="flag_audit_test")
        assert len(audit_entries) == 1
        assert audit_entries[0].action == AuditAction.FLAG_TOGGLE
        assert audit_entries[0].author == "lead_engineer"


class TestExperimentCanaryAndAssignment:
    def test_create_experiment_structure(self, clean_rollback_manager):
        ctl = ExperimentVariant(name="control", model="gemini-1.5-flash", weight=1.0)
        v1 = ExperimentVariant(name="candidate", model="gemini-1.5-pro", weight=1.0)

        exp = clean_rollback_manager.create_experiment(
            experiment_id="model_eval_01",
            name="Model Comparison",
            target_type=ExperimentTargetType.MODEL,
            control=ctl,
            variants=[v1],
            canary_percentage=20.0,
        )
        assert exp.status == ExperimentStatus.ACTIVE
        assert exp.canary_percentage == 20.0
        assert len(exp.version_history) == 1

    def test_canary_routing_distribution(self, clean_rollback_manager):
        ctl = ExperimentVariant(name="control", weight=1.0)
        v1 = ExperimentVariant(name="canary_v1", weight=1.0)

        clean_rollback_manager.create_experiment(
            experiment_id="canary_exp",
            name="Canary Test",
            control=ctl,
            variants=[v1],
            canary_percentage=25.0,
        )

        # Over 100 sessions, ~25% should reach canary_v1 and ~75% should receive control
        canary_count = 0
        for i in range(100):
            variant, was_fallback = clean_rollback_manager.assign_variant("canary_exp", f"sess-{i}")
            assert not was_fallback
            if variant.name == "canary_v1":
                canary_count += 1

        assert 15 <= canary_count <= 35


class TestAutomatedRollback:
    def test_automated_rollback_on_high_error_rate(self, clean_rollback_manager):
        ctl = ExperimentVariant(name="control", weight=1.0)
        cand = ExperimentVariant(name="buggy_variant", weight=1.0)
        triggers = RollbackTriggerConfig(
            max_error_rate=0.10,  # 10%
            min_sample_size=10,
            auto_rollback_enabled=True,
        )

        clean_rollback_manager.create_experiment(
            experiment_id="unstable_model",
            name="Unstable Model Experiment",
            control=ctl,
            variants=[cand],
            triggers=triggers,
        )

        # Simulate 8 successes and 2 errors (20% error rate > 10% threshold)
        for _ in range(8):
            clean_rollback_manager.record_turn_result(
                "unstable_model", "buggy_variant", latency_ms=100.0, is_error=False
            )
        triggered_1 = clean_rollback_manager.record_turn_result(
            "unstable_model", "buggy_variant", latency_ms=100.0, is_error=True
        )
        assert not triggered_1  # 1 error / 9 turns = 11%, but sample size < 10

        triggered_2 = clean_rollback_manager.record_turn_result(
            "unstable_model", "buggy_variant", latency_ms=100.0, is_error=True
        )
        assert triggered_2  # 10th sample: 2 errors / 10 turns = 20% > 10% -> Rollback triggered!

        exp = clean_rollback_manager.get_experiment("unstable_model")
        assert exp.status == ExperimentStatus.ROLLED_BACK
        assert exp.kill_switch is True
        assert "exceeded trigger threshold" in exp.last_rollback_reason

        # Ensure subsequent user turns seamlessly fall back to control with zero failure
        chosen, was_fallback = clean_rollback_manager.assign_variant("unstable_model", "user_xyz")
        assert chosen.name == "control"
        assert was_fallback is True

    def test_automated_rollback_on_high_latency(self, clean_rollback_manager):
        ctl = ExperimentVariant(name="control", weight=1.0)
        cand = ExperimentVariant(name="slow_variant", weight=1.0)
        triggers = RollbackTriggerConfig(
            max_p95_latency_ms=1000.0,  # 1.0s
            min_sample_size=10,
            auto_rollback_enabled=True,
        )

        clean_rollback_manager.create_experiment(
            experiment_id="slow_model",
            name="Slow Model Experiment",
            control=ctl,
            variants=[cand],
            triggers=triggers,
        )

        # Record 10 turns with 2500ms latency
        for _ in range(9):
            clean_rollback_manager.record_turn_result("slow_model", "slow_variant", latency_ms=2500.0, is_error=False)
        triggered = clean_rollback_manager.record_turn_result(
            "slow_model", "slow_variant", latency_ms=2500.0, is_error=False
        )
        assert triggered is True

        exp = clean_rollback_manager.get_experiment("slow_model")
        assert exp.status == ExperimentStatus.ROLLED_BACK
        assert "P95 latency" in exp.last_rollback_reason

    def test_automated_rollback_on_consecutive_errors(self, clean_rollback_manager):
        ctl = ExperimentVariant(name="control", weight=1.0)
        cand = ExperimentVariant(name="failing_variant", weight=1.0)
        triggers = RollbackTriggerConfig(
            max_consecutive_errors=3,
            min_sample_size=3,
            auto_rollback_enabled=True,
        )

        clean_rollback_manager.create_experiment(
            experiment_id="crashing_variant_exp",
            name="Crashing Experiment",
            control=ctl,
            variants=[cand],
            triggers=triggers,
        )

        clean_rollback_manager.record_turn_result("crashing_variant_exp", "failing_variant", 50.0, is_error=True)
        clean_rollback_manager.record_turn_result("crashing_variant_exp", "failing_variant", 50.0, is_error=True)
        triggered = clean_rollback_manager.record_turn_result(
            "crashing_variant_exp", "failing_variant", 50.0, is_error=True
        )
        assert triggered is True

        exp = clean_rollback_manager.get_experiment("crashing_variant_exp")
        assert exp.status == ExperimentStatus.ROLLED_BACK


class TestManualRollbackAndVersions:
    def test_version_snapshots_and_rollback(self, clean_rollback_manager):
        ctl = ExperimentVariant(name="control", weight=1.0)
        v1 = ExperimentVariant(name="cand_1", weight=1.0)

        exp = clean_rollback_manager.create_experiment(
            experiment_id="version_exp",
            name="Version Test",
            control=ctl,
            variants=[v1],
            canary_percentage=10.0,
        )
        assert exp.current_version == 1

        # Ramp to 25%
        clean_rollback_manager.set_canary_rollout("version_exp", 25.0, reason="Ramp 25%")
        assert exp.current_version == 2
        assert exp.canary_percentage == 25.0

        # Ramp to 50%
        clean_rollback_manager.set_canary_rollout("version_exp", 50.0, reason="Ramp 50%")
        assert exp.current_version == 3
        assert exp.canary_percentage == 50.0

        # Roll back to immediate previous version (v2: 25%)
        clean_rollback_manager.rollback("version_exp", reason="Rollback to previous ramp")
        assert exp.current_version == 4
        assert exp.canary_percentage == 25.0

        # Roll back to exact version 1 (10%)
        clean_rollback_manager.rollback("version_exp", to_version=1, reason="Rollback to initial v1")
        assert exp.current_version == 5
        assert exp.canary_percentage == 10.0

    def test_kill_and_resume(self, clean_rollback_manager):
        ctl = ExperimentVariant(name="control", weight=1.0)
        v1 = ExperimentVariant(name="cand_1", weight=1.0)

        exp = clean_rollback_manager.create_experiment(
            experiment_id="kill_resume_exp",
            name="Kill Resume Test",
            control=ctl,
            variants=[v1],
        )
        assert not exp.kill_switch

        clean_rollback_manager.kill_experiment("kill_resume_exp", reason="Emergency test")
        assert exp.kill_switch is True
        assert exp.status == ExperimentStatus.PAUSED

        clean_rollback_manager.resume_experiment("kill_resume_exp", reason="Resume test")
        assert exp.kill_switch is False
        assert exp.status == ExperimentStatus.ACTIVE


class TestAdminAPI:
    @pytest.fixture
    def client(self, monkeypatch):
        monkeypatch.setenv("ADMIN_TOKEN", "test-admin-secret-token")
        return TestClient(test_app, raise_server_exceptions=False)

    def test_admin_auth_gating(self, client):
        resp = client.get("/admin/experiments")
        assert resp.status_code in (403, 503)

        resp = client.get("/admin/experiments", headers={"X-Admin-Token": "bad-token"})
        assert resp.status_code == 403

    def test_experiment_crud_and_rollout_api(self, client):
        headers = {"X-Admin-Token": "test-admin-secret-token"}

        # 1. Create experiment
        payload = {
            "experiment_id": "api_test_exp",
            "name": "API Test Experiment",
            "target_type": "prompt",
            "control": {"name": "ctl", "weight": 1.0},
            "variants": [{"name": "cand", "weight": 1.0}],
            "canary_percentage": 10.0,
            "author": "api_admin",
            "reason": "Creation via API",
        }
        create_resp = client.post("/admin/experiments", json=payload, headers=headers)
        assert create_resp.status_code == 200
        assert create_resp.json()["status"] == "success"

        # 2. Get details
        get_resp = client.get("/admin/experiments/api_test_exp", headers=headers)
        assert get_resp.status_code == 200
        assert get_resp.json()["experiment"]["canary_percentage"] == 10.0

        # 3. Update canary rollout to 50%
        canary_resp = client.post(
            "/admin/experiments/api_test_exp/canary",
            json={"percentage": 50.0, "author": "api_admin", "reason": "Ramping up"},
            headers=headers,
        )
        assert canary_resp.status_code == 200
        assert canary_resp.json()["experiment"]["canary_percentage"] == 50.0

        # 4. Rollback via API
        rb_resp = client.post(
            "/admin/experiments/api_test_exp/rollback",
            json={"author": "api_admin", "reason": "Reverting canary ramp"},
            headers=headers,
        )
        assert rb_resp.status_code == 200
        assert rb_resp.json()["experiment"]["canary_percentage"] == 10.0

        # 5. Record turn metrics via API
        turn_resp = client.post(
            "/admin/experiments/api_test_exp/record-turn",
            json={"variant_name": "cand", "latency_ms": 120.0, "is_error": False},
            headers=headers,
        )
        assert turn_resp.status_code == 200
        assert turn_resp.json()["status"] == "recorded"

        # 6. Check metrics
        metrics_resp = client.get("/admin/experiments/api_test_exp/metrics", headers=headers)
        assert metrics_resp.status_code == 200
        assert "cand" in metrics_resp.json()["metrics"]

        # 7. Check audit trail
        audit_resp = client.get("/admin/experiments-audit-trail?experiment_id=api_test_exp", headers=headers)
        assert audit_resp.status_code == 200
        assert audit_resp.json()["count"] >= 3

    def test_feature_flag_api(self, client):
        headers = {"X-Admin-Token": "test-admin-secret-token"}

        # 1. Create flag
        flag_payload = {
            "flag_id": "api_flag",
            "name": "API Flag",
            "enabled": True,
            "rollout_percentage": 25.0,
            "allowed_users": ["user_1"],
        }
        post_resp = client.post("/admin/feature-flags", json=flag_payload, headers=headers)
        assert post_resp.status_code == 200
        assert post_resp.json()["status"] == "success"

        # 2. Evaluate flag (public endpoint)
        eval_resp = client.get("/admin/feature-flags/api_flag/evaluate?session_id=user_1")
        assert eval_resp.status_code == 200
        assert eval_resp.json()["is_enabled"] is True

        # 3. Update flag to 100%
        put_resp = client.put(
            "/admin/feature-flags/api_flag",
            json={"rollout_percentage": 100.0, "author": "ops"},
            headers=headers,
        )
        assert put_resp.status_code == 200
        assert put_resp.json()["flag"]["rollout_percentage"] == 100.0

        # 4. List flags
        list_resp = client.get("/admin/feature-flags", headers=headers)
        assert list_resp.status_code == 200
        assert len(list_resp.json()["flags"]) >= 1
