"""Comprehensive test suite for runtime confidence threshold tuning (#270).

Validates:
- Dynamic threshold configuration and boundary safeguards
- Per-topic Islamic category customization (fiqh, hadith, aqidah, tafsir)
- Source-specific confidence constraints
- Gradual stepwise adjustments
- Audit trail diffs and rollbacks
- A/B testing variant routing and metrics
- Automated tuning recommendations
- Admin REST API endpoints and authentication
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from confidence import ConfidenceBand, ConfidenceSignals, assess, band_for, thresholds
from confidence_tuning import (
    AuditAction,
    ConfidenceThresholdConfig,
    TopicThresholdConfig,
    default_threshold_config,
    reset_tuning_manager,
)
from confidence_tuning_api import router as confidence_tuning_router

test_app = FastAPI()
test_app.include_router(confidence_tuning_router)


@pytest.fixture(autouse=True)
def clean_tuning_manager():
    """Ensure a fresh manager instance for every test."""
    mgr = reset_tuning_manager()
    yield mgr
    reset_tuning_manager()


class TestDefaultConfiguration:
    def test_default_config_structure(self):
        cfg = default_threshold_config()
        assert cfg.low_threshold == 0.40
        assert cfg.high_threshold == 0.70
        assert cfg.scholar_queue_threshold == 0.40
        assert cfg.high_stakes_penalty == 0.15
        assert "fiqh" in cfg.topics
        assert "hadith" in cfg.topics
        assert "aqidah" in cfg.topics
        assert "quran" in cfg.sources

    def test_initial_audit_log(self, clean_tuning_manager):
        history = clean_tuning_manager.get_audit_trail()
        assert len(history) == 1
        assert history[0].action == AuditAction.INIT
        assert history[0].author == "system"


class TestSafeguardsAndValidation:
    def test_invalid_low_high_spread(self):
        with pytest.raises(ValueError, match="must exceed low_threshold"):
            ConfidenceThresholdConfig(low_threshold=0.70, high_threshold=0.65)

    def test_threshold_out_of_bounds(self):
        with pytest.raises(ValueError):
            ConfidenceThresholdConfig(low_threshold=0.01, high_threshold=0.70)

        with pytest.raises(ValueError):
            ConfidenceThresholdConfig(low_threshold=0.40, high_threshold=0.99)

    def test_invalid_high_stakes_penalty(self):
        with pytest.raises(ValueError):
            ConfidenceThresholdConfig(high_stakes_penalty=0.85)

    def test_topic_spread_validation(self):
        with pytest.raises(ValueError, match="must exceed low_threshold"):
            TopicThresholdConfig(low_threshold=0.80, high_threshold=0.82)


class TestDynamicThresholdsAndTopics:
    def test_effective_thresholds_fallback_to_global(self, clean_tuning_manager):
        t = clean_tuning_manager.get_effective_thresholds()
        assert t["low"] == 0.40
        assert t["high"] == 0.70

    def test_fiqh_topic_thresholds(self, clean_tuning_manager):
        t_global = clean_tuning_manager.get_effective_thresholds()
        t_fiqh = clean_tuning_manager.get_effective_thresholds(topic="fiqh")

        # Fiqh requires higher caution
        assert t_fiqh["low"] == 0.50
        assert t_fiqh["high"] == 0.75
        assert t_fiqh["high_stakes_penalty"] == 0.20
        assert t_fiqh["low"] > t_global["low"]

    def test_source_threshold_elevation(self, clean_tuning_manager):
        # Quran source has min_confidence=0.60, which elevates low threshold
        t_quran = clean_tuning_manager.get_effective_thresholds(source="quran")
        assert t_quran["low"] == 0.60
        assert t_quran["retrieval_min_score"] == 0.35

    def test_runtime_band_resolution(self, clean_tuning_manager):
        # Score 0.45 is uncertain under general, but abstain under fiqh (low=0.50)
        band_gen = clean_tuning_manager.resolve_band(0.45, topic="general")
        band_fiqh = clean_tuning_manager.resolve_band(0.45, topic="fiqh")

        assert band_gen == "uncertain"
        assert band_fiqh == "abstain"

    def test_scholar_queue_decision(self, clean_tuning_manager):
        # Non-religious never queues
        assert not clean_tuning_manager.should_queue_scholar(0.30, is_religious=False)

        # General religious below 0.40 queues
        assert clean_tuning_manager.should_queue_scholar(0.35, is_religious=True)
        # Score 0.45 does not queue for general (threshold 0.35/0.40)
        assert not clean_tuning_manager.should_queue_scholar(0.45, is_religious=True, topic="general")
        # But score 0.45 DOES queue for fiqh (threshold 0.50)
        assert clean_tuning_manager.should_queue_scholar(0.45, is_religious=True, topic="fiqh")


class TestGradualAdjustAndRollback:
    def test_gradual_adjustment_steps(self, clean_tuning_manager):
        initial_version = clean_tuning_manager.version
        entries = clean_tuning_manager.gradual_adjust(
            target_low=0.46,
            target_high=0.76,
            steps=3,
            author="ops_lead",
            reason="Gradual ramp to higher safety",
        )
        assert len(entries) == 3
        assert clean_tuning_manager.version == initial_version + 3
        assert clean_tuning_manager.current_config.low_threshold == 0.46
        assert clean_tuning_manager.current_config.high_threshold == 0.76

        for e in entries:
            assert e.action == AuditAction.GRADUAL_STEP
            assert e.author == "ops_lead"

    def test_rollback_to_previous(self, clean_tuning_manager):
        orig_cfg = clean_tuning_manager.current_config.model_copy(deep=True)
        orig_low = orig_cfg.low_threshold

        new_cfg = orig_cfg.model_copy(deep=True)
        new_cfg.low_threshold = 0.50
        new_cfg.high_threshold = 0.80
        clean_tuning_manager.update_config(new_cfg, author="admin", reason="Test bump")

        assert clean_tuning_manager.current_config.low_threshold == 0.50

        # Rollback
        rollback_entry = clean_tuning_manager.rollback(author="admin", reason="Emergency revert")
        assert rollback_entry.action == AuditAction.ROLLBACK
        assert clean_tuning_manager.current_config.low_threshold == orig_low

    def test_rollback_to_specific_version(self, clean_tuning_manager):
        v1_config = clean_tuning_manager.current_config.model_copy(deep=True)

        # Update 1
        cfg2 = v1_config.model_copy(deep=True)
        cfg2.low_threshold = 0.45
        clean_tuning_manager.update_config(cfg2, reason="Change 1")

        # Update 2
        cfg3 = v1_config.model_copy(deep=True)
        cfg3.low_threshold = 0.52
        clean_tuning_manager.update_config(cfg3, reason="Change 2")

        assert clean_tuning_manager.version == 3

        # Roll back specifically to version 1
        clean_tuning_manager.rollback(to_version=1, reason="Revert to baseline")
        assert clean_tuning_manager.current_config.low_threshold == 0.40


class TestABExperiments:
    def test_ab_experiment_lifecycle(self, clean_tuning_manager):
        candidate_cfg = clean_tuning_manager.current_config.model_copy(deep=True)
        candidate_cfg.low_threshold = 0.45
        candidate_cfg.high_threshold = 0.75

        exp = clean_tuning_manager.start_ab_experiment(
            experiment_id="exp-conf-01",
            name="Conservative Threshold Test",
            variant_b_config=candidate_cfg,
            traffic_split=0.50,
            author="ai_team",
        )
        assert exp.active is True
        assert clean_tuning_manager.ab_experiment is not None

        # Verify deterministic routing
        variant_name, active_cfg = clean_tuning_manager._determine_variant_config("user-12345")
        assert variant_name in ("variant_a", "variant_b")

        # Record turns under experiment
        clean_tuning_manager.record_turn(
            score=0.42,
            band="uncertain",
            is_religious=True,
            feedback=True,
            request_key="test-key-1",
        )
        assert (exp.metrics_a.total_evaluations + exp.metrics_b.total_evaluations) == 1

        # Stop and promote Variant B
        clean_tuning_manager.stop_ab_experiment(promote_variant_b=True, author="ai_team")
        assert clean_tuning_manager.ab_experiment is None
        assert clean_tuning_manager.current_config.low_threshold == 0.45


class TestTelemetryAndRecommendations:
    def test_recommendations_with_insufficient_data(self, clean_tuning_manager):
        recs = clean_tuning_manager.generate_recommendations()
        assert len(recs) == 1
        assert recs[0]["type"] == "insufficient_data"

    def test_recommendations_high_abstain_rate(self, clean_tuning_manager):
        # Simulate 10 turns with 8 abstains (80% abstain rate)
        for _ in range(8):
            clean_tuning_manager.record_turn(score=0.20, band="abstain", is_religious=False)
        for _ in range(2):
            clean_tuning_manager.record_turn(score=0.85, band="confident", is_religious=False, feedback=True)

        recs = clean_tuning_manager.generate_recommendations()
        types = [r["type"] for r in recs]
        assert "high_abstain_rate" in types
        abstain_rec = next(r for r in recs if r["type"] == "high_abstain_rate")
        assert abstain_rec["action"] == "lower_low_threshold"

    def test_recommendations_low_satisfaction(self, clean_tuning_manager):
        # Simulate turns with high confident answers but negative user feedback
        for _ in range(8):
            clean_tuning_manager.record_turn(score=0.85, band="confident", is_religious=False, feedback=False)
        for _ in range(2):
            clean_tuning_manager.record_turn(score=0.85, band="confident", is_religious=False, feedback=True)

        recs = clean_tuning_manager.generate_recommendations()
        types = [r["type"] for r in recs]
        assert "low_satisfaction_quality" in types
        sat_rec = next(r for r in recs if r["type"] == "low_satisfaction_quality")
        assert sat_rec["action"] == "raise_high_threshold"


class TestConfidenceModuleIntegration:
    def test_confidence_thresholds_reflects_tuned_manager(self, clean_tuning_manager):
        # Update config in tuning manager
        new_cfg = clean_tuning_manager.current_config.model_copy(deep=True)
        new_cfg.low_threshold = 0.48
        clean_tuning_manager.update_config(new_cfg)

        # confidence.thresholds() should reflect the update
        t = thresholds()
        assert t["low"] == 0.48

    def test_confidence_band_for_with_topic(self, clean_tuning_manager):
        # Fiqh low threshold is 0.50
        assert band_for(0.45, topic="fiqh") is ConfidenceBand.ABSTAIN
        assert band_for(0.55, topic="fiqh") is ConfidenceBand.UNCERTAIN

    def test_assess_with_signals_topic(self, clean_tuning_manager):
        signals = ConfidenceSignals(
            self_consistency=0.55,
            citation_verification=0.55,
            is_religious=True,
            topic="fiqh",
        )
        assessment = assess(signals)
        # Signals fused score is 0.55. Under fiqh (high=0.75, low=0.50), it is uncertain.
        assert assessment.band is ConfidenceBand.UNCERTAIN


class TestAdminAPIEndpoints:
    @pytest.fixture
    def client(self, monkeypatch):
        monkeypatch.setenv("ADMIN_TOKEN", "super-secret-admin-token")
        return TestClient(test_app, raise_server_exceptions=False)

    def test_auth_rejection(self, client):
        # Missing token
        resp = client.get("/admin/confidence/config")
        assert resp.status_code in (403, 503)

        # Invalid token
        resp = client.get(
            "/admin/confidence/config",
            headers={"X-Admin-Token": "wrong-token"},
        )
        assert resp.status_code == 403

    def test_get_and_update_config_api(self, client):
        headers = {"X-Admin-Token": "super-secret-admin-token"}

        # GET config
        resp = client.get("/admin/confidence/config", headers=headers)
        assert resp.status_code == 200
        data = resp.json()
        assert "config" in data
        assert data["config"]["low_threshold"] == 0.40

        # PUT config
        cfg = data["config"]
        cfg["low_threshold"] = 0.42
        cfg["high_threshold"] = 0.72
        update_resp = client.put(
            "/admin/confidence/config",
            json={"config": cfg, "reason": "API test update", "author": "tester"},
            headers=headers,
        )
        assert update_resp.status_code == 200
        assert update_resp.json()["status"] == "success"

        # Verify update persisted
        resp2 = client.get("/admin/confidence/config", headers=headers)
        assert resp2.json()["config"]["low_threshold"] == 0.42

    def test_topic_endpoints_api(self, client):
        headers = {"X-Admin-Token": "super-secret-admin-token"}

        # Add custom topic
        topic_payload = {
            "low_threshold": 0.52,
            "high_threshold": 0.78,
            "scholar_queue_threshold": 0.52,
            "high_stakes_penalty": 0.18,
            "description": "Inheritance law requires strict verification",
        }
        resp = client.put("/admin/confidence/topics/faraid", json=topic_payload, headers=headers)
        assert resp.status_code == 200
        assert resp.json()["status"] == "success"

        # Delete topic
        del_resp = client.delete("/admin/confidence/topics/faraid", headers=headers)
        assert del_resp.status_code == 200
        assert del_resp.json()["removed_topic"] == "faraid"

    def test_gradual_adjust_and_rollback_api(self, client):
        headers = {"X-Admin-Token": "super-secret-admin-token"}

        # Gradual adjust
        adj_payload = {
            "target_low": 0.44,
            "target_high": 0.74,
            "steps": 2,
            "reason": "Gradual API test",
            "author": "tester",
        }
        resp = client.post("/admin/confidence/gradual-adjust", json=adj_payload, headers=headers)
        assert resp.status_code == 200
        assert resp.json()["steps_applied"] == 2

        # Rollback
        rb_resp = client.post(
            "/admin/confidence/rollback",
            json={"reason": "Rollback test", "author": "tester"},
            headers=headers,
        )
        assert rb_resp.status_code == 200
        assert rb_resp.json()["status"] == "success"

    def test_ab_test_endpoints_api(self, client):
        headers = {"X-Admin-Token": "super-secret-admin-token"}

        # Fetch active config for Variant B
        cfg_resp = client.get("/admin/confidence/config", headers=headers)
        var_b = cfg_resp.json()["config"]
        var_b["low_threshold"] = 0.44

        start_payload = {
            "experiment_id": "api-ab-exp-1",
            "name": "API AB Experiment",
            "traffic_split": 0.50,
            "variant_b": var_b,
            "author": "api_tester",
        }
        resp = client.post("/admin/confidence/ab-test/start", json=start_payload, headers=headers)
        assert resp.status_code == 200
        assert resp.json()["status"] == "success"

        # Check status
        status_resp = client.get("/admin/confidence/ab-test", headers=headers)
        assert status_resp.status_code == 200
        assert status_resp.json()["active"] is True

        # Stop test
        stop_resp = client.post(
            "/admin/confidence/ab-test/stop",
            json={"promote_variant_b": False, "author": "api_tester"},
            headers=headers,
        )
        assert stop_resp.status_code == 200
        assert stop_resp.json()["status"] == "success"

    def test_audit_trail_and_metrics_api(self, client):
        headers = {"X-Admin-Token": "super-secret-admin-token"}

        # Audit trail
        audit_resp = client.get("/admin/confidence/audit-trail", headers=headers)
        assert audit_resp.status_code == 200
        assert "audit_trail" in audit_resp.json()

        # Metrics
        metrics_resp = client.get("/admin/confidence/metrics", headers=headers)
        assert metrics_resp.status_code == 200
        assert "global" in metrics_resp.json()

        # Recommendations
        recs_resp = client.get("/admin/confidence/recommendations", headers=headers)
        assert recs_resp.status_code == 200
        assert "recommendations" in recs_resp.json()

        # Feedback
        fb_resp = client.post(
            "/admin/confidence/feedback",
            json={
                "score": 0.85,
                "band": "confident",
                "is_religious": True,
                "topic": "fiqh",
                "feedback": True,
            },
            headers=headers,
        )
        assert fb_resp.status_code == 200
        assert fb_resp.json()["status"] == "recorded"
