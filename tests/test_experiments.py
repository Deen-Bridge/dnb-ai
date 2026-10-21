"""Offline rollback, canary, persistence, and concurrency regressions."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest
from pydantic import ValidationError

from experiments import ExperimentManager, RollbackPolicy
from prompts.registry import ExperimentConfig, PromptRegistry, Variant
from prompts.template import PromptTemplate


@pytest.fixture
def registry():
    registry = PromptRegistry()
    registry.register(PromptTemplate("stable", "1.0.0", "Stable response"))
    registry.register(PromptTemplate("candidate", "1.0.0", "Candidate in {response_language}", ("response_language",)))
    return registry


@pytest.fixture
def config():
    return ExperimentConfig(
        "test",
        Variant("control", "stable"),
        [Variant("candidate", "candidate", model_name="candidate-model")],
        traffic_percent=100,
    )


@pytest.fixture
def manager(registry, config):
    manager = ExperimentManager(registry, "stable-model")
    manager.register_experiment(config)
    manager.activate("test")
    return manager


def test_activation_is_explicit_and_deactivation_preserves_versions(registry, config):
    manager = ExperimentManager(registry, "stable-model")
    manager.register_experiment(config)
    assert manager.choose("session") is None
    manager.activate("test")
    assert manager.choose("session").variant.name == "candidate"
    manager.deactivate("test")
    assert manager.choose("session") is None
    assert len(manager.versions("test")) == 1
    assert [e["event"] for e in manager.events("test")] == ["registered", "activated", "deactivated"]


@pytest.mark.parametrize("percent", [0, 10, 50, 100])
def test_canary_percentage_and_sticky_assignment(registry, config, percent):
    manager = ExperimentManager(registry, "stable-model")
    manager.register_experiment(replace(config, traffic_percent=percent))
    manager.activate("test")
    treated = 0
    for i in range(1000):
        chosen = manager.choose(str(i))
        assert manager.choose(str(i)).variant.name == chosen.variant.name
        treated += not chosen.is_control
    assert abs(treated / 10 - percent) < 4


def test_legacy_weighted_ab_assignment_is_preserved(registry, config):
    config.traffic_percent = None
    config.variants[0].weight = 3
    manager = ExperimentManager(registry, "stable-model")
    manager.register_experiment(config)
    manager.activate("test")
    treated = sum(not manager.choose(str(i)).is_control for i in range(1000))
    assert 700 < treated < 800


def test_snapshots_pin_prompt_model_and_generation_settings(registry, config):
    config.variants[0].generation_config = {"temperature": 0.2}
    manager = ExperimentManager(registry, "stable-model")
    manager.register_experiment(config)
    manager.activate("test")
    registry.register(PromptTemplate("candidate", "2.0.0", "Replacement"))
    config.variants[0].model_name = "mutated"
    config.variants[0].generation_config["temperature"] = 1.9
    chosen = manager.choose("session")
    assert chosen.variant.model_name == "candidate-model"
    assert chosen.variant.generation_config == {"temperature": 0.2}
    assert chosen.template.version == "1.0.0"
    assert chosen.template.render(response_language="en") == "Candidate in en"


def test_error_threshold_needs_minimum_samples_and_strict_breach(registry, config):
    manager = ExperimentManager(registry, "stable-model")
    manager.register_experiment(config, RollbackPolicy(min_samples=2, max_error_rate=0.5))
    manager.activate("test")
    chosen = manager.choose("session")
    assert not manager.record(chosen, 10, True)
    assert not manager.record(chosen, 10, False)  # exactly at the threshold
    assert manager.record(chosen, 10, True)
    stable = manager.choose("session")
    assert stable.is_control and stable.rolled_back
    assert stable.variant.model_name == "stable-model"
    assert not manager.record(chosen, 10, True)  # late observation, no second rollback
    events = [event for event in manager.events("test") if event["event"] == "rolled_back"]
    assert len(events) == 1
    assert events[0]["reason"] == "error_rate"


def test_latency_threshold_uses_p95(registry, config):
    manager = ExperimentManager(registry, "stable-model")
    manager.register_experiment(config, RollbackPolicy(min_samples=3, max_error_rate=None, max_p95_latency_ms=100))
    manager.activate("test")
    chosen = manager.choose("session")
    assert not manager.record(chosen, 50, False)
    assert not manager.record(chosen, 50, False)
    assert manager.record(chosen, 101, False)
    assert manager.events("test")[-1]["reason"] == "p95_latency"


def test_metrics_window_is_bounded_and_control_errors_do_not_retrigger(manager, config):
    manager.register_experiment(config, RollbackPolicy(min_samples=3, window_size=3, max_error_rate=0.5))
    manager.activate("test")
    chosen = manager.choose("session")
    for error in [True, False, False, False]:
        assert not manager.record(chosen, 20, error)
    assert manager.metrics("test") == [
        {"version": 2, "variant": "candidate", "window_attempts": 3, "errors": 0, "error_rate": 0, "p95_latency_ms": 20}
    ]
    stable = manager.stable_decision("test")
    for _ in range(30):
        assert not manager.record(stable, 60000, True)
    assert manager.choose("session").variant.name == "candidate"


def test_rollback_targets_previous_stable_configuration_and_preserves_history(manager, config):
    config.control.model_name = "new-control"
    manager.register_experiment(config)
    assert manager.choose("session").version == 1
    manager.activate("test")
    manager.rollback("test")
    assert manager.choose("session").variant.model_name == "stable-model"
    assert len(manager.versions("test")) == 2
    with pytest.raises(ValueError, match="new configuration version"):
        manager.set_killed("test", False)
    manager.register_experiment(config)
    manager.activate("test")
    manager.mark_stable("test", 3)
    manager.rollback("test")
    assert manager.choose("session").variant.model_name == "new-control"


def test_manual_kill_and_resume_remain_compatible(manager):
    manager.set_killed("test", True)
    assert manager.choose("session").is_control
    manager.set_killed("test", False)
    assert not manager.choose("session").is_control


def test_experiment_created_disabled_can_be_resumed(registry, config):
    manager = ExperimentManager(registry, "stable-model")
    manager.register_experiment(replace(config, kill_switch=True))
    manager.activate("test")
    assert manager.choose("session").is_control
    manager.set_killed("test", False)
    assert not manager.choose("session").is_control


def test_restart_restores_pinned_templates_and_rollback_across_workers(tmp_path, registry, config):
    database = str(tmp_path / "experiments.sqlite3")
    first = ExperimentManager(registry, "stable-model", database, default_generation_config={"temperature": 0.7})
    first.register_experiment(config, RollbackPolicy(min_samples=1, max_error_rate=0))
    first.activate("test")
    chosen = first.choose("session")
    restarted = ExperimentManager(
        PromptRegistry(), "changed-default", database, default_generation_config={"temperature": 1.9}
    )
    assert restarted.choose("session").template.body == chosen.template.body
    assert restarted.choose("session").variant.generation_config == {"temperature": 0.7}
    with ThreadPoolExecutor(max_workers=2) as workers:
        results = list(workers.map(lambda worker: worker.record(chosen, 20, True), [first, restarted]))
    assert results.count(True) == 1
    assert first.choose("session").variant.model_name == "stable-model"
    assert first.events("test")[-1]["reason"] == "error_rate"


def test_deletion_and_id_reuse_do_not_reuse_inflight_versions(manager, config):
    old = manager.choose("session")
    manager.unregister_experiment("test")
    assert manager.choose("session") is None
    assert not manager.record(old, 20, True)
    assert manager.register_experiment(config) == 2
    manager.activate("test")
    assert not manager.record(old, 20, True)
    assert manager.choose("session").version == 2


def test_unpromoted_control_changes_never_reach_untreated_users(manager, config):
    config.control.model_name = "unverified-control"
    config.traffic_percent = 0
    manager.register_experiment(config)
    manager.activate("test")
    assert manager.choose("session").variant.model_name == "stable-model"
    manager.mark_stable("test", 2)
    assert manager.choose("session").variant.model_name == "unverified-control"


def test_late_observations_do_not_roll_back_a_new_active_revision(registry, config):
    manager = ExperimentManager(registry, "stable-model")
    policy = RollbackPolicy(min_samples=1, max_error_rate=0)
    manager.register_experiment(config, policy)
    manager.activate("test")
    old = manager.choose("session")
    manager.register_experiment(config, policy)
    manager.activate("test")
    assert not manager.record(old, 20, True)
    assert manager.choose("session").version == 2
    assert not manager.choose("session").is_control


def test_every_control_plane_state_change_is_audited(manager, config):
    manager.set_killed("test", True)
    manager.set_killed("test", False)
    manager.register_experiment(config)
    manager.activate("test")
    manager.mark_stable("test", 1)
    manager.rollback("test")
    manager.deactivate("test")
    manager.unregister_experiment("test")
    assert [event["event"] for event in manager.events("test")] == [
        "registered",
        "activated",
        "killed",
        "resumed",
        "registered",
        "activated",
        "marked_stable",
        "rolled_back",
        "deactivated",
        "deleted",
    ]


@pytest.mark.parametrize(
    "policy",
    [
        {"min_samples": 3, "window_size": 2},
        {"max_error_rate": -1},
        {"max_p95_latency_ms": 0},
        {"max_p95_latency_ms": float("inf")},
    ],
)
def test_invalid_policies_are_rejected(policy):
    with pytest.raises(ValidationError):
        RollbackPolicy(**policy)


@pytest.mark.parametrize("percent", [-1, 101, float("nan")])
def test_invalid_rollout_percentages_are_rejected(registry, config, percent):
    manager = ExperimentManager(registry, "stable-model")
    with pytest.raises(ValueError):
        manager.register_experiment(replace(config, traffic_percent=percent))


@pytest.mark.parametrize("change", ["weight", "duplicate", "template", "variables", "generation"])
def test_invalid_configurations_are_rejected_without_state_changes(registry, config, change):
    manager = ExperimentManager(registry, "stable-model")
    if change == "weight":
        config.variants[0].weight = float("nan")
    elif change == "duplicate":
        config.variants[0].name = "control"
    elif change == "template":
        config.variants[0].template_name = "missing"
    elif change == "variables":
        registry.register(PromptTemplate("candidate", "2.0.0", "{unsupported}", ("unsupported",)))
    else:
        config.variants[0].generation_config = {"unknown_option": 1}
    with pytest.raises(ValueError):
        manager.register_experiment(config)
    assert manager.list_experiments() == {}
