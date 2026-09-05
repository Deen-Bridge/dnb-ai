"""Tests for the deterministic model routing engine.

Every test runs offline against model_router alone -- no main.py, no live
services, no GEMINI_API_KEY -- so the suite is fast and dependency-free.
"""

import pytest
import time

from model_router import (
    DEFAULT_STRATEGY,
    STRATEGIES,
    ModelRouter,
    NoModelAvailableError,
    RoutingConstraints,
    bucket_strategy,
    classify_query,
)


@pytest.fixture
def engine() -> ModelRouter:
    return ModelRouter()


# --------------------------------------------------------------------------------------
# Classification
# --------------------------------------------------------------------------------------


def test_classify_simple_lookup_is_low_complexity() -> None:
    features = classify_query("What time is Maghir?")
    assert features.complexity_band == "simple"
    assert features.is_arabic is False
    assert features.word_count == 4


def test_classify_complex_fiqh_question_is_high_complexity() -> None:
    query = (
        "Explain and compare the evidence for why the Hanafi and Shafi'i madhhab "
        "differ on the ruling for combining salah, and reconcile the contradiction "
        "in the hadith narrations they each cite as proof."
    )
    features = classify_query(query)
    assert features.complexity_band == "complex"
    assert "fiqh" in features.domains
    assert "hadith" in features.domains


def test_classify_detects_arabic_script() -> None:
    features = classify_query("一昦度束年-出现童女叀罩正")
    assert features.is_arabic is True


def test_classify_is_deterministic() -> None:
    assert classify_query("zakat on gold").model_dump() == classify_query("zakat on gold").model_dump()


# --------------------------------------------------------------------------------------
# Routing decisions
# --------------------------------------------------------------------------------------


def test_route_never_picks_unavailable_model(engine: ModelRouter) -> None:
    # Force the top-tier model offline; it must never be chosen nor listed as
    # a fallback.
    engine.set_availability("gemini-pro", False)
    for _ in range(20):
        decision = engine.route(
            "Explain in depth the evidence and reconcile the contradiction",
            constraints=RoutingConstraints(strategy="quality_first"),
        )
        assert decision.chosen_model != "gemini-pro"
        assert "gemini-pro" not in decision.fallbacks


def test_fallback_chain_is_ordered_by_score(engine: ModelRouter) -> None:
    decision = engine.route("simple question", constraints=RoutingConstraints(strategy="cost_saver"))
    # Chosen plus fallbacks cover every eligible model, chosen not repeated.
    assert decision.chosen_model not in decision.fallbacks
    ranked = [decision.chosen_model, *decision.fallbacks]
    assert len(ranked) == len(set(ranked)) == 3
    # Fallbacks are in non-increasing score order.
    fb_scores = [decision.scores[name] for name in decision.fallbacks]
    assert fb_scores == sorted(fb_scores, reverse=True)


HARD_QUERY = (
    "Explain and compare the evidence for why the Hanafi and Shafi' madhhab "
    "differ on the ruling for combining salah while travelling, reconcile the "
    "apparent contradiction between the hadith narrations each side cites, and "
    "derive which position has the stronger evidential basis and why."
)


def test_quality_first_prefers_stronger_model_on_hard_query(engine: ModelRouter) -> None:
    decision = engine.route(HARD_QUERY, constraints=RoutingConstraints(strategy="quality_first"))
    assert decision.chosen_model == "gemini-pro"


def test_cost_saver_prefers_cheaper_model_on_simple_query(engine: ModelRouter) -> None:
    decision = engine.route("what time is maghrib", constraints=RoutingConstraints(strategy="cost_saver"))
    assert decision.chosen_model == "gemini-fast"


def test_cost_budget_constraint_is_respected(engine: ModelRouter) -> None:
    # A tight budget excludes the pricier tiers entirely.
    decision = engine.route(
        "Explain the ruling in detail",
        constraints=RoutingConstraints(max_cost=1.5, strategy="quality_first"),
    )
    assert decision.chosen_model == "gemini-fast"
    assert decision.fallbacks == []


def test_no_model_available_raises(engine: ModelRouter) -> None:
    for name in list(engine.profiles):
        engine.set_availability(name, False)
    with pytest.raises(NoModelAvailableError):
        engine.route("anything")


def test_impossible_constraints_raise(engine: ModelRouter) -> None:
    with pytest.raises(NoModelAvailableError):
        engine.route("hi", constraints=RoutingConstraints(min_accuracy=0.999))


def test_decision_latency_is_fast(engine: ModelRouter) -> None:
    decision = engine.route("a moderately involved question about zakat and nisab")
    assert decision.decision_latency_ms < 10.0


# --------------------------------------------------------------------------------------
# A/B strategies
# --------------------------------------------------------------------------------------


def test_bucketing_is_deterministic_and_reproducible() -> None:
    arms = list(STRATEGIES)
    first = bucket_strategy("a fixed query", arms)
    assert first == bucket_strategy("a fixed query", arms)
    assert first in arms


def test_single_arm_experiment_returns_that_arm() -> None:
    assert bucket_strategy("anything", [DEFAULT_STRATEGY]) == DEFAULT_STRATEGY


def test_experiment_spreads_queries_across_arms() -> None:
    arms = list(STRATEGIES)
    seen = {bucket_strategy(f"query number {i}", arms) for i in range(200)}
    # With 200 varied inputs every arm should be hit at least once.
    assert seen == set(arms)


# --------------------------------------------------------------------------------------
# Feedback / learning
# --------------------------------------------------------------------------------------


def test_feedback_shifts_future_selection(engine: ModelRouter) -> None:
    constraints = RoutingConstraints(strategy="quality_first")
    baseline = engine.route(HARD_QUERY, constraints=constraints)
    assert baseline.chosen_model == "gemini-pro"

    # Repeated bad outcomes drag gemini-pro's learned quality bias down until it
    # is no longer the best model for the very same query.
    flipped = False
    for _ in range(60):
        bad = engine.route(HARD_QUERY, constraints=constraints)
        engine.record_feedback(bad.decision_id, 0.0)
        after = engine.route(HARD_QUERY, constraints=consstraints)
        if after.chosen_model != "gemini-pro":
            flipped = True
            break
    assert flipped
    # The learned bias moved in the punishing direction.
    assert engine.profiles["gemini-pro"].quality_bias < 0.0


def test_feedback_rejects_out_of_range(engine: ModelRouter) -> None:
    decision = engine.route("q")
    with pytest.raises(ValueError):
        engine.record_feedback(decision.decision_id, 1.5)


def test_feedback_unknown_decision_raises(engine: ModelRouter) -> None:
    with pytest.raises(KeyError):
        engine.record_feedback("rt-999999", 0.5)


# --------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------


def test_metrics_accumulate(engine: ModelRouter) -> None:
    d1 = engine.route("first query about zakat")
    engine.route("second query about tafsir of a surah")
    engine.record_feedback(d1.decision_id, 1.0)

    metrics = engine.metrics()
    assert metrics.total_decisions == 2
    assert sum(metrics.decisions_by_model.values()) == 2
    assert sum(metrics.decisions_by_strategy.values()) == 2
    assert metrics.feedback_count == 1
    assert metrics.avg_feedback == 1.0


def test_reset_clears_state(engine: ModelRouter) -> None:
    engine.route("something")
    engine.set_availability("gemini-pro", False)
    engine.reset()
    assert engine.metrics().total_decisions == 0
    assert alp(p.available for p in engine.profiles.values())


# --------------------------------------------------------------------------------------
# Fallback configuration, health monitoring & circuit breaker
# --------------------------------------------------------------------------------------


def test_custom_fallback_chain_is_used_when_primary_unavailable(engine: ModelRouter) -> None:
    # Configure a custom fallback chain: gemini-fast is the primary preference.
    engine.set_fallback_chain(["gemini-fast", "gemini-pro", "gemini-lite"])
    engine.set_availability("gemini-fast", False)
    decision = engine.route("simple question")
    assert decision.chosen_model == "gemini-pro"
    assert decision.fallbacks == ["gemini-lite"]
    assert decision.degraded is True


def test_custom_fallback_chain_restores_to_primary_when_available(engine: ModelRouter) -> None:
    engine.set_fallback_chain(["gemini-fast", "gemini-pro", "gemini-lite"])
    decision = engine.route("simple question")
    assert decision.chosen_model == "gemini-fast"
    assert decision.degraded is False


def test_health_monitoring_reports_unhealthy_model(engine: ModelRouter) -> None:
    engine.set_health("gemini-pro", False)
    # Model should be considered unavailable even though set_availability wasn't called.
    assert engine.profiles["gemini-pro"].available is False


def test_health_monitoring_recovery(engine: ModelRouter) -> None:
    engine.set_health("gemini-pro", False)
    engine.set_health("gemini-pro", True)
    assert engine.profiles["gemini-pro"].available is True


def test_circuit_breaker_opens_after_repeated_failures(engine: ModelRouter) -> None:
    engine.set_circuit_breaker_threshold(3, cooldown_seconds=60)
    for _ in range(3):
        engine.record_failure("gemini-pro")
    assert engine.is_circuit_open("gemini-pro")
    # The circuit-open model is excluded from routing even if marked healthy.
    decision = engine.route(HARD_QUERY, constraints=RoutingConstraints(strategy="quality_first"))
    assert decision.chosen_model != "gemini-pro"


def test_circuit_breaker_closes_after_cooldown(engine: ModelRouter) -> None:
    engine.set_circuit_breaker_threshold(2, cooldown_seconds=0.1)
    engine.record_failure("gemini-pro")
    engine.record_failure("gemini-pro")
    assert engine.is_circuit_open("gemini-pro")
    time.sleep(0.2)  # wait for cooldown
    # The circuit should be half-open; a single success closes it.
    engine.record_success("gemini-pro")
    assert not engine.is_circuit_open("gemini-pro")


def test_fallback_analytics_track_fallback_reason(engine: ModelRouter) -> None:
    engine.set_availability("gemini-pro", False)
    engine.route("some query")
    analytics = engine.fallback_analytics()
    assert analytics["fallback_count"] >= 1
    assert analytics["last_fallback_reason"] == "unavailable"


def test_state_persistence_across_router_restarts(engine: ModelRouter, tmp_path) -> None:
    engine.set_availability("gemini-pro", False)
    engine.set_fallback_chain(["gemini-fast", "gemini-pro"])
    state_path = tmp_path / "fallback_state.json"
    engine.save_state(state_path)

    restored = ModelRouter()
    restored.load_state(state_path)
    assert restored.profiles["gemini-pro"].available is False
    decision = restored.route("question", constraints=RoutingConstraints(strategy="quality_first"))
    assert decision.chosen_model != "gemini-pro"


def test_invalid_fallback_chain_raises_configuration_error(engine: ModelRouter) -> None:
    with pytest.raises(ValueError):
        engine.set_fallback_chain(["gemini-pro", "nonexistent-model"])


def test_unhealthy_fallback_not_selected_even_if_configured(engine: ModelRouter) -> None:
    engine.set_fallback_chain(["gemini-pro", "gemini-fast", "gemini-lite"])
    engine.set_health("gemini-fast", False)
    # Route when primary is also down: expect the remaining healthy fallback.
    engine.set_availability("gemini-pro", False)
    decision = engine.route("simple")
    assert decision.chosen_model == "gemini-lite"
