"""Tests for adaptive teaching (#251) — deterministic, no live API calls."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import adaptive_teaching
from adaptive_teaching import (
    AdaptiveTeachingEngine,
    ComprehensionAssessment,
    Interaction,
    LearningProfile,
    LearningProfileStore,
    Pace,
    TeachingStyle,
    analyze_comprehension,
    detect_style_hint,
)
from answer_depth import DepthLevel


@pytest.fixture
def engine() -> AdaptiveTeachingEngine:
    return AdaptiveTeachingEngine()


@pytest.fixture
def profile() -> LearningProfile:
    return LearningProfile(user_id="learner-1")


@pytest.fixture
def client(monkeypatch, tmp_path) -> TestClient:
    monkeypatch.setattr(adaptive_teaching, "_engine", AdaptiveTeachingEngine())
    monkeypatch.setattr(
        adaptive_teaching,
        "_profile_store",
        LearningProfileStore(str(tmp_path / "profiles.json")),
    )
    app = FastAPI()
    app.include_router(adaptive_teaching.router)
    return TestClient(app)


# ---------------------------------------------------------------------------
# Comprehension analysis
# ---------------------------------------------------------------------------


def test_neutral_message_scores_neutral() -> None:
    assessment = analyze_comprehension("What is the ruling on praying while travelling?")
    assert assessment.score == pytest.approx(0.5)
    assert assessment.is_confused is False
    assert assessment.needs_examples is False
    assert assessment.confidence < 0.3


def test_confusion_wording_lowers_score() -> None:
    assessment = analyze_comprehension("I don't understand, can you explain again?")
    assert assessment.score < 0.3
    assert assessment.is_confused is True
    assert assessment.needs_examples is True
    assert "confusion" in assessment.signals
    assert assessment.confidence > 0.3


def test_mastery_wording_raises_score() -> None:
    assessment = analyze_comprehension("Got it, makes sense, thank you")
    assert assessment.score > 0.8
    assert assessment.is_confused is False
    assert "mastery" in assessment.signals


def test_advanced_vocabulary_raises_score_slightly() -> None:
    neutral = analyze_comprehension("What is the ruling on this?")
    advanced = analyze_comprehension("What is the evidence and the difference between the madhhab positions?")
    assert advanced.score > neutral.score
    assert "advanced_interest" in advanced.signals


def test_feedback_and_depth_requests_shape_the_score() -> None:
    assert analyze_comprehension(feedback="down").score < 0.3
    assert analyze_comprehension(feedback="up").score > 0.7
    assert analyze_comprehension(depth_request="simpler").score < 0.35
    assert analyze_comprehension(depth_request="deeper").score > 0.6


def test_assessment_serializes_to_json_safe_dict() -> None:
    payload = analyze_comprehension("I'm confused").to_dict()
    assert set(payload) == {
        "score",
        "confidence",
        "signals",
        "is_confused",
        "needs_examples",
        "style_hint",
        "reasons",
    }


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("Can you show me a diagram of the steps?", TeachingStyle.VISUAL),
        ("Why is that the ruling? What is the proof?", TeachingStyle.ANALYTICAL),
        ("Can you give me a practical example?", TeachingStyle.EXAMPLE_BASED),
        ("What is zakat?", None),
    ],
)
def test_detect_style_hint(message: str, expected: TeachingStyle | None) -> None:
    assert detect_style_hint(message) is expected


# ---------------------------------------------------------------------------
# Adaptation
# ---------------------------------------------------------------------------


def test_confusion_simplifies_depth_and_adds_examples(engine: AdaptiveTeachingEngine, profile: LearningProfile) -> None:
    result = engine.observe(profile, Interaction(message="I don't understand this at all"))

    assert result.profile.level is DepthLevel.BEGINNER
    assert result.profile.examples_multiplier > 1.0
    assert result.profile.pace is Pace.SLOW
    assert any("confusion" in item for item in result.adjustments)
    assert result.recommendation.examples >= 2


def test_repeated_success_increases_depth(engine: AdaptiveTeachingEngine, profile: LearningProfile) -> None:
    engine.observe(profile, Interaction(message="Got it, makes sense"))
    result = engine.observe(profile, Interaction(message="Understood, that's clear now"))

    assert result.profile.consecutive_successes >= 2
    assert result.profile.level is DepthLevel.ADVANCED
    assert any("increased depth" in item for item in result.adjustments)


def test_explicit_simpler_request_moves_exactly_one_level(
    engine: AdaptiveTeachingEngine, profile: LearningProfile
) -> None:
    result = engine.observe(profile, Interaction(message="Can you explain that simpler?", depth_request="simpler"))
    assert result.profile.level is DepthLevel.BEGINNER


def test_explicit_deeper_request_moves_one_level(engine: AdaptiveTeachingEngine, profile: LearningProfile) -> None:
    result = engine.observe(profile, Interaction(message="Go deeper on the evidence", depth_request="deeper"))
    assert result.profile.level is DepthLevel.ADVANCED


def test_depth_requests_are_clamped_to_the_scale(engine: AdaptiveTeachingEngine) -> None:
    profile = LearningProfile(user_id="top", level=DepthLevel.SCHOLARLY)
    deeper = engine.observe(profile, Interaction(depth_request="deeper"))
    assert deeper.profile.level is DepthLevel.SCHOLARLY

    low = LearningProfile(user_id="low", level=DepthLevel.ELI5)
    simpler = engine.observe(low, Interaction(depth_request="simpler"))
    assert simpler.profile.level is DepthLevel.ELI5


def test_positive_feedback_tracks_style_effectiveness(engine: AdaptiveTeachingEngine, profile: LearningProfile) -> None:
    result = engine.observe(
        profile, Interaction(message="That diagram helped", style=TeachingStyle.VISUAL, feedback="up")
    )

    assert result.profile.style_scores["visual"] > 0.5
    assert result.profile.preferred_style is TeachingStyle.VISUAL
    assert result.recommendation.style is TeachingStyle.VISUAL


def test_negative_feedback_moves_away_from_an_ineffective_style(
    engine: AdaptiveTeachingEngine,
) -> None:
    profile = LearningProfile(user_id="u", preferred_style=TeachingStyle.VISUAL)
    profile.style_scores["visual"] = 0.4
    engine.observe(profile, Interaction(style=TeachingStyle.VISUAL, feedback="down", message="Still unclear"))
    assert profile.preferred_style is TeachingStyle.BALANCED


def test_detected_style_hint_is_applied(engine: AdaptiveTeachingEngine, profile: LearningProfile) -> None:
    result = engine.observe(profile, Interaction(message="Can you show me a diagram or chart?"))
    assert result.profile.preferred_style is TeachingStyle.VISUAL


def test_pacing_follows_learning_speed(engine: AdaptiveTeachingEngine, profile: LearningProfile) -> None:
    profile.comprehension = 0.8
    profile.avg_response_seconds = 5
    profile.level = DepthLevel.ADVANCED
    result = engine.observe(profile, Interaction(message="Got it, makes sense", elapsed_seconds=5))
    assert result.profile.pace is Pace.FAST

    result = engine.observe(profile, Interaction(message="I don't understand", elapsed_seconds=90))
    assert result.profile.pace is Pace.SLOW


def test_topic_progress_is_tracked(engine: AdaptiveTeachingEngine, profile: LearningProfile) -> None:
    result = engine.observe(profile, Interaction(message="Got it", topic="zakat"))

    assert result.profile.topic_attempts["zakat"] == 1
    assert result.profile.topic_levels["zakat"] is DepthLevel.INTERMEDIATE
    assert 0.0 <= result.profile.topic_comprehension["zakat"] <= 1.0


def test_recommendation_switches_from_analysis_to_examples_when_comprehension_is_low(
    engine: AdaptiveTeachingEngine,
) -> None:
    profile = LearningProfile(user_id="u", comprehension=0.2, preferred_style=TeachingStyle.ANALYTICAL)
    recommendation = engine.recommend(profile)
    assert recommendation.style is TeachingStyle.EXAMPLE_BASED
    assert recommendation.include_analogies is True


def test_recommendation_is_topic_aware(engine: AdaptiveTeachingEngine) -> None:
    profile = LearningProfile(user_id="u")
    profile.topic_levels["inheritance"] = DepthLevel.SCHOLARLY
    recommendation = engine.recommend(profile, topic="inheritance")
    assert recommendation.level is DepthLevel.SCHOLARLY
    assert recommendation.indicator["position"] == 5
    assert "madhhab" in recommendation.prompt_block.lower()


def test_recommendation_does_not_mutate_the_profile(engine: AdaptiveTeachingEngine) -> None:
    profile = LearningProfile(user_id="u")
    before = profile.to_dict()
    engine.recommend(profile, style=TeachingStyle.VISUAL)
    assert profile.to_dict() == before


def test_observation_result_serializes(engine: AdaptiveTeachingEngine, profile: LearningProfile) -> None:
    payload = engine.observe(profile, Interaction(message="I'm confused", topic="salah")).to_dict()
    assert payload["assessment"]["is_confused"] is True
    assert payload["profile"]["user_id"] == "learner-1"
    assert payload["recommendation"]["depth"]["level"] == payload["profile"]["level"]
    assert isinstance(payload["adjustments"], list)


# ---------------------------------------------------------------------------
# Profile persistence
# ---------------------------------------------------------------------------


def test_profile_round_trips_through_dict() -> None:
    profile = LearningProfile(user_id="u", level=DepthLevel.ADVANCED, comprehension=0.72, pace=Pace.FAST)
    profile.topic_levels["zakat"] = DepthLevel.SCHOLARLY
    profile.style_scores["visual"] = 0.8

    restored = LearningProfile.from_dict(profile.to_dict())
    assert restored.level is DepthLevel.ADVANCED
    assert restored.pace is Pace.FAST
    assert restored.topic_levels["zakat"] is DepthLevel.SCHOLARLY
    assert restored.style_scores["visual"] == pytest.approx(0.8)


def test_profile_from_dict_tolerates_stale_values() -> None:
    restored = LearningProfile.from_dict(
        {
            "user_id": "legacy",
            "level": "wizard",
            "preferred_style": "telepathic",
            "pace": "warp",
            "style_scores": {"visual": 2.0, "unknown": 1.0},
            "examples_multiplier": 99,
            "topic_levels": {"zakat": "brief"},
        }
    )
    assert restored.level is DepthLevel.INTERMEDIATE
    assert restored.preferred_style is TeachingStyle.BALANCED
    assert restored.pace is Pace.STEADY
    assert restored.style_scores["visual"] == 1.0
    assert "unknown" not in restored.style_scores
    assert restored.examples_multiplier == 3.0
    assert restored.topic_levels["zakat"] is DepthLevel.BEGINNER


def test_store_persists_profiles(tmp_path) -> None:
    path = str(tmp_path / "profiles.json")
    store = LearningProfileStore(path)
    profile = store.get("u1")
    profile.level = DepthLevel.ELI5
    store.save(profile)

    reloaded = LearningProfileStore(path)
    assert reloaded.get("u1").level is DepthLevel.ELI5


def test_store_reset(tmp_path) -> None:
    store = LearningProfileStore(str(tmp_path / "profiles.json"))
    store.save(store.get("u1"))
    assert store.reset("u1") is True
    assert store.reset("u1") is False


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------


def test_styles_endpoint(client: TestClient) -> None:
    response = client.get("/adaptive/styles")
    assert response.status_code == 200
    payload = response.json()
    assert {item["style"] for item in payload["styles"]} == {
        "balanced",
        "visual",
        "analytical",
        "example_based",
    }
    assert len(payload["paces"]) == 3
    assert payload["depths"][0] == "eli5"


def test_profile_endpoint_returns_recommendation(client: TestClient) -> None:
    response = client.get("/adaptive/profile/learner")
    assert response.status_code == 200
    payload = response.json()
    assert payload["profile"]["level"] == "intermediate"
    assert payload["recommendation"]["depth"]["level"] == "intermediate"


def test_observe_endpoint_adapts_to_confusion(client: TestClient) -> None:
    response = client.post(
        "/adaptive/observe",
        json={
            "user_id": "learner",
            "message": "I don't understand, explain again",
            "topic": "zakat",
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["assessment"]["is_confused"] is True
    assert payload["profile"]["level"] == "beginner"
    assert payload["recommendation"]["examples"] >= 2
    assert any("confusion" in item for item in payload["adjustments"])


def test_observe_endpoint_accepts_explicit_depth_request(client: TestClient) -> None:
    response = client.post(
        "/adaptive/observe",
        json={"user_id": "learner", "message": "more detail please", "depth_request": "deeper"},
    )
    assert response.status_code == 200
    assert response.json()["profile"]["level"] == "advanced"


def test_observe_endpoint_validates_inputs(client: TestClient) -> None:
    assert client.post("/adaptive/observe", json={"user_id": "u", "feedback": "meh"}).status_code == 422
    assert client.post("/adaptive/observe", json={"user_id": "u", "depth_request": "sideways"}).status_code == 422
    assert client.post("/adaptive/observe", json={"user_id": "u", "style": "telepathic"}).status_code == 400


def test_profile_update_endpoint(client: TestClient) -> None:
    response = client.put("/adaptive/profile/learner", json={"style": "visual", "level": "advanced"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["profile"]["preferred_style"] == "visual"
    assert payload["recommendation"]["level"] == "advanced"

    assert client.put("/adaptive/profile/learner", json={"style": "psychic"}).status_code == 400
    assert client.put("/adaptive/profile/learner", json={"level": "doctorate"}).status_code == 400
    assert client.put("/adaptive/profile/learner", json={"pace": "warp"}).status_code == 400


def test_recommend_endpoint_without_observation(client: TestClient) -> None:
    client.put("/adaptive/profile/learner", json={"level": "eli5"})
    response = client.post("/adaptive/recommend", json={"user_id": "learner"})
    assert response.status_code == 200
    assert response.json()["recommendation"]["level"] == "eli5"


def test_reset_endpoint(client: TestClient) -> None:
    client.get("/adaptive/profile/learner")
    response = client.delete("/adaptive/profile/learner")
    assert response.status_code == 200
    assert response.json() == {"user_id": "learner", "reset": True}


def test_assessment_dataclass_exposes_reasons() -> None:
    assessment = analyze_comprehension("Still confused about the evidence", feedback="down")
    assert isinstance(assessment, ComprehensionAssessment)
    assert assessment.reasons
