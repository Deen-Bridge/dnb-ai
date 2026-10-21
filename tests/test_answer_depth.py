"""Tests for answer depth customization (#221, #263) — no live API calls."""

import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import answer_depth
from answer_depth import (
    DEPTH_CONFIGS,
    DEPTH_SCALE,
    DepthAdapter,
    DepthLevel,
    DepthSessionStore,
    StructuredAnswer,
    UserDepthPreferences,
    UserPreferencesStore,
    build_depth_indicator,
    build_depth_prompt_block,
    get_answer_config,
    normalize_depth,
    step_depth,
)


@pytest.fixture
def store() -> DepthSessionStore:
    return DepthSessionStore()


@pytest.fixture
def temp_preferences(tmp_path) -> UserPreferencesStore:
    return UserPreferencesStore(str(tmp_path / "depth_preferences.json"))


@pytest.fixture
def client(monkeypatch, tmp_path) -> TestClient:
    """API client with isolated in-memory threads and preferences."""
    monkeypatch.setattr(answer_depth, "_session_store", DepthSessionStore())
    monkeypatch.setattr(answer_depth, "_prefs_store", UserPreferencesStore(str(tmp_path / "prefs.json")))
    app = FastAPI()
    app.include_router(answer_depth.router)
    return TestClient(app)


# ---------------------------------------------------------------------------
# Scale and normalization
# ---------------------------------------------------------------------------


def test_scale_has_five_ordered_levels() -> None:
    assert [level.value for level in DEPTH_SCALE] == [
        "eli5",
        "beginner",
        "intermediate",
        "advanced",
        "scholarly",
    ]
    assert all(level in DEPTH_CONFIGS for level in DEPTH_SCALE)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("eli5", DepthLevel.ELI5),
        ("BEGINNER", DepthLevel.BEGINNER),
        (" detailed ", DepthLevel.ADVANCED),
        ("brief", DepthLevel.BEGINNER),
        ("standard", DepthLevel.INTERMEDIATE),
        ("expert", DepthLevel.SCHOLARLY),
    ],
)
def test_normalize_depth_accepts_canonical_and_legacy_names(raw: str, expected: DepthLevel) -> None:
    assert normalize_depth(raw) is expected


def test_normalize_depth_returns_none_for_empty_and_none() -> None:
    assert normalize_depth(None) is None
    assert normalize_depth("   ") is None


def test_normalize_depth_rejects_unknown_level() -> None:
    with pytest.raises(ValueError, match="eli5"):
        normalize_depth("postgraduate")


def test_step_depth_is_clamped_at_both_ends() -> None:
    assert step_depth(DepthLevel.ELI5, -1) is DepthLevel.ELI5
    assert step_depth(DepthLevel.SCHOLARLY, 1) is DepthLevel.SCHOLARLY
    assert step_depth(DepthLevel.INTERMEDIATE, -1) is DepthLevel.BEGINNER
    assert step_depth(DepthLevel.INTERMEDIATE, 1) is DepthLevel.ADVANCED


# ---------------------------------------------------------------------------
# Configuration and prompt instructions
# ---------------------------------------------------------------------------


def test_depth_configs_become_richer_as_level_rises() -> None:
    previous = DEPTH_CONFIGS[DEPTH_SCALE[0]]
    for level in DEPTH_SCALE[1:]:
        current = DEPTH_CONFIGS[level]
        assert current.max_length >= previous.max_length
        assert current.citation_density >= previous.citation_density
        assert current.terminology_complexity >= previous.terminology_complexity
        previous = current


def test_eli5_config_prefers_analogies_and_no_citations() -> None:
    config = DEPTH_CONFIGS[DepthLevel.ELI5]
    assert config.use_analogies is True
    assert config.citation_density == 0.0
    assert config.examples_per_concept >= 2


def test_prompt_block_states_depth_and_constraints() -> None:
    block = build_depth_prompt_block(DEPTH_CONFIGS[DepthLevel.ELI5])
    assert "Explain like I'm 5" in block
    assert "analogy" in block
    assert "Do not add citations" in block
    # Depth must never license changing the actual ruling.
    assert "ruling" in block


def test_advanced_prompt_requests_arabic_and_ikhtilaf() -> None:
    block = build_depth_prompt_block(DEPTH_CONFIGS[DepthLevel.ADVANCED])
    assert "Arabic" in block
    assert "ikhtilaf" in block


def test_indicator_exposes_control_state() -> None:
    indicator = build_depth_indicator(DepthLevel.ELI5)
    payload = indicator.to_dict()
    assert payload["position"] == 1
    assert payload["total"] == 5
    assert payload["can_go_simpler"] is False
    assert payload["can_go_deeper"] is True
    assert payload["visual"]["badge"]
    # JSON-safe for the response model.
    assert json.loads(json.dumps(payload))["level"] == "eli5"

    top = build_depth_indicator(DepthLevel.SCHOLARLY).to_dict()
    assert top["can_go_simpler"] is True
    assert top["can_go_deeper"] is False


# ---------------------------------------------------------------------------
# Preferences
# ---------------------------------------------------------------------------


def test_preferences_default_to_intermediate(temp_preferences: UserPreferencesStore) -> None:
    assert temp_preferences.has("user-1") is False
    prefs = temp_preferences.get("user-1")
    assert prefs.default_level is DepthLevel.INTERMEDIATE
    # Reading a default must not create a stored preference.
    assert temp_preferences.has("user-1") is False


def test_preferences_round_trip_through_disk(temp_preferences: UserPreferencesStore, tmp_path) -> None:
    temp_preferences.set_default_level("user-1", DepthLevel.ADVANCED)
    temp_preferences.set_topic_level("user-1", "zakat", DepthLevel.ELI5)

    reloaded = UserPreferencesStore(str(tmp_path / "depth_preferences.json"))
    prefs = reloaded.get("user-1")
    assert prefs.default_level is DepthLevel.ADVANCED
    assert prefs.per_topic_levels["zakat"] is DepthLevel.ELI5
    assert reloaded.has("user-1") is True


def test_preferences_accept_legacy_stored_levels() -> None:
    prefs = UserDepthPreferences.from_dict(
        {"user_id": "legacy", "default_level": "detailed", "per_topic_levels": {"hajj": "brief"}}
    )
    assert prefs.default_level is DepthLevel.ADVANCED
    assert prefs.per_topic_levels["hajj"] is DepthLevel.BEGINNER


def test_effective_level_priority(temp_preferences: UserPreferencesStore) -> None:
    temp_preferences.set_default_level("user-1", DepthLevel.BEGINNER)
    temp_preferences.set_topic_level("user-1", "zakat", DepthLevel.ADVANCED)

    assert temp_preferences.effective_level("user-1", requested=DepthLevel.ELI5) is DepthLevel.ELI5
    assert temp_preferences.effective_level("user-1", topic="zakat") is DepthLevel.ADVANCED
    assert temp_preferences.effective_level("user-1") is DepthLevel.BEGINNER


def test_get_answer_config_applies_saved_preferences(monkeypatch, temp_preferences) -> None:
    monkeypatch.setattr(answer_depth, "_prefs_store", temp_preferences)
    monkeypatch.setattr(answer_depth, "_adapter", DepthAdapter())
    temp_preferences.set_default_level("user-1", DepthLevel.SCHOLARLY)

    config = get_answer_config(user_id="user-1")
    assert config.level is DepthLevel.SCHOLARLY
    assert config.include_madhhab_comparison is True


def test_adapter_compresses_to_a_simpler_structure() -> None:
    full = StructuredAnswer(
        summary="Zakat is obligatory charity.",
        main_content="Long explanation.",
        sections=[
            answer_depth.AnswerSection(id="1", title="Evidence", content="...", level="primary"),
            answer_depth.AnswerSection(id="2", title="Detail", content="...", level="tertiary"),
        ],
        citations=[{"ref": f"c{i}"} for i in range(8)],
        metadata={"depth_level": "scholarly"},
    )
    compressed = DepthAdapter().compress_answer(full, DepthLevel.BEGINNER)
    assert compressed.sections == []
    assert len(compressed.citations) == 1
    assert compressed.metadata["compressed_from"] == "scholarly"


# ---------------------------------------------------------------------------
# Interactive controls
# ---------------------------------------------------------------------------


def test_record_original_keeps_delivered_answer(store: DepthSessionStore) -> None:
    thread = store.record_original("chat-1", "msg-1", "Original answer", DepthLevel.ADVANCED)
    assert thread.original_answer == "Original answer"
    assert thread.current_level is DepthLevel.ADVANCED
    assert thread.variations[0].answer == "Original answer"
    assert store.get("chat-1", "msg-1") is thread


def test_simpler_moves_one_step_and_preserves_original(store: DepthSessionStore) -> None:
    store.record_original("chat-1", "msg-1", "Original answer", DepthLevel.ADVANCED)
    adjustment = store.adjust("chat-1", "simpler", message_id="msg-1")

    assert adjustment.previous_level is DepthLevel.ADVANCED
    assert adjustment.current_level is DepthLevel.INTERMEDIATE
    assert adjustment.at_limit is False
    assert adjustment.original_answer == "Original answer"
    assert adjustment.indicator.position == 3
    assert "follow_up_prompt" in adjustment.to_dict()
    assert adjustment.to_dict()["regenerate_required"] is True
    pending = [v for v in adjustment.variations if v.pending]
    assert pending and pending[-1].level is DepthLevel.INTERMEDIATE


def test_adjust_ignores_previous_depth_when_stepping_multiple_times(store: DepthSessionStore) -> None:
    store.record_original("chat-1", "msg-1", "Original", DepthLevel.ADVANCED)
    store.adjust("chat-1", "simpler", message_id="msg-1")
    second = store.adjust("chat-1", "simpler", message_id="msg-1")
    third = store.adjust("chat-1", "simpler", message_id="msg-1")
    fourth = store.adjust("chat-1", "simpler", message_id="msg-1")

    assert second.current_level is DepthLevel.BEGINNER
    assert third.current_level is DepthLevel.ELI5
    assert third.at_limit is False
    assert fourth.at_limit is True
    assert fourth.current_level is DepthLevel.ELI5


def test_deeper_at_scholarly_reports_limit(store: DepthSessionStore) -> None:
    store.record_original("chat-1", None, "Original", DepthLevel.SCHOLARLY)
    adjustment = store.adjust("chat-1", "deeper")
    assert adjustment.at_limit is True
    assert adjustment.current_level is DepthLevel.SCHOLARLY


def test_set_level_explicitly(store: DepthSessionStore) -> None:
    store.record_original("chat-1", None, "Original", DepthLevel.BEGINNER)
    adjustment = store.adjust("chat-1", "set", level=DepthLevel.SCHOLARLY)
    assert adjustment.current_level is DepthLevel.SCHOLARLY
    assert adjustment.direction == "set"


def test_set_requires_a_level(store: DepthSessionStore) -> None:
    store.record_original("chat-1", None, "Original", DepthLevel.BEGINNER)
    with pytest.raises(ValueError, match="requires a level"):
        store.adjust("chat-1", "set")


def test_adjust_unknown_chat_raises_key_error(store: DepthSessionStore) -> None:
    with pytest.raises(KeyError):
        store.adjust("missing-chat", "simpler")


def test_record_variation_fulfils_pending_entry(store: DepthSessionStore) -> None:
    store.record_original("chat-1", "msg-1", "Original", DepthLevel.ADVANCED)
    store.adjust("chat-1", "simpler", message_id="msg-1")

    thread = store.record_variation("chat-1", DepthLevel.INTERMEDIATE, "Simpler answer", message_id="msg-1")
    assert thread.current_level is DepthLevel.INTERMEDIATE
    assert [v.pending for v in thread.variations] == [False, False]
    assert thread.variations[-1].answer == "Simpler answer"


def test_store_evicts_oldest_threads() -> None:
    bounded = DepthSessionStore(max_threads=2)
    bounded.record_original("chat-1", None, "a", DepthLevel.ELI5)
    bounded.record_original("chat-2", None, "b", DepthLevel.ELI5)
    bounded.record_original("chat-3", None, "c", DepthLevel.ELI5)
    assert len(bounded._threads) == 2  # noqa: SLF001 - bounded-store contract


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------


def test_levels_endpoint_lists_scale(client: TestClient) -> None:
    response = client.get("/depth/levels")
    assert response.status_code == 200
    payload = response.json()
    assert [item["level"] for item in payload["levels"]] == [level.value for level in DEPTH_SCALE]
    assert payload["aliases"]["brief"] == "beginner"


def test_register_and_adjust_through_api(client: TestClient) -> None:
    register = client.post(
        "/depth/register",
        json={"chat_id": "c1", "message_id": "m1", "answer": "Original", "level": "advanced"},
    )
    assert register.status_code == 200
    assert register.json()["thread"]["original_level"] == "advanced"

    adjust = client.post("/depth/adjust", json={"chat_id": "c1", "message_id": "m1", "direction": "simpler"})
    assert adjust.status_code == 200
    payload = adjust.json()
    assert payload["current_level"] == "intermediate"
    assert payload["original_answer"] == "Original"
    assert payload["depth"]["position"] == 3
    assert payload["follow_up_prompt"]


def test_adjust_unknown_chat_returns_404(client: TestClient) -> None:
    response = client.post("/depth/adjust", json={"chat_id": "nope", "direction": "deeper"})
    assert response.status_code == 404


def test_adjust_rejects_bad_direction(client: TestClient) -> None:
    client.post("/depth/register", json={"chat_id": "c1", "answer": "Original"})
    response = client.post("/depth/adjust", json={"chat_id": "c1", "direction": "sideways"})
    assert response.status_code == 400


def test_adjust_set_requires_level(client: TestClient) -> None:
    client.post("/depth/register", json={"chat_id": "c1", "answer": "Original"})
    response = client.post("/depth/adjust", json={"chat_id": "c1", "direction": "set"})
    assert response.status_code == 400


def test_adjust_rejects_unknown_level(client: TestClient) -> None:
    client.post("/depth/register", json={"chat_id": "c1", "answer": "Original"})
    response = client.post("/depth/adjust", json={"chat_id": "c1", "direction": "set", "level": "doctorate"})
    assert response.status_code == 400


def test_adjust_can_persist_the_new_level(client: TestClient) -> None:
    client.post("/depth/register", json={"chat_id": "c1", "answer": "Original", "level": "advanced"})
    response = client.post(
        "/depth/adjust",
        json={
            "chat_id": "c1",
            "direction": "simpler",
            "user_id": "u1",
            "persist_preference": True,
        },
    )
    assert response.status_code == 200
    prefs = client.get("/depth/preferences/u1").json()
    assert prefs["preferences"]["default_level"] == "intermediate"


def test_preferences_endpoints(client: TestClient) -> None:
    update = client.put("/depth/preferences/u1", json={"default_level": "eli5"})
    assert update.status_code == 200
    assert update.json()["effective"]["level"] == "eli5"

    topic = client.put("/depth/preferences/u1", json={"topic": "zakat", "level": "scholarly"})
    assert topic.status_code == 200
    assert topic.json()["preferences"]["per_topic_levels"]["zakat"] == "scholarly"

    fetched = client.get("/depth/preferences/u1").json()
    assert fetched["preferences"]["default_level"] == "eli5"


def test_preferences_reject_unknown_level(client: TestClient) -> None:
    response = client.put("/depth/preferences/u1", json={"default_level": "wizard"})
    assert response.status_code == 400


def test_depth_history_endpoint(client: TestClient) -> None:
    assert client.get("/depth/response/c1").status_code == 404
    client.post("/depth/register", json={"chat_id": "c1", "answer": "Original", "level": "beginner"})
    client.post("/depth/adjust", json={"chat_id": "c1", "direction": "deeper"})

    history = client.get("/depth/response/c1")
    assert history.status_code == 200
    thread = history.json()["thread"]
    assert thread["original_answer"] == "Original"
    assert thread["current_level"] == "intermediate"
    assert thread["original_level"] == "beginner"
    assert thread["variations"][0]["direction"] == "original"
